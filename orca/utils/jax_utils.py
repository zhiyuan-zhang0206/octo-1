from copy import deepcopy
import io
import logging
import os
import pickle
from typing import Any, Callable, Sequence, Union

import jax
from jax._src import xla_bridge as xb
from jax.experimental import multihost_utils
from jax.experimental.compilation_cache import compilation_cache
import jax.numpy as jnp
from jax.stages import Compiled
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import chlo, stablehlo
import numpy as np


def host_broadcast_str(x: str) -> str:
    """Broadcast_one_to_all, but with a string. Strings should all be the same length."""
    multihost_utils.assert_equal(
        len(x), f"String lengths are not equal: got {len(x)} for {jax.process_index()}"
    )
    encoded = np.array([ord(c) for c in x], dtype=np.uint8)
    encoded = multihost_utils.broadcast_one_to_all(encoded)
    return "".join([chr(u) for u in encoded])


def shard_along_axis(x: Any, devices: Sequence[jax.Device], axis: int = 0) -> jax.Array:
    """Shard a PyTree of arrays along a given axis, putting them on device in
    the process. Works in multi-host setting as long as PyTrees are equal on all
    hosts."""
    sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(devices, "x"),
        jax.sharding.PartitionSpec(*([None] * axis + ["x"])),
    )
    x = jax.tree_map(jnp.array, x)
    return jax.tree_map(
        lambda arr: jax.make_array_from_callback(
            arr.shape, sharding, lambda index: arr[index]
        ),
        x,
    )


def replicate(x: Any, devices: Sequence[jax.Device]) -> jax.Array:
    """Replicate a PyTree of arrays across devices. Works in multi-host setting
    as long as PyTrees are equal on all hosts."""
    sharding = jax.sharding.PositionalSharding(devices).replicate()
    x = jax.tree_map(jnp.array, x)
    return jax.tree_map(
        lambda arr: jax.make_array_from_callback(
            arr.shape, sharding, lambda index: arr[index]
        ),
        x,
    )


def initialize_compilation_cache(
    cache_dir=os.path.expanduser("~/.jax_compilation_cache"),
):
    """Initializes the Jax persistent compilation cache."""
    compilation_cache.initialize_cache(cache_dir)
    for logger in [logging.getLogger(name) for name in logging.root.manager.loggerDict]:
        logger.addFilter(
            lambda record: "Not writing persistent cache entry for"
            not in record.getMessage()
        )


def serialize_jax_fn(fn: Callable, *args, **kwargs) -> bytes:
    """
    Seralizes a Jax function using StableHLO and pickle. Only supports trivial
    shardings (no cross-device communication).
    """
    lowered = jax.jit(fn, backend="cpu").lower(*args, **kwargs)

    output = io.BytesIO()
    lowered.compiler_ir("stablehlo").operation.write_bytecode(file=output)
    hlo = output.getvalue()

    n_invals = len(lowered._lowering.compile_args["in_shardings"])
    n_outvals = len(lowered._lowering.compile_args["out_shardings"])

    objs_to_skip = [lowered._lowering._hlo]
    for k in ["backend", "in_shardings", "out_shardings", "device_assignment"]:
        objs_to_skip.append(lowered._lowering.compile_args[k])
    lowered = deepcopy(lowered, {id(obj): None for obj in objs_to_skip})

    return pickle.dumps((lowered, hlo, n_invals, n_outvals))


def deserialize_jax_fn(
    serialized: Union[bytes, str],
    device: jax.Device = jax.devices()[0],
) -> Compiled:
    """
    Deserializes and compiles a Jax function serialized using
    `serialize_jax_fn`. Forces computation onto a single device.
    """
    if isinstance(serialized, str):
        with open(serialized, "rb") as f:
            serialized = f.read()
    lowered, hlo, n_invals, n_outvals = pickle.loads(serialized)

    with ir.Context() as context:
        stablehlo.register_dialect(context)
        chlo.register_chlo_dialect(context)
        hlo = ir.Module.parse(hlo)

    lowered._lowering._hlo = hlo

    lowered._lowering.compile_args["backend"] = xb.get_device_backend(device)
    lowered._lowering.compile_args["device_assignment"] = [device]
    lowered._lowering.compile_args["in_shardings"] = [
        jax.sharding.PositionalSharding([device])
    ] * n_invals
    lowered._lowering.compile_args["out_shardings"] = [
        jax.sharding.PositionalSharding([device])
    ] * n_outvals

    return lowered.compile()
