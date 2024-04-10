import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'




from octo.model.octo_model import OctoModel

model = OctoModel.load_pretrained("/scratch/zhangzhiyuan/octo/octo-base")
# model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small")


from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
# download one example BridgeV2 image
IMAGE_URL = "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_12.jpg"
img = np.array(Image.open(requests.get(IMAGE_URL, stream=True).raw).resize((256, 256)))
plt.imshow(img)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# create obs & task dict, run inference
import jax
# add batch + time horizon 1
img = img[np.newaxis,np.newaxis,...]
observation = {"image_primary": img, "pad_mask": np.array([[True]])}
task = model.create_tasks(texts=["pick up the fork"])
action = model.sample_actions(observation, task, rng=jax.random.PRNGKey(0))
print(action.shape)
print(action)   # [batch, action_chunk, action_dim]


# salloc -N 1 --cpus-per-task=4 -t 10:00 -p HGX --qos=lv0b --account=research --mem=32GB --gres=gpu:1










