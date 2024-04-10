#!/bin/bash
#SBATCH -J hello              # 作业名是 test
#SBATCH -p DGX               # 提交到 cpu 分区
#SBATCH -N 1                    # 使用一个节点
#SBATCH --qos=lv0b           
#SBATCH --cpus-per-task=1    # 每个进程占用一个 cpu 核心
#SBATCH -t 5:00              # 任务最大运行时间是 5 分钟
#SBATCH -o test.out          # 将屏幕的输出结果保存到当前文件夹的 test.out

python hello.py                      # 执行我的 ./hello 程序