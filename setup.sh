docker run --name siqi_slime_opsd --gpus all --ipc=host -p 30000:30000 --shm-size=64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /mnt/disk1_from_server2/siqizhu4/opsd_slime:/root/slime_siqi \
  -v /mnt/disk1_from_server2/siqizhu4/ray_tmp:/tmp/ray_siqi \
  -v /mnt/disk1_from_server2/siqizhu4/checkpoints:/root/checkpoints_siqi \
  -v /mnt/disk1_from_server2/siqizhu4/hf_cache:/root/.cache/huggingface \
  -w /root/slime_siqi \
  -it slimerl/slime:latest /bin/bash


docker run --name siqi_slime_opsd --gpus all \
  --network host \
  --ipc=host --shm-size=64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /mnt/disk1_from_server2/siqizhu4/opsd_slime:/root/slime_siqi \
  -v /mnt/disk1_from_server2/siqizhu4/ray_tmp:/tmp/ray_siqi \
  -v /mnt/disk1_from_server2/siqizhu4/checkpoints:/root/checkpoints_siqi \
  -w /root/slime_siqi \
  -it slimerl/slime:latest /bin/bash

ray stop --force
pkill -9 ray
rm -rf /tmp/ray/*
rm -rf /tmp/ray_siqi/*

export HF_HOME=/mnt/disk1_from_server2/siqizhu4/hf_cache



docker exec -it siqi_slime_opsd bash

bash start.ps1

apt install iproute2 -y

export RAY_memory_usage_threshold=0.99
export RAY_memory_monitor_refresh_ms=0


export TRAIN_DATASET="open-thoughts/OpenThoughts-114k"
export TRAIN_CONFIG="metadata"
TRAIN_ANSWER_FORMAT=answer EVAL_ANSWER_FORMAT=answer bash examples/on_policy_distillation/run-qwen3-4B-opsd_pi.sh

TRAIN_ANSWER_FORMAT=boxed EVAL_ANSWER_FORMAT=boxed bash examples/on_policy_distillation/run-qwen3-4B-opsd_pi.sh


bash examples/on_policy_distillation/run-qwen3-4B-opsd_pi.sh


CUDA_VISIBLE_DEVICES=6,7 python -m sglang.launch_server \
    --model-path /root/checkpoints_siqi/models--Qwen--Qwen3-30B-A3B-Instruct-2507 \
    --host 0.0.0.0 \
    --port 30000 \
    --context-length 16384 \
    --tp-size 2


CUDA_VISIBLE_DEVICES=7,8 python -m sglang.launch_server \
    --model-path checkpoints/models--Qwen--Qwen3-30B-A3B-Instruct-2507 \
    --host 0.0.0.0 \
    --port 30000 \
    --context-length 65536 \
    --tp-size 2


export NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=2,3,4,5 python -m sglang.launch_server \
    --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --host 0.0.0.0 \
    --port 30000 \
    --context-length 32768 \
    --tp-size 4

base_url="http://172.22.224.251:30000/v1"


For server1: 
sed -n '1360,1370p' /sgl-workspace/sglang/python/sglang/srt/managers/tokenizer_manager.py

 然后把约 1365 行的：
  await self.send_to_scheduler.send_pyobj(obj)
  改为（去掉 await）：
  self.send_to_scheduler.send_pyobj(obj)

 或者用一行命令修复：
  sed -i 's/await
  self\.send_to_scheduler\.send_pyobj(obj)/self.send_to_scheduler.send_pyobj(obj)/' \
      /sgl-workspace/sglang/python/sglang/srt/managers/tokenizer_manager.py


# 删除data文件夹的git记录
git rm -r --cached data

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1