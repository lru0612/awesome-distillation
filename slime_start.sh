docker run --gpus all --ipc=host \
  --name slime-lhy \
  --ulimit memlock=-1 --ulimit stack=-1 \
  -v /home/luhongyu/awesome-distillation:/root/awesome-distillation \
  -v /home/luhongyu/OpenClaw-RL:/root/OpenClaw-RL \
  -v /home/luhongyu/data:/root/data \
  -v /mnt/data_from_server1/siqizhu4/lhy/models:/root/models \
  -v /mnt/data_from_server1/siqizhu4/lhy/output:/root/output \
  -it slimerl/slime:latest /bin/bash
