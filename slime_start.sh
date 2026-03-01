docker run --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /data/siqizhu4/lhy/awesome-distillation:/root/awesome-distillation \
  -v /data/siqizhu4/lhy/data:/root/data \
  -v /data/siqizhu4/lhy/models:/root/models \
  -v /mnt/data_from_server1/siqizhu4/lhy/output:/root/output \
  -it slimerl/slime:latest /bin/bash