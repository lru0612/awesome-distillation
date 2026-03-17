#!/bin/bash
set -e

# # 1. 下载 Qwen3-8B 模型
# echo "=== Downloading Qwen3-8B model ==="
# hf download Qwen/Qwen3-8B --local-dir /root/Qwen3-8B

# # 2. 下载 MATH-lighteval 数据集
# echo "=== Downloading MATH-lighteval dataset ==="
# hf download --repo-type dataset DigitalLearningGmbH/MATH-lighteval --local-dir /root/math

# # 3. 加载 Qwen3-8B 模型配置
# echo "=== Loading Qwen3-8B model config ==="
# source scripts/models/qwen3-8B.sh

# # 4. 将 HuggingFace 模型转换为 torch_dist 格式
# echo "=== Converting HF checkpoint to torch_dist format ==="
# CUDA_VISIBLE_DEVICES=6,7,8,9 PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
#     ${MODEL_ARGS[@]} \
#     --hf-checkpoint /root/Qwen3-8B \
#     --save /root/Qwen3-8B_torch_dist

# mkdir -p /root/slime/output/Qwen3-8B_opsd_slime/
# echo "=== All done! ==="


# # 1. 下载 Qwen3-8B 模型
# echo "=== Downloading Qwen3-8B model ==="
# hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir /root/Qwen3-4B-Instruct-2507

# # 2. 下载 MATH-lighteval 数据集
# echo "=== Downloading MATH-lighteval dataset ==="
# hf download --repo-type dataset DigitalLearningGmbH/MATH-lighteval --local-dir /root/math

# # 3. 加载 Qwen3-8B 模型配置
# echo "=== Loading Qwen3-8B model config ==="
# source scripts/models/qwen3-4B-Instruct-2507.sh

# # 4. 将 HuggingFace 模型转换为 torch_dist 格式
# echo "=== Converting HF checkpoint to torch_dist format ==="
# CUDA_VISIBLE_DEVICES=6,7,8,9 PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
#     ${MODEL_ARGS[@]} \
#     --hf-checkpoint /root/Qwen3-4B-Instruct-2507 \
#     --save /root/Qwen3-4B-Instruct-2507_torch_dist

# mkdir -p /root/slime/output/Qwen3-4B-Instruct-2507_opsd_slime/
# echo "=== All done! ==="


# 1. 下载 Qwen3-8B 模型
echo "=== Downloading Qwen3-8B model ==="
hf download Qwen/Qwen3-1.7B --local-dir /root/Qwen3-1.7B

# # 2. 下载 MATH-lighteval 数据集
# echo "=== Downloading MATH-lighteval dataset ==="
# hf download --repo-type dataset DigitalLearningGmbH/MATH-lighteval --local-dir /root/math

# 3. 加载 Qwen3-8B 模型配置
echo "=== Loading Qwen3-8B model config ==="
source scripts/models/qwen3-1.7B.sh

# 4. 将 HuggingFace 模型转换为 torch_dist 格式
echo "=== Converting HF checkpoint to torch_dist format ==="
CUDA_VISIBLE_DEVICES=2,5,7,9 PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-1.7B \
    --save /root/Qwen3-1.7B_torch_dist

mkdir -p /root/slime/output/Qwen3-1.7B_opsd_slime/
echo "=== All done! ==="
