MODEL_ARGS=(
   --swiglu
   --num-layers 32
   --hidden-size 4096
   --ffn-hidden-size 11008
   --num-attention-heads 32
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base 500000
   --vocab-size 100278
   --make-vocab-size-divisible-by 1
   --kv-channels 128
   --qk-layernorm
   --untie-embeddings-and-output-weights
)
