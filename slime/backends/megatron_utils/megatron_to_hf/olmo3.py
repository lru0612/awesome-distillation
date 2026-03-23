import re
import torch


def convert_olmo3_to_hf(args, name, param):
    # Global weights
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    num_heads = args.num_attention_heads
    # OLMo3 has no GQA: num_query_groups == num_attention_heads
    num_query_groups = getattr(args, "num_query_groups", None) or num_heads
    value_num_per_group = num_heads // num_query_groups

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()

        # Skip extra state (TransformerEngine FP8 state)
        if "_extra_state" in rest:
            return []

        # OLMo3 is post-norm only.  The pre-converted _torch_dist checkpoint uses
        # the standard Megatron architecture (no standalone post-norm modules).
        # The fused layernorm in linear_qkv / linear_fc1 approximates OLMo3's
        # post_attention_layernorm / post_feedforward_layernorm respectively.
        if rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
        if rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.post_feedforward_layernorm.weight", param)]

        # Standalone post-norm params (only present if --post-self-attn-layernorm
        # / --post-mlp-layernorm flags are used)
        if rest == "self_attention.post_self_attn_layernorm.weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
        if rest == "mlp.post_mlp_layernorm.weight":
            return [(f"model.layers.{layer_idx}.post_feedforward_layernorm.weight", param)]

        if rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]

        if rest == "self_attention.linear_qkv.weight":
            param = param.view(num_query_groups, -1, head_dim, args.hidden_size)
            q_param, k_param, v_param = torch.split(
                param, split_size_or_sections=[value_num_per_group, 1, 1], dim=1
            )
            q_param = q_param.reshape(-1, args.hidden_size)
            k_param = k_param.reshape(-1, args.hidden_size)
            v_param = v_param.reshape(-1, args.hidden_size)
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.weight", q_param),
                (f"model.layers.{layer_idx}.self_attn.k_proj.weight", k_param),
                (f"model.layers.{layer_idx}.self_attn.v_proj.weight", v_param),
            ]

        # QK layer norms: Megatron stores (head_dim,), HF expects (num_heads * head_dim,)
        if rest == "self_attention.q_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_norm.weight", param.repeat(num_heads))]
        if rest == "self_attention.k_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.k_norm.weight", param.repeat(num_heads))]

        if rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        if rest == "mlp.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]

    raise ValueError(f"Unknown OLMo3 parameter name: {name}")
