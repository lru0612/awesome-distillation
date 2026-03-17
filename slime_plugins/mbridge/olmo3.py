from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec

from mbridge.core import LLMBridge, register_model


@register_model("olmo3")
class OLMo3Bridge(LLMBridge):
    """
    Bridge for OLMo3 models (OLMo2-based architecture).

    Key differences from LLaMA:
      - Post-norm only: post_attention_layernorm (after attn) + post_feedforward_layernorm (after FFN)
        No input_layernorm / no pre-FFN norm.
      - QK layer norms: q_norm and k_norm per layer.
      - Full MHA: num_key_value_heads == num_attention_heads (no GQA).
    """

    # Direct (non-layer) weight mappings: Megatron name -> HF name
    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }

    # Per-layer attention weight mappings.
    # OLMo3 has no input_layernorm (pre-attn norm), so we reuse post_attention_layernorm
    # to initialize Megatron's fused linear_qkv.layer_norm_weight.  Both are [hidden_size]
    # RMSNorm scale tensors, so the shapes match and training will adapt from there.
    _ATTENTION_MAPPING = {
        "self_attention.linear_proj.weight": [
            "model.layers.{layer_number}.self_attn.o_proj.weight"
        ],
        "self_attention.linear_qkv.layer_norm_weight": [
            "model.layers.{layer_number}.post_attention_layernorm.weight"
        ],
        "self_attention.q_layernorm.weight": [
            "model.layers.{layer_number}.self_attn.q_norm.weight"
        ],
        "self_attention.k_layernorm.weight": [
            "model.layers.{layer_number}.self_attn.k_norm.weight"
        ],
        "self_attention.linear_qkv.weight": [
            "model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
    }

    # Per-layer MLP weight mappings.
    # OLMo3 has no pre-FFN norm, so we reuse post_feedforward_layernorm to initialize
    # Megatron's fused linear_fc1.layer_norm_weight.
    _MLP_MAPPING = {
        "mlp.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.gate_proj.weight",
            "model.layers.{layer_number}.mlp.up_proj.weight",
        ],
        "mlp.linear_fc1.layer_norm_weight": [
            "model.layers.{layer_number}.post_feedforward_layernorm.weight"
        ],
        "mlp.linear_fc2.weight": [
            "model.layers.{layer_number}.mlp.down_proj.weight"
        ],
    }

    def _build_config(self):
        return self._build_base_config(
            add_qkv_bias=False,
            qk_layernorm=True,
            post_self_attn_layernorm=True,
            post_mlp_layernorm=True,
        )

    def _get_transformer_layer_spec(self):
        return get_gpt_layer_with_transformer_engine_spec(
            post_self_attn_layernorm=True,
            post_mlp_layernorm=True,
        )

    def _weight_to_mcore_format(self, mcore_weights_name: str, hf_weights: list) -> object:
        # OLMo3's q_norm/k_norm have shape (hidden_size,) = (4096,) in HF,
        # covering all heads independently.  Megatron uses (head_dim,) = (128,)
        # with the same scale shared across heads.  Average the per-head slices
        # to produce a single representative (head_dim,) initialisation.
        if "q_layernorm.weight" in mcore_weights_name or "k_layernorm.weight" in mcore_weights_name:
            assert len(hf_weights) == 1
            w = hf_weights[0]  # (num_heads * head_dim,)
            head_dim = self.hf_config.hidden_size // self.hf_config.num_attention_heads
            return w.view(-1, head_dim).mean(dim=0).contiguous()
        return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        assert "_extra_state" not in mcore_weights_name, "extra_state should not be loaded"

        if mcore_weights_name in self._DIRECT_MAPPING:
            return [self._DIRECT_MAPPING[mcore_weights_name]]

        # Post-norms are standalone parameters in the Megatron state dict
        if "post_self_attn_layernorm" in mcore_weights_name:
            layer_number = mcore_weights_name.split(".")[2]
            return [f"model.layers.{layer_number}.post_attention_layernorm.weight"]
        if "post_mlp_layernorm" in mcore_weights_name:
            layer_number = mcore_weights_name.split(".")[2]
            return [f"model.layers.{layer_number}.post_feedforward_layernorm.weight"]

        if "self_attention" in mcore_weights_name:
            return self._weight_name_mapping_attention(mcore_weights_name)
        if "mlp" in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)

        raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")
