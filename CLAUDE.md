# CLAUDE.md

## Debug Workflow for OLMo3-7B OPSD Training

### Running Training Scripts
1. Make code changes on the host at `/home/luhongyu/awesome-distillation/`
2. Run training via: `docker exec slime-lhy bash /root/awesome-distillation/examples/on_policy_distillation/start.sh`
3. Wait for the run to complete (or fail), then check the latest log in `output/run_log/`
4. If errors occur, fix and re-run

### Key Architecture Notes
- **GPU**: NVIDIA RTX PRO 6000 Blackwell (SM120) — FlashAttention v3 needs SM<=90, flashinfer blocked for OLMo3
- **SGLang attention backend**: must use `triton` (set via `--sglang-attention-backend triton`)
- **OLMo3 is post-norm**: but the `_torch_dist` checkpoint uses standard Megatron fused layernorms (no `--post-self-attn-layernorm` / `--post-mlp-layernorm` flags)
- **Weight conversion mode**: use `raw` (default), NOT `bridge` — `AutoBridge` doesn't support `Olmo3ForCausalLM`
- **OLMo3 is full MHA** (no GQA): `--num-query-groups 32` must match `--num-attention-heads 32`

### Important File Paths
- Training scripts: `examples/on_policy_distillation/0316-run-OLMO3-7B-openthoughts-*.sh`
- Model config: `scripts/models/olmo3-7B.sh`
- OLMo3 weight converter: `slime/backends/megatron_utils/megatron_to_hf/olmo3.py`
- Model provider: `slime/backends/megatron_utils/model_provider.py`
- Checkpoint loading: `slime/backends/megatron_utils/checkpoint.py`
- Docker container: `slime-lhy` (code mounted at `/root/awesome-distillation/`)
