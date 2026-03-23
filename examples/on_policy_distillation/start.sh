# bash examples/on_policy_distillation/run-qwen3-8B-token-kl-weighted_inverse_0227.sh
# bash examples/on_policy_distillation/run-qwen3-8B-token-kl-weighted_linear_0227.sh
# bash examples/on_policy_distillation/0301-run-qwen3-8B-opsd_competition.sh
# bash examples/on_policy_distillation/0301-run-qwen3-8B-token-kl-weighted_inverse-competition.sh
bash examples/on_policy_distillation/gpu_monitor_run.sh 4 examples/on_policy_distillation/0316-run-OLMO3-7B-openthoughts-jsd.sh 15
bash examples/on_policy_distillation/gpu_monitor_run.sh 4 examples/on_policy_distillation/0316-run-OLMO3-7B-openthoughts-jsd--weighted_inverse.sh 15
bash examples/on_policy_distillation/gpu_monitor_run.sh 4 examples/on_policy_distillation/0316-run-OLMO3-7B-openthoughts-forwardKL.sh 15
bash examples/on_policy_distillation/gpu_monitor_run.sh 4 examples/on_policy_distillation/0316-run-OLMO3-7B-openthoughts-ReverseKL.sh 15