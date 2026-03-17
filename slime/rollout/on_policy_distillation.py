import aiohttp
import torch

from slime.rollout.rm_hub import grade_answer_verl
from slime.utils.types import Sample


async def reward_func(args, sample, **kwargs):
    payload = {
        # "text": sample.prompt + sample.response,
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            teacher_output = await resp.json()

    # Accuracy is used for logging/eval metrics; training reward remains zero in post_process_rewards.
    accuracy = 1.0 if grade_answer_verl(sample.response or "", sample.label or "") else 0.0
    return {
        "teacher_output": teacher_output,
        "accuracy": accuracy,
    }


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Process rewards from teacher model and extract teacher log probabilities.

    This function:
    1. Extracts teacher log-probs from the reward response (which contains sglang's logprob output)
    2. Trims them to match the response length
    3. Stores them in sample.teacher_log_probs for OPD KL penalty computation
    4. Returns scalar rewards (0.0 for pure distillation) compatible with GRPO/PPO

    Note: The reward_func calls the teacher server which returns token-level log-probs.
    For pure on-policy distillation without task rewards, we return 0.0 for each sample.
    The actual learning signal comes from the OPD KL penalty applied in compute_advantages_and_returns.
    """
    raw_rewards = []
    response_lengths = [sample.response_length for sample in samples]

    teacher_outputs = []
    for sample in samples:
        reward = sample.reward
        if isinstance(reward, dict) and "teacher_output" in reward:
            teacher_output = reward["teacher_output"]
            raw_rewards.append(float(reward.get("accuracy", 0.0)))
        else:
            # Backward-compatible path for historical checkpoints/scripts.
            teacher_output = reward
            raw_rewards.append(0.0)
        teacher_outputs.append(teacher_output)

    # Extract teacher log-probs from the sglang response
    teacher_log_probs = [
        torch.tensor([item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]], dtype=torch.float32)
        for reward in teacher_outputs
    ]
    teacher_log_probs = [
        t_log_prob[-response_length:]
        for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
    ]

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs

    # Return scalar rewards for GRPO/PPO advantage estimator
    # For pure on-policy distillation, we use 0.0 as the task reward.
    # The learning signal comes entirely from the OPD KL penalty.
    # If you have task rewards, you can add them here.
    scalar_rewards = [0.0] * len(samples)

    return raw_rewards, scalar_rewards
