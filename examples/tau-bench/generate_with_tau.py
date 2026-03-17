"""
Tau-Bench Integration for slime Training

This module provides the main interface for training agents in tau-bench environments
using the slime framework. It handles agent-environment interactions and converts
results to the format expected by slime's training pipeline.

OPSD teacher information modes (set via OPSD_MODE env var):
  conciseness (default): No extra metadata. Teacher gets efficiency guideline hint.
  gt:                    Passes ground-truth action sequence from the Task object.
                         Teacher system = wiki + exact tool calls to use.
  oracle:                Calls an external LLM to generate an expert solution plan.
                         Teacher system = wiki + LLM-generated plan.
                         Falls back to GT actions if the oracle call fails.
                         Configure oracle model via ORACLE_MODEL / ORACLE_API_BASE.
"""

import logging
import os
import sys
from typing import Any

# Ensure this file's directory is on the path for relative imports (needed when
# Ray ships the working directory to remote workers as a zip package)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tau_bench.envs import get_env
from tau_bench.types import RunConfig, RESPOND_ACTION_NAME
from trainable_agents import InteractionResult, Status, agent_factory

from slime.utils.types import Sample

# Set up logger for this module
logger = logging.getLogger(__name__)

# Tau-bench configuration
# TAU_CONFIGS = {
#     "env": "retail",  # Select between ["retail", "airline"]
#     "agent": "tool-calling",  # Select between ["tool-calling", "act", "react", "few-shot"]
#     "user_model": "gemini-2.5-flash-lite",  # Cheap Model for user simulator
#     "task_split": "train",  # Select between ["train", "test", "dev"] for retail
#     "user_strategy": "llm",  # Select between ["llm", "react", "verify", "reflection"]
#     "model_provider": "auto_router",  # Unused, required
#     "model": "qwen3-4b",  # Unused, required
#     "user_model_provider": "gemini",
# }
# # Replace with your actual API key for user sim
# GEMINI_API_KEY = "NONE"
# os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

TAU_CONFIGS = {
      "env": "retail",
      "agent": "tool-calling",
      "user_model": "/root/checkpoints_siqi/models--Qwen--Qwen3-30B-A3B-Instruct-2507",  # litellm 的 openai provider 格式
      "user_model_provider": "openai",          # 用 openai-compatible provider
      "task_split": "train",
      "user_strategy": "llm",
      "model_provider": "auto_router",
      "model": "/root/checkpoints_siqi/models--Qwen--Qwen3-30B-A3B-Instruct-2507",
  }

os.environ["OPENAI_API_BASE"] = "http://172.22.224.251:30000/v1"
os.environ["OPENAI_API_KEY"] = "EMPTY"
tau_config = RunConfig(**TAU_CONFIGS)

# ---------------------------------------------------------------------------
# OPSD mode configuration
# ---------------------------------------------------------------------------
# OPSD_MODE: "conciseness" | "gt" | "oracle"
#   conciseness — default; no extra metadata, teacher gets efficiency guideline.
#   gt          — ground-truth action sequence from Task is added to metadata.
#   oracle      — external LLM generates an expert plan; falls back to GT.
OPSD_MODE: str = os.environ.get("OPSD_MODE", "gt")

# Oracle LLM settings (only used when OPSD_MODE == "oracle").
# ORACLE_MODEL defaults to the user_model (Qwen3-30B).
# ORACLE_API_BASE defaults to OPENAI_API_BASE already set above.
ORACLE_MODEL: str = os.environ.get(
    "ORACLE_MODEL",
    TAU_CONFIGS["user_model"],
)
ORACLE_API_BASE: str = os.environ.get(
    "ORACLE_API_BASE",
    os.environ.get("OPENAI_API_BASE", ""),
)


async def _fetch_oracle_plan(
    wiki: str,
    instruction: str,
    tools_info: list[dict],
) -> str:
    """Call an external LLM to generate an expert solution plan for the task.

    The plan is injected into the teacher's system message so the teacher has
    privileged knowledge about *how* to solve the task, producing better logits
    over the student's response tokens.

    Args:
        wiki:        Agent knowledge base (policies / rules).
        instruction: Task instruction string from the Task object.
        tools_info:  OpenAI-format tool specifications.

    Returns:
        A concise step-by-step plan string, or "" on failure.
    """
    import openai

    tool_names = [
        t["function"]["name"] for t in tools_info if "function" in t
    ]

    user_msg = (
        f"Task: {instruction}\n\n"
        f"Available tools: {', '.join(tool_names)}\n\n"
        "Write a concise step-by-step plan to solve this task. "
        "For each step, specify the tool name and the key arguments. "
        "Do not execute the tools — just plan the solution path."
    )

    client = openai.AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        base_url=ORACLE_API_BASE,
    )
    response = await client.chat.completions.create(
        model=ORACLE_MODEL,
        messages=[
            {"role": "system", "content": wiki},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    return response.choices[0].message.content or ""


def res_to_sample(res: InteractionResult, task_index: int) -> Sample:
    """
    Convert InteractionResult to Sample format for slime training.

    This function transforms the tau-bench interaction result into the format
    expected by slime's training pipeline, handling status mapping and response
    length calculation.

    Args:
        res: InteractionResult from tau-bench agent
        task_index: Index of the task being processed

    Returns:
        Sample object for slime training
    """
    # Map tau-bench status to slime status
    status_mapping = {
        Status.COMPLETED: "completed",
        Status.TRUNCATED: "truncated",
        Status.ABORTED: "aborted",
    }
    status = status_mapping.get(res.status)

    # Debug logging for response tracking
    logger.debug(
        f"res_to_sample: response_length="
        f"{res.response_length if hasattr(res, 'response_length') else 'None'}, "
        f"loss_mask_len={len(res.loss_mask) if res.loss_mask else 'None'}, "
        f"tokens_len={len(res.tokens) if res.tokens else 'None'}"
    )

    # Create sample with basic information
    sample = Sample(
        index=task_index,
        prompt=res.prompt,
        tokens=res.tokens,
        response=res.response,
        reward={"tau_reward": float(res.reward or 0.0)},
        loss_mask=res.loss_mask,
        status=status,
        metadata=res.info,
    )

    # Ensure response_length is set correctly
    if hasattr(res, "response_length"):
        sample.response_length = res.response_length
    else:
        # Fallback: calculate from loss_mask if available
        if res.loss_mask:
            # loss_mask only contains response part, so length equals response_length
            sample.response_length = len(res.loss_mask)
        elif res.tokens:
            # If no loss_mask available, use total tokens as fallback
            sample.response_length = len(res.tokens)
        else:
            sample.response_length = 0
            logger.debug(f"res_to_sample: Set response_length={sample.response_length}")

    return sample


async def generate(args: dict[str, Any], sample: Sample, sampling_params: dict) -> Sample:
    """
    Generate a complete agent-environment interaction trajectory for tau-bench.

    This is the main entry point for slime training. It creates a tau-bench
    environment, initializes a trainable agent, and executes a full interaction
    trajectory. The result is converted to slime's Sample format for training.

    Args:
        args: Rollout arguments from slime training pipeline
        sample: Sample containing task index in prompt field
        sampling_params: LLM sampling parameters

    Returns:
        Sample object containing the complete interaction trajectory

    Raises:
        AssertionError: If partial rollout is requested (not supported)
    """
    # Validate arguments
    assert not args.partial_rollout, "Partial rollout is not supported for tau-bench interactions."

    # Extract task index from sample prompt
    task_index = int(sample.prompt)
    logger.info(f"Starting agent-environment interaction for task {task_index}")

    # Initialize tau-bench environment
    env = get_env(
        env_name=tau_config.env,
        user_strategy=tau_config.user_strategy,
        user_model=tau_config.user_model,
        user_provider=tau_config.user_model_provider,
        task_split=tau_config.task_split,
        task_index=task_index,
    )

    # Create trainable agent
    agent = agent_factory(
        tools_info=env.tools_info,
        wiki=env.wiki,
        config=tau_config,
        rollout_args=args,
        sampling_params=sampling_params,
    )

    # Execute agent-environment interaction
    # Note: The sample.prompt field contains the task index for repeatability
    interaction_result = await agent.asolve(env, agent.rollout_args, agent.sampling_params, task_index)

    # Convert to slime Sample format
    result_sample = res_to_sample(interaction_result, task_index)

    # Enrich metadata for OPSD teacher token construction.
    # initial_prompt_token_length and initial_user_message are already in
    # result_sample.metadata (written by _build_final_result via res.info).
    result_sample.metadata['wiki'] = agent.wiki
    result_sample.metadata['tools_info'] = env.tools_info

    # GT mode: add ground-truth action sequence from the Task object.
    # post_process_rewards reads metadata['task']['actions'] to build the
    # "correct sequence of actions" teacher hint.
    # Oracle mode also sets this so that if the oracle call fails, GT is the fallback.
    if OPSD_MODE in ("gt", "oracle"):
        gt_actions = [
            {"name": a.name, "kwargs": dict(a.kwargs)}
            for a in env.task.actions
            if a.name != RESPOND_ACTION_NAME
        ]
        result_sample.metadata['task'] = {"actions": gt_actions}
        logger.debug(
            f"task {task_index}: added {len(gt_actions)} GT actions to metadata"
        )

    # Oracle mode: call external LLM to generate an expert solution plan.
    # post_process_rewards prefers oracle_plan over GT actions when both are set.
    if OPSD_MODE == "oracle":
        try:
            oracle_plan = await _fetch_oracle_plan(
                wiki=agent.wiki,
                instruction=env.task.instruction,
                tools_info=env.tools_info,
            )
            result_sample.metadata['oracle_plan'] = oracle_plan
            logger.info(
                f"task {task_index}: oracle plan fetched "
                f"({len(oracle_plan)} chars): {oracle_plan[:120]}..."
            )
        except Exception as exc:
            logger.warning(
                f"task {task_index}: oracle plan fetch failed ({exc}); "
                "falling back to GT actions."
            )

    logger.info(f"Finished agent-environment interaction for task {task_index}")
    return result_sample
