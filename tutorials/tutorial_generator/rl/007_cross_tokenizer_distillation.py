# pyright: reportUndefinedVariable=false, reportMissingImports=false
"""Tutorial source for `007_cross_tokenizer_distillation` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "2 × 8×H200 + 1 × 4×B200 + CPU sandbox pool",
    "summary": "Cross-tokenizer agentic distillation on Toolathlon with live, environment-grounded rewards — DeepSeek V4 Flash teacher, Qwen3.6-35B-A3B student",
    "difficulty": "Advanced",
    "order": 60,
    "api_classes": [
        "Qwen3_6_35B",
        "DeploymentConfig",
        "EvalConfig",
        "EvalRowResult",
        "Qwen3_6_35b_Recipe",
        "TrainConfig",
    ],
}

from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # On-policy Distillation (OPD) Across Model Families

    Previously, we studied the reward objective of OPD: minimize the reverse-KL divergence of a teacher model and student model.
    While OPD provdies the Qwen3-4B model a smarter Qwen3-8B reference on our dataset's task, all the lightweight Qwen3 models are already distilled 
    from larger flagship models during post-training (see "Strong-to-Weak Distillation" in https://arxiv.org/pdf/2505.09388).
    By using models from two different families, we gain more information on probability regions that are low-value to a teacher but high-value to a student.

    Reverse-KL divergence is computed using per-token log-probabilties from the teacher and student; what if our models tokenizer the same text differently?
    We approximate the teacher's logprobs of student tokens by summing together logprobs for any sequence of text with mismatched tokens (this algorithm is borrowed
    from SimCT: https://arxiv.org/abs/2605.07711). The summed logprob over the sequence is included as if it were an additional token for the shared vocabulary.

    This tutorial teachers a student model, Qwen3.6-35B-A3B, from a teacher model, DeepSeek-V4 Flash, to produce a series of correct toolcalls from the Toolathlon
    dataset (https://toolathlon.xyz/introduction). The reward objective contains the negative reverse-KL, partial credit for schema-validated toolcalls, and a
    binary reward for terminal success/failure. Toolathlon provides Docker images for self-hosting MCP servers in Modal sandboxes and final evaluation of the
    environment state. Qwen3.6 is given an expert trajectory from DeepSeek-V3.2 up to the Kth tool call, where K is initialized to the N trajectory steps - 1.
    As the student model completes a task, K is decremented until the rollouts have completed or the task is started from the N=0 step (toolcall). 

    ### Steps
    1. Deploy DeepSeek V4 Flash as teacher on SGLang.
    2. Scope training + eval split from Toolathlon dataset. Use only tasks that do not require external accounts or store state in memory.
    3. Replay the ground-truth trajectory and snapshot the environment state at each step using Modal directory snapshots (https://modal.com/docs/guide/sandbox-snapshots).
    4. At rollout/eval time, mount the `(task, K)` directory snapshot into a warm Modal sandbox to restore the environment. 
    5. Define SimCT alignment helpers and the 4-signal reward (schema + live tool-exec success + structural match + terminal eval verdict).
    6. Train with prefix-conditioned multi-turn rollouts: restore the env at step `K`, let the student drive to completion, grade with the
       live verdict combined with partial credit, and distill the teacher's logprobs over the full trajectory (cross-tokenizer OPD + GRPO).
    7. Evaluate the base and trained student on held-out tasks with the live verdict and compare.
    """


@py_only
@markdown
def _run_instructions():
    """
    Run with:
    ```
    uv run python tutorials/rl/007_cross_tokenizer_distillation/007_cross_tokenizer_distillation.py
    ```
    Set PYTHONUNBUFFERED=1 if running inside of an agent shell (e.g. Claude Code) to see output.
    """


@notebook_only
@shell("%uv pip install -q 'modal>=1.4.3' git+https://github.com/modal-projects/training-gym.git@main")
def _install():
    pass


@code
def _imports():
    import asyncio
    import json
    import os
    import re

    import modal
    from modal_training_gym import (
        DeploymentConfig,
        EvalConfig,
        EvalRowResult,
        ModelDeployment,
        Qwen3_6_35B,
        TrainConfig,
        list_checkpoints,
    )
    from modal_training_gym.common.models.base import HFModelConfiguration

    from modal_training_gym.common.dataset import ToolathlonTrajectoryDataset

    from modal_training_gym.common.environments import (
        build_prefix_messages,
        build_snapshot_library,
        get_env_pool,
        tool_schemas_to_openai,
    )
    from modal_training_gym.deploy_recipes.sglang_recipe import (
        DeepSeek_V4_Flash_SglangRecipe,
        Qwen3_6_35b_SglangRecipe,
    )
    from modal_training_gym.train_recipes.slime_recipe import Qwen3_6_35b_Recipe


@markdown
def _deploy_intro():
    """
    ## Deploy DeepSeek-V4 Flash Teacher Model
    Modal training gym provides a sglang recipe for deploying the 284B-A13B. The FP4 checkpoint fits comfortably on a single
    8×H100 node (~150GB of weights on 640GB), with ample room for the KV cache at 64k context.
    """


@code
def _deploy_teacher():
    teacher_deployment = DeploymentConfig(
        model=HFModelConfiguration(model_name="deepseek-ai/DeepSeek-V4-Flash"),
        recipe=DeepSeek_V4_Flash_SglangRecipe(
            tp=8,
            gpu="H100",
            context_length=65536,
        ),
        app_name="dsv4-flash-test",
        served_model_name="deepseek-v4-flash",
    ).serve()
    print(f"Teacher URL: {teacher_deployment.url}")

    TEACHER_READY_TIMEOUT = 30 * 60
    teacher_deployment.wait_until_ready(timeout=TEACHER_READY_TIMEOUT)

    TEACHER_GENERATE_URL = f"{teacher_deployment.url}/generate"


@code
def _student_model():
    base_model = Qwen3_6_35B()


@markdown
def _dataset_intro():
    """
    ## Defining the Dataset Split

    Toolathlon provdies expert trajectories and MCP environments for benchmarking long-horizon agents on real-world tasks (https://arxiv.org/abs/2510.25726).
    We use Toolathlon's provided DeepSeek V3.2 trajectory as the ground-truth dataset and inject the first K steps (toolcalls) into the model's context during training.
    The task MCP servers that spin up in Modal sandboxes rely on directory snapshots, which save the task's file state at a step K.
    Environments that rely on storing state in memory or external account credentials are excluded because we cannot recover state reliably. 
    """


@code
def _dataset():
    EVAL_TASKS = ["excel-data-transformation", "ppt-analysis", "interview-report"]
    TRAIN_TASKS = [
        "arrange-workspace", "cooking-guidance", "detect-revised-terms", "dietary-health",
        "excel-market-research", "imagenet", "paper-checker", "privacy-desensitization",
        "university-course-selection",
    ]
    ALL_VALID_TASKS = TRAIN_TASKS + EVAL_TASKS
    MAX_TURNS = 30
    CURRICULUM_TAIL_MIN = 1
    CURRICULUM_PASS_RATE = 0.5
    CURRICULUM_STALL_WARN_ROUNDS = 5
    EVAL_TAIL_STEPS = 8
    ROLLOUT_LOG_EVERY = 8
    STUDENT_ENABLE_THINKING = False
    OPD_SKIP_ON_TEACHER_FAILURE = True

    dataset = ToolathlonTrajectoryDataset(
        train_tasks=TRAIN_TASKS,
        eval_tasks=EVAL_TASKS,
    )


@markdown
def _curriculum_intro():
    """
    ## Custom Student Training Curriculum

    The Toolathlon dataset has ~25 steps per-trajectory—too long-horizon for our student model to start hillclimbing on.
    A well-known technique in machine learning is gradually increasing the difficulty of training task, which has been
    found to speed up convergence and improve the quality of local optima (https://dl.acm.org/doi/epdf/10.1145/1553374.1553380).
    To expedite the process of training and increase accuracy on the Toolathlon terminal evaluation metric, which is our north star,
    we start the student model with the expert trajectory context up to the Nth step and initialize K to N-1 as we start training.
    See more details on the conditions for decrementing K over rollouts in the "Initializing from the Kth Expert Trajectory Step" section.
    """

@code
def _curriculum():
    def _iter_rollout_samples(result):
        container = getattr(result, "samples", result)
        stack = [container]
        while stack:
            item = stack.pop()
            if isinstance(item, (list, tuple)):
                stack.extend(item)
            elif item is not None:
                yield item

    def curriculum_rollout(args, rollout_id, data_source, evaluation=False):
        from slime.rollout.sglang_rollout import generate_rollout

        if evaluation:
            return generate_rollout(args, rollout_id, data_source, evaluation=True)

        if not hasattr(args, "curriculum_tail") or rollout_id == 0:
            args.curriculum_tail = CURRICULUM_TAIL_MIN
            args.curriculum_stall_rounds = 0

        T = args.curriculum_tail
        print(
            f"[curriculum] iter={rollout_id} tail={T} "
            f"stall_rounds={args.curriculum_stall_rounds}",
            flush=True,
        )

        result = generate_rollout(args, rollout_id, data_source, evaluation=False)

        verdicts = [
            bool(getattr(s, "toolathlon_task_passed", False))
            for s in _iter_rollout_samples(result)
            if hasattr(s, "toolathlon_task_passed")
        ]
        pass_rate = (sum(verdicts) / len(verdicts)) if verdicts else 0.0

        if pass_rate >= CURRICULUM_PASS_RATE:
            args.curriculum_tail = T + 1
            args.curriculum_stall_rounds = 0
            print(
                f"[curriculum] mastered tail={T} pass_rate={pass_rate:.2f} "
                f"-> advance to {args.curriculum_tail}",
                flush=True,
            )
        else:
            args.curriculum_stall_rounds += 1
            msg = (
                f"[curriculum] holding tail={T} pass_rate={pass_rate:.2f} "
                f"stall_rounds={args.curriculum_stall_rounds}"
            )
            if args.curriculum_stall_rounds >= CURRICULUM_STALL_WARN_ROUNDS:
                msg += (
                    f" — WARNING: stuck >= {CURRICULUM_STALL_WARN_ROUNDS} rounds; "
                    "check reward/parsing/env before expecting progress"
                )
            print(msg, flush=True)

        return result


@code
def _build_snapshots():
    print("--- Building Toolathlon snapshot library... ---")
    golden_by_task = dataset.golden_by_task()
    build_snapshot_library(golden_by_task)
    print("--- Snapshot library ready ---")




@markdown
def _reward_intro():
    """
    ## Reward Decomposition
    The reward function is broken up into four separate terms:
    - 0.25 * mean_partial assigns partial credit to each toolcall in the model's output for all toolcalls:
        - +0.20 if the toolcall parses properly (is not none and contains a name field)
        - +0.15 if the toolcall name is known
        - +0.15 if the toolcall validates against the tool's expected JSON schema
        - +0.50 * structural_match:
            - +0.4 if toolcall name matches the golden expert trajectory's toolcall
            - +0.15 if overlap between the toolcall's args and the golden toolcall's
            - +0.15 if full overlap between the toolcall's args and the golden toolcall's args
            - +0.3 if no args in the toolcall
            - +0.3 * the ratio of matching arg values between model and expert toolcall to shared toolcall args
    - +0.20 * partial_credit score (see above) for the very first toolcall
    - +0.10 * exec_score for all successful execution of toolcalls on the Toolathlon MCP environment
    - +0.45 if Toolathlon's terminal environment evaluation returns 1 for success
    """


@code
def _reward():
    _UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
    _TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}")
    _TMP_RE = re.compile(r"/tmp/[^\s/]+")

    def _normalize_value(v) -> str:
        s = str(v).strip().lower()
        s = _UUID_RE.sub("<uuid>", s)
        s = _TS_RE.sub("<timestamp>", s)
        s = _TMP_RE.sub("/tmp/<tmp>", s)
        return s

    def _coerce_args(call: dict | None) -> dict:
        if not call:
            return {}
        args = call.get("arguments", {})
        if isinstance(args, dict):
            return args
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
                return parsed if isinstance(parsed, dict) else {}
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}

    def _validates(call: dict, schema: dict) -> bool:
        try:
            import jsonschema
            jsonschema.validate(_coerce_args(call), schema)
            return True
        except Exception:
            return False

    def _score_structural_match(student_call: dict | None, expert_call: dict) -> float:
        if not student_call:
            return 0.0
        if student_call.get("name") != expert_call.get("name"):
            return 0.0
        score = 0.4

        s_vals = _coerce_args(student_call)
        e_vals = _coerce_args(expert_call)
        s_args = set(s_vals.keys())
        e_args = set(e_vals.keys())
        if s_args == e_args:
            score += 0.3
        elif s_args & e_args:
            score += 0.15
        else:
            return score

        shared = s_args & e_args
        if not shared:
            score += 0.3
        else:
            matches = sum(
                1 for k in shared
                if _normalize_value(s_vals.get(k)) == _normalize_value(e_vals.get(k))
            )
            score += 0.3 * (matches / len(shared))
        return score

    def _partial_credit(student_call: dict | None, expert_call: dict, tool_schemas: dict) -> float:
        if not student_call or "name" not in student_call:
            return 0.0
        score = 0.20
        spec = tool_schemas.get(student_call.get("name"))
        if spec is not None:
            score += 0.15
            schema = spec.get("parameters", spec) if isinstance(spec, dict) else spec
            if _validates(student_call, schema):
                score += 0.15
        score += 0.50 * _score_structural_match(student_call, expert_call)
        return min(1.0, score)

    def trajectory_reward(
        student_calls: list,
        exec_successes: list,
        expert_calls: list,
        tool_schemas: dict,
        task_passed: bool | None,
        tail_len: int,
    ) -> float:
        T = max(int(tail_len), 1)
        graded = min(len(student_calls), len(expert_calls), T)
        partial_sum = sum(
            _partial_credit(student_calls[j], expert_calls[j], tool_schemas)
            for j in range(graded)
        )
        mean_partial = partial_sum / T
        first_call_score = (
            _partial_credit(student_calls[0], expert_calls[0], tool_schemas)
            if graded and expert_calls else 0.0
        )
        successful_execs = sum(1.0 for ok in exec_successes if ok)
        exec_score = min(successful_execs, T) / T
        if task_passed is not None:
            verify_score = 1.0 if task_passed else 0.0
            return (
                0.25 * mean_partial
                + 0.20 * first_call_score
                + 0.10 * exec_score
                + 0.45 * verify_score
            )
        return mean_partial




@markdown
def _simct_explainer():
    """
    ## Cross-tokenizer alignment via SimCT Minimally Aligned Units (MTUs)

    To solve the problem of cross-tokenizer misalignment, where two tokenizers may not share the same token vocabularies or
    may merge tokens differently, we employ an algorithm called SimCT (https://arxiv.org/abs/2605.07711) that constructs
    Minimally Aligned Units (MAUs) between boundaries of character-aligned tokens.

    Here's an example where SimCT would create a MAU for normalizing both model's different vocabularies:
    For the sentence "I am happy today", the student model tokenizes it as ["I", "am", "ha", "pp", "y", "today"] and the teacher model tokenizes it as 
    ["I", "am", "hap", "py", "today"]. The "I" and "am" tokens match up perfectly, but the word "happy" is split across multiple tokens.
    
    SimCT greedily finds shared character boundaries between both tokenizations. Where boundaries don't align (the word "happy"),
    it groups the tokens into a MAU spanning ["ha", "pp", "y"] for the student and ["hap", "py"] for the teacher. 
    The MAU log-probability is the joint probability in log space, which is the sum of its constituent token logprobs 
    (or the log of the product of token probabilities). To integrate with slime's per-token KL, 
    we distribute the teacher's MAU logprob sum equally across student tokens in the MAU, 
    so that slime's per-token summation reconstructs the correct MAU-level reverse KL.

    Special tokens are excluded from alignment via `skip_special_tokens=True` during decoding, ensuring the
    character-level alignment only operates on actual content where the teacher's logprobs are meaningful.
    """


@code
def _mau_helpers():
    def align_cross_tokenizer(
        teacher_token_texts: list[str],
        teacher_logprobs: list[float],
        student_token_texts: list[str],
        return_coverage: bool = False,
    ):
        import torch

        def _offsets(texts):
            out, pos = [], 0
            for t in texts:
                out.append((pos, pos + len(t)))
                pos += len(t)
            return out

        t_off = _offsets(teacher_token_texts)
        s_off = _offsets(student_token_texts)
        t_bounds = {s for s, e in t_off} | {e for s, e in t_off}
        s_bounds = {s for s, e in s_off} | {e for s, e in s_off}
        shared = sorted(t_bounds & s_bounds)

        result = torch.zeros(len(student_token_texts), dtype=torch.float32)
        covered = torch.zeros(len(student_token_texts), dtype=torch.bool)
        for i in range(len(shared) - 1):
            lo, hi = shared[i], shared[i + 1]
            t_idx = [j for j, (s, e) in enumerate(t_off) if s >= lo and e <= hi]
            s_idx = [j for j, (s, e) in enumerate(s_off) if s >= lo and e <= hi]
            if t_idx and s_idx:
                mau_lp_sum = sum(teacher_logprobs[j] for j in t_idx)
                per_student_token = mau_lp_sum / len(s_idx)
                for j in s_idx:
                    result[j] = per_student_token
                    covered[j] = True
        if return_coverage:
            return result, covered
        return result




@markdown
def _rm_intro():
    """
    ## Reward function

    The reward function factors in the reverse KL divergence from the teacher's per-token logprobs and the aforementioned composite reward score.
    By including a reward aligned with the target task of toolcalling, we can help "reward-tilt" the student model 
    towards the reward-weighted version of the teacher's distribution rather than just the raw distribution. The following blogs show helpful 
    visualization of reward-tilting a student model: 
    https://emilianopp.github.io/Privileged-Information-Distillation-and-Self-Distillation/ (see Reward-Tilted Self-Distillation section).

    We also give the teacher *privileged information* the student never sees: when scoring the student's response, the teacher is conditioned on
    the remaining expert tool calls (`golden_calls[K:]`) and their observations. A teacher that already knows the intended solution places sharper,
    better-calibrated probability mass on the correct next tool call, so the reverse-KL term distills the student toward a stronger reference while
    the student keeps acting from its own privilege-free prompt at train and inference time (Learning Using Privileged Information; see the blog above).
    Two invariants make this safe: the scored *response* text stays byte-identical to what the student generated, and `logprob_start_len` is counted in
    the *teacher's* tokenizer so the returned logprobs begin exactly at the response — never leaking the privileged prefix into the distilled signal.
    """


@code
def _rm():
    _student_tokenizer_cache = {}

    def _get_student_tokenizer():
        if "tok" not in _student_tokenizer_cache:
            from transformers import AutoTokenizer
            _student_tokenizer_cache["tok"] = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3.6-35B-A3B", trust_remote_code=True
            )
        return _student_tokenizer_cache["tok"]

    _teacher_tokenizer_cache = {}

    def _get_teacher_tokenizer():
        # Used only to locate the response boundary in TEACHER tokens (below);
        # the teacher logits themselves come from the served SGLang endpoint.
        if "tok" not in _teacher_tokenizer_cache:
            from transformers import AutoTokenizer
            _teacher_tokenizer_cache["tok"] = AutoTokenizer.from_pretrained(
                "deepseek-ai/DeepSeek-V4-Flash", trust_remote_code=True
            )
        return _teacher_tokenizer_cache["tok"]

    def _privileged_prefix(label, K):
        """Privileged information the STUDENT never sees: the remaining expert tool
        calls (golden_calls[K:]) and their observations. Conditioning the teacher on
        the intended solution sharpens its distribution over the student's response,
        so the reverse-KL pulls the student toward a better-informed reference. The
        student still learns the task from its own privilege-free prompt — this is
        Learning Using Privileged Information / reward-tilted distillation."""
        golden_tail = (label.get("golden_calls") or [])[K:]
        obs_tail = (label.get("observations") or [])[K:]
        lines = ["[PRIVILEGED — reference solution for the remaining steps; do not reveal]"]
        for i, call in enumerate(golden_tail):
            call = call or {}
            arguments = json.dumps(call.get("arguments") or {}, ensure_ascii=False)
            lines.append(f"  step {K + i}: {call.get('name', '')}({arguments})")
            if i < len(obs_tail):
                lines.append(f"    -> observed: {str(obs_tail[i])[:500]}")
        return "\n".join(lines)

    def _teacher_response_boundary(teacher_tok, prefix_text, full_text):
        """Number of TEACHER tokens before the student response.

        SGLang tokenizes ``text`` itself, so ``logprob_start_len`` is counted in the
        teacher's vocabulary — not the student token count. ``prefix_text`` ends on a
        blank line so the boundary tokenizes cleanly; offset mapping (when the teacher
        ships a fast tokenizer) makes this robust to any boundary merge, with a plain
        prefix-length fallback otherwise."""
        try:
            offsets = teacher_tok(
                full_text, add_special_tokens=False, return_offsets_mapping=True
            )["offset_mapping"]
            return sum(1 for start, _end in offsets if start < len(prefix_text))
        except (TypeError, KeyError, NotImplementedError, ValueError):
            return len(teacher_tok(prefix_text, add_special_tokens=False)["input_ids"])

    async def cross_tokenizer_reward(args, sample, **kwargs):
        """Collect teacher log-probs with production-grade retry logic.

        The teacher is conditioned on PRIVILEGED context (the future expert
        trajectory) that the student's prompt omits, so its per-token distribution
        over the student's response is a stronger OPD target."""
        import aiohttp
        import random

        tokenizer = _get_student_tokenizer()

        # Split the student sequence into (prompt prefix, response). The response text
        # must stay byte-identical to what the student generated so the teacher scores
        # exactly those tokens and SimCT can align them in post-processing.
        resp_len = max(1, sample.response_length)
        split = max(0, len(sample.tokens) - resp_len)
        student_prefix_text = tokenizer.decode(sample.tokens[:split], skip_special_tokens=True)
        response_text = tokenizer.decode(sample.tokens[split:], skip_special_tokens=True)

        # Swap the student prefix for a PRIVILEGED prefix: the same context plus the
        # remaining expert calls the student never saw. It is excluded from the
        # returned logprobs (via logprob_start_len), so it only conditions the teacher.
        meta = getattr(sample, "metadata", {}) or {}
        label = json.loads(getattr(sample, "label", "{}") or "{}")
        K = int(meta.get("step_K", 0))
        teacher_prefix_text = f"{student_prefix_text}\n\n{_privileged_prefix(label, K)}\n\n"
        full_text = teacher_prefix_text + response_text

        # logprob_start_len is in TEACHER tokens. SGLang's logprob serialization also
        # degrades with context length (CPU backend, sglang #27196), so we still return
        # logprobs over the response only — the privileged prefix just conditions.
        teacher_tok = _get_teacher_tokenizer()
        prompt_length = _teacher_response_boundary(teacher_tok, teacher_prefix_text, full_text)

        payload = {
            "text": full_text,
            "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
            "return_logprob": True,
            # Only return logprobs for the student response tokens; exclude the prefix.
            "logprob_start_len": prompt_length,
            "return_text_in_logprobs": True,
        }
        response_token_count = resp_len

        # If the teacher fails to response after max_attempts with retries enabled,
        # then we exclude its logprobs from the OPD calculation and continue training.
        skip_opd = {"meta_info": {"input_token_logprobs": []}}

        # Add a timeout for the teacher request.    
        request_timeout = max(60, response_token_count * 10 // 1000)
        max_attempts = 20
        for attempt in range(max_attempts):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        args.rm_url, json=payload,
                        timeout=aiohttp.ClientTimeout(total=request_timeout),
                    ) as resp:
                        resp.raise_for_status()
                        return await resp.json()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                status = getattr(e, "status", None)
                if status == 400:
                    print(
                        f"[rm] OPD skipped: 400 for {len(sample.tokens)}-token sample",
                        flush=True,
                    )
                    return skip_opd
                is_retryable = status in (503, 502, 504, 429, None)
                if attempt == max_attempts - 1 or not is_retryable:
                    print(
                        f"[rm] OPD skipped: teacher /generate failed after {attempt + 1} "
                        f"attempts (status={status}, tokens={len(sample.tokens)}); "
                        "continuing training without OPD for this trajectory",
                        flush=True,
                    )
                    return skip_opd
                base = min(60.0, 2 ** min(attempt, 6))
                wait = base + random.uniform(0, base * 0.25)
                print(
                    f"[rm] teacher /generate attempt {attempt + 1}/{max_attempts} status={status}; "
                    f"retry in {wait:.1f}s",
                    flush=True,
                )
                await asyncio.sleep(wait)

        print("[rm] OPD skipped: teacher /generate exhausted all retries", flush=True)
        return skip_opd




@markdown
def _generate_intro():
    """
    ## Initializing from the K Expert Trajectory Step

    Following the reverse curriculum described in "Custom Student Training Curriculum", each rollout initializes the
    student's prompt from step K in Toolathlon's golden expert trajectory. As training progress, the index K will decrement
    by one every time the student passes Toolathlon's terminal environment evaluation and the pass rate of all student samples is
    greater than or equal to CURRICULUM_PASS_RATE. The sequence of rollout iterations will finish before K=0 or the student will continue
    training at K=0 for however many extra iterations there are. Additionally, the Toolathlon MCP environments are initialized to the
    state of the executed expert trajectory snapshot at the Kth step. This feature for Toolathlon was merged into the training-gym
    and leverages Modal directory snapshots for replayable state over the expert trajectory (https://modal.com/docs/guide/sandbox-snapshots#directory-snapshots).
    """


@code
def _generate():
    async def tool_step_generate(args, sample, sampling_params):
        from slime.rollout.sglang_rollout import GenerateState
        from slime.utils.http_utils import post
        from slime.utils.types import Sample

        state = GenerateState(args)
        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
        label = json.loads(getattr(sample, "label", "{}"))

        task_name = label["task_name"]
        N = label.get("total_steps", 1)
        golden_calls = label.get("golden_calls", [])
        tool_schemas = label.get("tool_schemas", {})

        T = int(getattr(args, "curriculum_tail", EVAL_TAIL_STEPS))
        K = max(0, N - T)
        max_turns = min(int(getattr(args, "max_turns", MAX_TURNS)), max(T * 2, 4))
        expert_call = golden_calls[K] if K < len(golden_calls) else {}

        prefix_msgs = build_prefix_messages(label, K)
        tools_list = tool_schemas_to_openai(tool_schemas)
        prompt_text = state.tokenizer.apply_chat_template(
            prefix_msgs, tools=tools_list, tokenize=False,
            add_generation_prompt=True, enable_thinking=STUDENT_ENABLE_THINKING,
        )
        prompt_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        base_render = state.tokenizer.apply_chat_template(
            prefix_msgs, tools=tools_list, tokenize=False,
            add_generation_prompt=False, enable_thinking=STUDENT_ENABLE_THINKING,
        )
        gen_suffix = prompt_text[len(base_render):]
        probe = state.tokenizer.apply_chat_template(
            prefix_msgs
            + [
                {"role": "assistant", "content": "\x01A\x01"},
                {"role": "tool", "content": "\x00OBS\x00"},
            ],
            tools=tools_list, tokenize=False,
            add_generation_prompt=True, enable_thinking=STUDENT_ENABLE_THINKING,
        )
        after_assistant = probe.split("\x01A\x01", 1)[1]
        obs_open, _rest = after_assistant.split("\x00OBS\x00", 1)
        obs_close = _rest[: len(_rest) - len(gen_suffix)]
        stop_tok = obs_open.split("\n", 1)[0]
        verbose = ROLLOUT_LOG_EVERY > 0 and (getattr(sample, "index", 0) % ROLLOUT_LOG_EVERY == 0)

        def _log(msg):
            print(f"[rollout:{task_name} T={T} K={K}] {msg}", flush=True)

        if verbose:
            _log(f"start N={N} prompt_tokens={len(prompt_ids)} max_turns={max_turns}")

        pool = get_env_pool()
        try:
            env = await asyncio.to_thread(pool.acquire, task_name, K)
        except Exception as e:
            _log(f"sandbox acquire failed: {e!r} — skipping rollout")
            sample.status = Sample.Status.ABORTED
            return sample
        trajectory_text = ""
        response_segments: list[tuple[str, int]] = []

        student_calls: list = []
        exec_successes: list = []
        finish_type = "stop"
        try:
            for turn in range(max_turns):
                output = await post(url, {
                    "text": prompt_text + trajectory_text,
                    "sampling_params": sampling_params,
                })
                finish_type = output["meta_info"]["finish_reason"]["type"]
                if finish_type == "abort":
                    sample.status = Sample.Status.ABORTED
                    return sample

                model_text = output["text"]
                trajectory_text += model_text
                response_segments.append((model_text, 1))

                actions = base_model.parse_response(model_text).tool_calls
                action = actions[0] if actions else None
                student_calls.append(
                    {"name": action.name, "arguments": action.arguments} if action else None
                )

                if action is None:  
                    if verbose:
                        _log(f"turn {turn} MALFORMED")
                    break
                if action.name in ("claim_done", "local-claim_done"):
                    if verbose:
                        _log(f"turn {turn} claim_done")
                    break

                try:
                    result = await asyncio.to_thread(env.step, action)
                    obs_text, is_error = result.observation.text, result.observation.is_error
                except Exception as e:
                    _log(f"turn {turn} sandbox error: {e!r} — ending episode")
                    exec_successes.append(False)
                    finish_type = "stop"
                    break
                exec_successes.append(not is_error)
                if verbose:
                    _log(f"turn {turn} tool={action.name} -> {'ERR' if is_error else 'ok'}")
                seg_open = (
                    obs_open[len(stop_tok):] if model_text.endswith(stop_tok) else obs_open
                )
                obs_segment = seg_open + obs_text[:2000] + obs_close + gen_suffix
                trajectory_text += obs_segment
                response_segments.append((obs_segment, 0))

                if finish_type == "length":
                    break

            try:
                verdict = await asyncio.to_thread(env.evaluate)
                task_passed = verdict.passed
            except Exception as e:
                _log(f"sandbox gone before evaluate: {e!r} — marking failed")
                task_passed = False
        finally:
            await asyncio.to_thread(pool.release, env)

        expert_calls = golden_calls[K:K + len(student_calls)]

        _log(
            f"done turns={len(student_calls)} "
            f"exec_ok={sum(exec_successes)}/{len(exec_successes)} "
            f"finish={finish_type} passed={bool(task_passed)}"
        )

        response_token_ids: list[int] = []
        loss_masks: list[int] = []
        for seg, trainable in response_segments:
            tids = state.tokenizer(seg, add_special_tokens=False)["input_ids"]
            response_token_ids += tids
            loss_masks += [trainable] * len(tids)

        sample.tokens = prompt_ids + response_token_ids
        sample.response_length = len(response_token_ids)
        sample.response = trajectory_text
        sample.loss_mask = loss_masks
        sample.status = (
            Sample.Status.TRUNCATED if finish_type == "length"
            else Sample.Status.COMPLETED
        )
        sample.metadata = {
            "student_calls": student_calls,
            "exec_successes": exec_successes,
            "expert_calls": expert_calls,
            "tool_schemas": tool_schemas,
            "task_passed": bool(task_passed),
            "tail_len": T,
            "curriculum_tail": T,
            "step_K": K,
        }

        sample.toolathlon_student_calls = student_calls
        sample.toolathlon_exec_successes = exec_successes
        sample.toolathlon_expert_calls = expert_calls
        sample.toolathlon_tool_schemas = tool_schemas
        sample.toolathlon_task_passed = bool(task_passed)
        sample.toolathlon_tail_len = T
        return sample




@code
def _post_process():
    def cross_tokenizer_post_process(args, samples, **kwargs):
        """Compute step rewards + align teacher logprobs via SimCT MAUs."""
        import torch

        tokenizer = _get_student_tokenizer()
        raw_rewards = [s.get_reward_value(args) for s in samples]

        rewards = []
        first_call_hits = 0
        for sample in samples:
            meta = getattr(sample, "metadata", {}) or {}
            sc_list = (
                getattr(sample, "toolathlon_student_calls", None)
                or meta.get("student_calls", [])
                or []
            )
            ec_list = (
                getattr(sample, "toolathlon_expert_calls", None)
                or meta.get("expert_calls", [])
                or []
            )
            sc0 = sc_list[0] if sc_list else None
            ec0 = ec_list[0] if ec_list else {}
            if sc0 and ec0 and sc0.get("name") == ec0.get("name"):
                first_call_hits += 1
            task_passed_from_attr = (
                getattr(sample, "toolathlon_task_passed", None)
                if hasattr(sample, "toolathlon_task_passed")
                else None
            )
            task_passed_value = (
                task_passed_from_attr if task_passed_from_attr is not None
                else meta.get("task_passed")
            )
            r = trajectory_reward(
                sc_list,
                getattr(sample, "toolathlon_exec_successes", None)
                or meta.get("exec_successes", [])
                or [],
                ec_list,
                getattr(sample, "toolathlon_tool_schemas", None)
                or meta.get("tool_schemas", {}),
                task_passed_value,
                getattr(sample, "toolathlon_tail_len", None)
                or meta.get("tail_len", 1),
            )
            rewards.append(r)
            try:
                sample.reward = r
            except Exception:
                pass

        cur_tail = getattr(args, "curriculum_tail", None)
        hit_rate = (first_call_hits / len(samples)) if samples else 0.0

        if rewards:
            n = len(samples)
            n_pass = sum(1 for s in samples if (getattr(s, "metadata", {}) or {}).get("task_passed"))
            all_exec = [
                ok for s in samples
                for ok in ((getattr(s, "metadata", {}) or {}).get("exec_successes") or [])
            ]
            exec_ok = (sum(1 for ok in all_exec if ok) / len(all_exec)) if all_exec else 0.0
            mean_reward = sum(rewards) / n
            print(
                f"[group] tail={cur_tail} "
                f"reward[min/mean/max]={min(rewards):.2f}/{mean_reward:.2f}/{max(rewards):.2f} "
                f"pass={n_pass}/{n} first_call={first_call_hits}/{n} exec_ok={exec_ok:.2f}",
                flush=True,
            )

        unaligned = total_resp = opd_dropped = 0
        for sample, reward in zip(samples, raw_rewards):
            r_meta = reward.get("meta_info", {})
            raw_logprobs = r_meta.get("input_token_logprobs", [])
            entries = raw_logprobs[1:]

            t_lps = [e[0] if e[0] is not None else 0.0 for e in entries if e is not None]
            t_texts = [e[2] if len(e) > 2 else "" for e in entries if e is not None]

            # Teacher logprob response starts at an index after the expert trajectory input prompt. 
            # Align the teacher's token logprobs with the start of the student response sequence.
            resp_tokens = sample.tokens[-sample.response_length:]
            s_texts = [tokenizer.decode([tid], skip_special_tokens=True) for tid in resp_tokens]
            aligned, covered = align_cross_tokenizer(t_texts, t_lps, s_texts, return_coverage=True)

            # If teacher failed to respond (OPD_SKIP_ON_TEACHER_FAILURE), then
            # return a NaN teacher tensor. The patched slime OPD term detects a tensor
            # with torch.isnan() and zeros out the OPD term. Before the patch in the training-gym (#155),
            # this custom function would have to return a tensor of zeros to prevent a ValueError in Slime.
            if not raw_logprobs and OPD_SKIP_ON_TEACHER_FAILURE:
                sample.teacher_log_probs = torch.full(
                    (len(aligned),), float("nan"), dtype=torch.float32
                )
                opd_dropped += 1
                continue

            sample.teacher_log_probs = aligned
            resp_covered = covered
            unaligned += int((~resp_covered).sum().item())
            total_resp += int(resp_covered.numel())

        if total_resp:
            gap_frac = unaligned / total_resp
            print(
                f"[align] teacher_align_gap={unaligned}/{total_resp} "
                f"({gap_frac:.1%}) response tokens unaligned — high => cross-tokenizer misalignment",
                flush=True,
            )
        if opd_dropped:
            print(
                f"[opd] disabled OPD for {opd_dropped}/{len(samples)} trajectories "
                "(teacher unavailable — NaN sentinel; GRPO gradient kept)",
                flush=True,
            )

        return rewards, rewards




@markdown
def _eval_base_intro():
    """
    ## Baseline eval

    Before training, we measure the Qwen3.6-35B-A3B on the held-out Tier A tasks excel-data-transformation, ppt-analysis, and 
    interview-report. Using our K curriculum and directory snapshotting technique, we initialize the agent context and sandbox
    environments to the Kth step where K is N - 8 to give us a reasonable baseline to beat with training.
    """


@code
def _eval_base():
    SERVED_CONTEXT_LEN = 131072
    RESPONSE_TOKEN_CAP = 8192
    CONTEXT_SAFETY_MARGIN = 512
    _RETRYABLE_STATUS = (429, 500, 502, 503, 504)

    EVAL_MAX_TURNS = EVAL_TAIL_STEPS * 2
    MAX_CONSECUTIVE_TOOL_ERRORS = 3
    DEPLOYMENT_READY_TIMEOUT = 1200 

    def _prompt_token_count(messages, tools=None) -> int:
        try:
            tok = _get_student_tokenizer()
            text = tok.apply_chat_template(
                messages, tools=tools, tokenize=False, add_generation_prompt=True
            )
            return len(tok(text, add_special_tokens=False)["input_ids"])
        except Exception:
            return sum(len(str(m.get("content", ""))) for m in messages) // 3

    def _chat(deployment, messages, tools=None, max_tokens=None, max_attempts=12):
        import random
        import time

        import requests as _requests

        if max_tokens is None:
            max_tokens = RESPONSE_TOKEN_CAP
        remaining = SERVED_CONTEXT_LEN - _prompt_token_count(messages, tools) - CONTEXT_SAFETY_MARGIN
        capped = min(max_tokens, remaining)
        if capped <= 0:
            return ""

        body = {
            "model": deployment.deployment_config.served_model_name,
            "messages": messages,
            "tools": tools,
            "temperature": 0.0,
            "max_tokens": capped,
            "chat_template_kwargs": {"enable_thinking": STUDENT_ENABLE_THINKING},
        }
        for attempt in range(max_attempts):
            try:
                resp = _requests.post(
                    f"{deployment.url}/v1/chat/completions", json=body, timeout=120
                )
                if resp.status_code in _RETRYABLE_STATUS:
                    raise _requests.HTTPError(f"retryable status {resp.status_code}", response=resp)
                resp.raise_for_status()
                msg = resp.json()["choices"][0]["message"]
                return msg.get("content") or msg.get("reasoning_content", "") or ""
            except (_requests.ConnectionError, _requests.Timeout, _requests.HTTPError) as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                retryable = status in _RETRYABLE_STATUS or status is None
                if attempt == max_attempts - 1 or not retryable:
                    raise
                wait = min(30.0, 2 ** attempt) + random.uniform(0, 1.0)
                time.sleep(wait)
        return ""

    def toolathlon_eval_fn(deployment: ModelDeployment, example: dict) -> EvalRowResult:
        label = json.loads(example.get("label", "{}"))
        task_name = label.get("task_name", "")
        N = label.get("total_steps", 1)
        golden_calls = label.get("golden_calls", [])
        K = 1
        expert_call = golden_calls[K] if K < len(golden_calls) else {}

        messages = build_prefix_messages(label, K)
        tools_list = tool_schemas_to_openai(label.get("tool_schemas", {}))

        import time as _time

        def _log(msg):
            print(f"[eval:{task_name} K={K}] {msg}", flush=True)

        deployment.wait_until_ready(timeout=DEPLOYMENT_READY_TIMEOUT)
        env = get_env_pool().acquire(task_name, K)
        first_call = None
        student_calls = []
        exec_successes = []
        last_response = ""
        consecutive_errors = 0
        exit_reason = "max_turns"
        done = False
        shaped_score = 0.0
        _log(f"start tail={EVAL_TAIL_STEPS} max_turns={EVAL_MAX_TURNS}")
        try:
            for turn in range(EVAL_MAX_TURNS):
                _t0 = _time.monotonic()
                response = _chat(deployment, messages, tools=tools_list)
                gen_s = _time.monotonic() - _t0
                last_response = response

                parsed = base_model.parse_response(response)
                actions = parsed.tool_calls
                if turn == 0: 
                    first_call = (
                        {"name": actions[0].name, "arguments": actions[0].arguments} if actions else None
                    )
                _log(f"turn {turn} gen={gen_s:.1f}s calls={[a.name for a in actions]}")
                if not actions:
                    exit_reason = "unparseable"
                    break

                observations: list[tuple[str, str]] = []
                for action in actions:
                    name = action.name
                    student_calls.append({"name": action.name, "arguments": action.arguments})
                    if name in ("claim_done", "local-claim_done"):
                        exit_reason = "claim_done"
                        done = True
                        break
                    try:
                        step_result = env.step(action)
                        obs, is_error = step_result.observation.text, step_result.observation.is_error
                    except Exception as e:
                        _log(f"  sandbox error on {name}: {e!r} — ending episode")
                        exec_successes.append(False)
                        exit_reason = "sandbox_error"
                        done = True
                        break
                    exec_successes.append(not is_error)
                    _log(f"  exec {name} -> {'ERR' if is_error else 'ok'}")
                    observations.append((name, obs))
                    consecutive_errors = consecutive_errors + 1 if is_error else 0
                    if consecutive_errors >= MAX_CONSECUTIVE_TOOL_ERRORS:
                        exit_reason = "repeated_errors"
                        done = True
                        break
                if done:
                    break

                call_ids = [f"call_t{turn}_{i}" for i in range(len(actions))]
                messages.append({
                    "role": "assistant",
                    "content": parsed.content,
                    "tool_calls": [
                        {
                            "id": cid,
                            "type": "function",
                            "function": {"name": a.name, "arguments": json.dumps(a.arguments)},
                        }
                        for cid, a in zip(call_ids, actions)
                    ],
                })
                for cid, (name, obs) in zip(call_ids, observations):
                    messages.append(
                        {"role": "tool", "tool_call_id": cid, "content": obs[:2000]}
                    )
            try:
                task_passed = env.evaluate().passed
            except Exception as e:
                _log(f"sandbox gone before evaluate: {e!r} — marking failed")
                task_passed = False
            expert_calls = golden_calls[K:K + len(student_calls)]
            shaped_score = trajectory_reward(
                student_calls,
                exec_successes,
                expert_calls,
                label.get("tool_schemas", {}),
                bool(task_passed),
                EVAL_TAIL_STEPS,
            )
            _log(
                f"done exit={exit_reason} passed={bool(task_passed)} "
                f"reward={shaped_score:.3f} exec_ok={sum(exec_successes)}/{len(exec_successes)}"
            )
        finally:
            get_env_pool().release(env)

        return EvalRowResult(
            score=shaped_score,
            response=last_response,
            metadata={
                "task": task_name,
                "step_K": K,
                "eval_tail": EVAL_TAIL_STEPS,
                "task_passed": bool(task_passed),
                "terminal_score": 1.0 if task_passed else 0.0,
                "shaped_reward": shaped_score,
                "exec_successes": sum(exec_successes),
                "exec_calls": len(exec_successes),
                "parsed_call": first_call is not None,
                "tool_match": bool(first_call and first_call.get("name") == expert_call.get("name")),
            },
        )

    student_recipe = Qwen3_6_35b_SglangRecipe(context_length=SERVED_CONTEXT_LEN)
    base_deployment = DeploymentConfig(model=base_model, recipe=student_recipe).serve()
    print(f"Student URL: {base_deployment.url}")

    eval_config = EvalConfig(dataset=dataset, eval_fn=toolathlon_eval_fn)
    print("--- Evaluating base student (shaped live reward + terminal verdict metadata)... ---")
    base_eval = eval_config.evaluate(base_deployment, debug=True, max_concurrency=4)
    print(f"Base shaped reward: {base_eval.mean:.3f}")




@markdown
def _train_intro():
    """
    ## Training

    Each rollout draws 12 prompts and 4 samples, resulting in 48 total trajectories. Training runs
    on a 8:H200 cluster with TP=2, and DP=2 for increased performance, and the rollouts run on a separate 8:H200
    connected to the training node via RDMA. The reward function uses GRPO to compute a group-relative advantage
    per sample. To prefer the reward of GRPO over the reverse KL loss, we set opd_kl_coef=0.3 in Slime.
    """


@code
def _train():

    training_run = TrainConfig(
        model=base_model,
        dataset=dataset,
        recipe=Qwen3_6_35b_Recipe(
            custom_rm_function=cross_tokenizer_reward,
            custom_generate_function=tool_step_generate,
            rollout_function=curriculum_rollout,
            capture_trace=True,
            trace_sample_limit=16,
            image_overlay=lambda img: img.pip_install(
                "modal~=1.4.3", "huggingface_hub~=1.12", "aiohttp~=3.13", "jsonschema~=4.23",
            ),

            gpu_type="H200",
            colocate=False,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            rollout_num_gpus=8,
            tensor_model_parallel_size=2,
            sequence_parallel=True,
            pipeline_model_parallel_size=2,
            context_parallel_size=2,
            expert_model_parallel_size=4,
            rollout_num_gpus_per_engine=8,
            sglang_dp_size=8,
            sglang_ep_size=8,
            sglang_cuda_graph_bs=[1, 2, 4, 8, 16, 24, 32, 48],
            sglang_max_running_requests=48,

            num_rollout=5,
            rollout_batch_size=12,
            n_samples_per_prompt=4,
            rollout_max_response_len=12288,
            rollout_temperature=0.6,
            sglang_mem_fraction_static=0.75,

            global_batch_size=16,
            lr=1e-6,
            save_interval=5,

            environment={
                "PYTHONPATH": "/root/Megatron-LM/:/root",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "NCCL_NVLS_ENABLE": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            },

            extra_config={
                "use_opd": True,
                "opd_type": "sglang",
                "opd_kl_coef": 0.3,
                "custom_reward_post_process_path": (
                    "007_cross_tokenizer_distillation"
                    ".cross_tokenizer_post_process"
                ),
                "rm_url": TEACHER_GENERATE_URL,
                "max_turns": MAX_TURNS,
                "log_multi_turn": True,
                "save_debug_rollout_data": "/checkpoints/debug/rollout_{rollout_id}.pt",
            },
        ),
    )

    print("--- Starting GRPO + cross-tokenizer OPD training... ---")
    print(f"  Teacher: DeepSeek V4 Flash")
    print(f"  Student: Qwen3.6-35B-A3B")
    print(f"  Dataset: Toolathlon Tier A, prefix-conditioned (task, K) rows")
    print(f"  Reward: schema + live exec + structural match + terminal eval verdict")
    train_result = training_run.train()
    print(f"Training run id: {train_result.training_run_id}")
    print("--- Training complete ---")


@markdown
def _eval_trained_intro():
    """
    ## Evaluate the trained student

    Deploy the last checkpoint and re-run the held-out Tier A tasks with the same evaluator from our earlier baseline.
    """


@code
def _eval_trained():
    checkpoint = list_checkpoints(train_result.training_run_id)[-1]
    print(f"Checkpoint: {checkpoint.path}")

    trained_deployment = DeploymentConfig(
        model=Qwen3_6_35B(),
        recipe=student_recipe,
        checkpoint=checkpoint,
        app_name="qwen3-6-35b-toolathlon-trained",
        served_model_name="qwen3-6-35b-toolathlon-trained",
    ).serve()
    print(f"Trained student URL: {trained_deployment.url}")

    print("--- Evaluating trained student (shaped live reward + terminal verdict metadata)... ---")
    trained_eval = eval_config.evaluate(trained_deployment, debug=True, max_concurrency=4)
    print(f"Trained shaped reward: {trained_eval.mean:.3f}")




@code
def _compare():
    def _frac(rows, key):
        if not rows:
            return 0.0
        return sum(1 for r in rows if r.metadata.get(key)) / len(rows)

    n_base = len(base_eval.rows)
    n_trained = len(trained_eval.rows)

    print(f"{'Metric':<25} {'Base':>10} {'Trained':>10} {'Delta':>10}")
    print("-" * 57)
    print(f"{'Eval rows':<25} {n_base:>10d} {n_trained:>10d} {'':>10}")
    print(f"{'Shaped reward':<25} {base_eval.mean:>10.3f} {trained_eval.mean:>10.3f} {trained_eval.mean - base_eval.mean:>+10.3f}")

    base_pass_rate = _frac(base_eval.rows, "task_passed")
    trained_pass_rate = _frac(trained_eval.rows, "task_passed")
    print(f"{'Terminal pass rate':<25} {base_pass_rate:>10.1%} {trained_pass_rate:>10.1%} {trained_pass_rate - base_pass_rate:>+10.1%}")

    if n_base and n_trained:
        base_parse = _frac(base_eval.rows, "parsed_call")
        trained_parse = _frac(trained_eval.rows, "parsed_call")
        print(f"{'Parsed tool call':<25} {base_parse:>10.1%} {trained_parse:>10.1%} {trained_parse - base_parse:>+10.1%}")

        base_match = _frac(base_eval.rows, "tool_match")
        trained_match = _frac(trained_eval.rows, "tool_match")
        print(f"{'First-call tool match':<25} {base_match:>10.1%} {trained_match:>10.1%} {trained_match - base_match:>+10.1%}")
    else:
        print("(no eval rows — check the eval dataset / deployment)")


@markdown
def _next_steps():
    """
    ## Next steps

    Ways to improve this tutorial:

    1. **Service-backed tasks (Tier B)**: extend beyond the snapshot-safe Tier A set to tasks whose state lives in-memory
    (k8s/kind, WooCommerce→MySQL, Canvas→Postgres, emails→poste.io). 
    2. **Different student/teacher model configuration**: If the gap between tokenizers is high before alignment, 
    try different cross-tokenization strategies (https://neurips.cc/virtual/2025/loc/san-diego/poster/119176).
    4. **Densify the KL with top-k logprobs**: today the OPD reverse KL uses a single per-token teacher logprob. SGLang can
    also return the teacher's top-k distribution at each position (`top_logprobs_num`); extending slime's OPD loss to consume
    that full distribution — rather than just the chosen-token logprob — would give a denser distillation signal (see
    `loss_patching_future.md`).
    """
