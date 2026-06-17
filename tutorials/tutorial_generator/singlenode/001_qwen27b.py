# pyright: reportUndefinedVariable=false
"""Tutorial source for `001_qwen27b` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "1 × 8×H100",
    "summary": "Train Qwen3.6-27B on DAPO-math with GRPO",
    "difficulty": "Advanced",
    "order": 1,
    "api_classes": [
        "HuggingFaceDataset",
        "DeploymentConfig",
        "ModelDeployment",
        "Qwen3_6_27B",
        "Qwen3_6_27b_Recipe",
        "TrainConfig",
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Training Qwen3.6-27B on DAPO-math

    This tutorial trains **Qwen3.6-27B** (a 27B-parameter *dense* model)
    on grade-school math problems from
    [DAPO-math-17k](https://huggingface.co/datasets/zhuzilin/dapo-math-17k).

    The loop:
    1. Load math problems from HuggingFace via `HuggingFaceDataset`.
    2. Score model outputs using slime's built-in `deepscaler` reward
       model, which extracts the final numerical answer and compares
       it to the ground truth.
    3. Feed that score back as a GRPO reward through SLIME.
    4. Compare base vs. trained accuracy.

    Qwen3.6-27B is a *hybrid* model: most layers use Gated DeltaNet
    (linear attention) with a periodic full Gated Attention layer, in a
    3:1 ratio. slime builds this layer pattern from a model-specific spec
    (`slime_plugins.models.qwen3_5`), and the training gym wires it up
    automatically because the model's `megatron_model_type` is set —
    which also triggers checkpoint pre-conversion from HF to Megatron
    format before training starts.
    """


@py_only
@markdown
def _run_instructions():
    """
    Run with:
    ```
    uv run modal run -d tutorials/singlenode/001_qwen27b/001_qwen27b.py
    ```
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@main")
def _install():
    pass


@code
def _imports():
    from modal_training_gym import (
        DeploymentConfig,
        HuggingFaceDataset,
        Qwen3_6_27B,
        TrainConfig,
        list_checkpoints,
    )
    from modal_training_gym.deploy_recipes.sglang_recipe import Qwen3_6_27b_SglangRecipe
    from modal_training_gym.train_recipes.slime_recipe import Qwen3_6_27b_Recipe


@markdown
def _dataset_intro():
    """
    ## Load DAPO-math from HuggingFace

    [DAPO-math-17k](https://huggingface.co/datasets/zhuzilin/dapo-math-17k)
    contains ~17k math problems with ground-truth answers. We use a
    small subset for this tutorial — 100 training samples and 20 for eval.
    """


@code
def _dataset():
    class MathDataset(HuggingFaceDataset):
        hf_repo = "zhuzilin/dapo-math-17k"
        input_column = "prompt"
        output_column = "label"
        output_format = "jsonl"
        apply_chat_template = True
        always_prepare = True

    dataset = MathDataset(n_rows=120)


@notebook_only
@markdown
def _dataset_preview():
    """
    Let's take a quick look at the dataset.
    """


@notebook_only
@code
def _dataset_preview_code():
    rows = dataset.load()
    for row in rows.select(range(2)):
        prompt = row["prompt"]
        if isinstance(prompt, list):
            prompt = prompt[0]["content"] if prompt else ""
        print(prompt[:200])
        print(f"  label: {row['label']}")
        print()


@markdown
def _train_intro():
    """
    ## Train with SLIME

    This dense model runs on a single 8×H100 node with TP4 and PP2,
    plus optimizer CPU offload. This
    is a single-node adaptation of the validated Miles Qwen3.5-27B config.

    Key points:
    - **`rm_type="deepscaler"`** — slime's built-in math reward that
      extracts and compares numerical answers. No custom reward function
      or sandbox needed.
    - The model's `megatron_model_type` triggers automatic checkpoint
      pre-conversion from HF to Megatron format before training starts.
    """


@code
def _train():
    training_run = TrainConfig(
        model=Qwen3_6_27B(),
        dataset=dataset,
        recipe=Qwen3_6_27b_Recipe(
            rm_type="deepscaler",
            rollout_max_response_len=4096,
            eval_max_response_len=4096,
            n_samples_per_eval_prompt=4,
            train_function_kwargs={"ephemeral_disk": 2_097_152},
        ),
    )
    print("Starting training...")
    train_result = training_run.train()
    print(f"Training run id: {train_result.training_run_id}")


@markdown
def _serve_eval_intro():
    """
    ## Serve and evaluate

    Serve the trained checkpoint and run a quick math eval.
    """


@code
def _serve_trained():
    checkpoint = list_checkpoints(train_result.training_run_id)[-1]
    trained_deployment = DeploymentConfig(
        model=Qwen3_6_27B(),
        recipe=Qwen3_6_27b_SglangRecipe(),
        checkpoint=checkpoint,
        app_name="qwen3-6-27b-math-serve",
        served_model_name="qwen3-6-27b-math",
    ).serve()
    print(f"Trained model URL: {trained_deployment.url}")
