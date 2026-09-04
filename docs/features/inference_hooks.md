# Inference Hooks

vLLM exposes two complementary ways to instrument model execution with PyTorch
forward hooks: a built-in NVTX-based system for CUDA profiling, and direct use
of PyTorch's hook API for any backend (CPU, MPS, XPU, CUDA).

## PyTorch forward hooks (all backends)

PyTorch's [`register_forward_hook`](https://pytorch.org/docs/stable/generated/torch.nn.Module.register_forward_hook.html)
and [`register_forward_pre_hook`](https://pytorch.org/docs/stable/generated/torch.nn.Module.register_forward_pre_hook.html)
work on every backend — CUDA, MPS (Apple Silicon), CPU, and XPU — because they
are pure Python callbacks invoked by `nn.Module.__call__`, not GPU-specific.
This is the right approach when running on a Mac or any non-CUDA platform.

### Attaching hooks after model load

The model is accessible through the `LLM` object after initialization. Hooks
must be registered **before** the first inference call.

```python
from vllm import LLM, SamplingParams

llm = LLM(model="ibm-granite/granite-4.2-3b", enforce_eager=True)

# Reach the underlying nn.Module
# The exact path depends on the executor backend; this works for the
# local (single-process) path used on CPU and MPS.
model = llm.llm_engine.model_executor.driver_worker.model_runner.model

# Inspect available submodule names
for name, _ in model.named_modules():
    print(name)
```

!!! note
    Pass `enforce_eager=True` to disable CUDA graph capture. Hooks are
    incompatible with CUDA graphs because the captured graph replays raw GPU
    operations and bypasses Python callbacks entirely. On CPU and MPS this is
    the default anyway.

### Recording FFN inputs and outputs

The example below captures the input and output of every layer's MLP for a
Granite-style model (gate/up projection merged into `gate_up_proj`, down
projection as `down_proj`). The same pattern applies to any other submodule.

```python
from vllm import LLM, SamplingParams

llm = LLM(model="ibm-granite/granite-4.2-3b", enforce_eager=True)
model = llm.llm_engine.model_executor.driver_worker.model_runner.model

captured: dict[str, list] = {}
hook_handles = []


def make_ffn_hooks(layer_idx: int):
    key_up_in  = f"layer{layer_idx}.gate_up_proj.input"
    key_up_out = f"layer{layer_idx}.gate_up_proj.output"
    key_dn_in  = f"layer{layer_idx}.down_proj.input"
    key_dn_out = f"layer{layer_idx}.down_proj.output"

    def pre_up(module, args, kwargs):
        captured.setdefault(key_up_in, []).append(args[0].detach().cpu())

    def post_up(module, args, output):
        captured.setdefault(key_up_out, []).append(output[0].detach().cpu())

    def pre_dn(module, args, kwargs):
        captured.setdefault(key_dn_in, []).append(args[0].detach().cpu())

    def post_dn(module, args, output):
        captured.setdefault(key_dn_out, []).append(output[0].detach().cpu())

    return pre_up, post_up, pre_dn, post_dn


for i, layer in enumerate(model.model.layers):
    pre_up, post_up, pre_dn, post_dn = make_ffn_hooks(i)
    hook_handles += [
        layer.mlp.gate_up_proj.register_forward_pre_hook(pre_up, with_kwargs=True),
        layer.mlp.gate_up_proj.register_forward_hook(post_up),
        layer.mlp.down_proj.register_forward_pre_hook(pre_dn, with_kwargs=True),
        layer.mlp.down_proj.register_forward_hook(post_dn),
    ]

outputs = llm.generate("Hello, world!", SamplingParams(max_tokens=8))

# Remove hooks when done
for h in hook_handles:
    h.remove()

# Inspect recorded tensors
for key, tensors in captured.items():
    print(key, [t.shape for t in tensors])
```

The `hook_handles` list holds [`RemovableHook`](https://pytorch.org/docs/stable/generated/torch.utils.hooks.RemovableHook.html)
objects — call `.remove()` on each to deregister when done. Failing to remove
hooks leaves them active for all subsequent calls.

### Hook-based activation sampling (CLI)

`tools/profiler/record_ffn_activations.py` provides a ready-to-run script that
wires up the hooks described above and saves the collected tensors to a NumPy
`.npz` archive.  It works on any backend — CPU, MPS (Apple Silicon), and CUDA —
because it uses plain PyTorch hooks with `enforce_eager=True`.

#### Environment

The script must be run with the repo-local virtualenv's Python.  The venv
lives at `.venv/` inside the repository root and is managed with `uv`:

```bash
# Create once if needed (from the vllm/ repo root):
uv venv --python 3.12
uv pip install numpy torch
```

On macOS `python` may not exist or may resolve to the system Homebrew Python
even after `source .venv/bin/activate`.  Always invoke the interpreter
explicitly to be safe:

```bash
.venv/bin/python tools/profiler/record_ffn_activations.py ...
```

Never use `python3` from the system or a bare `pip install` — all packages
must be installed inside `.venv/` via `uv pip`.

#### What is recorded

For each instrumented transformer layer the script records:

| Key in `.npz` | Shape | Description |
| --- | --- | --- |
| `layer<N>/gate_up_input` | `[T, H]` | Hidden states entering the MLP (`H` = hidden size, `T` = total tokens) |
| `layer<N>/down_input` | `[T, I]` | Post-SiluAndMul activations that feed `down_proj` (`I` = intermediate size) |
| `layer<N>/neuron_activity` | `[I]` | Mean absolute value of `down_input` across all `T` tokens |

Tensor chunks are streamed to temporary files on disk as they are observed,
so peak RAM during inference is bounded to roughly one batch's activations
regardless of how many prompts are processed.  The model weights are freed
before the final concatenation step.

Pass `--no-save-tensors` to skip the per-token matrices entirely and keep
only `neuron_activity` — the smallest possible output.

#### Command line

```bash
# Record all layers, 1 generated token per prompt
.venv/bin/python tools/profiler/record_ffn_activations.py \
    --model ibm-granite/granite-4.2-3b \
    --prompts "The capital of France is" "Once upon a time" \
    --output ffn_activations.npz

# Record only layers 0 and 15 (reduces memory and file size)
.venv/bin/python tools/profiler/record_ffn_activations.py \
    --model ibm-granite/granite-4.2-3b \
    --prompts "Hello world" \
    --layers 0 15 \
    --output ffn_activations.npz

# Generate 16 tokens and record all layers
.venv/bin/python tools/profiler/record_ffn_activations.py \
    --model ibm-granite/granite-4.2-3b \
    --prompts "Explain quantum computing" \
    --max-tokens 16 \
    --output ffn_activations.npz

# Sample 128 random chunks from a calibration set
.venv/bin/python tools/profiler/record_ffn_activations.py \
    --model ibm-granite/granite-4.2-3b \
    --calibration-set bartowski-imatrix-v5-semantic.txt \
    --num-chunks 128 \
    --seed 42 \
    --output ffn_activations.npz

# Use all chunks from a calibration set (no sampling)
.venv/bin/python tools/profiler/record_ffn_activations.py \
    --model ibm-granite/granite-4.2-3b \
    --calibration-set bartowski-imatrix-v5-semantic.txt \
    --output ffn_activations.npz
```

Full option reference:

```
--model                Model name or local path (required)
--prompts              One or more prompt strings (mutually exclusive with --calibration-set)
--calibration-set FILE Path to a text file; every non-empty line is one chunk/prompt
                       (mutually exclusive with --prompts)
--num-chunks N         Randomly sample N chunks from --calibration-set; omit to use all
--seed S               Random seed for --num-chunks sampling (default: no fixed seed)
--chunk-len N          Truncate each prompt/chunk to at most N tokens before inference
                       (default: 512); set to 0 to disable truncation
--batch-size N         Prompts per generate() call (default: 8); increase for throughput
                       at the cost of higher peak RAM
--no-save-tensors      Skip per-token tensors (gate_up_input, down_input);
                       record only neuron_activity. Reduces RAM for large calibration sets.
--layers N …           Layer indices to record; omit to record all layers
--max-tokens           Number of new tokens to generate per prompt (default: 1)
--output               Output .npz path (default: ffn_activations.npz)
--dtype                Model dtype, e.g. float16, bfloat16, auto (default: bfloat16)
--kv-cache-gb          KV cache size in GiB (default: 1.0); keep small on CPU/Mac
--max-model-len        Maximum sequence length (default: 2048)
```

!!! note
    The script sets `VLLM_ENABLE_V1_MULTIPROCESSING=0` automatically so the
    model runs in-process and hooks can be attached.  This is required for hook
    access and is safe for single-prompt recording runs.

#### Analysing the results

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.load("ffn_activations.npz")

# Shapes of what was recorded
for key in sorted(data.files):
    print(f"{key}: {data[key].shape}")

# Top-20 most active intermediate neurons in layer 0
activity = data["layer0/neuron_activity"]      # shape (intermediate_size,)
top20 = np.argsort(activity)[-20:][::-1]
print("Top neurons:", top20)
print("Activity values:", activity[top20])

# Plot neuron activity for layer 0 vs layer 15
fig, axes = plt.subplots(1, 2, figsize=(12, 3))
for ax, layer in zip(axes, [0, 15]):
    key = f"layer{layer}/neuron_activity"
    if key in data:
        ax.bar(range(len(data[key])), data[key])
        ax.set_title(f"Layer {layer} neuron activity")
        ax.set_xlabel("Neuron index")
        ax.set_ylabel("Mean |activation|")
plt.tight_layout()
plt.savefig("neuron_activity.png", dpi=150)
```

#### Architecture notes

For **Llama / Mistral / Qwen** the gate and up projections are fused into a
single `MergedColumnParallelLinear` called `gate_up_proj`.  Its output has
shape `[T, 2 * intermediate_size]`; the script splits this at the midpoint into
`gate_raw` and `up_raw` before saving.  `SiluAndMul` is applied to those two
halves in-place inside the `LlamaMLP.forward` method — the result is what
arrives at `down_proj` as `down_input`.

For models that keep gate and up as separate modules (e.g. some Mixtral
variants), the script falls back to attaching to whichever of `gate_proj` /
`up_proj` / `c_fc` it finds first.  In that case `gate_raw` and `up_raw` in
the output will both represent the same projection.

---

### Finding the right submodule path

Module paths vary by model family. Use `model.named_modules()` to list them:

```python
for name, module in model.named_modules():
    if "mlp" in name or "ffn" in name:
        print(name, type(module).__name__)
```

Common patterns:

| Model family | Gate/up projection | Down projection |
| --- | --- | --- |
| Llama / Mistral / Qwen | `model.layers[N].mlp.gate_up_proj` | `model.layers[N].mlp.down_proj` |
| GPT-2 / GPT-J | `transformer.h[N].mlp.c_fc` | `transformer.h[N].mlp.c_proj` |
| Bloom / Falcon | `transformer.h[N].mlp.dense_h_to_4h` | `transformer.h[N].mlp.dense_4h_to_h` |

### Memory considerations

`detach().cpu()` copies each activation tensor to host memory on every forward
pass. For large models or long sequences this can exhaust RAM quickly. Consider:

- Recording only specific layers (`i == 0` or a small index set).
- Storing only shapes (`tensor.shape`) rather than values when only dimensions
  are needed.
- Using an index counter and recording every N-th call.

---

## Built-in NVTX tracing (CUDA only)

[`vllm/utils/nvtx_pytorch_hooks.py`](../../../vllm/utils/nvtx_pytorch_hooks.py)
provides [`PytHooks`](../../../vllm/utils/nvtx_pytorch_hooks.py), which
automatically attaches NVTX range markers to every submodule. The markers carry
module name, input/output **shapes** (not values), trainable-parameter shapes,
and layer-type static parameters (e.g. `in_features`/`out_features`).

This is surfaced through `--enable-layerwise-nvtx-tracing` and is intended for
use with NVIDIA Nsight Systems. It does **not** record actual tensor values.

### Requirements

- NVIDIA GPU (the `torch.cuda.nvtx` package must be present).
- CUDA graphs **disabled** — NVTX markers are not Dynamo-traceable, so they
  cannot survive graph capture.
- Compilation mode must not be `STOCK_TORCH_COMPILE`.

### Enabling

=== "CLI"

    ```bash
    vllm serve ibm-granite/granite-4.2-3b \
        --enable-layerwise-nvtx-tracing \
        --no-enable-chunked-prefill
    ```

=== "Python"

    ```python
    from vllm import LLM
    from vllm.config import ObservabilityConfig

    llm = LLM(
        model="ibm-granite/granite-4.2-3b",
        observability_config=ObservabilityConfig(enable_layerwise_nvtx_tracing=True),
        enforce_eager=True,  # required: disables CUDA graphs
    )
    ```

### What is recorded

Each submodule produces two NVTX ranges — one at entry (with input shapes) and
one at exit (with output shapes). The range payload is a JSON-formatted string:

```json
{
  "Module": "GraniteModel.layers.0.mlp.gate_up_proj",
  "TrainableParams": {"weight": [8192, 4096]},
  "Inputs": [[2048, 4096]],
  "StaticParams": {"in_features": 4096, "out_features": 8192}
}
```

Ranges are visible in the Nsight Systems timeline under the Python thread.

### Profiling with Nsight Systems

```bash
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=granite_profile \
    python my_inference_script.py
```

Open the resulting `.nsys-rep` file in Nsight Systems and filter the NVTX row
by module name to isolate FFN layers.

---

## Comparison

| | PyTorch hooks | NVTX tracing |
| --- | --- | --- |
| Backends | All (CPU, MPS, CUDA, XPU) | CUDA only |
| Records actual values | ✅ | ❌ (shapes only) |
| Works with CUDA graphs | ❌ | ❌ |
| Profiling tool integration | Manual | Nsight Systems |
| Enabled via | Python API | `--enable-layerwise-nvtx-tracing` |
