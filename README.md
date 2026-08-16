<div align="center">

# gguf-fit

**Decide `llama-server` settings by reading the GGUF — without running a single token.**

[![CI](https://github.com/zephel01/gguf-fit/actions/workflows/ci.yml/badge.svg)](https://github.com/zephel01/gguf-fit/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue)](https://github.com/zephel01/gguf-fit)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[Install](#install) · [Why](#why) · [What it finds](#what-it-finds) · [VRAM model](#vram-model) · [Usage](#usage) · [日本語](README.ja.md)

</div>

---

Picking `--ctx-size`. Choosing a quantization. Working out whether `--spec-type draft-mtp`
does anything at all. These usually get settled by trial and error, one GPU-hour at a time.

**Most of them don't have to be.** The GGUF header and tensor list already contain the answers.

`gguf-fit` reads them. It `mmap`s the file, so a 27B GGUF takes about a second.

## Install

```bash
uv tool install git+https://github.com/zephel01/gguf-fit
```

Or run it once without installing anything:

```bash
uvx --from git+https://github.com/zephel01/gguf-fit gguf-probe /models/*.gguf
```

<details>
<summary>Using pip instead</summary>

```bash
pip install git+https://github.com/zephel01/gguf-fit
```

</details>

## Quick start

```bash
gguf-probe --json --out gguf.json /models/Qwen3.8-27B-GGUF/*.gguf
gguf-plan gguf.json --pick Q5_K_M
```

Your VRAM, physical core count, and whether `--device` even applies are **detected from the
machine** — no flags needed. Pass `--vram 24` when you want to plan for a *different* machine
than the one you're on.

```console
# ===== Qwen3.8-27B-Q5_K_M / ctx 65,536 / KV f16 =====
# estimate: model 18.47 + KV 4.25 + overhead 1.00 = 23.72 GiB / budget 24.0 GiB
# !! only 0.28 GiB of headroom. The overhead has a single calibration point, so a
#    measurement above it means a failed start
#   Use q8_0 for the KV cache (-ctk q8_0 -ctv q8_0), or drop ctx one step
# native ctx = 262,144  / no rope scaling
# hybrid attention: only 17/65 layers hold KV = 68 KB/token

# --- llama-server ---
# found 4 MTP tensors -> adding --spec-type draft-mtp
llama-server -m /models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q5_K_M.gguf \
  --port 8085 --device CUDA0 \
  -ngl 99 -fa on \
  --ctx-size 65536 --parallel 1 \
  --batch-size 2048 --ubatch-size 512 \
  --spec-type draft-mtp

# --- config.yaml (under models:) ---
  qwen3-8-27b-q5_k_m:
    type: openai
    base_url: "http://localhost:8085/v1"
    model: "auto"
    temperature: 1.0
    top_p: 0.95
    top_k: 20
    min_p: 0.0
    max_tokens: 49152   # 3/4 of ctx 65,536; the rest is for the prompt
    seed: 42            # only set this when runs: 1
    # sampling defaults for a thinking chat template
```

Output is English by default; `--lang ja` switches it to Japanese.

## Why

Two commands, split along a real seam:

| | |
| :-- | :-- |
| **`gguf-probe`** | Reads the file. Reports what is actually in it. |
| **`gguf-plan`** | Takes that plus a VRAM budget → a launch command and a config. |

Each GGUF field decides a specific setting:

| Field | Decides |
| :-- | :-- |
| `context_length` + RoPE scaling | The hard ceiling for `--ctx-size` |
| KV bytes per token | VRAM cost per token → `--ctx-size` from a budget, and whether `-ctk/-ctv` is needed |
| File size | What the weights occupy |
| MTP / `nextn` tensors | Whether `--spec-type draft-mtp` can work **at all** |
| Chat template | Thinking model? → which sampling defaults apply |

## What it finds

From reading 12 quantizations of one 27B model.

### The file name lies about the bit width

`UD-Q6_K_XL` has **55.1% of its weights in Q8_0** and is 3.0 GB larger than plain `Q6_K`.
Benchmarking the two against each other is not a comparison at the same bit width —
whatever the names suggest.

### "Dynamic" varies by *role*, not by *layer*

| | Varies type across layers? |
| :-- | :-- |
| Standard K-quants | **Yes** — Q4_K_M 3/24 roles, Q5_K_M 3/24, Q3_K_M 2/24 |
| Unsloth Dynamic | **No** — UD-Q4/Q5/Q6/Q8 are all 0/24 |

llama.cpp's own `use_more_bits` heuristic is what varies type per layer in the standard
K-quants. What UD varies is *which tensor role* gets more bits.

### Hybrid attention makes KV 4× cheaper than `block_count` implies

Qwen3.8-27B has 65 layers, but only **17** hold a KV cache. The other 48 carry a fused
`attn_qkv` tensor — linear attention, fixed-size recurrent state, no KV.

> [!IMPORTANT]
> Counting all 65 layers gives **260 KB/token**. The real figure is **68 KB/token**.
> `gguf-probe` counts only layers that actually have `attn_k`/`attn_v` tensors.

## VRAM model

```
usage = model file + KV cache + overhead
```

Overhead — compute buffers, CUDA context, speculative decoding — is calibrated against
measurements:

| | GiB |
| :-- | --: |
| Qwen3.8-27B-Q5_K_M file | 18.47 |
| KV cache, ctx 65,536, f16 | 4.25 |
| subtotal | 22.72 |
| **measured** (`llama-server`, RTX 5090) | **23.50** |
| **→ overhead** | **0.78** |

The default rounds up to **1.0 GiB**. Override with `--overhead` — and please open an issue
with whatever you measure.

> [!WARNING]
> **Units flip conclusions here.** GGUF's `size_gb` is bytes ÷ 10⁹ (**GB**). A GPU's "24GB"
> is **GiB** (÷ 2³⁰) — 25.77 GB in decimal. `gguf-fit` works entirely in GiB.

<details>
<summary><b>Quantized KV saves less than the arithmetic suggests</b></summary>

`q8_0` stores 32 values as 32 int8 bytes plus one fp16 scale — 34 bytes against f16's 64,
a ratio of **0.531**. Measured on one model at ctx 65,536:

| | measured |
| :-- | --: |
| KV f16 | 24,068 MiB |
| KV q8_0 | 22,362 MiB |
| saving | **1,706 MiB (1.67 GiB)** |
| predicted from 0.531 | 1.99 GiB |

The extra ~0.33 GiB is most likely dequantization scratch space. With one context length
measured, "the ratio is really 0.61" and "the ratio is 0.531 but overhead grows" fit equally
well; a second measurement at a different `--ctx-size` would separate them.

The default uses the theoretical 0.531, so **`gguf-plan` slightly over-estimates how much
`q8_0` saves.**

</details>

## Usage

<details open>
<summary><b>gguf-probe</b> — read</summary>

```bash
gguf-probe /models/Qwen3.8-27B-Q5_K_M.gguf          # one file
gguf-probe /models/Qwen3.8-27B-GGUF/*.gguf          # a directory, with a comparison table
gguf-probe --out report.txt /models/*.gguf          # save text (also printed)
gguf-probe --json --out gguf.json /models/*.gguf    # save JSON
gguf-probe --roles /models/one.gguf                 # every role whose type varies by layer
```

The JSON records the absolute path of each GGUF, so `gguf-plan` can write a real `-m` for you.

</details>

<details open>
<summary><b>gguf-plan</b> — decide</summary>

```bash
gguf-plan gguf.json --vram 24 --ctx 65536              # what fits in 24 GiB?
gguf-plan gguf.json --vram 24 --pick Q5_K_M            # largest sensible ctx
gguf-plan gguf.json --vram 24 --pick Q5_K_M --ctx 131072
```

`--kv auto` (the default) falls back to `q8_0`, with a stated reason, when `f16` doesn't fit.

Omit `--ctx` and it rounds **down** to a standard context size rather than filling the budget
to the brim: 69,632 would leave 0.02 GiB of headroom and gain 6% of context over 65,536.

</details>

<details>
<summary><b>As a library</b> — no GGUF file, no <code>gguf</code> package needed</summary>

```python
from gguf_fit import summarize_tensors, recommended_ctx

s = summarize_tensors(
    [("blk.3.attn_k.weight", "Q5_K"), ("blk.0.attn_qkv.weight", "Q5_K")],
    {"block_count": 65, "head_count_kv": 4, "key_length": 256, "value_length": 256},
)
s["kv_cache"]["bytes_per_token_f16"]

recommended_ctx(rec, vram_gib=24.0, kv_mode="q8_0", overhead=1.0)
```

The classification and VRAM math are pure functions — that's what makes the test suite run
without a single model file.

</details>

## Configuration

Every setting resolves in this order, and `--show-config` tells you which one won:

```
CLI flag  >  environment variable  >  config file  >  detected  >  built-in default
```

### What gets detected

| | How | Used for |
| :-- | :-- | :-- |
| **VRAM** | `nvidia-smi`; the largest card if several | `--vram` |
| **Unified memory** | Apple Silicon → 75% of RAM (macOS caps what the GPU may take) | `--vram` |
| **Physical cores** | `/proc/cpuinfo` `(physical id, core id)` pairs, `hw.physicalcpu` on macOS | `--threads` |
| **Whether CUDA applies** | no NVIDIA → `--device` is **omitted**, not guessed | launch command |

`--threads` uses **physical** cores, not logical ones: handing llama.cpp the SMT siblings
tends to make matrix multiplication slower, not faster.

> [!WARNING]
> With more than one NVIDIA card, `gguf-plan` still writes `CUDA0` and adds a warning.
> `nvidia-smi` lists in PCI order and CUDA numbers devices by its own ordering — **they do
> not match**. Guessing an index here silently grabs the wrong card, so the tool refuses to
> guess. Check with `llama-server --list-devices`.

### Writing it down

Detection is convenient but not portable. `--write-config` freezes the current values into a
file, so the same plan comes out on a machine without `nvidia-smi`, or when you're planning
for someone else's hardware.

```console
$ gguf-plan --write-config
[written] gguf-fit.toml

# detected hardware
#   GPU 0       NVIDIA GeForce RTX 5090  31.8 GiB (0.0 used)
#   RAM         31.0 GiB
#   CPU         16 physical / 32 logical

lang = "en"   # <- default
vram = 31.8   # <- detected
overhead = 1.0   # <- default
device = "CUDA0"   # <- detected
threads = 16   # <- detected
```

Each line records **where the value came from**, so six months later you can still tell what
you chose from what the machine reported. Values that could not be detected are commented
out rather than written as `None`, which would not parse. It refuses to overwrite an existing
file without `--force`.

```toml
# ./gguf-fit.toml   (or ~/.config/gguf-fit/config.toml, or $GGUF_FIT_CONFIG)
lang     = "ja"
vram     = 24.0
overhead = 1.0
device   = "CUDA0"
port     = 8085
```

```console
$ gguf-plan --show-config
settings in effect (cli > env > config file > default)

  config file: gguf-fit.toml

  lang        ja                       <- config
  vram        24.0                     <- config
  overhead    1.0                      <- default
  port        8085                     <- default
  device      CUDA0                    <- config
  model_path  (unset)                  <- default
```

Environment variables are the same names with a `GGUF_FIT_` prefix: `GGUF_FIT_LANG`,
`GGUF_FIT_VRAM`, and so on. Detection sits **below** the config file on purpose — a value you
wrote down should not be silently overridden by the machine.

> [!TIP]
> Only the first config file found is used — they are not merged. Merging makes "where did
> this value come from" unanswerable, and answering that is the point of this tool.
> Unknown keys, wrong types, and broken TOML are reported on stderr rather than ignored.

## What this cannot tell you

- **The imatrix.** Two files of the same quantization type may be calibrated differently and
  GGUF does not record it. What you get here is the **type assignment**.
- **Actual quality.** "71.7% of weights are Q5_K" is a fact; what it scores is not. This tool
  narrows the candidates *before* you spend GPU time — it does not replace measuring.
- **Multi-GPU splits.** `--vram` is one device's budget. Check against `nvidia-smi`.

## Development

```bash
uv sync            # .venv + dev tools
uv run pytest -q   # 47 tests, no GGUF file required
uv run ruff check .
```

`uv.lock` is committed and CI runs `uv sync --locked`, so adding a dependency without
refreshing the lock fails the build instead of drifting silently.

<details>
<summary>Using pip instead</summary>

```bash
pip install -e . --group dev   # pip >= 25.1
pytest -q
ruff check .
```

</details>

### Two things are pinned on purpose

- **The ruff rule set** (`[tool.ruff.lint] select`). ruff widens its defaults on minor
  releases — code that passed under 0.15 produced 10 findings under 0.16. Selecting
  explicitly keeps CI from turning red on its own.
- **The ruff version** (`>=0.16.3,<0.17`), for the same reason.

### Before you simplify something

Six bugs have lived in this code, every one of them producing plausible-looking output.
The tests pin each. Please don't refactor them away.

<details>
<summary>The list</summary>

1. **KV layers are not `block_count`.** Count only layers with `attn_k`/`attn_v`; layers with
   just `attn_qkv` are linear attention and hold no KV. Getting this wrong over-estimates 4×.
2. **"More than one layer signature ⇒ Dynamic" is too loose.** It flagged even Q8_0. Judge by
   whether *the same role* has different types across layers.
3. **A difference confined to the MTP block is not per-layer allocation.** IQ4_XS is IQ4_XS
   for 64 layers with `blk.64` in Q4_K — counting that is a false positive.
4. **`output.weight` must match exactly.** A substring match picks up 65 `attn_output.weight`.
5. **GB vs GiB.** Mixing them flips "does it fit on a 24GB card".
6. **A `#` comment inside a backslash-continued shell command** swallows the `\` and truncates
   the command. There is a test that runs the emitted command through `bash -n`.

</details>

## License

MIT
