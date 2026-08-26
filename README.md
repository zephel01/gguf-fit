<div align="center">

# gguf-fit

**Decide `llama-server` settings by reading the GGUF — without running a single token.**

[![CI](https://github.com/zephel01/gguf-fit/actions/workflows/ci.yml/badge.svg)](https://github.com/zephel01/gguf-fit/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue)](https://github.com/zephel01/gguf-fit)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[Install](#install) · [Why](#why) · [What it finds](#what-it-finds) · [VRAM model](#vram-model) · [Usage](#usage) · [Reference](docs/) · [日本語](README.ja.md)

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
gguf-fetch ornith-ai/Ornith-1.5-35B-A3B-GGUF --fit   # download only what fits
gguf-probe --json --out gguf.json /models/Qwen3.8-27B-GGUF/*.gguf
gguf-plan gguf.json --pick Q5_K_M
```

Your VRAM, physical core count, and whether `--device` even applies are **detected from the
machine** — no flags needed. Pass `--vram 24` when you want to plan for a *different* machine
than the one you're on.

```console
# ===== Qwen3.8-27B-Q5_K_M / ctx 65,536 / KV f16 =====
# estimate: model 18.47 + KV 4.32 + overhead 1.00 = 23.78 GiB / budget 24.0 GiB
# !! only 0.22 GiB of headroom. Inference itself allocated another 0.11-0.14 GiB
#    after load when measured, so this can still fail to start
#   Use q8_0 for the KV cache (-ctk q8_0 -ctv q8_0), or drop ctx one step
# native ctx = 262,144  / no rope scaling
# hybrid attention: only 17/65 layers hold KV = 68 KB/token
# KV f16 = 69.1 KB/token, measured here (gguf-calibrate), not derived from the GGUF

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

Four commands, split along real seams:

| | |
| :-- | :-- |
| **`gguf-fetch`** | Downloads from Hugging Face — but decides **which files fit before downloading them**. |
| **`gguf-probe`** | Reads the file. Reports what is actually in it. |
| **`gguf-plan`** | Takes that plus a VRAM budget → a launch command and a config. |
| **`gguf-calibrate`** | Measures this machine once, so the budget arithmetic is not a guess. |

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

Both the KV rate and the overhead are **measured**, not assumed. `gguf-calibrate` starts
`llama-server` at two context sizes and fits the line — the relationship is exactly linear,
so two points determine it. On Qwen3.8-27B-Q5_K_M / RTX 5090 / `-fa on` / `--spec-type
draft-mtp`:

| | measured | from the GGUF | ratio |
| :-- | --: | --: | --: |
| KV f16 | **69.1 KB/token** | 68.0 | 1.02 |
| KV q8_0 | **43.1 KB/token** | 36.1 | **1.19** |
| intercept | **19.15 GiB** | file is 18.47 | → overhead **0.68** |

Max error across the four points: **0 MiB**. The default overhead rounds up to **1.0 GiB**,
which also covers the extra ~0.03 GiB seen under a sustained real workload.

> [!IMPORTANT]
> These numbers move with the llama.cpp version, the backend, the launch flags, and any
> change to how quantization is implemented. Run `gguf-calibrate` on your own machine and
> put the result in `gguf-fit.toml`. `gguf-plan` says in its output whether the figure it
> used was measured or derived.

> [!WARNING]
> **Units flip conclusions here.** GGUF's `size_gb` is bytes ÷ 10⁹ (**GB**). A GPU's "24GB"
> is **GiB** (÷ 2³⁰) — 25.77 GB in decimal. `gguf-fit` works entirely in GiB.

<details>
<summary><b>Quantized KV saves less than the arithmetic suggests</b></summary>

`q8_0` stores 32 values as 32 int8 bytes plus one fp16 scale — 34 bytes against f16's 64,
a ratio of **0.531**. Measured, the ratio is **0.624**:

| at ctx 65,536 | measured | predicted from 0.531 |
| :-- | --: | --: |
| KV f16 | 4.32 GiB | 4.25 |
| KV q8_0 | **2.69 GiB** | 2.26 |
| saving | **1.62 GiB** | 2.02 |

The extra ~0.4 GiB is most likely dequantization scratch space — 0.531 describes the storage
format, not what the kernel needs while it runs.

This changes an answer: on a 24 GiB card, the arithmetic says `q8_0` reaches ctx 131,072,
and the measurement says it needs **24.5 GiB** and does not fit. An earlier version of this
README made that claim. `gguf-plan` uses the measured figure when the config file has one.

</details>

<details>
<summary><b>When you measure changes the answer too</b></summary>

Same server, same ctx 65,536, four readings (VRAM growth over idle):

| | MiB | |
| :-- | --: | --: |
| straight after loading | 23,922 | |
| after one 8-token request | 24,022 | **+100** |
| after one 2,048-token request | 24,032 | **+110** |
| during a sustained benchmark run | 24,064 | **+142** (drifting ±50) |

**Every one of those offsets was identical for f16 and for q8_0**, so none of it scales with
the KV cache — it is what inference itself allocates. And a 256× longer prompt bought only
10 MiB: the allocation is decided by *whether any inference ran at all*, not by prompt length.

That matters because mixing conditions corrupts the fit. Measuring ctx 32,768 after load and
ctx 65,536 during inference puts a constant into the slope and gives **73.5 KB/token**; with
both points taken the same way it is **69.1**. This project made that mistake.

Re-measuring with the warm-up in place is the check that the model is right: all four points
moved by exactly the same amount, **the slope did not change by a single byte**, and the
intercept alone rose 19.04 → 19.15 GiB. A constant belongs in the intercept, and that is
where it went.

`gguf-calibrate` waits for the server's own `/health`, sends one request whose prompt fills a
batch, measures after it, and prints how much that request added. It reaches +110 of the +142
a sustained run showed. The last ~32 MiB is not chased — longer prompts don't produce it, so
it comes from sustained operation (slot churn, KV defrag, speculative-decode graph variants).
That is why the default `overhead` leaves room instead of sitting exactly on the measurement.

</details>

## Usage

<details open>
<summary><b>gguf-fetch</b> — download, but decide first</summary>

```bash
gguf-fetch ornith-ai/Ornith-1.5-35B-A3B-GGUF                  # just the verdict
gguf-fetch ornith-ai/Ornith-1.5-35B-A3B-GGUF --fit            # the ones that fit
gguf-fetch ornith-ai/Ornith-1.5-35B-A3B-GGUF --pick Q5_K_M    # exactly this one
gguf-fetch ornith-ai/Ornith-1.5-35B-A3B-GGUF --all            # every GGUF in the repo
```

`hf download` is already fine at downloading. What it cannot tell you is **which of the
five quantizations in that repo fits your 24 GiB card** — normally you find that out by
downloading them all and running `gguf-probe` afterwards.

This reverses the order. The repo listing costs a few KB. The GGUF header lives at the
*front* of the file, so an HTTP `Range` request gets the layer structure, the native ctx,
and the MTP tensor count for a few MB. Then the same arithmetic `gguf-plan` uses decides
what to download.

```console
$ gguf-fetch ornith-ai/Ornith-1.5-35B-A3B-GGUF --vram 24 --fit
ornith-ai/Ornith-1.5-35B-A3B-GGUF  (main)
budget 24.0 GiB / overhead 1.0 GiB / KV auto / counts as fitting from ctx 16,384

quantization       file   max ctx (f16)   max ctx (q8_0)   verdict
------------------------------------------------------------------
Q4_K_M           20.22G         131,072          249,856   -> DOWNLOAD
Q5_K_M           23.61G              no               no   no
Q6_K             27.20G              no               no   no
Q8_0             35.21G              no               no   no
BF16             66.19G              no               no   no

# the KV figures come from the header of Q4_K_M (12.0 MiB transferred). Quantizations of
# the same model share the layer structure, so the same KV/token applies to every row.
# 1 vision projector(s) found; taking mmproj-Ornith-1.5-35B-BF16.gguf (0.84 GiB).
```

**172 GB of candidates, judged with 12 MiB of transfer.**

* **Why one header is enough.** Quantizing changes tensor *types*, not tensor *names*. So
  every quantization in a repo has the same layer structure — the same KV bytes per token,
  the same native ctx, the same MTP tensors. Only the file size moves. The output always
  names the file the figure came from. `--probe all` reads every header instead (more
  transfer); `--probe none` reads none and judges on file size alone, and says so.
* **The `bpw` column is measured.** File size ÷ parameter count. Parameter counts do not
  change with quantization, so one header gives it for every row. **This repo started from
  "the file name lies about the bit width" — here is that lie as a number:** `UD-Q6_K_XL`
  has a 6 in its name and measures **7.41 bpw**; `UD-Q8_K_XL` is **9.21**, heavier than
  `Q8_0` (8.51). `BF16` is the check digit: it must come out at exactly 16.00 (measured
  16.00 and 16.01). If it doesn't, the parameter count is wrong.
* **Three ways to narrow the field.**

  | | |
  | :-- | :-- |
  | `--min-bpw 4.5` | cut on **what is in the file** |
  | `--spread` | N spread across the bpw range, not the top N |
  | `--only 'UD-Q?_K_*'` / `--exclude 'IQ*'` | cut on **the name** — not a quality judgement |

  `--spread` exists because of a real result: `--top 3` on unsloth/Qwen3.8-27B returns
  26.1 / 27.0 / 29.3 GiB — 8.21 / 8.51 / 9.21 bpw, **all three in the same bit tier**.
  That is 83 GiB downloaded for less than one step of comparison.
* **What `--fit` picks.** The largest quantizations that fit, `--top 3` of them by default.
  Not one: near the budget line the estimate is an estimate, and having the next one down
  on disk is worth more than the disk it costs. A quantization only counts as fitting if it
  reaches `--min-ctx` (default 16,384) — something that loads but only reaches ctx 4k is
  not a usable answer.
* **`mmproj`.** Vision projectors are picked up automatically (the smallest one, if a repo
  ships several). `--mmproj all` / `--mmproj none`.
* **Sharded GGUFs.** `-00001-of-00003` files are grouped into one candidate and their sizes
  summed. Counted separately they would look like three small models that all fit.
* **A directory named after a quantization holds the model.** Newer unsloth repos put the
  weights in `UD-IQ1_S/`, `Q4_K_M/`, `BF16/` … with nothing but a README at the root
  (`unsloth/Qwen3.8-Flash-Next-GGUF` is all three shards inside `UD-IQ1_S/`). Those are
  candidates, labelled by the directory — `--pick UD-IQ1_S` works, and so does the download.
* **Any other subdirectory is not a candidate — but it is still reachable.** `MTP/`,
  `imatrix/`, `original/` and friends hold things that are not the model, so they stay out
  of the fit table (treating one as the representative throws the KV rate off by 17× — see
  [the bug list](#before-you-simplify-something)). The same goes for a `mtp`/`draft` file
  sitting inside a quantization directory. A draft/MTP file is nonetheless used
  *together with* the model, and the model alone won't do.

  ```bash
  gguf-fetch <repo> --fit --extras mtp    # bring the draft/MTP file along
  gguf-fetch <repo> --fit --extras all    # everything in the subdirectories
  gguf-fetch <repo> --pick mtp            # just that one, by name
  ```

  The default is `none` because **what those files are for is not something this tool can
  read.** Their existence is always reported; asking for them is your call.
* **Before it writes anything** it checks free disk space, prints the exact `hf download`
  command, and asks. `--dry-run` prints and stops; `-y` skips the question.

Downloading is `hf download`'s job — this hands it a file list and gets out of the way.
Set `models_dir` in the config file to stop typing `--dir`. `HF_ENDPOINT` and `HF_TOKEN`
are honoured, so mirrors and gated repos work.

</details>

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

`--target {llama-server,ollama,lmstudio}` (default `llama-server`) switches the output format:

```bash
gguf-plan gguf.json --vram 24 --pick Q5_K_M --ctx 131072 --target ollama
gguf-plan gguf.json --vram 24 --pick Q5_K_M --ctx 131072 --target lmstudio
```

The estimate, headroom warning, and "measured vs. derived KV" line are identical across all
three targets — only the launch format changes. Neither Ollama's `Modelfile` nor LM Studio's
documented load interface exposes everything `llama-server`'s CLI does. `llama-server` gets the
exact flags; the other two get the best approximate number available, clearly labeled as such:

* **GPU layer count** — `num_gpu` was dropped from `docs.ollama.com/modelfile`, but Ollama's own
  maintainers confirm it still works ([ollama/ollama#13986](https://github.com/ollama/ollama/issues/13986)).
  gguf-fit only ever plans full offload, so the `Modelfile` sets `PARAMETER num_gpu 99` as an
  approximate, best-effort hint (mirroring `-ngl 99`) — not a guarantee. LM Studio's load API and
  `lms` CLI only take a 0–1 fraction (`lms load --gpu 0.5`), no layer count at all, so its output
  sticks to `--gpu max` and adds the total layer count as a comment for context.
* **KV cache quantization (`q8_0`)** — Ollama controls this with the server-wide
  `OLLAMA_KV_CACHE_TYPE` environment variable, not per model; the emitted `Modelfile` says so
  instead of inventing a `PARAMETER` for it. LM Studio's documented load API/CLI has no
  equivalent at all — if a plan needs `q8_0` to fit the budget, the LM Studio output says
  plainly that the budget may not be reachable there.
* **MTP / `--spec-type draft-mtp`** — Ollama has a `draft_num_predict` parameter for
  speculative decoding, but whether it drives MTP tensors the way llama.cpp's `--spec-type
  draft-mtp` does hasn't been checked, so nothing is set for it.

</details>

<details open>
<summary><b>gguf-calibrate</b> — measure, so the estimate stops being a guess</summary>

```bash
gguf-calibrate --model /models/Qwen3.8-27B-Q5_K_M.gguf \
  --ctx 32768,65536 --kv f16,q8_0 \
  --extra "--spec-type draft-mtp"        # use the flags you actually run with
```

It starts `llama-server` at each context size, waits for `/health`, sends one request whose
prompt fills a batch (`--warmup-tokens`, default 2048), reads the VRAM delta from
`nvidia-smi`, and kills the server. Four points take a few minutes and no benchmark.

```console
calibration result

  f16     69.1 KB/token   intercept 19.15 GiB   (2 points, max error 0 MiB)
  q8_0    43.1 KB/token   intercept 19.11 GiB   (2 points, max error 0 MiB)

  q8_0 / f16 = 0.624   (the naive 34/64-byte figure would say 0.531)

  one request added 110 MiB on top of the load-time figure; that is what these
  numbers include

--- paste into gguf-fit.toml ---
kv_f16_bytes = 70720   # 69.1 KB/token, 2 points
kv_q8_bytes = 44096    # 43.1 KB/token, 2 points
```

`--write-config` puts those two lines in the config file for you instead of leaving you to
copy them. Re-running replaces the block and keeps everything else, including your comments;
the result is parsed before it is written, so a run can't leave you with a broken file.

```bash
gguf-calibrate --model /models/your.gguf --write-config
# [updated] gguf-fit.toml
```

`gguf-plan` then uses the measured figures instead of the GGUF arithmetic. Requires
`nvidia-smi`; `--no-warmup` skips the request but reads ~110 MiB low.

Two identical runs produced the same four figures to the MiB, so **once is enough** — if the
numbers move later, something about the environment moved.

> [!NOTE]
> At least two `--ctx` values are required. One point cannot separate the slope from the
> intercept — "the KV ratio is 0.61" and "the ratio is 0.531 but overhead grew" predict the
> same number at a single context length. `gguf-calibrate` refuses rather than pick one.

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

# where gguf-fetch puts downloads (a per-repo subdirectory is created under it)
models_dir = "/mnt/data/models"

# from gguf-calibrate. Present → used instead of the GGUF arithmetic.
kv_f16_bytes = 70720
kv_q8_bytes  = 44096
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
- **A repo holding more than one model.** `gguf-fetch --probe one` (the default) applies one
  header to every row. That is correct for quantizations *of the same model*; it is wrong if
  an unrelated model shares the repo. `--probe all` reads them all.

## Development

```bash
uv sync            # .venv + dev tools
uv run pytest -q   # 265 tests, no GGUF file and no network required
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
7. **Not every GGUF in a repo is the model.** unsloth/Qwen3.8-27B-GGUF ships
   `MTP/mtp-...-Q4_0.gguf` (1.28 GiB). `gguf-fetch` treated it as a candidate, its `Q4_0`
   label collided with the real `Qwen3.8-27B-Q4_0.gguf` (14.95 GiB), and — being the
   smallest — **it was picked as the representative**, applying its 4.0 KB/token (1 KV layer
   of 65) to every row. The real model is 68.0 KB/token (17 of 65): **17× out**. The table
   looked entirely reasonable. Fix: try the largest first, and refuse any representative
   that fails `n_tensors >= block_count`.
8. **…and "not at the root" is not the same as "not the model".** The fix for 7 above was
   *root-level files only*, which was too blunt: `unsloth/Qwen3.8-Flash-Next-GGUF` keeps all
   three shards in `UD-IQ1_S/` and nothing at the root, so every candidate was thrown away
   and `gguf-fetch` reported **"no GGUF in this repo"** while the files sat right there.
   Fix: a subdirectory whose *name is itself a quantization label* (`UD-IQ1_S`, `Q4_K_M`,
   `BF16`) holds the model; `MTP`, `imatrix`, `original` do not parse as one and stay out.
   The bpw and `n_tensors` guards still run on top of that.
9. **Calibration is per-model but the config file is global.** 69.1 KB/token measured on
   Qwen3.8-27B was applied to Ornith-1.5-35B, whose KV costs 22.0 — understating max ctx by
   nearly 3×. `gguf-calibrate` now records `kv_measured_on` and `kv_derived_f16_bytes`, and
   both commands warn when the derived figures disagree by more than 1.15×.

</details>

## License

MIT
