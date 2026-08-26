# Command reference

Every flag of the four commands, with its default and the reason the default is
what it is. Generated against the actual `argparse` definitions — if this file
and `--help` disagree, `--help` is right and this file is stale.

Common to all four: `--config`, `--show-config`, `--lang`. See
[configuration.md](configuration.md) for how a value is resolved.

- [gguf-fetch](#gguf-fetch)
- [gguf-probe](#gguf-probe)
- [gguf-plan](#gguf-plan)
- [gguf-calibrate](#gguf-calibrate)

---

## gguf-fetch

Downloads GGUF files from Hugging Face — after deciding which ones fit.

```
gguf-fetch <repo> [mode] [options]
```

`<repo>` is a Hugging Face repo id (`ornith-ai/Ornith-1.5-35B-A3B-GGUF`).
With no mode flag it prints the verdict table and downloads nothing.

### Modes (mutually exclusive)

| Flag | Default | What it does |
| :-- | :-- | :-- |
| *(none)* | — | Judge only. Nothing is downloaded. |
| `--fit` | off | Download the largest quantizations that fit the budget. |
| `--pick NAME` | — | Download exactly this one. Matches the quant label first, then any filename substring. Reaches files that are not candidates (`--pick mtp`). |
| `--all` | off | Download every candidate in the repo — root-level GGUFs plus anything inside a directory named after a quantization (`UD-IQ1_S/`, `Q4_K_M/`). |

### Choosing what `--fit` takes

| Flag | Default | Notes |
| :-- | --: | :-- |
| `--top N` | `3` | How many to take. Not 1: near the budget line the estimate is an estimate, and the next one down is worth its disk. |
| `--min-ctx N` | `16384` | A quantization only counts as fitting if it reaches this ctx. Something that loads but stops at ctx 4k is not an answer. |
| `--min-bpw F` | *unset* | Drop anything under this measured bits-per-weight. Cuts on what is in the file, not on the name. |
| `--spread` | off | Take N spread across the bpw range instead of the top N. Without it, a 24-quant repo returns three files in the same bit tier. |
| `--only GLOB` | *unset* | Keep only candidates matching the glob (label or filename, case-insensitive). Repeatable. **Name matching, not a quality judgement.** |
| `--exclude GLOB` | *unset* | Drop candidates matching the glob. Repeatable. |

### Companion files

| Flag | Default | Notes |
| :-- | :-- | :-- |
| `--mmproj {auto,all,none}` | `auto` | Vision projector. `auto` takes the smallest one; without it a multimodal model cannot see. |
| `--extras {none,mtp,all}` | `none` | GGUFs in subdirectories that are **not** named after a quantization (`MTP/`, `imatrix/`, `original/`), plus `mtp`/`draft` files found inside a quantization directory. They are never judged for fit — they are not the model. `mtp` takes draft/MTP files only. Default is `none` because what those files are for is not something this tool can read. |

### How much to read before deciding

| Flag | Default | Notes |
| :-- | :-- | :-- |
| `--probe {one,all,none}` | `one` | `one` reads a single representative header (~12 MiB) and applies its KV rate to every row — valid because quantizing changes tensor *types*, not *names*. `all` reads every candidate. `none` reads nothing and judges on file size alone, and says so. |

### Budget

| Flag | Default | Notes |
| :-- | :-- | :-- |
| `--vram F` | detected | GiB. Detected from the machine; pass it to plan for a different one. |
| `--overhead F` | `1.0` | GiB. Compute buffers, CUDA context, speculative decoding. |
| `--kv {f16,q8_0,auto}` | `auto` | `auto` falls back to `q8_0` when f16 does not reach `--min-ctx`. |
| `--llama-server PATH` | `llama-server` | Repeatable, comma-separated accepted. Resolved by the same function `gguf-plan` uses, so both commands see the same devices. |

### Output and execution

| Flag | Default | Notes |
| :-- | :-- | :-- |
| `--dir PATH` | `.` (or `models_dir`) | A subdirectory named after the repo is created under it. |
| `--revision REF` | `main` | Branch, tag or commit. |
| `--dry-run` | off | Print the `hf download` command and stop. Still prints it when the disk is too small. |
| `-y`, `--yes` | off | Skip the confirmation. |
| `--json` | off | Candidates, verdicts, bpw and the selection as JSON. |
| `--hf-bin PATH` | `hf`, then `huggingface-cli` | Which executable runs the download. |
| `--refresh` | off | Ignore `vram`/`threads`/`device` in the config file and re-detect. |

### What it refuses to do

- Write when free disk space is smaller than the download (`--dry-run` still prints the command; the exit code stays 1).
- Continue when the server ignores an HTTP `Range` request — that is the last guard against silently receiving 21 GB.
- Treat a root-level file as a quantization when its bits-per-weight falls outside 0.5–33. `imatrix_unsloth.gguf` is 0.004 bpw against a 27B parameter count.
- Borrow a KV rate from a file that fails `n_tensors >= block_count`.

It warns, but proceeds, when the destination is inside a git working tree.

---

## gguf-probe

Reads local GGUF files and reports what is in them.

```
gguf-probe [paths...] [options]
```

Paths may contain glob characters (`*?[`), which are expanded by the tool
itself, so quoting them is safe.

| Flag | Default | Notes |
| :-- | :-- | :-- |
| `--json` | off | Machine-readable. The JSON records each file's **absolute path**, so `gguf-plan` can write a real `-m`. |
| `--roles` | off | Every tensor role whose type varies by layer. |
| `--out PATH` | *unset* | Also save to this file. The result still goes to stdout. |

Reading more than one language model adds a comparison table.

---

## gguf-plan

Turns a `gguf-probe --json` file plus a VRAM budget into a launch command.

```
gguf-plan <gguf.json> [options]
```

Without `--pick` it prints the fit table for every quantization. With `--pick`
it emits a launch command and a config for that one.

| Flag | Default | Notes |
| :-- | :-- | :-- |
| `--pick NAME` | *unset* | Part of a file name. |
| `--ctx N` | rounded down | Omitted, it picks the largest "round" ctx that fits, not the largest that fits — 69,632 leaves 0.02 GiB of headroom for 6% more context. |
| `--kv {f16,q8_0,auto}` | `auto` | `auto` switches to `q8_0` with a stated reason when f16 does not fit. |
| `--target {llama-server,ollama,lmstudio}` | `llama-server` | Only `llama-server` can express everything; the other two are approximations and say so. |
| `--vram F` | detected | GiB. |
| `--overhead F` | `1.0` | GiB. |
| `--model-path PATH` | from the JSON | What to write after `-m`. |
| `--port N` | `8085` | |
| `--device ID` | detected | `CUDA0` / `ROCm0` / `Vulkan0`. Omitted entirely when no NVIDIA device is found rather than guessed. |
| `--threads N` | physical cores | **Physical**, not logical — handing llama.cpp the SMT siblings can make matrix multiplication slower. |
| `--llama-server PATH` | `llama-server` | Repeatable. A CUDA build cannot see ROCm devices, so list every build you use. |
| `--write-config [PATH]` | `gguf-fit.toml` | Freeze the current values into a config file. Refuses to overwrite without `--force`. |
| `--force` | off | Allow `--write-config` to overwrite. |
| `--refresh` | off | Re-detect `vram`/`threads`/`device` instead of using the config file's. |

---

## gguf-calibrate

Measures this machine once, so the budget arithmetic stops being a guess.

```
gguf-calibrate --model /models/your.gguf [options]
```

Starts `llama-server` at each ctx, waits for `/health`, pushes one request
through, reads the increase with `nvidia-smi`, and fits a line. Needs
`nvidia-smi`. It runs no benchmark.

| Flag | Default | Notes |
| :-- | :-- | :-- |
| `--model PATH` | *required* | The GGUF to measure with. |
| `--ctx LIST` | `32768,65536` | **At least two values.** One point cannot separate the slope from the intercept — "the KV ratio is 0.61" and "the ratio is still 0.531 but overhead grew" produce the same single number. |
| `--kv LIST` | `f16,q8_0` | Which KV types to fit. |
| `--extra "ARGS"` | *unset* | Extra `llama-server` flags. **Pass the same ones you use in production** — the figures move with them. |
| `--warmup-tokens N` | `2048` | Prompt length for the one warm-up request, matching the default `--batch-size`. Going higher does not help: 8 → 2048 tokens moved the result by 10 MiB. |
| `--no-warmup` | off | Skip the request. The result then reads about 110 MiB low. |
| `--port N` | `18085` | Deliberately not 8085, so it does not collide with a server you are already running. |
| `--device ID` | detected | |
| `--threads N` | physical cores | |
| `--llama-server PATH` | `llama-server` | |
| `--write-config [PATH]` | `gguf-fit.toml` | Writes the calibrated block, replacing only that block and keeping your comments. The file is re-parsed as TOML before it is written, so a failure never leaves a broken file behind. It also records **which model** the figures came from. |

Two runs under the same conditions agreed to within 1 MiB across four points,
so measuring once is enough. A later change means the environment changed.
