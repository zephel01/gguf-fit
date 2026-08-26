# Changelog

Notable changes. Japanese: [CHANGELOG.ja.md](CHANGELOG.ja.md).

This project has not cut a release yet; `version` in `pyproject.toml` is still
`0.1.0`. Dates below are commit dates.

## Unreleased

### Added

- **`gguf-fetch`** — a fourth command. Downloads GGUF files from Hugging Face,
  but decides which ones fit **before** downloading them: the repo listing costs
  a few KB, one GGUF header comes over an HTTP `Range` request (~12 MiB), and
  the same arithmetic `gguf-plan` uses picks the files. Judging the five
  quantizations of Ornith-1.5-35B — 172 GB of candidates — transferred 12.0 MiB.
  Modes: verdict only, `--fit`, `--pick`, `--all`.
- **Measured bits-per-weight** in the verdict table, from file size ÷ parameter
  count. Parameter counts do not change with quantization, so one header covers
  every row at no extra transfer. `BF16` is the check digit — it must come out
  at 16.00, and does. This makes the file-name problem legible: `UD-Q6_K_XL`
  measures 7.41 bpw, and `UD-Q8_K_XL` at 9.21 is heavier than `Q8_0` at 8.51.
- **Narrowing for repos with many quantizations**: `--min-bpw` cuts on content,
  `--spread` takes N across the bpw range instead of the top N, `--only` /
  `--exclude` cut on name globs. On a 24-quant repo, `--fit --top 3` had been
  returning three files inside one bit tier.
- **`--extras {none,mtp,all}`** — GGUFs in subdirectories that are not named
  after a quantization (`MTP/`, `imatrix/`) are not candidates, but a draft/MTP
  file pairs with the model and can now ride along. Default `none`.
- `--dry-run`, `-y`, `--json`, `--revision`, `--mmproj`, `--probe`, `--hf-bin`.
- **`docs/`** — command, configuration and architecture references.
- `kv_measured_on` and `kv_derived_f16_bytes` are recorded by
  `gguf-calibrate --write-config`, so a calibration can be matched to the model
  it was taken on.

### Fixed

- **A model kept in a quantization-named directory was unreachable.**
  `unsloth/Qwen3.8-Flash-Next-GGUF` puts all three shards in `UD-IQ1_S/` and
  leaves nothing but a README at the root. The "root-level files only" rule
  below threw every candidate away, so `gguf-fetch` reported **no GGUF in this
  repo** and there was no way to download files that were plainly there. A
  subdirectory whose name is itself a quantization label (`UD-IQ1_S`, `Q4_K_M`,
  `BF16`) is now a candidate, labelled by the directory so `--pick UD-IQ1_S`
  works. `MTP`, `imatrix` and `original` do not parse as quantization labels and
  stay out, as does an `mtp`/`draft` file inside a quantization directory; the
  bpw and `n_tensors >= block_count` guards still apply on top.
- **Not every GGUF in a repo is the model.** `unsloth/Qwen3.8-27B-GGUF` ships
  `MTP/mtp-…-Q4_0.gguf` (1.28 GiB). It was treated as a candidate, its `Q4_0`
  label collided with the real 14.95 GiB file, and — being the smallest — it was
  picked as the representative, applying its 4.0 KB/token (1 KV layer of 65) to
  every row. The real model is 68.0 KB/token (17 of 65): **17× out**, in a table
  that looked entirely reasonable. The representative is tried largest-first and
  must satisfy `n_tensors >= block_count`; colliding labels are spelled out.
  (This originally also restricted candidates to root-level files — see the
  entry above for why that part was replaced.)
- **A root-level file that is not weights.** The same repo has
  `imatrix_unsloth.gguf` at the root — 13 MiB, 0.004 bpw against a 27B parameter
  count — where the subdirectory rule cannot catch it. `--spread` was selecting
  it as the low end of the range. Candidates outside 0.5–33 bpw are now dropped,
  by measurement rather than by name.
- **A calibration measured on one model was applied to another.** 69.1 KB/token
  from Qwen3.8-27B was used for Ornith-1.5-35B, whose KV costs 22.0,
  understating max ctx by nearly 3×. Both commands now compare derived figures
  and warn past 1.15×.
- **`gguf-fetch` ignored `llama_servers`.** It read only the plural config key,
  missing the singular key, the environment variable and `--llama-server`. On a
  machine with CUDA, ROCm and Vulkan builds side by side, `gguf-plan` and
  `gguf-fetch` could therefore assume different GPUs and produce different
  budgets. Both now resolve through `_config.resolve_llama_servers()`.
- **`gguf-fetch --show-config` hid `device`.** The budget is taken from a
  device's capacity, so omitting it failed to answer where the number came from.
  It now prints `device`, `llama_servers` and the detected-hardware block, like
  `gguf-plan`.
- **`gguf-fetch` printed none of `gguf-plan`'s budget warnings** — mixed
  backends, low free memory, driver/runtime disagreement, VRAM mismatch. Shared
  as `plan.budget_warnings()`.
- **"MB" that was computed in MiB.** In a tool with a warning box about GB
  versus GiB. The transfer figure now reads `MiB`.
- **`--pick` could not reach a file that was not a candidate**, while the output
  told the reader to use it for exactly that.
- **`--dry-run` stayed silent when the disk was too small**, printing no command
  at all. It now prints the command and keeps the non-zero exit code.
- **Japanese tables were padded by character count**, so every column with a
  Japanese cell drifted. Padded by display width now, in `gguf-plan` too.

### Guards

- Free disk space is checked before writing.
- A server that ignores an HTTP `Range` request stops the run rather than
  delivering the whole file.
- Downloading into a git working tree warns — with `*.gguf` in `.gitignore`,
  `git status` stays clean and 72 GiB goes unnoticed.

## 2026-08-16

- `gguf-plan --target ollama` / `lmstudio`, with GPU-layer hints marked as
  approximations.
- `gguf-calibrate` — measure this machine instead of baking in constants.
  Warm-up so the measurement includes what inference itself allocates;
  `--write-config` writes the result without disturbing the rest of the file.
- `gguf-plan` uses measured KV rates when the config has them.
- Hardware detection via `llama-server --list-devices` as the primary source,
  with AMD support, multiple binaries, driver-level cross-checking of "free",
  and the VRAM budget taken from the device that will actually be used.
- `--refresh`, and a guard against a machine-specific config file travelling to
  another machine.
- First commits: `gguf-probe`, `gguf-plan`.
