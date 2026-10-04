# Changelog

Notable changes. Japanese: [CHANGELOG.ja.md](CHANGELOG.ja.md).

This project has not cut a release yet; `version` in `pyproject.toml` is still
`0.1.0`. Dates below are commit dates.

## Unreleased

### Added

- **`gguf-fetch` can download from Pirate Face** (`--source`, `--via`, `--aria2-bin`),
  for models Hugging Face has removed. A `pirateface.co` URL, or a bare `owner/name` that
  Hugging Face answers 401/403/404/410/451 for, reads the file list, pinned revision, magnet
  and per-file SHA-256 from the Pirate Face page and downloads only the selected files from
  the swarm with `aria2c`, then verifies them. Pirate Face's web-seed turned out to be a
  302 to Hugging Face at the pinned commit, so it cannot survive a removal; `--via auto`
  uses it only as a fallback (`hf download --revision <sha>`). While Hugging Face still has
  the revision the verdict is unchanged; without it the verdict is by file size only and
  says so. JSON gained `source`, `magnet`, `sizes_approximate`.
  **The page lists the whole Hugging Face repo, but the torrent may hold only part of it**
  (checked against a real swarm: Ornith-1.5-9B-uncensored-GGUF lists 12 quantizations,
  the torrent holds only Q4_K_M). When the swarm is the route that will be used, the
  26 KiB torrent metadata is read *before* judging, files not in the torrent are left out
  of the candidates (and say so), and the same torrent is reused for the download. `--dry-run`
  therefore reads that metadata (it still downloads nothing). If the metadata cannot be
  read, the table is shown as before with a warning that it assumes every listed file exists.
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
- **`gguf-fetch --pick` takes several names.** `--pick Q5_K_M,Q6_K,Q8_0`,
  `--pick Q5_K_M, Q6_K, Q8_0` (spaces after the commas are fine) and
  `--pick Q5_K_M --pick Q6_K` all mean the same thing, and the files go down in
  one `hf download`. Each name is matched on its own; **if any one matches
  nothing, it stops before downloading** rather than bringing back two of the
  three. `--pick A B owner/repo` still finds the repo at the end.

### Fixed

- **`mmproj` named after the model** (mradermacher: `<model>.mmproj-Q8_0.gguf`) was read
  as a weights candidate, so `Q8_0` and `f16` collided with the projector's labels and the
  table showed full file names. It is now recognised as a vision projector anywhere in the name.
- **Diffusion-model GGUFs were evaluated as LLMs.** Repos such as
  Qwen-Image (`general.architecture = qwen_image21`, no `block_count` /
  `context_length` / vocabulary) printed "could not compute the KV cache size"
  for every quantization and re-read every header. Language-model detection now
  looks at the metadata shape rather than a name list, so new diffusion
  architectures need no update. Such repos are judged by file size against the
  budget with one header read and a one-line note; mmproj files are still not
  taken as the main model.

- **GGUFs without `attention.key_length` were rejected as "not the main model".**
  Older converters omit the key (e.g. third-party Qwen2.5 quants, and
  `Qwen/Qwen2.5-0.5B-Instruct-GGUF`), so no KV size was computed and
  `gguf-fetch` turned down every quantization — 771 tensors for 64 blocks was
  reported as if the tensor count were the problem. The head dimension is now
  taken as `embedding_length / head_count`, the same default llama.cpp uses
  (Qwen2.5-32B: 128 → 256 KiB/token), and `gguf-probe` says when it did so.
  When the KV size still cannot be computed, `gguf-fetch` now names the missing
  metadata instead of blaming the tensor count.
- **A split model whose first shard holds no tensors got no KV figure.** In
  `unsloth/Qwen3.8-Flash-Next-GGUF` the `-00001-of-00003` shard is 10.9 MB of
  metadata and tokenizer with zero tensors; the weights start in shard 2. Only
  that shard was read, so there was nothing to count: `n_tensors >= block_count`
  refused it and the table fell back to file size — no bpw, no max ctx. Taking
  KV layers from `block_count` instead would have been wrong by **4×** (48 blocks,
  but only 12 carry `attn_k`/`attn_v`; the rest are linear attention). When the
  first shard has no tensors, the other shards' headers are now read and merged
  (`merge_shard_headers`): tensors concatenated, `n_params` summed, metadata taken
  from whichever shard has it. 34.4 MiB transferred instead of 10.4 MiB, against a
  67.56 GiB download, and the extra read is announced.
- **"The KV figures come from the header of `<nothing>`."** When a header was
  fetched but none was usable, the source line printed with an empty filename.
  It now says no usable header was found, and reports the transfer.
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
