# Configuration reference

Every key that may appear in `gguf-fit.toml`, where it can come from, and which
source wins. `--show-config` prints the answer for the machine you are on, with
the origin of each value; this file is the map behind it.

## Precedence

```
CLI flag  >  environment variable  >  config file  >  detected  >  built-in default
```

Detection sits **below** the config file on purpose. A value you wrote down
should not be silently overridden by the machine. `--refresh` inverts that for
`vram` / `threads` / `device` when you have changed hardware.

## Where the file is looked for

Only the **first one found is used**. Files are never merged — merging makes
"where did this value come from" unanswerable, and answering that is the point
of the tool.

1. `--config PATH`
2. `$GGUF_FIT_CONFIG`
3. `./gguf-fit.toml`
4. `~/.config/gguf-fit/config.toml` (`$XDG_CONFIG_HOME` respected) — on Windows, `%APPDATA%\gguf-fit\config.toml`

Unknown keys, wrong types and broken TOML are reported on stderr, not ignored.
Not noticing that your config file is inert is the worst outcome.

## Environment variables

Any key becomes `GGUF_FIT_` + the key in upper case: `GGUF_FIT_LANG`,
`GGUF_FIT_VRAM`, `GGUF_FIT_MODELS_DIR`. List-valued keys are comma-separated.

`gguf-fetch` additionally honours two variables that belong to
`huggingface_hub`, not to this tool: `HF_ENDPOINT` (mirrors) and `HF_TOKEN` /
`HUGGING_FACE_HUB_TOKEN` / `HUGGINGFACE_TOKEN` (gated repos). A token is also
read from `$HF_HOME/token` or `~/.cache/huggingface/token`.

## Keys

### Output

| Key | Type | Default | Notes |
| :-- | :-- | :-- | :-- |
| `lang` | str | `"en"` | `"en"` or `"ja"`. Flag names, JSON keys and generated config contents are never translated — machines read those. |

### Budget

| Key | Type | Default | Notes |
| :-- | :-- | :-- | :-- |
| `vram` | float | detected | GiB for **one** device. Detected from `llama-server --list-devices`, else `nvidia-smi`; on Apple Silicon, 75% of RAM. |
| `overhead` | float | `1.0` | GiB. Measured 0.68 on one machine, rounded up to absorb the ~0.03 GiB more that appears under sustained load. |

### Calibrated KV rates

Written by `gguf-calibrate --write-config`. Present, they beat the figure
derived from the GGUF — which runs low: f16 measured 69.1 KB/token against a
derived 68.0, and q8_0 measured 43.1 against a derived 36.1.

| Key | Type | Default | Notes |
| :-- | :-- | :-- | :-- |
| `kv_f16_bytes` | float | *unset* | Bytes per token, f16. |
| `kv_q8_bytes` | float | *unset* | Bytes per token, q8_0. |
| `kv_measured_on` | str | *unset* | Which file the figures came from. For humans. |
| `kv_derived_f16_bytes` | float | *unset* | That model's GGUF-derived rate, for machine comparison. |

> **These are per-model values.** The KV rate is decided by the layer
> structure, so 69.1 KB/token measured on Qwen3.8-27B does not describe
> Ornith-1.5-35B, whose KV costs 22.0 — applying it understated max ctx by
> nearly 3×. When `kv_derived_f16_bytes` is present and the model in front of
> you derives a figure more than 1.15× away from it, both `gguf-plan` and
> `gguf-fetch` say so. Older config files without that key get no warning,
> because there is nothing to compare against.

### Hardware detection

| Key | Type | Default | Notes |
| :-- | :-- | :-- | :-- |
| `llama_servers` | list | `["llama-server"]` | Every build you use. **A CUDA build's `--list-devices` does not show ROCm devices**, so with one entry you will miss real GPUs. |
| `llama_server` | str | — | Singular form of the same thing, for people with one build. |
| `device` | str | detected | `CUDA0` / `ROCm0` / `Vulkan0`. |
| `threads` | int | physical cores | |

### Generated launch command

| Key | Type | Default | Notes |
| :-- | :-- | :-- | :-- |
| `port` | int | `8085` | |
| `model_path` | str | from the probe JSON | Rarely needed. |

### gguf-fetch

| Key | Type | Default | Notes |
| :-- | :-- | :-- | :-- |
| `models_dir` | str | `.` | Downloads go into a per-repo subdirectory under this. Worth setting: the default is the current directory, so running inside a source checkout puts tens of GB there — and with `*.gguf` in `.gitignore`, `git status` stays clean and you forget. |
| `hf_bin` | str | `hf`, then `huggingface-cli` | |

## Machine-specific by nature

`gguf-fit.toml` is in `.gitignore` and should stay there. `--write-config`
records VRAM and core counts; carried to another machine, it plans for a GPU
that is not there. `gguf-fit.example.toml` is the template that *is* tracked.

Every line written by `--write-config` carries its origin as a comment
(`<- detected`, `<- default`, `<- cli`), so six months later you can still tell
which numbers you chose and which the machine offered.
