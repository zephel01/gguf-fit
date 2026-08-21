# Architecture

Where things live, and which boundaries are load-bearing. Written for the
version of me who comes back in six months and wants to "simplify" something.

## Module map

```
src/gguf_fit/
  __init__.py     public API surface
  probe.py        read a local GGUF                 -> gguf-probe
  plan.py         budget arithmetic, launch output  -> gguf-plan
  calibrate.py    measure this machine              -> gguf-calibrate
  fetch.py        decide, then download             -> gguf-fetch
  _ggufhdr.py     GGUF header parser over bytes
  _hardware.py    GPU / RAM / CPU detection
  _config.py      config resolution with provenance
  _messages.py    en/ja message catalogue, display width
```

| Module | Depends on | Talks to the outside world |
| :-- | :-- | :-- |
| `_messages` | — | no |
| `_ggufhdr` | — (optionally `gguf`, for type names it does not know) | no |
| `_config` | `_messages` | reads the config file |
| `_hardware` | — | runs `nvidia-smi`, `llama-server --list-devices`, reads sysfs |
| `probe` | `_config`, `_messages`, `gguf` | mmaps GGUF files |
| `plan` | `probe`, `_config`, `_hardware`, `_messages` | reads the probe JSON |
| `calibrate` | `_hardware`, `probe` (only when writing a config) | starts `llama-server`, HTTP, `nvidia-smi` |
| `fetch` | `plan`, `probe`, `_ggufhdr`, `_config`, `_hardware`, `_messages` | HTTP to Hugging Face, runs `hf download` |

`fetch` importing `plan` is deliberate: the estimate must exist in exactly one
place. Two copies of the arithmetic is how the same model ends up with two
different answers.

## The pure core

Classification and budget arithmetic are pure functions. That is why the test
suite runs with no GGUF file and no network.

- `probe.summarize_tensors(tensors, meta)` — takes `(name, type)` pairs, returns the quant mix, hybrid-attention finding and KV size
- `plan.max_ctx` / `recommended_ctx` / `kv_gib` / `file_gib` / `headroom_gib` — take a record dict and a budget
- `_ggufhdr.parse_header(data, want)` — takes `bytes`
- `fetch.group_files` / `quant_label` / `bits_per_weight` / `choose` / `filter_candidates` — take plain data

Network and process boundaries are thin wrappers around those. Keep them thin.

## Two GGUF readers, on purpose

`probe.py` uses the `gguf` package's `GGUFReader`, which mmaps — it needs the
file on disk. `gguf-fetch` has to decide *before* the file exists, so
`_ggufhdr.py` parses the header out of a byte range fetched over HTTP.

They meet at `summarize_tensors`: both hand it `(tensor name, type name)`
pairs, so the KV arithmetic downstream is identical either way.

`_ggufhdr` signals "not enough bytes yet" with `TruncatedGGUF`, carrying how
many bytes are needed, and "this will never parse" with `ValueError`. Collapsing
those two into one exception means either giving up on a file that one more
range request would have read, or fetching forever.

## Boundaries that are load-bearing

**`_messages` holds every human-facing string.** Adding one means adding both
languages; a test fails otherwise. Flag names, JSON keys and generated config
contents stay untranslated — machines read those.

**`_config.resolve()` returns the value *and* where it came from.** Being able
to answer "where did this number come from" is the product. Anything that
resolves a setting behind its back breaks that.

**`_hardware` does not import `_messages`.** It reports machine facts, not
prose. The translated warnings that use those facts live in
`plan.budget_warnings()`, which `gguf-plan` and `gguf-fetch` both call — so one
cannot warn about mixed backends while the other stays silent.

**Do not lift `probe` or `fetch` into `__init__` as a name.** Rebinding a
submodule name to a function makes `from gguf_fit import probe` return the
function. A test caught that once. The functions are exported as `read_gguf`,
`group_files`, `quant_label`, `parse_gguf_header`.

## Where the estimate comes from

```
usage = model file + KV cache + overhead
```

- **model file** — the file size, in GiB. GGUF's `size_gb` is bytes ÷ 10⁹, a different unit; everything here is GiB.
- **KV cache** — bytes/token × ctx. Bytes/token counts only layers that actually hold `attn_k`/`attn_v`, not `block_count`. Getting that wrong over-estimates by 4× on a hybrid-attention model.
- **overhead** — 1.0 GiB by default, measured 0.68, rounded up.

`gguf-calibrate` replaces the first two with measurements. The output always
says which of the two it used.

## Testing

295 tests, no GGUF file and no network required.

- Header parsing is exercised with synthetic GGUF bytes built in the test file.
- `gguf-fetch`'s HTTP paths run against an `http.server` on localhost with `HF_ENDPOINT` pointed at it — including a handler that deliberately ignores `Range`, to prove the client stops.
- Hardware detection runs against recorded `nvidia-smi` / `--list-devices` output.

Test names state the defect being prevented, not the function being called.
Several of them exist because a bug shipped once; see "Before you simplify
something" in the README.

## Adding a command

1. A module with `main() -> int` in `src/gguf_fit/`
2. An entry in `[project.scripts]`
3. Messages in `_messages.py`, both languages
4. Config keys in `_config.KNOWN_KEYS` if it reads any
5. `_hardware.detect()` through `_config.resolve_llama_servers()`, so it sees the same devices as everything else
6. `plan.budget_warnings()` if it spends a VRAM budget
