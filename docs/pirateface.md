# Downloading from Pirate Face / the swarm

When a model has been removed from Hugging Face (HF), `gguf-fetch` can download it through the [Pirate Face](https://pirateface.co) index and the BitTorrent swarm. This page covers installation and usage. Every flag is listed in [commands.md](commands.md).

## When to use it

- You want a model that has been removed from HF.
- The HF repository now answers 401/403/404/410/451.

While HF still has the model, downloads come from HF as before. You do not need to change anything.

## Installation

Three things are needed, **each for a different purpose**.

| For | You need | How |
| :-- | :-- | :-- |
| Everything | Python 3.10+ and `gguf-fit` | "The tool" below |
| Downloading from the swarm | `aria2c` | macOS: `brew install aria2` / Debian-based: `apt install aria2`. On other systems install the package named `aria2` |
| Downloading from HF (`--via hf`, and the fallback of `--via auto`) | `hf` (formerly `huggingface-cli`) | `pip install -U huggingface_hub` |

### The tool

```bash
uv tool install git+https://github.com/zephel01/gguf-fit
```

To try it once without installing:

```bash
uvx --from git+https://github.com/zephel01/gguf-fit gguf-fetch --help
```

With pip: `pip install git+https://github.com/zephel01/gguf-fit`.

### Check that it worked

```bash
aria2c --version | head -1
gguf-fetch --help | grep -E -- '--(source|via|aria2-bin)'
```

If the second command prints all three flags, Pirate Face support is installed.

## Usage

```bash
# 1. Judge only (downloads nothing; reads only the swarm metadata)
gguf-fetch https://pirateface.co/mradermacher/Ornith-1.5-9B-uncensored-GGUF --dry-run --fit

# 2. Download what fits, from the swarm
gguf-fetch https://pirateface.co/mradermacher/Ornith-1.5-9B-uncensored-GGUF --fit --dir ~/models

# 3. Pick a quantization and use only the swarm
gguf-fetch https://pirateface.co/mradermacher/Ornith-1.5-9B-uncensored-GGUF \
  --pick Q4_K_M --mmproj none --via swarm --dir ~/models
```

How you choose files (`--fit`, `--pick`, `--top`, `--mmproj`, ...) is the same as for HF.

### Which source is used (`--source`)

| You pass | `auto` (default) does |
| :-- | :-- |
| `https://pirateface.co/owner/name` | Pirate Face |
| `https://huggingface.co/owner/name` | HF |
| `owner/name` | HF first. Switches to Pirate Face only when HF answers 401/403/404/410/451, not on a network error |

`--source hf` or `--source pirateface` pins it.

### How files are downloaded (`--via`)

| `--via` | Behavior |
| :-- | :-- |
| `auto` (default) | Try the swarm first; if it fails, switch to `hf download --revision <pinned commit>`. Without `aria2c`, goes straight to `hf` |
| `swarm` | Magnet and `aria2c` only. No fallback on failure |
| `hf` | `hf download` at the pinned revision. The swarm is not used |

### What happens

1. It reads the pinned revision, the magnet, per-file SHA-256 and approximate sizes from the Pirate Face page.
2. When the swarm is the route that will be used, it reads the torrent metadata (tens of KiB, a few seconds) **before judging**, and only the files **actually in the torrent** stay candidates. Files that are not in it are named and left out.
3. Only the selected files are fetched with `aria2c --select-file`.
4. They are moved directly under the destination and checked against the page's SHA-256. On a mismatch it does not say "done" and exits with code 1.

**A file listed on the page is not necessarily in the torrent.** `mradermacher/Ornith-1.5-9B-uncensored-GGUF`, checked in 2026-10, listed 12 quantizations on the page, but the torrent held only `Q4_K_M` (5.2 GiB). That is why step 2 narrows the candidates. If `--pick` names a quantization the torrent lacks, it stops with "matched nothing" before any transfer starts.

While HF still has the revision, sizes and GGUF headers come from HF and the verdict is unchanged. Once it is gone, headers cannot be read, so the verdict is **by file size only**, and says so (sizes are rounded to 0.1 GB). KV cache and max ctx are unknown until you measure after downloading.

With `--via auto` and HF alive, either route works, so the torrent is not read first.

### `--dry-run`

It prints the commands and stops; the model is not downloaded. When the swarm is the route, it still reads the torrent metadata for the judgement.

### Where files land

```
<dir>/<repo name>/
    Model.Q4_K_M.gguf          <- the selected file (same layout as hf download)
    <torrent name>/...          <- fragments of neighbouring files (only when present)
```

`aria2c` also fetches the ends of neighbouring files that share a piece with a selected file, so fragments can be left over. They are not usable; delete `<torrent name>/` by hand if you do not want them. The tool never deletes them.

Running the same command again does not download files whose SHA-256 already matches (it re-reads the metadata only).

## Configuration file

To avoid repeating flags, put them in `gguf-fit.toml`:

```toml
models_dir = "~/models"
source = "auto"        # auto / hf / pirateface
aria2_bin = "aria2c"   # when aria2c lives elsewhere
```

Environment variables: `HF_ENDPOINT` and `HF_TOKEN` (HF mirrors, gated repositories) and `PIRATEFACE_ENDPOINT` (a Pirate Face mirror). Precedence is in [configuration.md](configuration.md).

## Troubleshooting

| Symptom | Cause and fix |
| :-- | :-- |
| `aria2c was not found` (exit 127) | Install `aria2c`, or pass its path with `--aria2-bin`. Not needed with `--via hf` |
| `could not get the torrent metadata` after waiting 180 s | No peer answered, or the swarm is empty. The exit code is aria2c's. `--via auto` switches to HF (which fails if HF no longer has the model) |
| `the torrent holds none of the GGUF files listed on the page` | That torrent contains no GGUF. Look for another source |
| `--pick X matched nothing. Available: ...` | X is not in the torrent. Choose from "Available" |
| `SHA-256 does NOT match` (exit 1) | A damaged file is never reported as done. Do not use it; delete it and run again |
| `could not read ... on Pirate Face` | A network problem, or the page layout changed. Without the page, the Pirate Face route cannot proceed. If HF still has the model, use `--source hf` |

## Notes

- Pirate Face's web-seed is a 302 redirect to HF at the pinned commit. Once HF has removed the model, only the swarm works.
- You are responsible for checking the origin and licence of files fetched from the swarm. The SHA-256 check confirms the file matches Pirate Face's record; it does not vouch for the model's safety.
- The commands that will run are printed as they are. If in doubt, check with `--dry-run` first.
