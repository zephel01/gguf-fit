# gguf-fit documentation

The [project README](../README.md) is the introduction: what the tool decides,
why the numbers are what they are, and enough usage to get going. These pages
are the reference behind it.

| | |
| :-- | :-- |
| [commands.md](commands.md) | Every flag of the four commands, with defaults and the reason for each default |
| [configuration.md](configuration.md) | Every `gguf-fit.toml` key, the precedence chain, environment variables |
| [architecture.md](architecture.md) | Module map, the pure core, which boundaries are load-bearing |
| [../CHANGELOG.md](../CHANGELOG.md) | What changed and when |

Japanese: [README.ja.md](README.ja.md) · [commands.ja.md](commands.ja.md) ·
[configuration.ja.md](configuration.ja.md) ·
[architecture.ja.md](architecture.ja.md) ·
[../CHANGELOG.ja.md](../CHANGELOG.ja.md)

## Which page answers what

- *"What does this flag do, and why is the default 3?"* → commands
- *"I set `vram` and it did not take effect"* → configuration, then `--show-config`
- *"Why does the estimate look like that?"* → the README's VRAM model section, then architecture
- *"Where do I add a message / a config key / a command?"* → architecture

## What is deliberately not here

Anything that is already true in the code and printed at runtime. `--help`
lists the flags, `--show-config` says where every value came from, and the
output of `gguf-plan` states whether the KV figure was measured or derived.
Documentation that repeats those goes stale silently; the runtime output cannot.
