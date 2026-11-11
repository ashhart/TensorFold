# Serve config files

`tensorfold serve --config FILE.toml` reads serve flags from a TOML file, so a launch can be checked in,
reviewed and reused. The command line still works as before, and it wins over the file.

```toml
# glm.toml
model = "zai-org/GLM-5.3-Flash"
host = "0.0.0.0"
port = 8080
context = 131072
parallel = "auto"
temperature = 0.6
thinking = false
dashboard = true
alias = ["glm", "glm-flash"]
api-key-file = "/etc/tensorfold/keys"
```

```sh
tensorfold serve --config glm.toml                 # the file names the model
tensorfold serve --config glm.toml --port 9000     # the command line overrides the file
tensorfold serve other/model --config glm.toml     # a model on the command line overrides the file's
```

vLLM and SGLang read a YAML file through the same `--config` flag and with the same precedence; a file
moves over by rewriting `key: value` as `key = value`, quoting its strings and renaming keys to
TensorFold's flags.

## Keys

A key is a [serve flag](../README.md#serve-flags) without its `--`. `-` and `_` are the same, so
`max-tokens` and `max_tokens` both set `--max-tokens`; a file that has both is refused. `model` names the
model, as the positional argument does. Every key is at the top level of the file.

| Flag kind | Value in the file | Example |
| --- | --- | --- |
| Takes a value | A string or a number | `port = 8080`, `host = "0.0.0.0"`, `parallel = "auto"` |
| Repeatable | One value, or an array with one item a repeat | `alias = ["glm", "glm-flash"]` |
| Switch | `true` or `false` | `dashboard = true` |

A number is passed as written, without its `_` separators; the flag checks it as it checks a typed
value. `false` for a switch that has a `--no-` twin selects the twin: `thinking = false` is
`--no-thinking`. `false` for any other switch leaves it off, which is its default.

## Precedence

The command line wins over the file, and the file wins over the built-in defaults. `serve` reads the
file's flags first and the command line's after them, so a later value replaces an earlier one. Three
consequences follow:

- A repeatable flag adds up. With `alias = ["a"]` in the file and `--alias b` on the command line, the
  server answers to both.
- A switch the file turns on stays on unless it has a `--no-` twin. To serve without it, use another file
  or remove the line.
- Flags that refuse each other do so across the file and the command line. A file with `unix-socket`
  cannot be combined with `--port`.

A relative path in a value, such as `chat-template = "template.jinja"`, is relative to the directory
`serve` runs in, not to the file.

## What the file can hold

The reader takes a strict subset of TOML 1.0, so every file it accepts means the same to any TOML tool:

- comments, blank lines, LF or CRLF line ends;
- bare keys (`A-Z a-z 0-9 _ -`) and quoted keys;
- basic strings `"..."` with TOML's escapes, and literal strings `'...'`;
- decimal integers and floats, `true` and `false`;
- arrays of those, on one line or several, with a trailing comma.

It refuses the rest by line: tables (`[section]`), dotted keys, inline tables, nested arrays, multi-line
strings, dates and times, hexadecimal, octal and binary integers, a key defined twice, a byte order mark,
and a file over 1 MiB. Tables are refused, not ignored, so that a later release can give them a meaning.

## Errors

A problem with the file is a usage error that names the file and the line. Nothing is loaded.

```text
tensorfold serve: error: argument --config: glm.toml line 7: unknown key 'prot'
```

A key for a flag that this build does not serve is refused in the same way. When a flag refuses the
value the file gave it, `serve` prints the flag's own message and then names the file:

```text
tensorfold serve: error: argument --port: invalid port: '99999'
tensorfold serve: --config glm.toml supplied 9 options, read before the command line's
```

The startup log names the file and how many options it gave, never a value.

## Limits

- One `--config` a command, and a file cannot name another file. The path must end in `.toml`.
- The file holds serve flags only. Environment variables (`TF_*`, `TENSORFOLD_*`) are still set in the
  environment; [profiles](profiles.md) describes the registered switches.
- `tensorfold benchmark` and `tensorfold service install` do not read the file.
- Prefer `api-key-file` to `api-key` in a file that is checked in, so that no key is stored with it.
