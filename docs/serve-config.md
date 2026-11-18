# Serve config files

`tensorfold serve --config FILE.toml` reads serve flags from a TOML file, so a launch can be checked in,
reviewed and reused. The command line still works as before, and it wins over the file.

```toml
# glm.toml
model = "zai-org/GLM-5.3-Flash"
host = "127.0.0.1"
port = 8080
context = 131072
parallel = "auto"
temperature = 0.6
thinking = false
dashboard = true
alias = ["glm", "glm-flash"]
chat-template = "templates/glm.jinja"   # beside glm.toml
api-key-file = "/etc/tensorfold/keys"
```

```sh
tensorfold serve --config glm.toml                 # the file names the model
tensorfold serve --config glm.toml --port 9000     # the command line overrides the file
tensorfold serve other/model --config glm.toml     # a model on the command line overrides the file's
```

## Keys

A key is a [serve flag](../README.md#serve-flags) without its `--`. `-` and `_` are the same, so
`max-tokens` and `max_tokens` both set `--max-tokens`; a file that has both is refused. `model` names the
model, as the positional argument does. Every key is at the top level of the file.

| Flag kind | Value in the file | Example |
| --- | --- | --- |
| Takes a value | A string or a number | `port = 8080`, `host = "127.0.0.1"`, `parallel = "auto"` |
| Repeatable | One value, or an array of strings with one item a repeat | `alias = ["glm", "glm-flash"]` |
| Switch | `true` or `false` | `dashboard = true` |

A number is passed as written, without its `_` separators and, for an integer, without a leading `+`;
the flag checks it as it checks a typed value. A pair of switches has one key, the positive one: write `thinking = false` for `--no-thinking`; the key
`no-thinking` is refused and the message names the form to use. A switch with no positive twin keeps its
name (`no-drafts = true`), and `false` for a switch with no `--no-` twin leaves it off, its default.

## Precedence

The command line wins over the file, and the file wins over the built-in defaults. `serve` reads the
file's flags first and the command line's after them, so a later value replaces an earlier one. Three
consequences follow:

- A repeatable flag on the command line replaces the file's whole list. With `api-key = ["old"]` in the
  file and `--api-key new` on the command line, only `new` is accepted. A list the command line does not
  name stays the file's.
- A switch the file turns on stays on unless it has a `--no-` twin. To serve without it, use another file
  or remove the line.
- Flags that refuse each other do so across the file and the command line. A file with `unix-socket`
  cannot be combined with `--port`.

## Paths

A relative path in the file starts at the file's folder, not at the directory `serve` runs in, so a
file and what it names move together. This holds for the flags that name a file or folder:
`chat-template`, `api-key-file`, `unix-socket`, `voices`, `speed-up`, `prompt-cache-disk`, `learn-dir`,
`recall-dir`, `slide-graph`, `record`, `compact-memory` and `fsr-route-trace`. An absolute path passes
as written. No shell reads the file, so nothing expands `~`: these flags refuse a value that starts
with it, and the message asks for the whole path.

`model` and `drafter` take a folder or a Hugging Face repo id, and `org/name` could be either. Start a
folder with `./` or `../` to place it beside the file: `model = "./weights/glm"`. Any other text passes as
written and resolves as it does on the command line, where a leading `~/` is the home folder.

A path on the command line is still relative to the directory `serve` runs in.

## What the file can hold

The reader takes a strict subset of TOML 1.0, so every file it accepts means the same to any TOML tool:

- comments, blank lines, LF or CRLF line ends;
- bare keys (`A-Z a-z 0-9 _ -`) and quoted keys;
- basic strings `"..."` with TOML's escapes, and literal strings `'...'`;
- decimal integers and floats, `true` and `false`;
- arrays of strings, on one line or several, with a trailing comma.

It refuses the rest by line: tables (`[section]`), dotted keys, inline tables, nested arrays, an array
item that is not a string, multi-line strings, dates and times, hexadecimal, octal and binary integers, a key defined twice, a byte order mark,
and a file over 1 MiB. Tables are refused, not ignored, so that a later release can give them a meaning.

## Errors

A problem with the file is a usage error that names the file and the line. Nothing is loaded.

On macOS and Linux the file is opened once, checked and read through that one handle:

- a link is not followed, and anything that is not a regular file is refused;
- a file that its group or other users can write is refused, because it chooses what is served
  (`chmod go-w FILE`);
- a file that sets `api-key` is refused while its group or other users can read it (`chmod 600 FILE`, or
  use `api-key-file`).

```text
tensorfold serve: error: argument --config: glm.toml line 7: unknown key 'prot'
```

A key for a flag that this build does not serve is refused in the same way. Each value is checked by
its flag as a typed value is, so the flag's own message comes with the line:

```text
tensorfold serve: error: argument --config: glm.toml line 3: argument --port: invalid port: '99999'
```

A refusal that involves two flags, such as `unix-socket` in the file with `--port` on the command line,
has no single line. `serve` prints it as usual and then names the file:

```text
tensorfold serve: error: argument --unix-socket: not allowed with --port
tensorfold serve: --config glm.toml supplied 9 options, read before the command line's
```

The startup log names the file and how many options it gave, never a value.

## Limits

- One `--config` a command, and a file cannot name another file. The path must end in `.toml`.
- The file holds serve flags only. Environment variables (`TF_*`, `TENSORFOLD_*`) are still set in the
  environment; [profiles](profiles.md) describes the registered switches.
- `tensorfold benchmark` and `tensorfold service install` do not read the file. TensorFold Web refuses
  `--config` as an extra flag, because a file could set the flags that Settings manages.
- Prefer `api-key-file` to `api-key` in a file that is checked in, so that no key is stored with it.
