# TensorFold Web launcher

Run `tensorfold web [ARGS...]` to open the separately installed `tensorfold-web` application.
Arguments after `web` reach the application unchanged, including its own `--help`.
The native binary needs no model or GPU to launch it.

If the application is missing, a release binary explains it and asks `Install it now with uv? [y/N]`.
Only `y` or `yes`, ignoring case and surrounding spaces, permits installation.
The installer runs `uv tool install` with one argument pinned to the binary's release tag and the `web` subdirectory.
For example, a binary built with `-Dversion=1.2.0` prints:

```sh
uv tool install "tensorfold-web @ git+https://github.com/ashhart/TensorFold@v1.2.0#subdirectory=web"
```

After installation, the launcher checks PATH and then `uv tool dir --bin` for the executable.
On systems where process replacement is available it execs the application; on Windows it waits for the child.
The application's exit status is retained.

With either stdin or stdout redirected, without `uv`, or with `TENSORFOLD_WEB_NO_PROMPT=1`, the launcher prints manual
commands and exits 1. It never installs in those cases. A negative or empty answer does the same.
An already installed application starts without prompting in all of these cases.
Builds without an explicit `-Dversion` never install automatically: their printed command targets the manifest version,
which is identified as a manual suggestion rather than the development build's release tag.

The launcher sets `TENSORFOLD_BIN` to the absolute path of its own executable, replacing any inherited value.
TensorFold Web must use that path when starting TensorFold so it runs the binary the user selected.
The handoff does not add arguments to the web application's command line.

Run `zig build test-web-cli-host` for the host-only command tests.
They use temporary PATH directories and fake local executables, with no package downloads or network calls.
