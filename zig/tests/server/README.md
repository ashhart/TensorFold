# Native server tests

`zig build test-golden` runs the native fake server against the frozen request and response files in `golden/`.
The expected responses remain independent regression fixtures after the Python server's retirement.

The Python files here support the native tests:

- `corpus.py`, `fake_text.py`, `wire.py` and `freeze_cases.py` build the golden request corpus.
- `security_cases.py` and `security_check.py` exercise the native server's HTTP and authentication boundaries.
- `test_memory_http.py` supplies the HTTP checks used by memory receipts.

Regenerate request cases with `python3 -B zig/tests/server/freeze_cases.py`.
Review changes to expected responses separately; do not replace them wholesale with the candidate server's output.
The retired Python server, its scripted scheduler and the live Python-versus-Zig parity runner are no longer required.
