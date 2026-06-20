# Privacy And Telemetry

TensorFold is a local runtime. The core CLI does not send telemetry to a remote
service.

## Local Data

TensorFold may read:

- model files you pass on the command line
- expert pack directories you create
- prompt text sent to the local server
- local benchmark or canary descriptors you choose to run

## Logs

Runtime logs can include prompts, generated text, model paths, memory telemetry,
and benchmark commands. Treat logs as local artifacts unless you have scrubbed
them.

## Release Scrub

Before publishing docs, examples, logs, or support bundles, run:

```bash
python3 tools/check_public_scrub.py
```

The scrub gate checks the public product surface for local home paths, local
model-cache paths, private names, coordination notes, and email-like secrets.
It intentionally excludes private research files such as handoff notes and
findings logs.
