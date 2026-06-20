# Security Policy

TensorFold is alpha software. Treat it as local developer tooling unless you
have reviewed and hardened the deployment yourself.

## Supported Versions

| Version | Supported |
| --- | --- |
| 0.1.x | Security fixes accepted |

## Reporting A Vulnerability

Until the public GitHub repository is configured, report vulnerabilities
privately to the repository owner through GitHub. Do not open a public issue for
secrets, path leaks, command execution bugs, or denial-of-service issues.

When reporting, include:

- TensorFold version or commit
- operating system and Python version
- whether MLX/Metal was involved
- a minimal reproduction
- whether model files or generated artifacts are needed to reproduce

## Scope

Security-sensitive areas include:

- OpenAI-compatible server behavior
- file path handling
- safetensors parsing
- expert pack generation and loading
- public release scrub checks
- telemetry that could expose local paths or prompt contents

TensorFold does not execute downloaded model code directly, but model files and
tokenizers are still untrusted inputs. Keep serving endpoints bound to
`127.0.0.1` unless you have added authentication and network hardening.
