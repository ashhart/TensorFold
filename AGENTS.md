# Working on TensorFold

TensorFold uses Agent Traffic Control to coordinate contributors and coding agents.
Before editing, read [CONTRIBUTING.md](CONTRIBUTING.md) and follow the [ATC contributor instructions](docs/atc/SKILL.md).
For native-engine work, also follow the [native contributor skill](docs/recipes/SKILL.md); read any additional AGENTS.md in the folders you change.

Inspect `atc board --json` and check the issue before starting work.
Claim only free work, start only after a successful claim or confirmation that the issue is yours, and coordinate with the holder when it is taken.
Follow the [README workflow](README.md#contributing-with-agents) to publish an anonymous session on the [agent radar](https://tensorfold.dev/atc), pulse while working and land when the session stops.
Show a GitHub name with `--github-name` only when the human opts in.

Use `atc who <file> --json` to inspect overlapping edits and `atc overlap --json` before opening a pull request.
Resolve reported conflicts and link the issue in the pull request description so other contributors can see the work in progress.
