# Working on TensorFold

## Ask the human before installing or publishing

Following this file installs software and reports activity to a website, so tell the human what each step does and get a clear yes first.

- **Installing ATC.** `npm install -g github:ashhart/ATC` installs whatever is on the maintainer's ATC `main` branch that day, with no fixed version. Install a tag the human has looked at instead, for example `npm install -g github:ashhart/ATC#v0.9.6`. ATC's optional Claude Code plugin runs a hook before every shell command and every file edit, and those hooks can block the command or edit.
- **Public check-ins are on by default here.** This repository commits `.github/atc-presence.json`, so a new `atc claim` or `atc start` publishes unless you pass `--local`. Publishing posts a `/atc presence` comment on the issue from the human's GitHub account. It then sends the issue number, a session token, that comment's ID, the work state and whether the session shows the GitHub name (the name itself is not sent) to `https://tensorfold.dev`, which the TensorFold project runs, and sends them again on every pulse.
- **"Anonymous" only hides the name on the public radar.** The comment ID ties each session to the human's GitHub account for the site operator, and anyone reading the issue sees the comment while the session is open. The site also sees the machine's IP address and when it works.
- **Working without ATC.** A draft pull request whose description says `Fixes #<issue>` also counts as a claim (see [the ATC instructions](docs/atc/SKILL.md)).

Pass `--local` to `claim` and `start` unless the human has agreed to public check-ins.

## Coordinating with ATC

TensorFold uses Agent Traffic Control to coordinate contributors and coding agents.
Before editing, read [CONTRIBUTING.md](CONTRIBUTING.md) and follow the [ATC contributor instructions](docs/atc/SKILL.md).
For native-engine work, also follow the [native contributor skill](docs/recipes/SKILL.md); read any additional AGENTS.md in the folders you change.

Inspect `atc board --json` and check the issue before starting work.
Claim only free work, start only after a successful claim or confirmation that the issue is yours, and coordinate with the holder when it is taken.
Use ATC 0.7.0 or later from an updated clone: without `--local`, a successful new claim starts a public session on the [agent radar](https://tensorfold.dev/atc), as described above.
Keep its returned session number, pulse only while working and land when the session stops; for work already yours, use `atc start ISSUE --json`.
Follow the [contributing workflow](CONTRIBUTING.md#contributing-with-agents); use `--local` with claim or start unless the human has agreed to public check-ins.
Show a GitHub name with `--github-name` only when the human opts in.

Use `atc who <file> --json` to inspect overlapping edits and `atc overlap --json` before opening a pull request.
Resolve reported conflicts and link the issue in the pull request description so other contributors can see the work in progress.
Before opening or updating a pull request, run the [pull request skill](docs/pr/SKILL.md) pre-flight.
