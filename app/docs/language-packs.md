# Language packs

Choose a language in **Settings → Preferences → App and onboarding**, then save.
The first-run welcome screen also has a language selector before model setup.
Packs are bundled and work offline; selecting one does not download a model or
contact a translation service.

The interface choices are English, Spanish, Latin American Spanish, Simplified
Chinese, Traditional Chinese, Japanese, Korean, Russian, Polish, Latin, Hindi,
Thai, Vietnamese and Indonesian. Latin American Spanish uses the Spanish pack
with regional number and date formatting. Traditional Chinese starts from the
same authored Chinese wording with traditional glyphs; regional terminology and
all translations still benefit from native-speaker review.

**Use this language for new assistant replies** is on by default. It adds an
ephemeral response preference to TensorFold's own model requests, including local
helper requests, without rewriting stored messages or recovery snapshots. Users
can request another language in a message or turn the preference off. Fluency
depends on the selected model. External ACP sessions and the delegated OMP agent
keep their own response-language behavior; their protocol has no verified
preference field in this integration. Fixed benchmark and capability probes keep
their reproducible inputs.

App-authored labels, explanatory text, known states, new starter prompts, menus
and tray actions use the selected pack. Existing messages, user drafts, code,
commands, filenames, model/account names and raw provider or tool diagnostics
stay original. Unknown failures have localized surrounding text and an original
diagnostic. OS dialogs, external sign-in pages, third-party documentation and
paired mobile interfaces follow their own language settings. Switching languages
does not change approvals, provider selection, security settings or tool names.

The language preference uses existing profile settings and protected host
storage. Stale saves are refused, failed saves retain the authoritative setting,
and profile changes read the current profile rather than a shared locale cache.
Dates, relative times and numbers use the selected locale; Latin uses an explicit
numeric local date where the platform has no Latin date data.

## Adding or correcting a pack

`shared/languages.ts` owns the fixed identifiers and safe response instructions.
Authored English source phrases live in bounded JSON files in
`src/lib/preferences/sources/`; translated phrases live in
`src/lib/preferences/packs/<language>/`. Use whole sentences and count messages
with named parameters, not translated fragments with English plural suffixes.
Translate only authored text, never arbitrary conversation or diagnostic data.
Keep keys and placeholder sets exact, avoid bidi controls and keep each file
below 600 lines. Explicit unchanged technical terms are checked during review.

Run `npm run languages:index` after editing the packs. The generated index gates
the selector on exact source coverage and creates the small native menu pack.
`npm run verify` checks completeness, duplicates, parameters, bounds, generated
indexes, hook ownership, protected profile persistence, menu immutability and
language-aware request behavior. Browser checks use synthetic profiles and fake
services; native keyboard/IME, OS menu rendering and real-model fluency need
separate release qualification. No dependency was added for localization.
