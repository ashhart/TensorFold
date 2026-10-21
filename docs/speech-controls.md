# Speech control allowlist

`tensorfold serve MODEL --allowed-controls 'sighs,laughs'` filters the `input` of
`POST /v1/audio/speech` before UTF-8, empty-input, length and prompt checks,
tokenization, and prompt-derived seeding. It does not affect chat, completions,
or the standalone generation tool. Omitting the flag leaves input byte-identical
and does not allocate a filtered copy. An empty list (`--allowed-controls ''`)
strips recognized delivery controls. Names compare after Unicode NFC normalization,
lowercasing, whitespace folding and removal of trailing `!?.…`. Empty
comma-separated entries are ignored. Names must be valid UTF-8 without square
brackets; novel descriptors remain accepted.

## Open vocabulary and public examples

The [official vLLM-Omni recipe](https://recipes.vllm.ai/fishaudio/drama-3)
states that there is no fixed tag list: tags are ordinary text and any short
descriptor is accepted. LIST therefore accepts novel descriptors such as
`gentle robot`; it does not reject an otherwise valid name as unknown.

| Example | Public source |
| --- | --- |
| `sighs`, `laughs` | [Drama model card, Point tags](https://huggingface.co/fishaudio/drama-3#point-tags) |
| `laughing`, `whisper`, `angry`, `excited`, `pause` | [vLLM-Omni recipe, Event and emotion tags](https://recipes.vllm.ai/fishaudio/drama-3) |

`drama_cues.txt` lists only words used in tags in Fish Audio's public Drama 3
and Emotion Control docs, the Drama 3 model card and the Drama 3 technical
report. These examples are not an exhaustive list.

## Stripping rules

Listed square-bracket controls are retained with their normalized names. An
unlisted bracket is removed only if every word is in `drama_cues.txt` and at least
one word is a cue rather than a modifier. Unknown brackets, notes, nested
contextual brackets, indexes, Markdown links, quoted syntax, unclosed brackets
and brackets beyond the control scan bound are preserved.

The documented scoped tags `whisper`, `emphasis`, `soft`, `fast`, `slow`,
`stress` and `forceful` are removed case-insensitively, including inside
preserved brackets; their enclosed text stays. Other angle-bracket text stays.

Removing a control preserves the surrounding whitespace run with more line
breaks (the preceding run wins ties), and removes leading whitespace when
there is no preceding text. Whitespace elsewhere is unchanged. An input that
becomes empty is rejected by the existing speech validation.

This is a lexical filter, not a general prompt-injection or content-safety policy.
