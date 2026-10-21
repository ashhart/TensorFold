# Speech silence redraws

`tensorfold serve MODEL --silence-floor-dbfs F --silence-seconds S --silence-redraws N`
draws a whole `POST /v1/audio/speech` reply again when the silence guard ended it
with `silence`, up to N times (0 to 255). Omitted or 0, it never redraws, never
reads `X-Deadline-S`, and leaves every reply unchanged. Text models refuse
`--silence-redraws` at startup, an explicit 0 included. `--silence-redraws` of 1
or more needs an enabled silence guard; `--silence-redraw-budget-s` and
`--silence-redraw-controls` need `--silence-redraws` of 1 or more.

## Seeds

Redraw k (1 to N) seeds its prompt with `TENSORFOLD_SEED_SALT` plus k; the first
take keeps the stock seed, salt plus 0. Redraws are therefore deterministic. With
the same prompt, redraw k under salt s is the same draw as the first take under
salt s + k, so a sweep over salts should space compared salts more than N apart.

## Budget

A redraw starts only when t + d fits the budget, where t is the time since the
route received the request, before reading its body, and d is how long the
previous take ran. The budget is the lower of `--silence-redraw-budget-s` and the
request's `X-Deadline-S` header, each a positive finite number of seconds. With
neither, only N bounds redraws. A malformed `X-Deadline-S` is an HTTP 400 while
redraws are on, including streams. The budget decides starts; a running take is not cut.
If the engine is busy, submission retries after a sleep until the latest start
that leaves room for the previous take's duration. Without a deadline or server
budget, a busy submission keeps the previous audio rather than waiting unboundedly.

## Reply

The reply is the last take that finished with audio, sent with its own
`X-Finish-Reason`. A redraw that ends `stop` or `length` is sent. A redraw that
ends `silence` replaces the previous take and redrawing continues. When a redraw
does not start (budget spent, engine busy) or ends without audio, the previous
take is sent. The server keeps a second PCM buffer, sized like the first, while a
reply redraws. It logs each redraw's finish reason and why any redraw went unused.

With `--silence-redraws` above zero, every speech response carries
`X-Silence-Redraws` as a decimal integer, including zero and error replies.
It counts redraws admitted to the engine, not trips or attempted submissions;
a redraw that later fails still counts. Omission or an explicit zero leaves
the header absent.

## Controls

Without `--silence-redraw-controls`, a redraw speaks the same input. With
`--silence-redraw-controls LIST`, every redraw speaks the input, after any
`--allowed-controls` filtering, filtered by LIST under the
[speech control allowlist](speech-controls.md) grammar and stripping rules:
listed controls stay and unlisted controls are removed. An explicitly empty list
(`--silence-redraw-controls ''`) instead selects plain speech: every closed
square-bracket span and documented lowercase scoped tag is replaced by a space,
a leading `Direction:` line followed by `Line:` is removed, and Unicode
whitespace is folded and trimmed. If no such span is found, input is unchanged.
This empty-list setting reproduces the reference plain-controls redraw transform.
If the filtered input cannot be spoken, for example because nothing is left,
no redraw starts and the silenced take is sent. No control list is built in.

## Streams

Streams are never redrawn. A stream sends each decoded prefix as the engine
publishes it, so audio may already have reached the client when the guard trips,
and replacing it would mean buffering the stream and changing its semantics.
A stream the guard ends finishes with the `silence` trailer exactly as without
redraws. With redraws enabled, its initial headers include `X-Silence-Redraws: 0`;
this is not a trailer. Its `X-Deadline-S` is validated even though streams never redraw.
