# Speech silence redraws

`tensorfold serve MODEL --silence-floor-dbfs F --silence-seconds S --silence-redraws N`
draws a whole `POST /v1/audio/speech` reply again when the silence guard ended it
with `silence`, up to N times (0 to 255). Omitted or 0, it never redraws, never
reads `X-Deadline-S`, and leaves every reply unchanged. Text models refuse
`--silence-redraws` at startup, an explicit 0 included. `--silence-redraws` of 1
or more needs an enabled silence guard; `--silence-redraw-budget-s`,
`--silence-redraw-controls` and the [short reply](#short-replies) flags need
`--silence-redraws` of 1 or more.

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
`X-Finish-Reason`. A redraw that ends `silence`, or a [short](#short-replies)
`stop`, replaces the previous take and redrawing continues; any other finish is
sent. When a redraw does not start (budget spent, engine busy) or ends without
audio, the previous take is sent. The server keeps a second PCM buffer, sized
like the first, while a reply redraws. It logs each redraw's finish reason and
why any redraw went unused.

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
no redraw starts and the first take is sent. No control list is built in.

## Short replies

A take can also collapse into an early `stop`: speech ends long before the text
is spoken, with no silent tail for the guard to trip on. With
`--short-reply-ratio R` and `--short-reply-chars-per-s CPS`, a whole take that
ends `stop` with less than R × chars / CPS seconds of audio is redrawn as a
silenced one is, under the same count, budget and controls. chars counts the
code points of the text the take spoke, bracketed controls included: the input
after any `--allowed-controls` filtering, or for a redraw, after any
`--silence-redraw-controls` filtering. R is above 0 and below 1. Texts of fewer
than 200 characters are never judged, since a fixed lead-in and tail dominate
their duration, and neither are `length` finishes. Neither flag has a default;
each needs the other, and without them no `stop` is redrawn.

No rate is built in: it depends on the voice, its pace and its language.
Calibrate both flags on the voice you serve:

1. Serve it as usual, without the short reply flags.
2. Send whole replies of 200 characters or more that represent your traffic,
   controls included.
3. For each reply that finished `stop` and that you heard to be complete, divide
   its characters, counted as above, by its seconds of audio: the `samples` of
   the server's reply log line divided by the sample rate in the WAV header.
4. Set CPS to the median of those rates. Each complete reply's share is
   seconds × CPS / chars; set R below the lowest share, with a margin, so that
   no complete reply would have been redrawn.
5. Check that the replies you heard cut short fall below R. If they overlap the
   complete ones, the ratio cannot tell them apart on that voice: leave the
   flags off.

Calibrate again when the served voice or the kind of text changes.

## Streams

Streams are never redrawn. A stream sends each decoded prefix as the engine
publishes it, so audio may already have reached the client when the guard trips,
and replacing it would mean buffering the stream and changing its semantics.
A stream the guard ends finishes with the `silence` trailer exactly as without
redraws. With redraws enabled, its initial headers include `X-Silence-Redraws: 0`;
this is not a trailer. Its `X-Deadline-S` is validated even though streams never redraw.
