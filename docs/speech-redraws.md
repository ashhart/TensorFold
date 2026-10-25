# Speech redraws

`tensorfold serve MODEL --silence-floor-dbfs F --silence-seconds S --silence-redraws N`
draws a whole `POST /v1/audio/speech` reply again when the silence guard ended it
with `silence`, up to N times (0 to 255). Omitted or 0, with the register guard
off, it never redraws, never reads `X-Deadline-S`, and leaves every reply
unchanged. Text models refuse `--silence-redraws` at startup, an explicit 0
included, and the register flags. `--silence-redraws` of 1 or more needs an
enabled silence guard; `--silence-redraw-controls` and the
[short reply](#short-replies) flags need `--silence-redraws` of 1 or more, and
`--silence-redraw-budget-s` needs it or the [register guard](#register).

## Why Drama 3 needs it

Sampled at temperature 1.0, top-k 30 and top-p 0.9, Drama 3 sometimes speaks
the first sentence and then stalls: the rest of the reply is silence or a flat
low hum until the frame cap. The seed comes from the prompt, so a stalling input
stalls on every request. On one DGX Spark with `fishaudio/drama-3`, 2 of 9
paragraph-length inputs stalled on the default prompt path and 3 of 9 with
`TF_DRAMA_PREFILL_ROWS=128`, on different inputs; a separate port of the
reference sampler stalled on 3 of 16 seeds (#1275).

With the guard below, none of 18 whole replies stalled. Each stalled input
recovered in one redraw, and every reply that had not stalled kept the same
bytes. Take it as a starting point and calibrate the floor and durations on the
voice you serve, as the sections below describe:

```sh
tensorfold serve DRAMA3_DIR --silence-floor-dbfs=-30 --silence-seconds=4 \
  --silence-mid-s=2 --silence-mid-min-voiced-s=1 \
  --silence-redraws=3 --silence-redraw-budget-s=90
```

A floor of -30 dBFS also catches the hum. Streams are not redrawn (see
[Streams](#streams)): a stalled stream ends early, with the rest of the text
unspoken.

## Seeds

Without an explicit request `seed`, redraw k (1 to N) seeds its prompt with `TENSORFOLD_SEED_SALT` plus k; the first
take keeps the stock seed, salt plus 0. Redraws are therefore deterministic. With
the same prompt, redraw k under salt s is the same draw as the first take under
salt s + k, so a sweep over salts should space compared salts more than N apart,
or N + 1 with the register guard.

An explicit request `seed` instead uses that exact integer for the first take
and seed + k modulo 2^64 for each redraw, counting both silence and register
attempts. Requested `temperature`, `top_p` and `top_k` remain in effect through
all redraws and control filtering; see [speech sampling](speech-sampling.md).
The seed rule does not make timing-based redraw budgets deterministic.

## Budget

A redraw starts only when t + d fits the budget, where t is the time since the
route received the request, before reading its body, and d is how long the
previous take ran. The budget is the lower of `--silence-redraw-budget-s` and the
request's `X-Deadline-S` header, each a positive finite number of seconds. With
neither, only the retry counts bound redraws. A malformed `X-Deadline-S` is an
HTTP 400 while silence redraws or the register guard are on, including streams.
The budget decides starts; a running take is not cut.
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
`--silence-redraw-controls LIST`, silence redraws speak the input, after any
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
A direct register redraw keeps the current prompt; it does not apply the silence
redraw filter. If a silence redraw already filtered that prompt, the register
redraw inherits it.

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

## Register

`--register-redraw-st ST --register-redraw-high PERCENT --register-high-hz HZ`
draws a whole reply once more when a phrase after the first opens far above the
others in pitch, and sends the take with the smaller jump. The three flags have
no defaults and are given together: ST above 0, PERCENT above 0 to 100, HZ 75 to
600. Without them no reply is measured.

The guard measures takes that do not trip the silence retry rule. It reads pitch every 10 ms
by normalized cross-correlation over 20 ms windows of the audio averaged down to
8 to 16 kHz. This autocorrelation detector uses Praat's published To Pitch (ac)
parameter defaults: 75 to 600 Hz, a voicing
threshold of 0.45, a silence threshold of 0.03 of the take's peak and an octave
cost of 0.01. Unvoiced runs of 0.25 s part phrases, a phrase voiced for less than
0.25 s is skipped, and a phrase's opening is the median pitch of its first 0.25 s
voiced. A phrase after the first trips when its opening is ST semitones or more
above the median opening of the other phrases and PERCENT or more of its voiced
frames lie above HZ. The take's jump is the largest such excess. Audio sampled
below 8 kHz, all-zero audio, and takes with fewer than two retained phrases are
not measured. A measured take with no qualifying jump counts as zero.
The detector is not Praat's analysis: its values can differ from Praat's.

A take that trips is drawn once more, using the next seed under the [seed rule](#seeds) after any
silence redraws, if t + d fits the budget. The redraw is sent only if it does not
end as `silence` and has a strictly smaller jump; ties retain the first take.
A `length` redraw can replace a `stop` take. An unmeasured redraw can win only
when the silence guard is enabled; with it off, the first take is retained.
An unmeasurable first take does not trip.
The log gives the jump, the redraw's finish and why a first take is retained.

A register redraw that trips silence reenters the silence retry loop with its
remaining count and the same request budget. Only after that loop ends is its
last take compared with the held register take. The held audio is never
overwritten by those retries. Exhausting silence retries does not make a
silenced take eligible for register measurement.

With the register guard configured, whole successful replies always carry
`X-Register-Trip` and `X-Register-Redraws`, each a single `0` or `1`.
Trip is latched when any measured take trips. Redraw counts engine admission,
even when the first take is retained. Streams, error replies and guard-off
replies omit both headers. Register retries do not increment `X-Silence-Redraws`.

### Calibrating on your voice

The thresholds describe a voice's pitch, so no value suits every voice. Calibrate
them on the voice you serve:

1. Serve without the register flags, keep a set of whole replies, and mark those
   with a phrase that sounds stitched in from a higher register.
2. Choose trial HZ, ST and PERCENT values and serve the same inputs and seed salts
   with the flags enabled. Calibrate against the server's own `register jump=`
   log lines, not Praat measurements.
3. Compare logged trips with the marked replies. Raise thresholds when ordinary
   phrases trip; lower them when marked high phrases do not trip. The log reports
   qualifying jumps only, so absence of a line does not establish a measured zero.
4. Repeat on held-out inputs before using the thresholds for redraw selection.

## Streams

Streams are never redrawn or measured. A stream sends each decoded prefix as the
engine publishes it, so audio may already have reached the client when a guard
trips, and replacing it would mean buffering the stream and changing its semantics.
A stream the guard ends finishes with the `silence` trailer exactly as without
redraws. With redraws enabled, its initial headers include `X-Silence-Redraws: 0`;
this is not a trailer. Its `X-Deadline-S` is validated even though streams never redraw.
