# Speech sampling

`POST /v1/audio/speech` accepts these optional fields for Drama 3. Omission or
`null` keeps the existing default for that field. They apply to WAV replies,
whole PCM replies, streamed PCM and catalog voices.

| Field | Accepted JSON value | Default |
| --- | --- | --- |
| `temperature` | A finite number at least 0, within the `f32` range. A positive value must remain positive when converted to `f32`. | `1` |
| `top_p` | A finite number greater than 0 and at most 1, still greater than 0 after conversion to `f32`. | `0.9` |
| `top_k` | An integer from 1 through 64. | `30` |
| `seed` | An integer from 0 through 18446744073709551615. | The existing hash of the prompt and `TENSORFOLD_SEED_SALT`. |

Numbers encoded as strings and booleans are refused. `top_k` and `seed` must
use integer JSON syntax: `30.0` and `1e3` are refused. Seeds are parsed as exact
64-bit unsigned integers, including values above 9007199254740991; clients
whose JSON number type loses precision must preserve the integer when encoding
the request. Invalid fields return HTTP 400 before submission to the engine.

For `temperature` and `top_p`, the JSON parser first produces an `f64` value;
the bounds above apply to that value and its conversion to `f32`. Extremely
small floating literals such as `1e-400` and `-1e-400` already become zero in
that parser, so both request temperature 0; they are refused for `top_p`.
A positive parsed value such as `1e-100` is refused if it becomes zero only
when narrowed to `f32`.

```sh
curl http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","input":"Hello there.","temperature":0.8,"top_p":0.9,"top_k":30,"seed":42}' \
  --output speech.wav
```

The route passes these settings to the native sampler. On Metal and CUDA, `temperature`
of 0 or `top_k` of 1 selects its existing greedy path, which bypasses repetition
aware sampling (RAS). Other runtimes may handle repetition differently; matching
fields and seeds do not promise identical audio across backends or checkpoints.
The HTTP limit of 64 candidates fits both native Metal and CUDA admission.

An explicit `seed` is used exactly for the first take. If configured
[speech redraws](speech-redraws.md) admit another take, its seed is the original
seed plus the take's attempt number (1, 2, ...), wrapping modulo 2^64. This
number includes both silence and register redraws. All takes keep the requested
sampling values, including when controls are stripped or a reference prompt is
rebuilt. With no seed, the existing prompt hash and server salt plus attempt
number are unchanged. Streams keep their existing no-redraw behavior.

Seed selection is deterministic. The final take can still change if timing or
request deadlines admit a different number of redraws.

The API design references the public
[Sogni-AI/drama3-mlx generation interface](https://github.com/Sogni-AI/drama3-mlx/blob/ecc235039ccd5f982f66a6982184f57432354ab4/drama3_mlx/model.py#L79).
TensorFold's parsing and request propagation are an independent implementation;
this does not change the native sampling algorithms.
