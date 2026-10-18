# Local GUI benchmarks

Open Models, My models, Benchmarks, or use Benchmark in an eligible model's
Performance details.
The feature uses the bundled TensorFold 1.0.12 `benchmark` command and an installed
Hugging Face checkpoint already present in the configured cache.
Linked-folder models are currently unsupported; a run never downloads a model.

The standard community-v1 run uses fixed Code and Chat fixtures, temperatures 1
and 0, five measured repetitions per cell, a 4,096-token context and a 256-token
output limit.
Four warm-ups plus 20 measured requests produce 24 requests in total.
Serial baseline disables speculative drafting; the standard mode can use a
matching cached drafter, recorded in the receipt.
Saved model profiles are not changed to run this test.

Finish local chats, workers and external clients using the local API first.
TensorFold reserves the local engine, pauses its idle app-managed server and
starts a separate owned loopback benchmark server with prompt reuse and snapshots
disabled.
It restores the previous loaded model and its exact profile afterward, including
after cancellation or a failed run; restoration failures are shown separately.
Other work on the Mac can affect measurements.

Closing the drawer leaves the run active.
Reopen Benchmarks to follow it, cancel it or select a previous local result.
Cancellation requests a graceful stop and then uses bounded owned-process cleanup;
an early stop can have no receipt or only partial attempts, never a claimed complete
standard result.
Quitting closes admission and drains the owned benchmark without reloading a model
just before the app exits.

Each Code/Chat and temperature cell shows its valid sample count and outcomes.
Medians and ranges appear only for complete cells; early end-of-output, failed
warm-ups, errors and unmeasured delivery remain visible.
HTTP delivery speed, first-text latency and server-reported decode speed are
different measurements.
Missing metrics stay unreported, including server decode duration when the engine
does not provide the field expected by the native receipt parser.
No quality score, power usage or peak-memory estimate is inferred from this test.

Results use the existing protected profile store and retain recorded hardware,
engine, checkpoint and settings metadata.
Storage failures are visible and a validated receipt can be exported explicitly.
Review and export JSON opens a native save chooser; the exported object uses the
strict community receipt schema, excluding conversations, generated answers,
credentials and operator paths.
Basic copy previews the text that will be copied.
The Community benchmarks button only opens the existing public website.

No contributor account is needed to measure or export locally.
The app does not sign up, submit or publish results automatically.
The website performs its own checkpoint and protocol checks, and measurements
remain community-reported rather than independently verified.
See [community benchmark documentation](https://github.com/ashhart/TensorFold/blob/main/docs/community-benchmarks.md)
for the public submission workflow.
