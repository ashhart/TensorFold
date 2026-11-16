//! Force native Nemotron's actual prompt and completion bodies to compile; never run this object.
const tf = @import("tensorfold");
const lanes = @import("lanes");

export fn nemotronPrefillWidthsCompile(b: *tf.nemotron.backend.Metal, s: *lanes.Stream) u8 {
    const backend = b.backend();
    backend.prefill(s) catch return 1;
    // the stepped pass and its cancel (#3021), which the lane host calls between rounds
    _ = backend.vtable.prefill_step.?(backend.ptr, s, 0) catch return 1;
    backend.vtable.cancel_prefill.?(backend.ptr, s) catch return 1;
    const first = backend.first(s, s.prompt_len) catch return 1;
    _ = backend.read(first) catch return 1;
    backend.release(s);
    return 0;
}
