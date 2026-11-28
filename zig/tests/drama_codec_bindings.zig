//! Compiles codec_net.zig's schedules and weight_names.zig's load order over one backend's bindings.
const d3 = @import("drama3");

comptime {
    const C = d3.codec.Codec;
    for (.{ &C.load, &C.fork, &C.poison, &C.destroy, &C.reset, &C.decode, &C.encode }) |f| _ = f;
    const O = d3.tts.CodecOnly;
    for (.{ &O.open, &O.close, &O.decode, &O.encode, &O.weightBytes }) |f| _ = f;
    _ = &d3.weights.load;
}
