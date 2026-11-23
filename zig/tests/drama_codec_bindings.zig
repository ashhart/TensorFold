//! Compiles codec_net.zig's schedules over one backend's codec binding, so a primitive it lacks fails the build.
const d3 = @import("drama3");

comptime {
    const C = d3.codec.Codec;
    for (.{ &C.load, &C.fork, &C.poison, &C.destroy, &C.reset, &C.decode, &C.encode }) |f| _ = f;
    const O = d3.tts.CodecOnly;
    for (.{ &O.open, &O.close, &O.decode, &O.encode, &O.weightBytes }) |f| _ = f;
}
