//! KV-path RMS + SWA RoPE/FP8 store. This owns only the norm feeding the SWA producer.
const std = @import("std");
const cuda = @import("cuda");

/// Mirrors kv_glue.cu; norm_out is an optional contiguous bf16 debug output, not used in production.
pub const Args = extern struct {
    x: u64 = 0,
    w: u64 = 0,
    cs: u64 = 0,
    v: u64 = 0,
    s: u64 = 0,
    pos: u64 = 0,
    sl: u64 = 0,
    norm_out: u64 = 0,
    x_stride: i64 = 512,
    cs_stride: i64 = 64,
    v_stride: i64 = 576,
    s_stride: i64 = 8,
    inv_k: f32 = 1.0 / 512.0,
    eps: f32 = 1e-20,
    ring: c_int = 256,
    rows: c_int = 0,
};
comptime {
    std.debug.assert(@sizeOf(Args) == 112 and @offsetOf(Args, "inv_k") == 96 and @offsetOf(Args, "ring") == 104);
}

pub const Functions = struct {
    norm_store: cuda.Function,
    pub fn resolve(m: cuda.Module) !Functions {
        return .{ .norm_store = try m.function("kv_norm_store") };
    }
};

pub const Ops = struct {
    f: *const Functions,
    stream: cuda.Stream,
    pub fn normStore(o: Ops, a: Args, n: usize) !void {
        if (n == 0 or n > 64 or a.x_stride < 512 or a.cs_stride < 64 or a.v_stride < 576 or a.s_stride < 8 or
            a.ring <= 0 or (a.ring & (a.ring - 1)) != 0 or a.inv_k != 1.0 / 512.0 or
            (a.rows != 0 and a.rows != 1) or (a.rows == 1 and a.sl == 0)) return error.Shape;
        var args: cuda.Args = .{};
        args.add(a);
        try cuda.launch.launch(o.f.norm_store, .{ .grid = .{ .x = @intCast(n) }, .block = .{ .x = 128 } }, o.stream, &args);
    }
};
