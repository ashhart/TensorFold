//! dots.mocr's own CUDA launch beside the dense engine's: the q|k|v bias and Torch-order RoPE (dots_mocr.cu).
const cuda = @import("cuda");

/// Threads a (row, head) block takes: dims j and j + 64 of a 128-dim head.
const threads = 64;

pub const Kernels = struct {
    mod: cuda.Module,
    rope: cuda.Function,

    pub fn load(ctx: *const cuda.Context) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(ctx.d, cuda.kernels.dots_mocr);
        errdefer mod.unload();
        return .{ .mod = mod, .rope = try mod.function("tf_dots_qkv_rope") };
    }

    pub fn deinit(k: *Kernels) void {
        k.mod.unload();
    }

    /// q [rows, h, 128] and kv [rows, k hk | v hk] from fp32 acc [rows, (h + 2 hk) 128] plus bias, rotated at pos.
    pub fn qkvRope(k: *const Kernels, s: cuda.Stream, acc: u64, bias: u64, inv: u64, pos: u64, q: u64, kv: u64, rows: usize, h: usize, hk: usize) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        for ([_]u64{ acc, bias, inv, pos, q, kv }) |v| a.add(v);
        a.add(@as(c_int, @intCast(h)));
        a.add(@as(c_int, @intCast(hk)));
        try cuda.launch.launch(k.rope, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(h + 2 * hk), .z = 1 }, .block = .{ .x = threads } }, s, &a);
    }
};
