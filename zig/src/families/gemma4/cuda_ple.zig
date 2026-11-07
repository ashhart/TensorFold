//! gemma4_ple.cu: the E-series per-layer inputs from the projected rows and the token table's rows (#1222).
const std = @import("std");
const cuda = @import("cuda");

/// tf_gemma4_ple_rows's scalars, each a bf16 value as MLX multiplies by it, and the norm's epsilon.
pub const Scalars = struct { scale: f32, token: f32, mix: f32, eps: f32 };

/// The part each norm runs over: hidden_size_per_layer_input, 256 on every E-series checkpoint.
pub const width = 256;

pub const PleRows = struct {
    mod: cuda.Module,
    fn_: cuda.Function,

    pub fn load(d: *const cuda.Driver) !PleRows {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.gemma4_ple);
        errdefer mod.unload();
        return .{ .mod = mod, .fn_ = try mod.function("tf_gemma4_ple_rows") };
    }

    pub fn unload(g: *PleRows) void {
        g.mod.unload();
    }

    /// out [rows, parts 256] from projected rows p and table rows t (packed alike) and the norm's weight w [256].
    pub fn run(g: *const PleRows, s: cuda.Stream, p: u64, t: u64, w: u64, out: u64, rows: usize, parts: usize, c: Scalars) !void {
        if (parts == 0 or rows * parts * width > std.math.maxInt(i32)) return error.RowShape;
        for ([_]u64{ p, t, out }) |v| if (v % 16 != 0) return error.RowShape;
        if (rows == 0) return;
        var a: cuda.Args = .{};
        for ([_]u64{ p, t, w, out }) |v| a.add(v);
        a.add(@as(i32, @intCast(parts)));
        for ([_]f32{ c.scale, c.token, c.mix, c.eps }) |v| a.add(v);
        try cuda.launch.launch(g.fn_, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(parts), .z = 1 }, .block = .{ .x = 64 } }, s, &a);
    }
};
