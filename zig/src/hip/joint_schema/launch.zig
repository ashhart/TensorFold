//! The joint schema head's kernels (joint_schema/joint_schema.hip) resolved from their group and launched.
const std = @import("std");
const kernels = @import("../kernels.zig");
const launch = @import("../launch.zig");
const Launcher = @import("../launches.zig").Launcher;
const Function = @import("../module.zig").Function;
const Stream = @import("../stream.zig").Stream;
const Error = @import("../runtime.zig").Error;

/// A block's threads in every row kernel.
pub const threads = 256;

/// The kernels, each named tf_js_ and its field.
pub const Kernels = struct {
    layer_norm: Function,
    gelu: Function,
    softmax_rows: Function,
    widen: Function,
    round: Function,
    span_sums: Function,
    span_means: Function,
    option_summary: Function,
    logits: Function,
    combine: Function,
};

pub const Fns = struct {
    k: Kernels,

    pub fn load(l: *const Launcher) Error!Fns {
        const m = l.mods[@backingInt(kernels.Group.joint_schema)];
        var k: Kernels = undefined;
        inline for (@typeInfo(Kernels).@"struct".field_names) |name| @field(k, name) = try m.function("tf_js_" ++ name);
        return .{ .k = k };
    }

    /// Affine LayerNorm over `rows` rows of `dim` BF16 values (FP32 statistics), into y.
    pub fn layerNorm(f: Fns, s: Stream, x: u64, w: u64, b: u64, y: u64, rows: usize, dim: usize, eps: f32) Error!void {
        if (rows == 0) return;
        if (dim == 0 or rows > std.math.maxInt(u32)) return error.Invalid;
        var a: launch.Args = .{};
        for ([_]u64{ x, w, b, y }) |p| try a.add(p);
        try a.add(@as(c_int, @intCast(dim)));
        try a.add(eps);
        try go(f.k.layer_norm, s, .{ .x = @intCast(rows) }, &a);
    }

    pub fn gelu(f: Fns, s: Stream, x: u64, n: usize) Error!void {
        var a: launch.Args = .{};
        try a.add(x);
        try a.add(@as(c_longlong, @intCast(n)));
        try flat(f.k.gelu, s, n, &a);
    }

    /// `rows` rows of `cols` FP32 scores to their softmax in place.
    pub fn softmaxRows(f: Fns, s: Stream, x: u64, rows: usize, cols: usize) Error!void {
        if (rows == 0) return;
        if (cols == 0 or cols > std.math.maxInt(c_int) or rows > std.math.maxInt(u32)) return error.Invalid;
        var a: launch.Args = .{};
        try a.add(x);
        try a.add(@as(c_int, @intCast(cols)));
        try go(f.k.softmax_rows, s, .{ .x = @intCast(rows) }, &a);
    }

    /// BF16 to FP32 (`widen`) or FP32 to BF16 (`round`) over n values.
    pub fn convert(f: Fns, s: Stream, to_f32: bool, x: u64, y: u64, n: usize) Error!void {
        var a: launch.Args = .{};
        try a.add(x);
        try a.add(y);
        try a.add(@as(c_longlong, @intCast(n)));
        try flat(if (to_f32) f.k.widen else f.k.round, s, n, &a);
    }

    /// FP32 sums[span][dim] plus span rows inside prompt rows first..first+count, in row order; `ids` reads a table.
    pub fn spanSums(f: Fns, s: Stream, rows: u64, ids: u64, first: usize, count: usize, dim: usize, spans: u64, n: usize, sums: u64) Error!void {
        if (n == 0 or count == 0) return;
        var a: launch.Args = .{};
        try a.add(rows);
        try a.add(ids);
        for ([_]usize{ first, count, dim }) |v| try a.add(@as(c_int, @intCast(v)));
        try a.add(spans);
        try a.add(sums);
        try go(f.k.span_sums, s, .{ .x = @intCast((dim + threads - 1) / threads), .y = @intCast(n) }, &a);
    }

    /// BF16 means of `n` spans' FP32 sums.
    pub fn spanMeans(f: Fns, s: Stream, sums: u64, spans: u64, n: usize, dim: usize, out: u64) Error!void {
        if (n == 0) return;
        var a: launch.Args = .{};
        try a.add(sums);
        try a.add(spans);
        try a.add(@as(c_int, @intCast(dim)));
        try a.add(out);
        try go(f.k.span_means, s, .{ .x = @intCast((dim + threads - 1) / threads), .y = @intCast(n) }, &a);
    }

    /// Step 5's option summaries, grouped by `first` (questions + 1 ints); `weights`: an FP32 slot an option.
    pub fn optionSummary(f: Fns, s: Stream, routed: u64, fields: u64, first: u64, questions: usize, width: usize, weights: u64, out: u64) Error!void {
        if (questions == 0) return;
        var a: launch.Args = .{};
        for ([_]u64{ routed, fields, first }) |p| try a.add(p);
        try a.add(@as(c_int, @intCast(width)));
        try a.add(@as(f32, @floatCast(@sqrt(@as(f64, @floatFromInt(width))))));
        try a.add(weights);
        try a.add(out);
        try go(f.k.option_summary, s, .{ .x = @intCast(questions) }, &a);
    }

    pub const Logits = struct { fields: u64, opts: u64, qv: u64, g: u64, lex: u64, owner: u64, width: usize, hidden: usize, prior_scale: f32, prior: u64, cosine: u64, feat: u64 };

    /// Step 7 before the scorer for `options` options: prior and cosine (FP32 holding BF16) and the scorer's rows.
    pub fn logits(f: Fns, s: Stream, x: Logits, options: usize) Error!void {
        if (options == 0) return;
        var a: launch.Args = .{};
        for ([_]u64{ x.fields, x.opts, x.qv, x.g, x.lex, x.owner }) |p| try a.add(p);
        try a.add(@as(c_int, @intCast(x.width)));
        try a.add(@as(c_int, @intCast(x.hidden)));
        try a.add(x.prior_scale);
        for ([_]u64{ x.prior, x.cosine, x.feat }) |p| try a.add(p);
        try go(f.k.logits, s, .{ .x = @intCast(options) }, &a);
    }

    /// The gated logits from the prior, the cosine and the scorer's residual.
    pub fn combine(f: Fns, s: Stream, prior: u64, cosine: u64, res: u64, n: usize, joint_scale: f32, gate: f32, out: u64) Error!void {
        var a: launch.Args = .{};
        for ([_]u64{ prior, cosine, res }) |p| try a.add(p);
        try a.add(@as(c_int, @intCast(n)));
        try a.add(joint_scale);
        try a.add(gate);
        try a.add(out);
        try flat(f.k.combine, s, n, &a);
    }
};

fn go(f: Function, s: Stream, grid: launch.Dim3, a: *launch.Args) Error!void {
    try launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, s, a);
}

fn flat(f: Function, s: Stream, n: usize, a: *launch.Args) Error!void {
    if (n == 0) return;
    try go(f, s, .{ .x = @intCast((n + threads - 1) / threads) }, a);
}
