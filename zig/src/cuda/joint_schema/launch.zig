//! The joint schema head's kernels (kernels/cuda/joint_schema.cu) loaded from their image and launched.
const std = @import("std");
const kernels = @import("cuda").kernels;
const launch = @import("cuda").launch;
const Driver = @import("cuda").Driver;
const Module = @import("cuda").Module;
const Function = @import("cuda").Function;
const Stream = @import("cuda").Stream;

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
    lex_sums: Function,
    span_means: Function,
    option_summary: Function,
    logits: Function,
    combine: Function,
    add: Function,
    broadcast: Function,
};

pub const Fns = struct {
    mod: Module,
    k: Kernels,

    pub fn load(d: *const Driver) !Fns {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.joint_schema);
        errdefer mod.unload();
        var k: Kernels = undefined;
        inline for (@typeInfo(Kernels).@"struct".field_names) |name| @field(k, name) = try mod.function("tf_js_" ++ name);
        return .{ .mod = mod, .k = k };
    }

    pub fn unload(f: *Fns) void {
        f.mod.unload();
    }

    /// Affine LayerNorm over `rows` rows of `dim` BF16 values (FP32 statistics), into y.
    pub fn layerNorm(f: *const Fns, s: Stream, x: u64, w: u64, b: u64, y: u64, rows: usize, dim: usize, eps: f32) !void {
        if (rows == 0) return;
        if (dim == 0 or dim > std.math.maxInt(i32) or rows > std.math.maxInt(u32)) return error.Invalid;
        var a: launch.Args = .{};
        for ([_]u64{ x, w, b, y }) |p| a.add(p);
        a.add(@as(i32, @intCast(dim)));
        a.add(eps);
        try go(f.k.layer_norm, s, .{ .x = @intCast(rows) }, &a);
    }

    pub fn gelu(f: *const Fns, s: Stream, x: u64, n: usize) !void {
        var a: launch.Args = .{};
        a.add(x);
        a.add(@as(i64, @intCast(n)));
        try flat(f.k.gelu, s, n, &a);
    }

    /// `rows` rows of `cols` FP32 scores to their softmax in place.
    pub fn softmaxRows(f: *const Fns, s: Stream, x: u64, rows: usize, cols: usize) !void {
        if (rows == 0) return;
        if (cols == 0 or cols > std.math.maxInt(i32) or rows > std.math.maxInt(u32)) return error.Invalid;
        var a: launch.Args = .{};
        a.add(x);
        a.add(@as(i32, @intCast(cols)));
        try go(f.k.softmax_rows, s, .{ .x = @intCast(rows) }, &a);
    }

    /// BF16 to FP32 (`widen`) or FP32 to BF16 (`round`) over n values.
    pub fn convert(f: *const Fns, s: Stream, to_f32: bool, x: u64, y: u64, n: usize) !void {
        var a: launch.Args = .{};
        a.add(x);
        a.add(y);
        a.add(@as(i64, @intCast(n)));
        try flat(if (to_f32) f.k.widen else f.k.round, s, n, &a);
    }

    /// FP32 sums[span][dim] plus span rows inside prompt rows first..first+count of `rows`, in row order.
    pub fn spanSums(f: *const Fns, s: Stream, rows: u64, first: usize, count: usize, dim: usize, spans: u64, n: usize, sums: u64) !void {
        if (n == 0 or count == 0) return;
        var a: launch.Args = .{};
        a.add(rows);
        for ([_]usize{ first, count, dim }) |v| a.add(@as(i32, @intCast(v)));
        a.add(spans);
        a.add(sums);
        try go(f.k.span_sums, s, .{ .x = @intCast((dim + threads - 1) / threads), .y = @intCast(n) }, &a);
    }

    /// FP32 sums[span][dim] plus the packed lm_head rows (`lane`, qmmf's BF16 layout) of each span's tokens `ids`.
    pub fn lexSums(f: *const Fns, s: Stream, lane: u64, ids: u64, dim: usize, spans: u64, n: usize, sums: u64) !void {
        if (n == 0) return;
        if (dim % 64 != 0) return error.Invalid;
        var a: launch.Args = .{};
        a.add(lane);
        a.add(ids);
        a.add(@as(i32, @intCast(dim)));
        a.add(spans);
        a.add(sums);
        try go(f.k.lex_sums, s, .{ .x = @intCast((dim + threads - 1) / threads), .y = @intCast(n) }, &a);
    }

    /// BF16 means of `n` spans' FP32 sums.
    pub fn spanMeans(f: *const Fns, s: Stream, sums: u64, spans: u64, n: usize, dim: usize, out: u64) !void {
        if (n == 0) return;
        var a: launch.Args = .{};
        a.add(sums);
        a.add(spans);
        a.add(@as(i32, @intCast(dim)));
        a.add(out);
        try go(f.k.span_means, s, .{ .x = @intCast((dim + threads - 1) / threads), .y = @intCast(n) }, &a);
    }

    /// Step 5's option summaries, grouped by `first` (questions + 1 ints); `weights`: an FP32 slot an option.
    pub fn optionSummary(f: *const Fns, s: Stream, routed: u64, fields: u64, first: u64, questions: usize, width: usize, weights: u64, out: u64) !void {
        if (questions == 0) return;
        var a: launch.Args = .{};
        for ([_]u64{ routed, fields, first }) |p| a.add(p);
        a.add(@as(i32, @intCast(width)));
        a.add(@as(f32, @floatCast(@sqrt(@as(f64, @floatFromInt(width))))));
        a.add(weights);
        a.add(out);
        try go(f.k.option_summary, s, .{ .x = @intCast(questions) }, &a);
    }

    pub const Logits = struct { fields: u64, opts: u64, qv: u64, g: u64, lex: u64, owner: u64, width: usize, hidden: usize, prior_scale: f32, prior: u64, cosine: u64, feat: u64 };

    /// Step 7 before the scorer for `options` options: prior and cosine (FP32 holding BF16) and the scorer's rows.
    pub fn logits(f: *const Fns, s: Stream, x: Logits, options: usize) !void {
        if (options == 0) return;
        var a: launch.Args = .{};
        for ([_]u64{ x.fields, x.opts, x.qv, x.g, x.lex, x.owner }) |p| a.add(p);
        a.add(@as(i32, @intCast(x.width)));
        a.add(@as(i32, @intCast(x.hidden)));
        a.add(x.prior_scale);
        for ([_]u64{ x.prior, x.cosine, x.feat }) |p| a.add(p);
        try go(f.k.logits, s, .{ .x = @intCast(options) }, &a);
    }

    /// The gated logits from the prior, the cosine and the scorer's residual.
    pub fn combine(f: *const Fns, s: Stream, prior: u64, cosine: u64, res: u64, n: usize, joint_scale: f32, gate: f32, out: u64) !void {
        var a: launch.Args = .{};
        for ([_]u64{ prior, cosine, res }) |p| a.add(p);
        a.add(@as(i32, @intCast(n)));
        a.add(joint_scale);
        a.add(gate);
        a.add(out);
        try flat(f.k.combine, s, n, &a);
    }

    /// out = bf16(x + y) over n values.
    pub fn add(f: *const Fns, s: Stream, x: u64, y: u64, out: u64, n: usize) !void {
        var a: launch.Args = .{};
        for ([_]u64{ x, y, out }) |p| a.add(p);
        a.add(@as(i64, @intCast(n)));
        try flat(f.k.add, s, n, &a);
    }

    /// `rows` copies of one row of n BF16 values at `row` into out.
    pub fn broadcast(f: *const Fns, s: Stream, row: u64, out: u64, rows: usize, n: usize) !void {
        if (rows == 0 or n == 0) return;
        const total = rows * n;
        var a: launch.Args = .{};
        a.add(row);
        a.add(out);
        a.add(@as(i32, @intCast(n)));
        a.add(@as(i64, @intCast(total)));
        try flat(f.k.broadcast, s, total, &a);
    }
};

fn go(f: Function, s: Stream, grid: launch.Dim3, a: *launch.Args) !void {
    try launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, s, a);
}

fn flat(f: Function, s: Stream, n: usize, a: *launch.Args) !void {
    if (n == 0) return;
    try go(f, s, .{ .x = @intCast((n + threads - 1) / threads) }, a);
}
