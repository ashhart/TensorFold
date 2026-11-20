//! TensorFold's CUDA GLPv3 additive launch ABI. Concept and format: Matt Suiche/weightless (spec
//! b181a07d3980fc48abe39d95ddb044459dd0b868); directional-steering research (Arditi et al., NeurIPS 2024) is
//! background only, nothing here projects or ablates; the additive convention is llama.cpp's; the reader and
//! startup convention is TensorFold #1835. The kernels and their arithmetic contract: kernels/cuda/deepseek_v41/glp.cuh.

const std = @import("std");
const cuda = @import("cuda");

/// One CTA per row; the 5120-column width cap matches the contract's declared launch layout.
pub const NT = 320;
pub const max_width = 5120;
pub const chain = 128;
pub const max_world = 8;

/// The mode-less entry points of the add-only image: distinct from any earlier project-and-add image's symbols, so
/// a stale image fails to resolve instead of running under another argument layout.
pub const stream_symbol = "dsv41_glp_add_stream";
pub const writer_symbol = "dsv41_glp_add_writer";
pub const final_symbol = "dsv41_glp_add_final";

/// glp.cuh's StreamArgs: the four copies [R, 4 width] edited in place (x' = x + alpha d), the tap [R, width] the
/// mean of the steered copies. Strides are in elements.
pub const StreamArgs = extern struct {
    x: u64,
    xs: i64,
    tap: u64 = 0,
    ts: i64 = 0,
    dir: u64,
    alpha: f32,
    R: c_int,
    width: c_int,
};

/// glp.cuh's WriterArgs: the gathered partials [world, R, width] reduced in rank order, edited, rewritten in place.
pub const WriterArgs = extern struct {
    g: u64,
    gr: i64,
    gs: i64,
    dir: u64,
    alpha: f32,
    R: c_int,
    width: c_int,
    world: c_int,
};

/// glp.cuh's FinalArgs: the steered streams' collapse by `pre` and the final norm into `out`.
pub const FinalArgs = extern struct {
    x: u64,
    xs: i64,
    pre: u64,
    nw: u64,
    out: u64,
    eps: f32,
    nwbf16: c_int,
    R: c_int,
    width: c_int,
};

comptime {
    std.debug.assert(@sizeOf(StreamArgs) == 56 and @offsetOf(StreamArgs, "dir") == 32 and @offsetOf(StreamArgs, "alpha") == 40 and @offsetOf(StreamArgs, "R") == 44 and @offsetOf(StreamArgs, "width") == 48);
    std.debug.assert(@sizeOf(WriterArgs) == 48 and @offsetOf(WriterArgs, "dir") == 24 and @offsetOf(WriterArgs, "alpha") == 32 and @offsetOf(WriterArgs, "R") == 36 and @offsetOf(WriterArgs, "world") == 44);
    std.debug.assert(@sizeOf(FinalArgs) == 56 and @offsetOf(FinalArgs, "eps") == 40 and @offsetOf(FinalArgs, "width") == 52);
}

pub const Error = error{ BadWidth, BadStride, BadRanks, Misaligned, BadStrength, Overlap };

fn aligned(p: u64) bool {
    return p % 16 == 0;
}

/// Bytes (bf16) a strided [rows, cols] region spans from its first element; exact in 128 bits.
fn extent(rows: c_int, stride: i64, cols: i64) i128 {
    return ((@as(i128, rows) - 1) * stride + cols) * 2;
}

fn checkRow(width: c_int, rows: c_int, alpha: f32) Error!void {
    if (width <= 0 or @rem(width, 8) != 0 or width > max_width) return error.BadWidth;
    if (rows < 0) return error.BadStride;
    // 0 (either sign) is the exact bypass the kernel returns on; a negative or nonfinite strength is never launched
    if (!std.math.isFinite(alpha) or alpha < 0) return error.BadStrength;
}

/// A fold call's refusals: the width, the strides and the alignments the kernel's 16-byte vectors need.
pub fn checkStream(a: StreamArgs) Error!void {
    try checkRow(a.width, a.R, a.alpha);
    if (a.xs < 0 or @rem(a.xs, 8) != 0 or a.xs < 4 * @as(i64, a.width) and a.R > 1) return error.BadStride;
    if (!aligned(a.x) or !aligned(a.dir)) return error.Misaligned;
    if (a.tap == 0) return;
    if (!aligned(a.tap) or @rem(a.ts, 8) != 0 or a.ts < a.width and a.R > 1) return error.Misaligned;
    if (a.ts < 0) return error.BadStride;
    if (a.R == 0) return;
    // the bf16 copies' bytes and the bf16 tap's bytes never share one
    const xe = @as(i128, a.x) + extent(a.R, a.xs, 4 * @as(i64, a.width));
    const te = @as(i128, a.tap) + extent(a.R, a.ts, a.width);
    if (a.x < te and a.tap < xe) return error.Overlap;
}

/// A writer call's refusals: the ranks the kernel holds, the strides and the alignments.
pub fn checkWriter(a: WriterArgs) Error!void {
    try checkRow(a.width, a.R, a.alpha);
    if (a.world < 1 or a.world > max_world) return error.BadRanks;
    if (a.gs < 0 or @rem(a.gs, 8) != 0 or @rem(a.gr, 8) != 0 or a.gs < a.width and a.R > 1) return error.BadStride;
    if (!aligned(a.g) or !aligned(a.dir)) return error.Misaligned;
    // rank slots are whole, disjoint regions: a rank stride that reaches past the last row's end
    if (a.world > 1 and a.R > 0 and (a.gr <= 0 or @as(i128, a.gr) < @as(i128, a.R - 1) * a.gs + a.width)) return error.BadStride;
}

/// A final call's refusals: the width is whole chains, the strides and alignments the vectors need.
pub fn checkFinal(a: FinalArgs) Error!void {
    if (a.width <= 0 or @rem(a.width, chain) != 0 or a.width > max_width) return error.BadWidth;
    if (a.R < 0 or @rem(a.xs, 8) != 0 or a.xs < 4 * @as(i64, a.width) and a.R > 1) return error.BadStride;
    if (!aligned(a.x) or !aligned(a.nw) or !aligned(a.out) or !aligned(a.pre)) return error.Misaligned;
}

pub const Functions = struct {
    stream: cuda.Function,
    writer: cuda.Function,
    final: cuda.Function,

    pub fn resolve(m: cuda.Module) !Functions {
        return .{ .stream = try m.function(stream_symbol), .writer = try m.function(writer_symbol), .final = try m.function(final_symbol) };
    }
};

pub const Ops = struct {
    f: *const Functions,
    s: cuda.Stream,

    fn go(o: Ops, f: cuda.Function, rows: usize, shared: u32, args: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = @intCast(NT) }, .shared = shared }, o.s, args);
    }

    /// The fold's edit on `a.R` rows of four copies, in place (the steered copies' tap beside, when given).
    pub fn stream(o: Ops, a: StreamArgs) !void {
        try checkStream(a);
        if (a.R == 0) return;
        var args: cuda.Args = .{};
        args.add(a);
        try o.go(o.f.stream, @intCast(a.R), 0, &args);
    }

    /// The writer's reduction and edit of the gathered partials, in place: rank 0's slot edited, the others -0.0.
    pub fn writer(o: Ops, a: WriterArgs) !void {
        try checkWriter(a);
        if (a.R == 0) return;
        var args: cuda.Args = .{};
        args.add(a);
        try o.go(o.f.writer, @intCast(a.R), 0, &args);
    }

    /// The head's collapse and norm of the steered streams.
    pub fn final(o: Ops, a: FinalArgs) !void {
        try checkFinal(a);
        if (a.R == 0) return;
        var args: cuda.Args = .{};
        args.add(a);
        try o.go(o.f.final, @intCast(a.R), @intCast(2 * @as(usize, @intCast(a.width))), &args);
    }
};

test "glp ops: a call's refusals (width, strides, ranks, alignment, strength)" {
    const ok: StreamArgs = .{ .x = 256, .xs = 20480, .dir = 4096, .alpha = 1, .R = 4, .width = 5120 };
    try checkStream(ok);
    var bad = ok;
    bad.width = 5121;
    try std.testing.expectError(error.BadWidth, checkStream(bad));
    bad.width = 5128; // a whole group of eight, past the served width
    try std.testing.expectError(error.BadWidth, checkStream(bad));
    bad = ok;
    bad.xs = 5120;
    try std.testing.expectError(error.BadStride, checkStream(bad));
    bad = ok;
    bad.xs = -20480;
    try std.testing.expectError(error.BadStride, checkStream(bad));
    bad = ok;
    bad.dir = 4100;
    try std.testing.expectError(error.Misaligned, checkStream(bad));
    bad = ok;
    bad.x = 264; // 8-byte aligned: the kernel's 16-byte vectors need more
    try std.testing.expectError(error.Misaligned, checkStream(bad));
    bad = ok;
    bad.alpha = 0;
    try checkStream(bad);
    bad.alpha = -0.0;
    try checkStream(bad);
    bad.alpha = -0.5;
    try std.testing.expectError(error.BadStrength, checkStream(bad));
    bad.alpha = std.math.nan(f32);
    try std.testing.expectError(error.BadStrength, checkStream(bad));
    bad.alpha = std.math.inf(f32);
    try std.testing.expectError(error.BadStrength, checkStream(bad));
    const w: WriterArgs = .{ .g = 256, .gr = 40960, .gs = 5120, .dir = 4096, .alpha = 0.5, .R = 8, .width = 5120, .world = 2 };
    try checkWriter(w);
    var wb = w;
    wb.world = 9;
    try std.testing.expectError(error.BadRanks, checkWriter(wb));
    wb = w;
    wb.world = 0;
    try std.testing.expectError(error.BadRanks, checkWriter(wb));
    wb = w;
    wb.alpha = std.math.nan(f32);
    try std.testing.expectError(error.BadStrength, checkWriter(wb));
    for ([_]i64{ 0, -40960, 5120, 40960 - 8 }) |gr| {
        wb = w;
        wb.gr = gr;
        try std.testing.expectError(error.BadStride, checkWriter(wb)); // rank slots overlap the rows, or run backwards
    }
    wb = w;
    wb.world = 1;
    wb.gr = 0;
    try checkWriter(wb); // one rank has no slot stride to use
    var tb = ok;
    tb.tap = 256 + 8 * 5120; // inside the copies' bytes
    tb.ts = 5120;
    try std.testing.expectError(error.Overlap, checkStream(tb));
    tb.tap = 256 + 2 * 4 * 5120 * 4; // the first byte past the copies
    try checkStream(tb);
    tb.tap = 240; // starts just before the copies and runs into them
    try std.testing.expectError(error.Overlap, checkStream(tb));
    tb = ok;
    tb.tap = 256 + 2 * 4 * 5120 * 4;
    tb.ts = 5124; // not a whole group of eight
    try std.testing.expectError(error.Misaligned, checkStream(tb));
    const f: FinalArgs = .{ .x = 256, .xs = 20480, .pre = 512, .nw = 1024, .out = 2048, .eps = 1e-6, .nwbf16 = 0, .R = 3, .width = 5120 };
    try checkFinal(f);
    var fb = f;
    fb.width = 5000;
    try std.testing.expectError(error.BadWidth, checkFinal(fb));
    fb = f;
    fb.pre = 516;
    try std.testing.expectError(error.Misaligned, checkFinal(fb));
}
