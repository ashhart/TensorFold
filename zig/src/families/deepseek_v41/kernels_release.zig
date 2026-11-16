//! Launches of the release-format kernels (release_proj.cu, release_glue.cu), sm_121a only.
const std = @import("std");
const cuda = @import("cuda");
pub const split = @import("release_split.zig").split;

/// Mirrors RelActq: x bf16 rows at byte offset x_off and row stride x_ld (elements); rt optional (0: none).
pub const Actq = extern struct {
    x: u64,
    x_off: u64 = 0,
    x_ld: i64,
    codes: u64,
    scales: u64,
    rt: u64 = 0,
    m: c_int,
    k: c_int,
};

/// Mirrors RelRms: Actq after the reference RMSNorm with f32 weight w; normed (optional) keeps the bf16 norm.
pub const Rms = extern struct {
    x: u64,
    x_off: u64 = 0,
    x_ld: i64,
    w: u64,
    codes: u64,
    scales: u64,
    rt: u64 = 0,
    normed: u64 = 0,
    eps: f32,
    m: c_int,
    k: c_int,
    pad: c_int = 0,
};

/// Mirrors RelProj. FP8: a codes and sa scales (views), w u8 [n,k], ws [n/32,k/32]. BF16: a bf16 x, w bf16 [n,k].
pub const Proj = extern struct {
    a: u64,
    a_off: u64 = 0,
    a_ld: i64,
    sa: u64 = 0,
    sa_off: u64 = 0,
    sa_ld: i64 = 0,
    w: u64,
    ws: u64 = 0,
    y: u64,
    y_ld: i64,
    y_col: i64 = 0,
    part: u64 = 0,
    m: c_int,
    n: c_int,
    k: c_int,
    s: c_int = 1,
    f32: c_int = 0,
    pad: c_int = 0,
};

/// Mirrors RelSum: partials [s,m,n] FP32 into y.
pub const Sum = extern struct {
    part: u64,
    y: u64,
    y_ld: i64,
    y_col: i64 = 0,
    m: c_int,
    n: c_int,
    s: c_int,
    f32: c_int = 0,
};

comptime {
    std.debug.assert(@sizeOf(Actq) == 56 and @sizeOf(Rms) == 80 and @sizeOf(Proj) == 120 and @sizeOf(Sum) == 48);
}

/// A staged tile's geometry (release_proj.cu's Cfg): rows and columns a CTA, threads, stages, dynamic shared bytes.
pub const Tile = struct { rows: u32, cols: u32, threads: u32, smem: u32 };
pub const Form = enum { fp8, bf16 };

/// The tile a launch of m rows takes: decode (to 16 rows), mid (to 64) or prompt; the per-element steps are the same.
pub const Size = enum { decode, mid, prompt };

pub fn sizeOf(m: usize) Size {
    return if (m <= 16) .decode else if (m <= 64) .mid else .prompt;
}

pub fn tile(form: Form, size: Size) Tile {
    const g: [5]u32 = switch (size) {
        .decode => .{ 16, 64, 64, 4, 128 },
        .mid => .{ 64, 64, 128, 4, 64 },
        .prompt => .{ 128, 256, 512, 4, 64 },
    };
    const rows = g[0];
    const cols = g[1];
    const steps = g[4] / 32;
    const scales: u32 = if (form == .fp8) (rows * steps + cols / 32 * steps + 127) & ~@as(u32, 127) else 0;
    return .{ .rows = rows, .cols = cols, .threads = g[2], .smem = g[3] * ((rows + cols) * g[4] + scales) };
}

pub const glue_threads = 256;
pub const warp_threads = 128;
/// The most CTAs a grid-stride glue launch takes.
const most_ctas = 4096;

pub const Functions = struct {
    actq: cuda.Function,
    rms_actq: cuda.Function,
    /// [form][decode, mid, prompt]
    staged: [2][3]cuda.Function,
    /// [form]: the stage-free warp tiles (a persistent kernel's form, checked bit-equal)
    warp: [2]cuda.Function,
    sum: cuda.Function,

    /// `proj`: the dsv41_rel image, `glue`: dsv41_rel_glue.
    pub fn resolve(proj: cuda.Module, glue: cuda.Module) !Functions {
        var f: Functions = .{
            .actq = try glue.function("tf_dsv41_rel_actq"),
            .rms_actq = try glue.function("tf_dsv41_rel_rms_actq"),
            .staged = .{
                .{ try proj.function("tf_dsv41_rel_fp8"), try proj.function("tf_dsv41_rel_fp8_mid"), try proj.function("tf_dsv41_rel_fp8_prompt") },
                .{ try proj.function("tf_dsv41_rel_bf16"), try proj.function("tf_dsv41_rel_bf16_mid"), try proj.function("tf_dsv41_rel_bf16_prompt") },
            },
            .warp = .{ try proj.function("tf_dsv41_rel_fp8_warp"), try proj.function("tf_dsv41_rel_bf16_warp") },
            .sum = try proj.function("tf_dsv41_rel_sum"),
        };
        for ([_]Form{ .fp8, .bf16 }, &f.staged) |form, *row| for (row, [_]Size{ .decode, .mid, .prompt }) |fun, size| {
            const t = tile(form, size);
            if (t.smem > 48 * 1024) try fun.allowDynamicShared(t.smem);
        };
        return f;
    }
};

/// Bytes of FP32 partials a projection of m rows needs (0 when it does not split).
pub fn partBytes(m: usize, n: usize, k: usize) usize {
    const s = split(k, n);
    return if (s == 1) 0 else @as(usize, s) * m * n * 4;
}

fn aligned(ptr: u64, off: u64, unit: u64) bool {
    return (ptr + off) % unit == 0;
}

/// The launch's shape checks: whole 32-wide groups and columns, views wide enough, 16-byte rows for the stages.
pub fn checkProj(p: Proj, form: Form) !void {
    if (p.m <= 0 or p.n <= 0 or p.k <= 0 or @rem(p.n, 32) != 0 or @rem(p.k, 32) != 0) return error.Shape;
    const k: i64 = p.k;
    if (p.a_ld < k or @rem(p.a_ld, @as(i64, if (form == .fp8) 16 else 8)) != 0 or !aligned(p.a, p.a_off, 16)) return error.Shape;
    if (p.w == 0 or p.w % 16 != 0 or p.y == 0 or p.y_col < 0 or p.y_ld < p.y_col + p.n) return error.Shape;
    if (p.y % @as(u64, if (p.f32 != 0) 4 else 2) != 0) return error.Shape;
    if (form == .fp8 and (p.sa == 0 or p.ws == 0 or p.sa_ld < @divExact(k, 32))) return error.Shape;
    const s = p.s;
    if (!(s == 1 or s == 2 or s == 4 or s == 8) or @rem(@divExact(p.k, 32), s) != 0) return error.Shape;
    if (s > 1 and (p.part == 0 or p.part % 8 != 0)) return error.Shape;
}

fn ctas(n: u64, per: u64) u32 {
    return @intCast(@min((n + per - 1) / per, most_ctas));
}

pub const Ops = struct {
    f: *const Functions,
    s: cuda.Stream,

    pub fn actq(o: Ops, a: Actq) !void {
        if (a.m <= 0 or a.k <= 0 or @rem(a.k, 32) != 0 or a.x_ld < a.k or a.codes == 0 or a.scales == 0) return error.Shape;
        const blocks = @as(u64, @intCast(a.m)) * @as(u64, @intCast(@divExact(a.k, 32)));
        var args: cuda.Args = .{};
        args.add(a);
        try cuda.launch.launch(o.f.actq, .{ .grid = .{ .x = ctas(blocks, 8) }, .block = .{ .x = glue_threads } }, o.s, &args);
    }

    pub fn rmsActq(o: Ops, a: Rms) !void {
        if (a.m <= 0 or a.k <= 0 or @rem(a.k, 32) != 0 or a.x_ld < a.k or a.w == 0 or a.codes == 0 or a.scales == 0) return error.Shape;
        var args: cuda.Args = .{};
        args.add(a);
        try cuda.launch.launch(o.f.rms_actq, .{ .grid = .{ .x = ctas(@intCast(a.m), 1) }, .block = .{ .x = glue_threads } }, o.s, &args);
    }

    /// The FP8 projection at its served split (`partBytes` of scratch at p.part when it splits).
    pub fn fp8(o: Ops, p: Proj) !void {
        return o.project(served(p), .fp8, null);
    }

    pub fn bf16(o: Ops, p: Proj) !void {
        return o.project(served(p), .bf16, null);
    }

    fn served(p: Proj) Proj {
        var q = p;
        q.s = @intCast(split(@intCast(p.k), @intCast(p.n)));
        return q;
    }

    /// A projection at the slice count p.s; `grid` (tests) launches fewer CTAs that loop over every tile.
    pub fn project(o: Ops, p: Proj, form: Form, grid: ?u32) !void {
        try checkProj(p, form);
        const m: u32 = @intCast(p.m);
        const size = sizeOf(m);
        const t = tile(form, size);
        const tiles = ((m + t.rows - 1) / t.rows) * ((@as(u32, @intCast(p.n)) + t.cols - 1) / t.cols) * @as(u32, @intCast(p.s));
        var args: cuda.Args = .{};
        args.add(p);
        const fun = o.f.staged[@backingInt(form)][@backingInt(size)];
        try cuda.launch.launch(fun, .{ .grid = .{ .x = grid orelse tiles }, .block = .{ .x = t.threads }, .shared = t.smem }, o.s, &args);
        return o.finish(p);
    }

    /// The stage-free warp tiles (16 x 32 a warp from global memory) on `grid` CTAs of four warps.
    pub fn projectWarps(o: Ops, p: Proj, form: Form, grid: u32) !void {
        try checkProj(p, form);
        var args: cuda.Args = .{};
        args.add(p);
        try cuda.launch.launch(o.f.warp[@backingInt(form)], .{ .grid = .{ .x = grid }, .block = .{ .x = warp_threads } }, o.s, &args);
        return o.finish(p);
    }

    fn finish(o: Ops, p: Proj) !void {
        if (p.s == 1) return;
        try o.sum(.{ .part = p.part, .y = p.y, .y_ld = p.y_ld, .y_col = p.y_col, .m = p.m, .n = p.n, .s = p.s, .f32 = p.f32 });
    }

    pub fn sum(o: Ops, a: Sum) !void {
        if (a.m <= 0 or a.n <= 0 or !(a.s == 2 or a.s == 4 or a.s == 8) or a.part == 0 or a.y_ld < a.y_col + a.n) return error.Shape;
        const total = @as(u64, @intCast(a.m)) * @as(u64, @intCast(a.n));
        var args: cuda.Args = .{};
        args.add(a);
        try cuda.launch.launch(o.f.sum, .{ .grid = .{ .x = ctas(total, glue_threads) }, .block = .{ .x = glue_threads } }, o.s, &args);
    }
};

test "release launches refuse ragged groups, narrow views, odd splits and missing scratch" {
    const ok: Proj = .{ .a = 64, .a_ld = 5120, .sa = 8, .sa_ld = 160, .w = 64, .ws = 8, .y = 64, .y_ld = 1280, .m = 3, .n = 1280, .k = 5120 };
    try checkProj(ok, .fp8);
    var p = ok;
    p.k = 5100;
    try std.testing.expectError(error.Shape, checkProj(p, .fp8));
    p = ok;
    p.a_ld = 4096;
    try std.testing.expectError(error.Shape, checkProj(p, .fp8));
    p = ok;
    p.a_off = 8;
    try std.testing.expectError(error.Shape, checkProj(p, .fp8));
    p = ok;
    p.s = 3;
    try std.testing.expectError(error.Shape, checkProj(p, .fp8));
    p = ok;
    p.s = 8;
    try std.testing.expectError(error.Shape, checkProj(p, .fp8));
    p.part = 64;
    try checkProj(p, .fp8);
    p = ok;
    p.y_col = 8;
    try std.testing.expectError(error.Shape, checkProj(p, .fp8));
    p = ok;
    p.sa_ld = 100;
    try std.testing.expectError(error.Shape, checkProj(p, .fp8));
    p = ok;
    p.y = 65;
    try std.testing.expectError(error.Shape, checkProj(p, .fp8));
    p = ok;
    p.y = 66;
    try checkProj(p, .fp8);
    p.f32 = 1;
    try std.testing.expectError(error.Shape, checkProj(p, .fp8));
    p.y = 68;
    try checkProj(p, .fp8);
    try checkProj(.{ .a = 64, .a_ld = 520, .w = 64, .y = 64, .y_ld = 128, .m = 1, .n = 128, .k = 512 }, .bf16);
    try std.testing.expectEqual(@as(u32, 41472), tile(.fp8, .decode).smem);
    try std.testing.expectEqual(@as(u32, 99840), tile(.fp8, .prompt).smem);
    try std.testing.expectEqual(@as(u32, 98304), tile(.bf16, .prompt).smem);
    for ([_]Size{ .decode, .mid, .prompt }) |size| try std.testing.expect(tile(.fp8, size).smem <= 101376);
    try std.testing.expectEqual(Size.mid, sizeOf(17));
    try std.testing.expectEqual(Size.prompt, sizeOf(65));
    try std.testing.expectEqual(@as(usize, 4 * 2048 * 1280 * 4), partBytes(2048, 1280, 5120));
    try std.testing.expectEqual(@as(usize, 0), partBytes(2048, 5120, 576));
}
