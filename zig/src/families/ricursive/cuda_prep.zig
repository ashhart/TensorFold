//! Ripple's attention prep on CUDA (ricursive_prep.cu): glue.metal's rc_prep, q dense and k, v into a ring or pages.

const std = @import("std");
const cuda = @import("cuda");

const Stream = cuda.Stream;
const Function = cuda.Function;
/// A stream's pages as window_attention's paged prompt reads them; `stride` (a slot's table entries) is prep_rows'.
pub const Pages = cuda.window_attention.Pages;

const head_dim = 128;

/// A warp a (head, row), 8 warps a block: tf_rc_prep's and tf_rc_prep_rows' __launch_bounds__.
const warps = 8;
const threads = 32 * warps;

fn int(x: usize) c_int {
    return @intCast(x);
}

fn ceil(n: usize, d: usize) u32 {
    return @intCast((n + d - 1) / d);
}

pub const Kernels = struct {
    mod: cuda.Module,
    prep: Function,
    prep_rows: Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.ricursive_prep);
        errdefer mod.unload();
        return .{ .mod = mod, .prep = try mod.function("tf_rc_prep"), .prep_rows = try mod.function("tf_rc_prep_rows") };
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
        k.* = undefined;
    }
};

/// One projection's bf16 output rows (ricursive_prep.cuh Rows): the first row and the elements between rows.
pub const Rows = extern struct {
    x: u64,
    stride: i32,
    pad: i32 = 0,

    pub fn of(x: u64, stride: usize) Rows {
        return .{ .x = x, .stride = int(stride) };
    }
};

/// `rows` rows at pos + row: q, and k and v at ring slot (pos + row) % ring or in pages; `rope` false: NoPE, normed.
pub const Prep = struct { xq: Rows, xk: Rows, xv: Rows, inv: u64, qw: u64, kw: u64, q: u64, kc: u64, vc: u64, rows: usize, heads: usize, kv_heads: usize, pos: usize, ring: usize, rope: bool, eps: f32, pg: ?Pages = null };

/// The writes the kernel takes: a ring at least the call's rows, or pages each holding `tokens` whole rows.
pub fn admit(p: Prep) !void {
    if (p.pos + p.rows > std.math.maxInt(i32)) return error.PrepShape;
    if (p.pg) |pg| {
        if (pg.tokens == 0 or pg.tokens > std.math.maxInt(i32) or pg.page_elems < pg.tokens * p.kv_heads * head_dim or pg.page_elems > std.math.maxInt(i63)) return error.PageShape;
    } else if (p.rows > p.ring) return error.RingTooSmall; // a call's rows would overwrite their own slots
}

/// A block of 8 heads across, a row down.
pub fn grid(p: Prep) cuda.Dim3 {
    return .{ .x = ceil(p.heads + p.kv_heads, warps), .y = @intCast(p.rows), .z = 1 };
}

pub fn prep(k: *const Kernels, s: Stream, p: Prep) !void {
    return prepVia(k.prep, null, s, p);
}

/// The launch on `f`: the served tf_rc_prep, or (keep not null) contract_controls' tf_ctl_rc_prep keeping those points.
pub fn prepVia(f: Function, keep: ?i32, s: Stream, p: Prep) !void {
    if (p.rows == 0) return;
    try admit(p);
    var a: cuda.Args = .{};
    for ([_]Rows{ p.xq, p.xk, p.xv }) |r| a.add(r);
    for ([_]u64{ p.inv, p.qw, p.kw, p.q, p.kc, p.vc }) |v| a.add(v);
    for ([_]usize{ p.rows, p.heads, p.kv_heads, p.pos, p.ring }) |v| a.add(int(v));
    a.add(@as(c_int, @intFromBool(p.rope)));
    a.add(p.eps);
    a.add(if (p.pg) |pg| pg.table else @as(u64, 0)); // null: the ring
    a.add(@as(i64, if (p.pg) |pg| @intCast(pg.page_elems) else 0));
    a.add(@as(c_int, if (p.pg) |pg| @intCast(pg.tokens) else 0));
    if (keep) |v| a.add(@as(c_int, v));
    try cuda.launch.launch(f, .{ .grid = grid(p), .block = .{ .x = threads }, .shared = 0 }, s, &a);
}

/// tf_rc_prep_rows: row r at pos[r] of stream slot[r] (device int32, replayable); rows a ring apart must not share.
pub const PrepRows = struct { xq: Rows, xk: Rows, xv: Rows, inv: u64, qw: u64, kw: u64, q: u64, kc: u64, vc: u64, rows: usize, heads: usize, kv_heads: usize, pos: u64, slot: ?u64 = null, ring: usize, rope: bool, eps: f32, pg: ?Pages = null };

/// The writes tf_rc_prep_rows takes: a ring (positions on the device, unchecked), or pages of `tokens` whole rows.
pub fn admitRows(p: PrepRows) !void {
    if (p.pg) |pg| {
        if (pg.tokens == 0 or pg.tokens > std.math.maxInt(i32) or pg.page_elems < pg.tokens * p.kv_heads * head_dim or pg.page_elems > std.math.maxInt(i63) or pg.stride > std.math.maxInt(i32)) return error.PageShape;
    } else if (p.ring == 0 or p.ring > std.math.maxInt(i32)) return error.RingTooSmall;
}

/// tf_rc_prep's grid: a block of 8 heads across, a row down.
pub fn gridRows(p: PrepRows) cuda.Dim3 {
    return .{ .x = ceil(p.heads + p.kv_heads, warps), .y = @intCast(p.rows), .z = 1 };
}

pub fn prepRows(k: *const Kernels, s: Stream, p: PrepRows) !void {
    return prepRowsVia(k.prep_rows, null, s, p);
}

/// The launch on `f`: the served tf_rc_prep_rows, or (keep not null) contract_controls' tf_ctl_rc_prep_rows.
pub fn prepRowsVia(f: Function, keep: ?i32, s: Stream, p: PrepRows) !void {
    if (p.rows == 0) return;
    try admitRows(p);
    var a: cuda.Args = .{};
    for ([_]Rows{ p.xq, p.xk, p.xv }) |r| a.add(r);
    for ([_]u64{ p.inv, p.qw, p.kw, p.q, p.kc, p.vc }) |v| a.add(v);
    for ([_]usize{ p.rows, p.heads, p.kv_heads }) |v| a.add(int(v));
    a.add(p.pos);
    a.add(p.slot orelse @as(u64, 0)); // null: stream 0
    a.add(int(p.ring));
    a.add(@as(c_int, @intFromBool(p.rope)));
    a.add(p.eps);
    a.add(if (p.pg) |pg| pg.table else @as(u64, 0)); // null: the rings
    a.add(@as(i64, if (p.pg) |pg| @intCast(pg.page_elems) else 0));
    a.add(@as(c_int, if (p.pg) |pg| @intCast(pg.tokens) else 0));
    a.add(@as(c_int, if (p.pg) |pg| @intCast(pg.stride) else 0));
    if (keep) |v| a.add(@as(c_int, v));
    try cuda.launch.launch(f, .{ .grid = gridRows(p), .block = .{ .x = threads }, .shared = 0 }, s, &a);
}

test "launch geometry: 88 heads in 11 blocks a row, Rows as the kernel's struct" {
    const p: Prep = .{ .xq = Rows.of(0, 0), .xk = Rows.of(0, 0), .xv = Rows.of(0, 0), .inv = 0, .qw = 0, .kw = 0, .q = 0, .kc = 0, .vc = 0, .rows = 2048, .heads = 80, .kv_heads = 8, .pos = 0, .ring = 3072, .rope = true, .eps = 1e-6 };
    const g = grid(p);
    try std.testing.expectEqual(@as(c_uint, 11), g.x);
    try std.testing.expectEqual(@as(c_uint, 2048), g.y);
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(Rows));
}

test "admit: a ring holds the call's rows; pages hold 512 whole rows of 8 heads" {
    const t = std.testing;
    var p: Prep = .{ .xq = Rows.of(0, 0), .xk = Rows.of(0, 0), .xv = Rows.of(0, 0), .inv = 0, .qw = 0, .kw = 0, .q = 0, .kc = 0, .vc = 0, .rows = 2048, .heads = 80, .kv_heads = 8, .pos = 6000, .ring = 3072, .rope = true, .eps = 1e-6 };
    try admit(p);
    p.ring = 2047;
    try t.expectError(error.RingTooSmall, admit(p));
    p.pg = .{ .table = 0, .page_elems = 512 * 8 * 128, .tokens = 512 };
    try admit(p); // pages: no ring
    p.pg.?.page_elems -= 1;
    try t.expectError(error.PageShape, admit(p));
    p.pg = .{ .table = 0, .page_elems = 0, .tokens = 0 };
    try t.expectError(error.PageShape, admit(p));
}

test "rows: tf_rc_prep's grid; a ring of some rows, or pages of whole rows and an int32 table stride" {
    const t = std.testing;
    var p: PrepRows = .{ .xq = Rows.of(0, 0), .xk = Rows.of(0, 0), .xv = Rows.of(0, 0), .inv = 0, .qw = 0, .kw = 0, .q = 0, .kc = 0, .vc = 0, .rows = 64, .heads = 80, .kv_heads = 8, .pos = 0, .ring = 3072, .rope = true, .eps = 1e-6 };
    const g = gridRows(p);
    try t.expectEqual(@as(c_uint, 11), g.x);
    try t.expectEqual(@as(c_uint, 64), g.y);
    try admitRows(p);
    p.rows = 4096; // several streams' rows: more than one ring holds, each in its own
    try admitRows(p);
    p.ring = 0;
    try t.expectError(error.RingTooSmall, admitRows(p));
    p.pg = .{ .table = 0, .page_elems = 512 * 8 * 128, .tokens = 512, .stride = 256 };
    try admitRows(p); // pages: no ring
    p.pg.?.stride = 1 << 31;
    try t.expectError(error.PageShape, admitRows(p));
    p.pg = .{ .table = 0, .page_elems = 512 * 8 * 128 - 1, .tokens = 512, .stride = 256 };
    try t.expectError(error.PageShape, admitRows(p));
}
