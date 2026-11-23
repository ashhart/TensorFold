//! A shared round's attention past 8 streams: the real attentionSegs on a driver that runs the kernels' row mapping.
const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const t = std.testing;

/// nemotron_attention.cu's AttnSegs, as attentionSegs passes it by value.
const Segs = extern struct { kc: [glue.max_segs]u64, vc: [glue.max_segs]u64, meta: [glue.max_segs]u64, pt: [glue.max_segs]u64, ps: i64, row0: [glue.max_segs]c_int, n: c_int };

const base_shape: glue.Attn = .{ .nqkv = 6144, .heads = 32, .kv_heads = 2, .dim = 128, .nch = 2 };
const qd: u64 = base_shape.heads * base_shape.dim;
const kvd: u64 = base_shape.kv_heads * base_shape.dim;
const page = 512; // paged_attn.cuh's PAGE

/// Device memory by address (a value a row), each meta row's position and each page table's first page.
const Device = struct {
    var mem: std.AutoHashMapUnmanaged(u64, u64) = .empty;
    var pos: std.AutoHashMapUnmanaged(u64, i64) = .empty;
    var pages: std.AutoHashMapUnmanaged(u64, u64) = .empty;
    var fns: glue.Fns = undefined;

    fn get(at: u64) u64 {
        return mem.get(at) orelse 0;
    }
    fn put(at: u64, v: u64) void {
        mem.put(t.allocator, at, v) catch @panic("oom");
    }
    fn arg(comptime T: type, args: [*]?*anyopaque, i: usize) T {
        return @as(*align(1) const T, @ptrCast(args[i].?)).*;
    }
    /// key_at: contiguous rows, or the page the stream's table names at `ps` elements a page.
    fn key(cache: u64, k: i64, pt: u64, ps: i64) u64 {
        const kk: u64 = @bitCast(k);
        if (pt == 0) return cache +% kk *% kvd *% 2;
        const chunk = pages.get(pt).? *% @as(u64, @bitCast(ps));
        return cache +% (chunk +% (kk % page) *% kvd) *% 2;
    }

    const Planes = struct { po: u64, pm: u64, pl: u64 };

    fn kvRow(qkv: u64, row: u64, kc: u64, vc: u64, at: i64, pt: u64, ps: i64) void {
        const v = get(qkv + row * base_shape.nqkv * 2);
        put(key(kc, at, pt, ps), v);
        put(key(vc, at, pt, ps), v +% 1);
    }
    /// A row's partial output, maxima and sums: its query against the stream's keys and values 0..at.
    fn attnRow(qkv: u64, row: u64, kc: u64, vc: u64, at: i64, pt: u64, ps: i64, p: Planes) void {
        var acc = get(qkv + row * base_shape.nqkv * 2) *% 31;
        var k: i64 = 0;
        while (k <= at) : (k += 1) acc = acc *% 1000003 +% get(key(kc, k, pt, ps)) *% 7 +% get(key(vc, k, pt, ps));
        put(p.po + row * base_shape.nch * qd * 4, acc);
        put(p.pm + row * base_shape.nch * base_shape.heads * 4, acc ^ 0x55);
        put(p.pl + row * base_shape.nch * base_shape.heads * 4, acc ^ 0xaa);
    }
    /// The merge: a row's output from its three planes, and the output's group sums.
    fn mergeRow(p: Planes, row: u64, out: u64, xs: u64) void {
        const o = get(p.po + row * base_shape.nch * qd * 4) *% 3 +% get(p.pm + row * base_shape.nch * base_shape.heads * 4) *% 5 +% get(p.pl + row * base_shape.nch * base_shape.heads * 4);
        put(out + row * qd * 2, o);
        put(xs + row * (qd / 64) * 4, o ^ 0x77);
    }

    fn segOf(s: *const Segs, rq: c_int) usize {
        var k: usize = 0;
        while (k + 1 < s.n and s.row0[k + 1] <= rq) k += 1;
        return k;
    }

    fn launch(f: cuda.abi.Function, gx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, raw: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        const a = raw.?;
        const h = f;
        for (0..gx) |rq| {
            if (h == fns.kv_write.handle) { // qkv kc vc meta, 3 ints, pt ps
                kvRow(arg(u64, a, 0), rq, arg(u64, a, 1), arg(u64, a, 2), pos.get(arg(u64, a, 3)).? + @as(i64, @intCast(rq)), arg(u64, a, 7), arg(i64, a, 8));
            } else if (h == fns.attn_chunk.handle or h == fns.attn_chunk_pf.handle) { // qkv kc vc meta po pm pl, 6 ints, scale, pt ps
                attnRow(arg(u64, a, 0), rq, arg(u64, a, 1), arg(u64, a, 2), pos.get(arg(u64, a, 3)).? + @as(i64, @intCast(rq)), arg(u64, a, 14), arg(i64, a, 15), .{ .po = arg(u64, a, 4), .pm = arg(u64, a, 5), .pl = arg(u64, a, 6) });
            } else if (h == fns.attn_merge.handle) { // po pm pl meta out xs
                mergeRow(.{ .po = arg(u64, a, 0), .pm = arg(u64, a, 1), .pl = arg(u64, a, 2) }, rq, arg(u64, a, 4), arg(u64, a, 5));
            } else {
                const s = arg(Segs, a, 0);
                const k = segOf(&s, @intCast(rq));
                const p = pos.get(s.meta[k]).? + @as(i64, @intCast(rq)) - s.row0[k];
                if (h == fns.kv_write_segs.handle) { // segs qkv
                    kvRow(arg(u64, a, 1), rq, s.kc[k], s.vc[k], p, s.pt[k], s.ps);
                } else if (h == fns.attn_chunk_segs.handle) { // segs qkv po pm pl
                    attnRow(arg(u64, a, 1), rq, s.kc[k], s.vc[k], p, s.pt[k], s.ps, .{ .po = arg(u64, a, 2), .pm = arg(u64, a, 3), .pl = arg(u64, a, 4) });
                } else if (h == fns.attn_merge_segs.handle) { // segs po pm pl out xs
                    mergeRow(.{ .po = arg(u64, a, 1), .pm = arg(u64, a, 2), .pl = arg(u64, a, 3) }, rq, arg(u64, a, 4), arg(u64, a, 5));
                } else return 1;
            }
        }
        return 0;
    }
};

const Buffers = struct { qkv: u64, po: u64, pm: u64, pl: u64, out: u64, xs: u64 };

/// A round's layout: rows a stream (1-2, or 2 each), a gap between streams, paged or contiguous caches.
const Layout = struct { full: bool = false, gap: usize = 0, paged: bool = false };

fn rowsOf(l: Layout, i: usize) usize {
    return if (l.full) 2 else 1 + i % 2;
}
fn cacheOf(i: usize) [4]u64 {
    return .{ 0x1000_0000 + i * 0x10_0000, 0x2000_0000 + i * 0x10_0000, 0x3000_0000 + i * 0x100, 0x3800_0000 + i * 0x100 };
}
fn posOf(i: usize) usize {
    return 5 + 3 * i;
}
fn psOf(l: Layout) i64 {
    return @intCast(shapeOf(l).ps);
}
fn shapeOf(l: Layout) glue.Attn {
    var s = base_shape;
    if (l.paged) s.ps = 1 << 20;
    return s;
}

/// A round of streams `first..first + n` from row 0 of `b`, `l`'s layout; `plant` moves one stream's first query.
fn round(g: glue.Glue, l: Layout, first: usize, n: usize, b: Buffers, parts: ?usize, plant: ?usize) !void {
    var segs: [16]glue.AttnSeg = undefined;
    var row: usize = 0;
    for (segs[0..n], first..) |*s, i| {
        const c = cacheOf(i);
        s.* = .{ .kc = c[0], .vc = c[1], .meta = c[2], .row0 = row, .rows = rowsOf(l, i), .pt = if (l.paged) c[3] else 0 };
        for (0..rowsOf(l, i)) |r| Device.put(b.qkv + (row + r) * base_shape.nqkv * 2, 100_000 + i * 100 + r + @intFromBool(plant == i and r == 0));
        row += rowsOf(l, i) + l.gap;
    }
    // whole, as the forward and the MTP head call it; or in parts of `parts`, as they called it before REV60 1
    const step = parts orelse n;
    var k0: usize = 0;
    while (k0 < n) : (k0 += step) try g.attentionSegs(segs[k0..@min(n, k0 + step)], b.qkv, b.po, b.pm, b.pl, b.out, b.xs, shapeOf(l));
}

/// Every stream's output and group-sum rows, the keys and values it wrote, and earlier ones overwritten.
const Seen = struct { out: [16][2][2]u64 = @splat(@splat(@splat(0))), keys: [16][2][2]u64 = @splat(@splat(@splat(0))), corrupt: usize = 0 };

fn fresh(l: Layout, n: usize) !void {
    Device.mem.clearRetainingCapacity();
    Device.pos.clearRetainingCapacity();
    Device.pages.clearRetainingCapacity();
    for (0..n) |i| {
        const c = cacheOf(i);
        try Device.pos.put(t.allocator, c[2], @intCast(posOf(i)));
        try Device.pages.put(t.allocator, c[3], 3 + 2 * i); // each stream's first page somewhere of its own
        for (0..posOf(i)) |k| { // the keys and values the stream already holds
            Device.put(Device.key(c[0], @intCast(k), if (l.paged) c[3] else 0, psOf(l)), 1000 * i + k);
            Device.put(Device.key(c[1], @intCast(k), if (l.paged) c[3] else 0, psOf(l)), 1000 * i + k + 7);
        }
    }
}

/// Each stream's output and group-sum row addresses in a round of streams from `first` at `b`.
fn rowsAt(l: Layout, at: *[16][2][2]u64, first: usize, n: usize, b: Buffers, gap: usize) void {
    var row: usize = 0;
    for (first..first + n) |i| {
        for (0..rowsOf(l, i)) |r| at[i][r] = .{ b.out + (row + r) * qd * 2, b.xs + (row + r) * (qd / 64) * 4 };
        row += rowsOf(l, i) + gap;
    }
}

fn seen(l: Layout, n: usize, at: *const [16][2][2]u64) Seen {
    var s: Seen = .{};
    const ps = psOf(l);
    for (0..n) |i| {
        const c = cacheOf(i);
        const pt = if (l.paged) c[3] else 0;
        const p: i64 = @intCast(posOf(i));
        for (0..rowsOf(l, i)) |r| {
            s.out[i][r] = .{ Device.get(at[i][r][0]), Device.get(at[i][r][1]) };
            s.keys[i][r] = .{ Device.get(Device.key(c[0], p + @as(i64, @intCast(r)), pt, ps)), Device.get(Device.key(c[1], p + @as(i64, @intCast(r)), pt, ps)) };
        }
        for (0..posOf(i)) |k| { // the keys and values the stream held before the round
            if (Device.get(Device.key(c[0], @intCast(k), pt, ps)) != 1000 * i + k) s.corrupt += 1;
            if (Device.get(Device.key(c[1], @intCast(k), pt, ps)) != 1000 * i + k + 7) s.corrupt += 1;
        }
    }
    return s;
}

const wide: Buffers = .{ .qkv = 0x5000_0000, .po = 0x6000_0000, .pm = 0x6800_0000, .pl = 0x6900_0000, .out = 0x7000_0000, .xs = 0x7800_0000 };
const narrow = [2]Buffers{
    .{ .qkv = 0x8000_0000, .po = 0x9000_0000, .pm = 0x9800_0000, .pl = 0x9900_0000, .out = 0xa000_0000, .xs = 0xa800_0000 },
    .{ .qkv = 0xb000_0000, .po = 0xc000_0000, .pm = 0xc800_0000, .pl = 0xc900_0000, .out = 0xd000_0000, .xs = 0xd800_0000 },
};

/// The reference: streams 0..7 in one round, the rest in another, each from row 0 of its own buffers.
fn reference(g: glue.Glue, l: Layout, n: usize) !Seen {
    try fresh(l, n);
    try round(g, l, 0, 8, narrow[0], null, null);
    try round(g, l, 8, n - 8, narrow[1], null, null);
    var at: [16][2][2]u64 = undefined;
    rowsAt(l, &at, 0, 8, narrow[0], l.gap);
    rowsAt(l, &at, 8, n - 8, narrow[1], l.gap);
    return seen(l, n, &at);
}

fn wideRound(g: glue.Glue, l: Layout, n: usize, parts: ?usize, plant: ?usize) !Seen {
    try fresh(l, n);
    try round(g, l, 0, n, wide, parts, plant);
    var at: [16][2][2]u64 = undefined;
    rowsAt(l, &at, 0, n, wide, l.gap);
    return seen(l, n, &at);
}

fn expectSame(l: Layout, n: usize, want: Seen, got: Seen) !void {
    try t.expectEqual(@as(usize, 0), got.corrupt); // no stream's earlier keys or values overwritten
    for (0..n) |i| for (0..rowsOf(l, i)) |r| {
        try t.expectEqual(want.out[i][r], got.out[i][r]); // output and its group sums
        try t.expectEqual(want.keys[i][r], got.keys[i][r]); // the new key and value
    };
}

fn rig(d: *cuda.Driver) glue.Glue {
    d.api.cuLaunchKernel = Device.launch;
    inline for (@typeInfo(glue.Fns).@"struct".field_names, 0..) |name, i| @field(Device.fns, name) = .{ .d = d, .handle = @ptrFromInt(0x100 + i) };
    return .{ .set = null, .f = &Device.fns, .s = .{ .d = d, .handle = null }, .emb = undefined, .norms = undefined, .moe = undefined };
}

fn deinit() void {
    Device.mem.deinit(t.allocator);
    Device.pos.deinit(t.allocator);
    Device.pages.deinit(t.allocator);
    Device.mem, Device.pos, Device.pages = .{ .empty, .empty, .empty }; // the next test starts afresh
}

test "wide round walk: 9 to 16 streams' attention reads and writes each stream's own rows and cache, as 8-stream rounds do" {
    defer deinit();
    var d: cuda.Driver = undefined;
    const g = rig(&d);
    // 1-2 rows a stream or 2 each (32 rows at 16 streams); contiguous caches or paged; streams' rows apart or not
    const layouts = [_]Layout{ .{}, .{ .full = true }, .{ .paged = true }, .{ .full = true, .paged = true }, .{ .gap = 1 } };
    for (layouts) |l| for (9..17) |n| {
        const want = try reference(g, l, n);
        try t.expectEqual(@as(usize, 0), want.corrupt);
        // whole, as the callers pass it now, and in the old callers' parts of 8
        for ([_]?usize{ null, glue.max_segs }) |parts| try expectSame(l, n, want, try wideRound(g, l, n, parts, null));
    };
}

test "wide round walk: the attention comparison catches one stream's moved query (a planted control)" {
    defer deinit();
    var d: cuda.Driver = undefined;
    const g = rig(&d);
    const l: Layout = .{};
    const want = try reference(g, l, 12);
    const got = try wideRound(g, l, 12, null, 10);
    try t.expectEqual(@as(usize, 0), got.corrupt);
    for (0..12) |i| {
        const same = std.meta.eql(want.out[i], got.out[i]) and std.meta.eql(want.keys[i], got.keys[i]);
        try t.expectEqual(i != 10, same); // stream 10 alone differs: its output, sums and new key and value
    }
}
