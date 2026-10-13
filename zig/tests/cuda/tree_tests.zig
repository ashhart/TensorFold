//! tree_attention.cu against core/attention_ref.zig, and one stream's rows alone giving the same bits.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");

const tree = cuda.tree_attention;

fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn bf(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}

const Set = struct {
    gpu: check.Gpu,
    a: std.mem.Allocator,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,

    fn up(x: *Set, bytes: []const u8) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(x.gpu.d, bytes);
        try x.bufs.append(x.gpu.gpa, b);
        return b.ptr;
    }

    fn zero(x: *Set, s: cuda.Stream, n: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(x.gpu.d, @max(n, 16));
        try x.bufs.append(x.gpu.gpa, b);
        try b.fill8(0xff, s.handle);
        return b.ptr;
    }

    fn down(x: *Set, comptime T: type, ptr: u64, n: usize) ![]T {
        const o = try x.a.alloc(T, n);
        const b: cuda.DeviceBuffer = .{ .d = x.gpu.d, .ptr = ptr, .len = n * @sizeOf(T) };
        try b.download(0, std.mem.sliceAsBytes(o));
        return o;
    }
};

/// One call over `rows` (global window rows of the given streams, in order): their outputs.
fn call(x: *Set, s: cuda.Stream, g: *const tree.TreeAttention, d: usize, rows: []const usize, row_stream_all: []const i32, parents_all: []const i32, qs: []const u16, kns: []const u16, vns: []const u16, h: usize, hk: usize, streams_all: []const i32, offs: u64, origin: u64, ng: usize, scale: f32) ![]u16 {
    const n = rows.len;
    var local: std.AutoHashMap(usize, i32) = .init(x.a);
    for (rows, 0..) |r, i| try local.put(r, @intCast(i));
    const rs = try x.a.alloc(i32, n);
    const par = try x.a.alloc(i32, n);
    const q = try x.a.alloc(u16, n * h * d);
    const kn = try x.a.alloc(u16, n * hk * d);
    const vn = try x.a.alloc(u16, n * hk * d);
    for (rows, 0..) |r, i| {
        rs[i] = row_stream_all[r];
        par[i] = if (parents_all[r] < 0) -1 else local.get(@intCast(parents_all[r])).?;
        @memcpy(q[i * h * d ..][0 .. h * d], qs[r * h * d ..][0 .. h * d]);
        @memcpy(kn[i * hk * d ..][0 .. hk * d], kns[r * hk * d ..][0 .. hk * d]);
        @memcpy(vn[i * hk * d ..][0 .. hk * d], vns[r * hk * d ..][0 .. hk * d]);
    }
    // the streams' table for these rows: each stream's first row and rows among them, its committed keys
    const st = try x.a.dupe(i32, streams_all);
    for (0..st.len / 4) |k| {
        st[4 * k] = 0;
        st[4 * k + 1] = 0;
    }
    for (rows, 0..) |r, i| {
        const k: usize = @intCast(row_stream_all[r]);
        if (st[4 * k + 1] == 0) st[4 * k] = @intCast(i);
        st[4 * k + 1] += 1;
    }
    const streams_dev = try x.up(std.mem.sliceAsBytes(st));
    const paths = try x.zero(s, n * tree.max_depth * 4);
    const depths = try x.zero(s, n * 4);
    try g.paths(s, try x.up(std.mem.sliceAsBytes(par)), paths, depths, n);
    const sc_base = try x.zero(s, tree.TreeAttention.scratchBytes(n, hk, ng, d));
    const out = try x.zero(s, n * h * d * 2);
    const rws: tree.Rows = .{ .row_stream = try x.up(std.mem.sliceAsBytes(rs)), .streams = streams_dev, .offs = offs, .origin = origin, .paths = paths, .depths = depths };
    try g.run(s, try x.up(std.mem.sliceAsBytes(q)), try x.up(std.mem.sliceAsBytes(kn)), try x.up(std.mem.sliceAsBytes(vn)), rws, tree.TreeAttention.scratch(sc_base, n, hk, ng, d), out, n, h, hk, d, ng, scale);
    try s.synchronize();
    return x.down(u16, out, n * h * d);
}

fn case(gpu: check.Gpu, d: usize, h: usize, hk: usize, seed: u64, worst_all: *f64) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var g = try tree.TreeAttention.load(gpu.d);
    defer g.unload();
    var x: Set = .{ .gpu = gpu, .a = a };
    defer {
        for (x.bufs.items) |*b| b.free();
        x.bufs.deinit(gpu.gpa);
    }
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const committed = [_]usize{ 5, 1500, 4200 };
    const widths = [_]usize{ 9, 33, 17 };
    // the caches: stream s's rows at element offset cache_at[s] of one buffer (keys, then values)
    var cache_rows: usize = 0;
    for (committed) |p| cache_rows += p;
    const kcache = try a.alloc(u16, 2 * cache_rows * hk * d);
    for (kcache) |*v| v.* = bf16(rng.floatNorm(f32));
    var offs: [2 * committed.len]i64 = undefined;
    var at: usize = 0;
    for (committed, 0..) |p, si| {
        offs[2 * si] = @intCast(at * hk * d);
        offs[2 * si + 1] = @intCast((cache_rows + at) * hk * d);
        at += p;
    }
    var total: usize = 0;
    for (widths) |w| total += w;
    const row_stream = try a.alloc(i32, total);
    const parents = try a.alloc(i32, total);
    var streams: [4 * committed.len]i32 = undefined;
    var start: usize = 0;
    for (widths, committed, 0..) |w, p, si| {
        for (0..w) |j| {
            row_stream[start + j] = @intCast(si);
            parents[start + j] = if (j == 0 or rng.uintLessThan(u32, 6) == 0) -1 else @intCast(start + rng.uintLessThan(usize, j));
        }
        for ([_]usize{ start, w, p, 0 }, 0..) |v, k| streams[4 * si + k] = @intCast(v);
        start += w;
    }
    const qs = try a.alloc(u16, total * h * d);
    const kns = try a.alloc(u16, total * hk * d);
    const vns = try a.alloc(u16, total * hk * d);
    for (qs) |*v| v.* = bf16(rng.floatNorm(f32));
    for (kns) |*v| v.* = bf16(rng.floatNorm(f32));
    for (vns) |*v| v.* = bf16(rng.floatNorm(f32));
    const origin = try x.up(std.mem.sliceAsBytes(kcache));
    const offs_dev = try x.up(std.mem.sliceAsBytes(&offs));
    var ng: usize = 0;
    for (committed, widths) |p, w| ng = @max(ng, tree.TreeAttention.groups(p + w));
    const scale: f32 = @floatCast(1 / @sqrt(@as(f64, @floatFromInt(d))));
    const all = try a.alloc(usize, total);
    for (all, 0..) |*r, i| r.* = i;
    const got = try call(&x, s, &g, d, all, row_stream, parents, qs, kns, vns, h, hk, &streams, offs_dev, origin, ng, scale);
    // the reference: committed keys, then the path root first
    var worst: f64 = 0;
    const kd = try a.alloc(f64, (4200 + 128) * d);
    const vd = try a.alloc(f64, kd.len);
    const keys = try a.alloc(usize, 4200 + 128);
    const q = try a.alloc(f64, d);
    const want = try a.alloc(f64, d);
    const tol = try a.alloc(f64, d);
    var path: [tree.max_depth]usize = undefined;
    for (0..total) |r| {
        const si: usize = @intCast(row_stream[r]);
        const p = committed[si];
        var depth: usize = 0;
        var cur: i32 = @intCast(r);
        while (cur >= 0) : (cur = parents[@intCast(cur)]) {
            path[depth] = @intCast(cur);
            depth += 1;
        }
        std.mem.reverse(usize, path[0..depth]);
        for (0..hk) |kh| {
            for (0..p) |j| for (0..d) |e| {
                const ko: usize = @intCast(offs[2 * si]);
                const vo: usize = @intCast(offs[2 * si + 1]);
                kd[j * d + e] = bf(kcache[ko + (j * hk + kh) * d + e]);
                vd[j * d + e] = bf(kcache[vo + (j * hk + kh) * d + e]);
            };
            for (0..depth) |j| for (0..d) |e| {
                kd[(p + j) * d + e] = bf(kns[(path[j] * hk + kh) * d + e]);
                vd[(p + j) * d + e] = bf(vns[(path[j] * hk + kh) * d + e]);
            };
            for (0..p + depth) |j| keys[j] = j;
            for (0..h / hk) |hh| {
                const hd = kh * (h / hk) + hh;
                for (q, 0..) |*v, e| v.* = bf(qs[(r * h + hd) * d + e]);
                core.attention_ref.head(q, kd, vd, keys[0 .. p + depth], d, scale, want, tol);
                for (want, tol, 0..) |w, t, e| worst = @max(worst, @abs(bf(got[(r * h + hd) * d + e]) - w) / t);
            }
        }
    }
    try check.expect(worst <= 1, "tree attention dim {d}: within float64 tolerance (worst {d:.3})", .{ d, worst });
    worst_all.* = @max(worst_all.*, worst);
    // stream 1's rows alone (re-indexed): the same bits
    const only = try a.alloc(usize, widths[1]);
    for (only, 0..) |*r, i| r.* = widths[0] + i;
    const alone = try call(&x, s, &g, d, only, row_stream, parents, qs, kns, vns, h, hk, &streams, offs_dev, origin, ng, scale);
    try check.sameBytes("tree attention: one stream's rows alone", std.mem.sliceAsBytes(alone), std.mem.sliceAsBytes(got[widths[0] * h * d ..][0 .. widths[1] * h * d]));
    // one shared block walking every (row, group), and a grid sized for a long context: the same bits
    g.shared_blocks = 1;
    const strided = try call(&x, s, &g, d, all, row_stream, parents, qs, kns, vns, h, hk, &streams, offs_dev, origin, ng, scale);
    try check.sameBytes("tree attention: one shared block", std.mem.sliceAsBytes(strided), std.mem.sliceAsBytes(got));
    g.shared_blocks = 96;
    const wide = try call(&x, s, &g, d, all, row_stream, parents, qs, kns, vns, h, hk, &streams, offs_dev, origin, 129, scale);
    try check.sameBytes("tree attention: groups for a 262k context", std.mem.sliceAsBytes(wide), std.mem.sliceAsBytes(got));
    // one warp a block (a device with SMs to spare): stream 2's rows alone, its full groups too, the batch's bits
    g.sms = 1 << 20;
    defer g.sms = 0;
    const third = try a.alloc(usize, widths[2]);
    for (third, 0..) |*r, i| r.* = widths[0] + widths[1] + i;
    try check.expect(widths[2] * ng * hk <= tree.slice_items, "tree attention: the one-warp case fits the slices", .{});
    const one = try call(&x, s, &g, d, third, row_stream, parents, qs, kns, vns, h, hk, &streams, offs_dev, origin, ng, scale);
    try check.sameBytes("tree attention: one warp a block", std.mem.sliceAsBytes(one), std.mem.sliceAsBytes(got[(widths[0] + widths[1]) * h * d ..][0 .. widths[2] * h * d]));
    g.shared_blocks = 1;
    const one_strided = try call(&x, s, &g, d, third, row_stream, parents, qs, kns, vns, h, hk, &streams, offs_dev, origin, ng, scale);
    try check.sameBytes("tree attention: one warp a block, one shared block", std.mem.sliceAsBytes(one_strided), std.mem.sliceAsBytes(one));
    g.shared_blocks = 96;
}

pub fn run(gpu: check.Gpu) !void {
    var worst: f64 = 0;
    try case(gpu, 128, 48, 4, 41, &worst); // Kolibri
    try case(gpu, 256, 24, 4, 42, &worst); // Qwen3.8-27B
    check.pass("tree attention: dim 128 and 256, committed keys and paths within float64 tolerance (worst {d:.3}), a stream's bits alone, any group slots, four-warp or one-warp blocks", .{worst});
}
