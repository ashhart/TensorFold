//! A shared round's attention past 8 streams: the real attentionSegs on a driver that runs the kernels' row mapping.
const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const t = std.testing;

/// nemotron_attention.cu's AttnSegs, as attentionSegs passes it by value.
const Segs = extern struct { kc: [glue.max_segs]u64, vc: [glue.max_segs]u64, meta: [glue.max_segs]u64, pt: [glue.max_segs]u64, ps: i64, row0: [glue.max_segs]c_int, n: c_int };

const shape: glue.Attn = .{ .nqkv = 6144, .heads = 32, .kv_heads = 2, .dim = 128, .nch = 2 };
const qd: u64 = shape.heads * shape.dim;
const kvd: u64 = shape.kv_heads * shape.dim;

/// Device memory by address, one value a row (a key's, a query's, an output's), and each meta row's position.
const Device = struct {
    var mem: std.AutoHashMapUnmanaged(u64, u64) = .empty;
    var pos: std.AutoHashMapUnmanaged(u64, i64) = .empty;
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
    fn key(cache: u64, k: i64) u64 {
        return cache +% @as(u64, @bitCast(k)) *% kvd *% 2; // contiguous rows (no page table), as the kernels' key_at
    }

    /// One row's keys and values written at its position, its query read, its output: the kernels' indexing.
    fn kvRow(qkv: u64, row: u64, kc: u64, vc: u64, at: i64) void {
        const v = get(qkv + row * shape.nqkv * 2);
        put(key(kc, at), v);
        put(key(vc, at), v +% 1);
    }
    fn attnRow(qkv: u64, row: u64, kc: u64, vc: u64, at: i64, po: u64) void {
        var acc = get(qkv + row * shape.nqkv * 2) *% 31;
        var k: i64 = 0;
        while (k <= at) : (k += 1) acc = acc *% 1000003 +% get(key(kc, k)) *% 7 +% get(key(vc, k)); // causal: keys 0..at
        put(po + row * shape.nch * qd * 4, acc);
    }
    fn mergeRow(po: u64, row: u64, out: u64) void {
        put(out + row * qd * 2, get(po + row * shape.nch * qd * 4));
    }

    fn segOf(s: *const Segs, rq: c_int) usize {
        var k: usize = 0;
        while (k + 1 < s.n and s.row0[k + 1] <= rq) k += 1;
        return k;
    }

    fn launch(f: cuda.abi.Function, gx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, raw: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        const args = raw.?;
        const h = f;
        for (0..gx) |rq| {
            if (h == fns.kv_write.handle or h == fns.attn_chunk.handle or h == fns.attn_chunk_pf.handle) {
                const p = pos.get(arg(u64, args, 3)).? + @as(i64, @intCast(rq));
                if (h == fns.kv_write.handle) kvRow(arg(u64, args, 0), rq, arg(u64, args, 1), arg(u64, args, 2), p) else attnRow(arg(u64, args, 0), rq, arg(u64, args, 1), arg(u64, args, 2), p, arg(u64, args, 4));
            } else if (h == fns.attn_merge.handle) {
                mergeRow(arg(u64, args, 0), rq, arg(u64, args, 4));
            } else {
                const s = arg(Segs, args, 0);
                const k = segOf(&s, @intCast(rq));
                const p = pos.get(s.meta[k]).? + @as(i64, @intCast(rq)) - s.row0[k];
                if (h == fns.kv_write_segs.handle) kvRow(arg(u64, args, 1), rq, s.kc[k], s.vc[k], p) else if (h == fns.attn_chunk_segs.handle) attnRow(arg(u64, args, 1), rq, s.kc[k], s.vc[k], p, arg(u64, args, 2)) else if (h == fns.attn_merge_segs.handle) mergeRow(arg(u64, args, 1), rq, arg(u64, args, 4)) else return 1;
            }
        }
        return 0;
    }
};

const Buffers = struct { qkv: u64, po: u64, pm: u64, pl: u64, out: u64, xs: u64 };

/// Stream i: its rows, position and caches; its query rows' values are its own.
fn rowsOf(i: usize) usize {
    return 1 + i % 2;
}
fn cacheOf(i: usize) [3]u64 {
    return .{ 0x1000_0000 + i * 0x10_0000, 0x2000_0000 + i * 0x10_0000, 0x3000_0000 + i * 0x100 };
}

/// A round of streams `first..first + n` from row 0 of `b`: each stream's keys before its position set, its queries.
fn round(g: glue.Glue, first: usize, n: usize, b: Buffers, parts: ?usize) !void {
    var segs: [16]glue.AttnSeg = undefined;
    var row: usize = 0;
    for (segs[0..n], first..) |*s, i| {
        const c = cacheOf(i);
        s.* = .{ .kc = c[0], .vc = c[1], .meta = c[2], .row0 = row, .rows = rowsOf(i) };
        for (0..rowsOf(i)) |r| Device.put(b.qkv + (row + r) * shape.nqkv * 2, 100_000 + i * 100 + r);
        row += rowsOf(i);
    }
    // whole, as the forward and the MTP head call it now; or in parts of `parts`, as they called it before
    const step = parts orelse n;
    var k0: usize = 0;
    while (k0 < n) : (k0 += step) try g.attentionSegs(segs[k0..@min(n, k0 + step)], b.qkv, b.po, b.pm, b.pl, b.out, b.xs, shape);
}

/// Every stream's output rows and the keys and values it wrote, after `run`.
const Seen = struct { out: [16][2]u64 = @splat(@splat(0)), keys: [16][2][2]u64 = @splat(@splat(@splat(0))), corrupt: usize = 0 };

fn fresh(n: usize) !void {
    Device.mem.clearRetainingCapacity();
    Device.pos.clearRetainingCapacity();
    for (0..n) |i| {
        const c = cacheOf(i);
        try Device.pos.put(t.allocator, c[2], @intCast(5 + 3 * i));
        for (0..5 + 3 * i) |k| { // the keys the stream already holds
            Device.put(Device.key(c[0], @intCast(k)), 1000 * i + k);
            Device.put(Device.key(c[1], @intCast(k)), 1000 * i + k + 7);
        }
    }
}

fn seen(n: usize, rows_at: []const [2]u64) Seen {
    var s: Seen = .{};
    for (0..n) |i| {
        const c = cacheOf(i);
        const p: i64 = @intCast(5 + 3 * i);
        for (0..rowsOf(i)) |r| {
            s.out[i][r] = Device.get(rows_at[i][r]);
            s.keys[i][r] = .{ Device.get(Device.key(c[0], p + @as(i64, @intCast(r)))), Device.get(Device.key(c[1], p + @as(i64, @intCast(r)))) };
        }
        for (0..@intCast(p)) |k| if (Device.get(Device.key(c[0], @intCast(k))) != 1000 * i + k) {
            s.corrupt += 1; // a key the stream held before the round
        };
    }
    return s;
}

const wide: Buffers = .{ .qkv = 0x5000_0000, .po = 0x6000_0000, .pm = 0x6800_0000, .pl = 0x6900_0000, .out = 0x7000_0000, .xs = 0x7800_0000 };
const narrow = [2]Buffers{
    .{ .qkv = 0x8000_0000, .po = 0x9000_0000, .pm = 0x9800_0000, .pl = 0x9900_0000, .out = 0xa000_0000, .xs = 0xa800_0000 },
    .{ .qkv = 0xb000_0000, .po = 0xc000_0000, .pm = 0xc800_0000, .pl = 0xc900_0000, .out = 0xd000_0000, .xs = 0xd800_0000 },
};

/// Each stream's output row addresses in a round of streams from `first` at `b`.
fn outRows(out: *[16][2]u64, first: usize, n: usize, b: Buffers) void {
    var row: usize = 0;
    for (first..first + n) |i| {
        for (0..rowsOf(i)) |r| out[i][r] = b.out + (row + r) * qd * 2;
        row += rowsOf(i);
    }
}

test "wide round walk: 9 to 16 streams' attention reads and writes each stream's own rows and cache, as 8-stream rounds do" {
    defer Device.mem.deinit(t.allocator);
    defer Device.pos.deinit(t.allocator);
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Device.launch;
    inline for (@typeInfo(glue.Fns).@"struct".field_names, 0..) |name, i| @field(Device.fns, name) = .{ .d = &d, .handle = @ptrFromInt(0x100 + i) };
    const g: glue.Glue = .{ .set = null, .f = &Device.fns, .s = .{ .d = &d, .handle = null }, .emb = undefined, .norms = undefined, .moe = undefined };
    for (9..17) |n| {
        // the reference: streams 0..7 in one round, the rest in another, each from row 0 of its own buffers
        try fresh(n);
        try round(g, 0, 8, narrow[0], null);
        try round(g, 8, n - 8, narrow[1], null);
        var at: [16][2]u64 = undefined;
        outRows(&at, 0, 8, narrow[0]);
        outRows(&at, 8, n - 8, narrow[1]);
        const want = seen(n, &at);
        try t.expectEqual(@as(usize, 0), want.corrupt);
        outRows(&at, 0, n, wide);
        for ([_]?usize{ null, glue.max_segs }) |parts| {
            try fresh(n);
            try round(g, 0, n, wide, parts);
            const got = seen(n, &at);
            try t.expectEqual(@as(usize, 0), got.corrupt); // no stream's earlier keys overwritten
            for (0..n) |i| {
                try t.expectEqualSlices(u64, want.out[i][0..rowsOf(i)], got.out[i][0..rowsOf(i)]);
                for (0..rowsOf(i)) |r| try t.expectEqual(want.keys[i][r], got.keys[i][r]);
            }
        }
    }
}
