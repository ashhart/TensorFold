//! exl3-kv bench: KV placement timings, bf16 against int8 and the host index keys.

const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("../../check.zig");
const rig = @import("../../flashnext_rig.zig");
const Kv = flash.exl3.kv;
const Rig = rig.Rig;
const bf16 = rig.bf16;

const q_heads = 24;
const kv_heads = 2;
const head_dim = 256;
const half = 32;
const idw = 2052;
const top = 512;
const ratio = 4;
const nch = 5;

const Bench = struct {
    s: cuda.Stream,
    ev0: cuda.Event,
    ev1: cuda.Event,
    reps: usize = 20,

    fn time(b: *Bench, ctx: anytype, comptime f: fn (@TypeOf(ctx)) anyerror!void) !f64 {
        for (0..3) |_| try f(ctx);
        try b.s.synchronize();
        var best: f64 = std.math.inf(f64);
        for (0..5) |_| {
            try b.ev0.record(b.s);
            for (0..b.reps) |_| try f(ctx);
            try b.ev1.record(b.s);
            try b.ev1.synchronize();
            best = @min(best, @as(f64, try b.ev0.elapsedMs(b.ev1)) / @as(f64, @floatFromInt(b.reps)));
        }
        return best;
    }
};

/// A cache of `cap` tokens in one place: bf16 K/V and its int8 codes + scales (same values, packed on the device).
const Place = struct {
    kc: u64,
    vc: u64,
    c: Kv.Kv8.Cache,
};

const Attn = struct {
    g: *const flash.Glue,
    k8: *const Kv.Kv8,
    s: cuda.Stream,
    p: Place,
    int8: bool,
    q: u64,
    pos: u64,
    po: u64,
    pm: u64,
    pl: u64,
    ids: u64,
    nk: u64,
    spr: u64,
    out: u64,
    rows: usize,
    chunks: usize,

    fn go(a: *const Attn) anyerror!void {
        if (a.int8) {
            try a.k8.attnChunks(a.s, a.q, a.p.c, a.pos, a.po, a.pm, a.pl, a.ids, a.nk, a.spr, a.rows, q_heads, kv_heads, a.chunks, true);
            try a.k8.attnMerge(a.s, a.po, a.pm, a.pl, a.pos, a.out, a.nk, a.spr, a.rows, q_heads, kv_heads, true, true);
        } else {
            try a.g.attnChunks(a.s, a.q, a.p.kc, a.p.vc, a.pos, a.po, a.pm, a.pl, a.ids, a.nk, a.spr, a.rows, kv_heads, a.chunks, true, .{});
            try a.g.attnMerge(a.s, a.po, a.pm, a.pl, a.pos, a.out, a.nk, a.spr, a.rows, kv_heads, true);
        }
    }
};

/// `rows` consecutive sparse rows ending at `end0 + rows`: row 0 picks `top` random blocks, each next row keeps a
fn overlapIds(rng: std.Random, a: std.mem.Allocator, end0: usize, rows: usize, keep: f64, ids: []i32, nk: []i32) !void {
    const complete0 = (end0 + 1) / ratio;
    var cur = try std.DynamicBitSetUnmanaged.initEmpty(a, complete0 + rows);
    var got: usize = 0;
    while (got < top) {
        const b = rng.uintLessThan(usize, complete0);
        if (cur.isSet(b)) continue;
        cur.set(b);
        got += 1;
    }
    for (0..rows) |row| {
        const end = end0 + row + 1;
        const complete = end / ratio;
        if (row > 0) {
            // drop (1 - keep) of the set, refill from all complete blocks
            var drop: usize = @intFromFloat(@round((1 - keep) * @as(f64, top)));
            while (drop > 0) {
                const b = rng.uintLessThan(usize, complete);
                if (!cur.isSet(b)) continue;
                cur.unset(b);
                drop -= 1;
            }
            while (cur.count() < top) {
                const b = rng.uintLessThan(usize, complete);
                if (!cur.isSet(b)) cur.set(b);
            }
        }
        var k: usize = 0;
        var it = cur.iterator(.{});
        while (it.next()) |b| {
            for (0..ratio) |j| ids[row * idw + k * ratio + j] = @intCast(b * ratio + j);
            k += 1;
        }
        var n: usize = top * ratio;
        for (ratio * complete..end) |t| {
            ids[row * idw + n] = @intCast(t);
            n += 1;
        }
        nk[row] = @intCast(n);
    }
}

const Copy = struct {
    s: cuda.Stream,
    dst: u64,
    src: [*]const u8,
    bytes: usize,
    pieces: usize,
    fn go(c: *const Copy) anyerror!void {
        const d = c.s.d;
        const each = c.bytes / c.pieces;
        for (0..c.pieces) |i| try d.check(d.api.cuMemcpyHtoDAsync_v2(c.dst + i * each, c.src + i * each, each, c.s.handle), "cuMemcpyHtoDAsync");
    }
};

const Gather = struct {
    k8: *const Kv.Kv8,
    s: cuda.Stream,
    src_k: u64,
    src_v: u64,
    dst_k: u64,
    dst_v: u64,
    ids: u64,
    n: usize,
    row: usize,
    fn go(g: *const Gather) anyerror!void {
        try g.k8.gatherRows(g.s, g.src_k, g.dst_k, g.ids, g.n, g.row);
        try g.k8.gatherRows(g.s, g.src_v, g.dst_v, g.ids, g.n, g.row);
    }
};

pub fn benchMode(gpu: check.Gpu, cap: usize) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0xbe7c4);
    const rng = prng.random();
    var r = try Rig.init(gpu, a, rng);
    defer r.deinit();
    var g = try flash.Glue.load(gpu.d);
    defer g.deinit();
    var k8 = try Kv.Kv8.load(gpu.d);
    defer k8.deinit();
    var b: Bench = .{ .s = r.s, .ev0 = try cuda.Event.init(gpu.d, true), .ev1 = try cuda.Event.init(gpu.d, true) };
    {
        var free_b: usize = 0;
        var total_b: usize = 0;
        try gpu.d.check(gpu.d.api.cuMemGetInfo_v2(&free_b, &total_b), "cuMemGetInfo");
        std.debug.print("  device after context + the attention modules: free {d:.3} of {d:.3} GiB (context and modules {d:.3} GiB)\n", .{ @as(f64, @floatFromInt(free_b)) / Kv.gib, @as(f64, @floatFromInt(total_b)) / Kv.gib, @as(f64, @floatFromInt(total_b - free_b)) / Kv.gib });
    }
    const nrow = cap * kv_heads;
    const kv_bytes = nrow * head_dim * 2;
    const cb = Kv.Kv8.cacheBytes(cap, kv_heads);
    std.debug.print("== exl3-kv bench: cache of {d} tokens, one attention layer: bf16 K+V {d:.1} MiB, int8 codes+scales {d:.1} MiB\n", .{ cap, @as(f64, @floatFromInt(2 * kv_bytes)) / 1048576.0, @as(f64, @floatFromInt(2 * (cb.codes + cb.scales))) / 1048576.0 });

    // the cache's values: random bf16 made on the device side once, then packed; the host copies get the same bytes
    var dev_k = try cuda.DeviceBuffer.alloc(gpu.d, kv_bytes);
    defer dev_k.free();
    var dev_v = try cuda.DeviceBuffer.alloc(gpu.d, kv_bytes);
    defer dev_v.free();
    var host_k = try cuda.HostBuffer.allocMapped(gpu.d, kv_bytes);
    defer host_k.free();
    var host_v = try cuda.HostBuffer.allocMapped(gpu.d, kv_bytes);
    defer host_v.free();
    {
        const hk = host_k.slice(u16);
        const hv = host_v.slice(u16);
        for (hk) |*x| x.* = bf16(rng.floatNorm(f32) * 0.5);
        for (hv) |*x| x.* = bf16(rng.floatNorm(f32) * 0.5);
        try dev_k.upload(0, host_k.bytes);
        try dev_v.upload(0, host_v.bytes);
    }
    var dev_c: [4]cuda.DeviceBuffer = undefined;
    var host_c: [4]cuda.HostBuffer = undefined;
    for (0..4) |i| {
        const n = if (i % 2 == 0) cb.codes else cb.scales;
        dev_c[i] = try cuda.DeviceBuffer.alloc(gpu.d, n);
        host_c[i] = try cuda.HostBuffer.allocMapped(gpu.d, n);
    }
    defer for (&dev_c, &host_c) |*x, *y| {
        x.free();
        y.free();
    };
    try k8.packRows(r.s, dev_k.ptr, dev_c[0].ptr, dev_c[1].ptr, nrow, true);
    try k8.packRows(r.s, dev_v.ptr, dev_c[2].ptr, dev_c[3].ptr, nrow, true);
    try r.s.synchronize();
    for (0..4) |i| try dev_c[i].download(0, host_c[i].bytes);
    const on_dev: Place = .{ .kc = dev_k.ptr, .vc = dev_v.ptr, .c = .{ .k8 = dev_c[0].ptr, .ks = dev_c[1].ptr, .v8 = dev_c[2].ptr, .vs = dev_c[3].ptr } };
    const on_host: Place = .{ .kc = try host_k.device(), .vc = try host_v.device(), .c = .{ .k8 = try host_c[0].device(), .ks = try host_c[1].device(), .v8 = try host_c[2].device(), .vs = try host_c[3].device() } };

    const max_rows = 256;
    const q = r.bfs(max_rows * q_heads * head_dim, 1);
    const dq = try r.dev(std.mem.sliceAsBytes(q));
    const po = try r.out(max_rows * nch * q_heads * head_dim * 4);
    const pm = try r.out(max_rows * nch * q_heads * 4);
    const pl = try r.out(max_rows * nch * q_heads * 4);
    const out = try r.out(max_rows * q_heads * head_dim * 2);
    const ids = try a.alloc(i32, max_rows * idw);
    const nk = try a.alloc(i32, max_rows);
    const spr = try a.alloc(i32, max_rows);

    const Case = struct { name: []const u8, rows: usize, end0: usize, sparse: bool, keep: f64 };
    const cases = [_]Case{
        .{ .name = "decode 1 row, sparse at the cache end", .rows = 1, .end0 = cap - 2, .sparse = true, .keep = 0 },
        .{ .name = "decode window 4 rows (MTP verify), keep 0.95", .rows = 4, .end0 = cap - 5, .sparse = true, .keep = 0.95 },
        .{ .name = "prompt 256 rows, sparse, keep 0.95", .rows = 256, .end0 = cap - 257, .sparse = true, .keep = 0.95 },
        .{ .name = "prompt 256 rows, sparse, keep 0.80", .rows = 256, .end0 = cap - 257, .sparse = true, .keep = 0.80 },
        .{ .name = "prompt 256 rows, sparse, keep 0 (worst case)", .rows = 256, .end0 = cap - 257, .sparse = true, .keep = 0 },
        .{ .name = "prompt 256 rows, dense (keys 1,795..2,050)", .rows = 256, .end0 = 1794, .sparse = false, .keep = 0 },
    };
    for (cases) |c| {
        @memset(ids, 0);
        if (c.sparse) {
            try overlapIds(rng, a, c.end0, c.rows, c.keep, ids, nk);
        } else for (0..c.rows) |row| {
            nk[row] = @intCast(c.end0 + row + 1);
        }
        for (spr[0..c.rows]) |*x| x.* = @intFromBool(c.sparse);
        // distinct keys these rows read (what crosses PCIe at least once)
        var seen = try std.DynamicBitSetUnmanaged.initEmpty(a, cap);
        var reads: usize = 0;
        for (0..c.rows) |row| {
            const n: usize = @intCast(nk[row]);
            reads += n;
            for (0..n) |i| seen.set(if (c.sparse) @intCast(ids[row * idw + i]) else i);
        }
        const distinct = seen.count();
        const pos0: i32 = @intCast(c.end0 + 1);
        const dpos = try r.dev(std.mem.sliceAsBytes(&[_]i32{ pos0, 0, 0, 0 }));
        const dids = try r.dev(std.mem.sliceAsBytes(ids[0 .. c.rows * idw]));
        const dnk = try r.dev(std.mem.sliceAsBytes(nk[0..c.rows]));
        const dspr = try r.dev(std.mem.sliceAsBytes(spr[0..c.rows]));
        const chunks = if (c.sparse) nch else (c.end0 + c.rows + 1 + 511) / 512;
        var ms: [4]f64 = undefined;
        for ([_]bool{ false, true }, 0..) |int8, ii| for ([_]Place{ on_dev, on_host }, 0..) |p, pi| {
            const at: Attn = .{ .g = &g, .k8 = &k8, .s = r.s, .p = p, .int8 = int8, .q = dq, .pos = dpos, .po = po, .pm = pm, .pl = pl, .ids = dids, .nk = dnk, .spr = dspr, .out = out, .rows = c.rows, .chunks = chunks };
            b.reps = if (pi == 1 and c.rows > 4) 3 else 20;
            ms[ii * 2 + pi] = try b.time(&at, Attn.go);
        };
        // the int8 kernel with the next tile in flight at every row count (Kv.ahead_rows raised)
        var ms_ahead: f64 = ms[2];
        if (c.rows > Kv.ahead_rows) {
            const keep_rows = Kv.ahead_rows;
            Kv.ahead_rows = 1 << 20;
            defer Kv.ahead_rows = keep_rows;
            const at: Attn = .{ .g = &g, .k8 = &k8, .s = r.s, .p = on_dev, .int8 = true, .q = dq, .pos = dpos, .po = po, .pm = pm, .pl = pl, .ids = dids, .nk = dnk, .spr = dspr, .out = out, .rows = c.rows, .chunks = chunks };
            b.reps = 20;
            ms_ahead = try b.time(&at, Attn.go);
        }
        const key_bf: f64 = 2 * kv_heads * head_dim * 2; // one key's K and V, bf16, both heads
        const key_i8: f64 = @floatFromInt(Kv.kvRowBytes(.int8, kv_heads));
        const once_bf = @as(f64, @floatFromInt(distinct)) * key_bf / 1e6;
        const once_i8 = @as(f64, @floatFromInt(distinct)) * key_i8 / 1e6;
        std.debug.print("RESULT {s}: keys read {d}, distinct {d} ({d:.2} MB bf16 / {d:.2} MB int8 once)\n", .{ c.name, reads, distinct, once_bf, once_i8 });
        std.debug.print("  chunks+merge ms: bf16 device {d:.4}  bf16 host-mapped {d:.4} ({d:.1} GB/s of distinct)  int8 device {d:.4} (read-ahead kernel {d:.4})  int8 host-mapped {d:.4} ({d:.1} GB/s of distinct)\n", .{ ms[0], ms[1], once_bf / ms[1], ms[2], ms_ahead, ms[3], once_i8 / ms[3] });
    }

    // explicit staging alternatives for one decode row's 2,051 keys
    {
        const n = 2051;
        const row_bytes = kv_heads * head_dim * 2; // one key, both heads, K (V the same)
        const bytes = 2 * n * row_bytes;
        var pinned = try cuda.HostBuffer.alloc(gpu.d, bytes);
        defer pinned.free();
        var staging = try cuda.DeviceBuffer.alloc(gpu.d, bytes);
        defer staging.free();
        const one: Copy = .{ .s = r.s, .dst = staging.ptr, .src = pinned.bytes.ptr, .bytes = bytes, .pieces = 1 };
        b.reps = 20;
        const t_one = try b.time(&one, Copy.go);
        const blocks: Copy = .{ .s = r.s, .dst = staging.ptr, .src = pinned.bytes.ptr, .bytes = 1024 * (bytes / 1024), .pieces = 1024 };
        const t_blocks = try b.time(&blocks, Copy.go);
        try overlapIds(rng, a, cap - 2, 1, 0, ids, nk);
        const dids = try r.dev(std.mem.sliceAsBytes(ids[0..n]));
        const gk: Gather = .{ .k8 = &k8, .s = r.s, .src_k = on_host.kc, .src_v = on_host.vc, .dst_k = staging.ptr, .dst_v = staging.ptr + n * row_bytes, .ids = dids, .n = n, .row = row_bytes };
        const t_gather = try b.time(&gk, Gather.go);
        // the selection's ids to the host and back to a waiting thread: one D2H and a stream synchronize
        var hid = try cuda.HostBuffer.alloc(gpu.d, n * 4);
        defer hid.free();
        var best: f64 = std.math.inf(f64);
        for (0..50) |_| {
            const t0 = check.now(gpu.io);
            try gpu.d.check(gpu.d.api.cuMemcpyDtoHAsync_v2(hid.bytes.ptr, dids, n * 4, r.s.handle), "cuMemcpyDtoHAsync");
            try r.s.synchronize();
            best = @min(best, @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e6);
        }
        const mb = @as(f64, @floatFromInt(bytes)) / 1e6;
        std.debug.print("RESULT staging one decode row's 2,051 keys (K+V bf16, {d:.2} MB): one pinned H2D {d:.4} ms ({d:.1} GB/s); 1,024 block copies {d:.4} ms; gather kernel from mapped host {d:.4} ms ({d:.1} GB/s); ids D2H + stream sync {d:.4} ms\n", .{ mb, t_one, mb / t_one, t_blocks, t_gather, mb / t_gather, best });
    }
    // the indexer's raw keys in mapped host memory: upstream's pooling reads them only when a block of 4 completes
    {
        const ikc_bytes = cap * 128 * 2;
        var ikc_dev = try cuda.DeviceBuffer.alloc(gpu.d, ikc_bytes);
        defer ikc_dev.free();
        var ikc_host = try cuda.HostBuffer.allocMapped(gpu.d, ikc_bytes);
        defer ikc_host.free();
        for (ikc_host.slice(u16)) |*x| x.* = bf16(rng.floatNorm(f32));
        try ikc_dev.upload(0, ikc_host.bytes);
        const kw = try r.dev(std.mem.sliceAsBytes(r.f32s(128, 0.5, 1.5)));
        const inv = try a.alloc(f32, half);
        for (inv, 0..) |*o, i| o.* = @floatCast(std.math.pow(f64, 10_000_000.0, -@as(f64, @floatFromInt(i)) / half));
        const dinv = try r.dev(std.mem.sliceAsBytes(inv));
        const Pool = struct {
            g: *const flash.Glue,
            s: cuda.Stream,
            ikc: u64,
            pooled: u64,
            pos: u64,
            kw: u64,
            inv: u64,
            rows: usize,
            fn go(p: *const @This()) anyerror!void {
                try p.g.qsaPool(p.s, p.ikc, p.pooled, p.pos, p.kw, p.inv, 1e-6, p.rows, .{});
            }
        };
        for ([_]usize{ 1, 256 }) |rows| {
            const pos0: i32 = @intCast(cap - 256);
            const dpos = try r.dev(std.mem.sliceAsBytes(&[_]i32{ if (rows == 1) pos0 + 3 else pos0, 0, 0, 0 }));
            const pd = try r.out((cap / 4 + 1) * 128 * 2);
            const ph = try r.out((cap / 4 + 1) * 128 * 2);
            const on_d: Pool = .{ .g = &g, .s = r.s, .ikc = ikc_dev.ptr, .pooled = pd, .pos = dpos, .kw = kw, .inv = dinv, .rows = rows };
            const on_h: Pool = .{ .g = &g, .s = r.s, .ikc = try ikc_host.device(), .pooled = ph, .pos = dpos, .kw = kw, .inv = dinv, .rows = rows };
            b.reps = 20;
            const td = try b.time(&on_d, Pool.go);
            const th = try b.time(&on_h, Pool.go);
            const n = (cap / 4 + 1) * 128;
            const same = std.mem.eql(u16, try r.back(u16, pd, n), try r.back(u16, ph, n));
            std.debug.print("RESULT indexer pooling of {d} row(s) at the cache end: raw keys on the device {d:.4} ms, in mapped host memory {d:.4} ms; pooled rows {s}\n", .{ rows, td, th, if (same) "bit-equal" else "DIFFER" });
            try check.expect(same, "pooling from mapped host keys differs", .{});
        }
    }
    // a prompt-cache snapshot (cuda_snap.zig copies with cuMemcpyDtoDAsync) into mapped pinned host memory and back
    {
        const bytes: usize = 256 << 20;
        var dev = try cuda.DeviceBuffer.alloc(gpu.d, bytes);
        defer dev.free();
        var host = try cuda.HostBuffer.allocMapped(gpu.d, bytes);
        defer host.free();
        const hp = try host.device();
        const Dtod = struct {
            s: cuda.Stream,
            dst: u64,
            src: u64,
            n: usize,
            fn go(c: *const @This()) anyerror!void {
                try c.s.d.check(c.s.d.api.cuMemcpyDtoDAsync_v2(c.dst, c.src, c.n, c.s.handle), "cuMemcpyDtoDAsync");
            }
        };
        b.reps = 3;
        const save: Dtod = .{ .s = r.s, .dst = hp, .src = dev.ptr, .n = bytes };
        const t_save = try b.time(&save, Dtod.go);
        const back: Dtod = .{ .s = r.s, .dst = dev.ptr, .src = hp, .n = bytes };
        const t_back = try b.time(&back, Dtod.go);
        const gb = @as(f64, @floatFromInt(bytes)) / 1e9;
        std.debug.print("RESULT a 256 MiB snapshot through cuMemcpyDtoDAsync to/from mapped pinned host memory: save {d:.2} ms ({d:.1} GB/s), restore {d:.2} ms ({d:.1} GB/s)\n", .{ t_save, gb / t_save * 1e3, t_back, gb / t_back * 1e3 });
    }
    check.pass("exl3-kv bench: timings above (an idle card under gpurun)", .{});
}
