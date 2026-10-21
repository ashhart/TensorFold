//! exl3-pool: the Flash Next EXL3 expert offload pool (families/flashnext/cuda_exl3_pool.zig, cuda/exl3_pool.zig) on
//! upstream's `exl3_experts.Experts`, with real tensors of the pack (port of exl3-flashnext-window-wip's exl3-offload).
//! Host bank: the memory-mapped shards. Reference: the same layer fully resident (513 experts on the device, tables
//! pointing at them). Every comparison is byte for byte: the pool only changes which device address a pointer table
//! entry holds, so the kernels must produce the same bits.
//!
//!   exl3-pool <pack> equal [layers]   pool at 10, 24, 64 lines vs fully resident over rounds forcing misses, evictions,
//!                                     split passes (16-row chunks) and prefetch; line bytes, tables, staged-bytes ledger
//!   exl3-pool <pack> graph [layers]   runSlots captured once over the pool's tables, replayed after line swaps
//!   exl3-pool <pack> bench [layer]    miss cost a expert and a round (staged and pinned-direct), copy/compute overlap
//!   exl3-pool <pack> hit <picks.bin>...  hit rate vs line count: synthetic skewed 200 tokens, and real router picks
//!   exl3-pool <pack> [layers]         equal + graph
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const check = @import("check.zig");

const st = core.safetensors;
const exl3e = cuda.exl3_experts;
const fpool = flash.exl3_pool;
const policy = fpool.offload;
const Gpu = check.Gpu;

const dims = 2560;
const width = 640;
const routed_n = 512;
const table_n = 513;
const top_k = 10;
const slots = 11;
const max_rows = 16;
const shard_count = 7;
const projs = [3][]const u8{ "gate_proj", "up_proj", "down_proj" };

const Pack = struct {
    files: [shard_count]st.File = undefined,
    open_n: usize = 0,

    fn open(p: *Pack, gpu: Gpu, dir: []const u8) !void {
        var buf: [512]u8 = undefined;
        for (0..shard_count) |i| {
            const path = try std.fmt.bufPrint(&buf, "{s}/model-{d:0>5}-of-{d:0>5}.safetensors", .{ dir, i + 1, shard_count });
            p.files[i] = try st.File.open(gpu.gpa, gpu.io, path);
            p.open_n += 1;
        }
    }

    fn close(p: *Pack, io: std.Io) void {
        for (p.files[0..p.open_n]) |*f| f.close(io);
    }

    fn get(p: *const Pack, comptime fmt: []const u8, args: anytype) !st.Tensor {
        var buf: [200]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, fmt, args);
        for (p.files[0..p.open_n]) |*f| if (f.get(name)) |t| return t;
        std.debug.print("missing tensor {s}\n", .{name});
        return error.MissingTensor;
    }
};

// ---------------------------------------------------------------------------------------------------- one layer

/// One layer's experts: host trellis slices (mmap), the stacked scales and the fully resident reference on the device.
const LayerData = struct {
    layer: usize,
    routed: [routed_n][3][]const u8,
    shared: [3][]const u8,
    scale_bufs: [6]cuda.DeviceBuffer,
    res_mem: cuda.DeviceBuffer,
    res_tab: cuda.DeviceBuffer,
    res: exl3e.Layer,

    fn load(gpu: Gpu, pack: *const Pack, layer: usize) !*LayerData {
        const gpa = gpu.gpa;
        const ld = try gpa.create(LayerData);
        errdefer gpa.destroy(ld);
        ld.layer = layer;
        // stacked scales [E, ·]: gate suh [D] svh [I], up suh [D] svh [I], down suh [I] svh [D]
        const sizes = [6]usize{ dims, dims, width, width, width, dims }; // suh_g suh_u svh_g svh_u suh_d svh_d
        var host: [6][]u8 = undefined;
        for (&host, sizes) |*h, n| h.* = try gpa.alloc(u8, table_n * n * 2);
        defer for (host) |h| gpa.free(h);
        var total: usize = 0;
        for (0..table_n) |e| {
            for (projs, 0..) |pn, p| {
                const tr = if (e < routed_n)
                    try pack.get("model.language_model.layers.{d}.mlp.experts.{d}.{s}.trellis", .{ layer, e, pn })
                else
                    try pack.get("model.language_model.layers.{d}.mlp.shared_expert.{s}.trellis", .{ layer, pn });
                const su = if (e < routed_n)
                    try pack.get("model.language_model.layers.{d}.mlp.experts.{d}.{s}.suh", .{ layer, e, pn })
                else
                    try pack.get("model.language_model.layers.{d}.mlp.shared_expert.{s}.suh", .{ layer, pn });
                const sv = if (e < routed_n)
                    try pack.get("model.language_model.layers.{d}.mlp.experts.{d}.{s}.svh", .{ layer, e, pn })
                else
                    try pack.get("model.language_model.layers.{d}.mlp.shared_expert.{s}.svh", .{ layer, pn });
                const k: usize = if (p == 2) width else dims;
                const n: usize = if (p == 2) dims else width;
                if (tr.rank != 3 or tr.dim(0) * 16 != k or tr.dim(1) * 16 != n) return error.BadShape;
                if (su.bytes.len != k * 2 or sv.bytes.len != n * 2) return error.BadShape;
                if (e < routed_n) ld.routed[e][p] = tr.bytes else ld.shared[p] = tr.bytes;
                total += (tr.bytes.len + 255) / 256 * 256;
                const su_i: usize = switch (p) {
                    0 => 0,
                    1 => 1,
                    else => 4,
                };
                const sv_i: usize = switch (p) {
                    0 => 2,
                    1 => 3,
                    else => 5,
                };
                @memcpy(host[su_i][e * k * 2 ..][0 .. k * 2], su.bytes);
                @memcpy(host[sv_i][e * n * 2 ..][0 .. n * 2], sv.bytes);
            }
        }
        var made: usize = 0;
        errdefer for (ld.scale_bufs[0..made]) |*b| b.free();
        for (&ld.scale_bufs, host) |*b, h| {
            b.* = try cuda.DeviceBuffer.fromHost(gpu.d, h);
            made += 1;
        }
        // the fully resident reference: every expert's trellis back to back, its own tables
        ld.res_mem = try cuda.DeviceBuffer.alloc(gpu.d, total);
        errdefer ld.res_mem.free();
        var ptrs: [3][table_n]u64 = undefined;
        var k2s: [3][table_n]i32 = undefined;
        var at: usize = 0;
        const tiles = (dims / 16) * (width / 16);
        for (0..table_n) |e| for (0..3) |p| {
            const bytes = if (e < routed_n) ld.routed[e][p] else ld.shared[p];
            try ld.res_mem.upload(at, bytes);
            ptrs[p][e] = ld.res_mem.ptr + at;
            k2s[p][e] = @intCast(bytes.len / (tiles * 16));
            at += (bytes.len + 255) / 256 * 256;
        };
        const tl = cuda.exl3_pool.Tables.init(table_n);
        const tab = try gpa.alloc(u8, tl.bytes);
        defer gpa.free(tab);
        @memset(tab, 0);
        for (0..3) |p| {
            @memcpy(tab[tl.ptr[p]..][0 .. table_n * 8], std.mem.sliceAsBytes(&ptrs[p]));
            @memcpy(tab[tl.k2[p]..][0 .. table_n * 4], std.mem.sliceAsBytes(&k2s[p]));
        }
        ld.res_tab = try cuda.DeviceBuffer.fromHost(gpu.d, tab);
        ld.res = .{
            .gate = ld.res_tab.ptr + tl.ptr[0],
            .up = ld.res_tab.ptr + tl.ptr[1],
            .down = ld.res_tab.ptr + tl.ptr[2],
            .gate_k2 = ld.res_tab.ptr + tl.k2[0],
            .up_k2 = ld.res_tab.ptr + tl.k2[1],
            .down_k2 = ld.res_tab.ptr + tl.k2[2],
            .suh_g = ld.scale_bufs[0].ptr,
            .suh_u = ld.scale_bufs[1].ptr,
            .svh_g = ld.scale_bufs[2].ptr,
            .svh_u = ld.scale_bufs[3].ptr,
            .suh_d = ld.scale_bufs[4].ptr,
            .svh_d = ld.scale_bufs[5].ptr,
            .experts = table_n,
            .dims = dims,
            .width = width,
            .codebook = .mul1,
            .act_mode = 1,
        };
        std.debug.print("  layer {d}: 513 experts mapped, fully resident reference {d:.0} MiB; routed k2 gate/up/down {d}/{d}/{d}, shared {d}/{d}/{d}\n", .{
            layer, @as(f64, @floatFromInt(total)) / 1048576.0, k2s[0][0], k2s[1][0], k2s[2][0], k2s[0][512], k2s[1][512], k2s[2][512],
        });
        return ld;
    }

    fn free(ld: *LayerData, gpa: std.mem.Allocator) void {
        ld.res_tab.free();
        ld.res_mem.free();
        for (&ld.scale_bufs) |*b| b.free();
        gpa.destroy(ld);
    }

    fn scales(ld: *const LayerData) fpool.Scales {
        return .{ .suh_g = ld.scale_bufs[0].ptr, .suh_u = ld.scale_bufs[1].ptr, .svh_g = ld.scale_bufs[2].ptr, .svh_u = ld.scale_bufs[3].ptr, .suh_d = ld.scale_bufs[4].ptr, .svh_d = ld.scale_bufs[5].ptr };
    }

    fn source(ld: *const LayerData) fpool.Source {
        return .{ .routed = &ld.routed, .shared = ld.shared };
    }
};

// ---------------------------------------------------------------------------------------------------- inputs

/// Skewed expert choice: Zipf(s) over a fixed random permutation of the 512 ids.
const Zipf = struct {
    perm: [routed_n]u16,
    cdf: [routed_n]f64,

    fn init(seed: u64, s: f64) Zipf {
        var z: Zipf = undefined;
        var prng = std.Random.DefaultPrng.init(seed);
        for (&z.perm, 0..) |*p, i| p.* = @intCast(i);
        prng.random().shuffle(u16, &z.perm);
        var sum: f64 = 0;
        for (&z.cdf, 0..) |*c, i| {
            sum += 1.0 / std.math.pow(f64, @floatFromInt(i + 1), s);
            c.* = sum;
        }
        for (&z.cdf) |*c| c.* /= sum;
        return z;
    }

    /// One row: 10 distinct routed ids then the shared expert.
    fn row(z: *const Zipf, r: std.Random, out: []i32) void {
        var n: usize = 0;
        while (n < top_k) {
            const u = r.float(f64);
            var i: usize = 0;
            while (i + 1 < routed_n and z.cdf[i] < u) i += 1;
            const id: i32 = z.perm[i];
            if (std.mem.indexOfScalar(i32, out[0..n], id) != null) continue;
            out[n] = id;
            n += 1;
        }
        out[top_k] = 512;
    }
};

fn bf16Bits(v: f32) u16 {
    const b: u32 = @bitCast(v);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

/// The device buffers of one engine side (pool or resident): x, picks, weights, scratch, out.
const Io = struct {
    x: cuda.DeviceBuffer,
    picks: cuda.DeviceBuffer,
    wts: cuda.DeviceBuffer,
    out: cuda.DeviceBuffer,
    sbuf: cuda.DeviceBuffer,
    sc: exl3e.Scratch,

    fn init(gpu: Gpu, l: exl3e.Layer) !Io {
        var v: Io = undefined;
        v.x = try cuda.DeviceBuffer.alloc(gpu.d, max_rows * dims * 2);
        v.picks = try cuda.DeviceBuffer.alloc(gpu.d, max_rows * slots * 4);
        v.wts = try cuda.DeviceBuffer.alloc(gpu.d, max_rows * slots * 4);
        v.out = try cuda.DeviceBuffer.alloc(gpu.d, max_rows * dims * 4);
        v.sbuf = try cuda.DeviceBuffer.alloc(gpu.d, try exl3e.scratchBytes(l, max_rows, slots));
        v.sc = try exl3e.scratch(v.sbuf.ptr, l, max_rows, slots);
        return v;
    }

    fn free(v: *Io) void {
        v.x.free();
        v.picks.free();
        v.wts.free();
        v.out.free();
        v.sbuf.free();
    }

    fn set(v: *Io, x: []const u16, picks: []const i32, wts: []const f32) !void {
        try v.x.upload(0, std.mem.sliceAsBytes(x));
        try v.picks.upload(0, std.mem.sliceAsBytes(picks));
        try v.wts.upload(0, std.mem.sliceAsBytes(wts));
    }

    fn y(v: Io, gpa: std.mem.Allocator, rows: usize) ![]u8 {
        const out = try gpa.alloc(u8, rows * slots * dims * 4);
        try v.sbuf.d.check(v.sbuf.d.api.cuMemcpyDtoH_v2(out.ptr, v.sc.y, out.len), "cuMemcpyDtoH");
        return out;
    }
};

fn call(ex: *const exl3e.Experts, s: cuda.Stream, l: exl3e.Layer, v: Io, picks: u64, rows: usize, combine: bool) !void {
    if (combine) {
        try ex.run(s, l, v.x.ptr, dims, .bf16, picks, v.wts.ptr, rows, v.out.ptr, v.sc);
    } else {
        try ex.runSlots(s, l, v.x.ptr, dims, .bf16, picks, rows, v.sc);
    }
}

/// One round through the pool: plan, then each pass staged, computed and released.
fn poolRound(pool: *fpool.Pool, ex: *const exl3e.Experts, s: cuda.Stream, v: Io, picks: []const i32, rows: usize, combine: bool) !*const fpool.Round {
    const r = try pool.plan(picks, rows);
    const l = pool.tables();
    for (0..r.n_passes) |j| {
        try pool.stage(r, j, s);
        try call(ex, s, l, v, pool.picksFor(r, j, v.picks.ptr), rows, combine);
        try pool.release(r, j, s);
    }
    return r;
}

fn genRound(z: *const Zipf, rng: std.Random, rows: usize, x: []u16, picks: []i32, wts: []f32) void {
    for (x[0 .. rows * dims]) |*b| b.* = bf16Bits((rng.float(f32) - 0.5) * 2.0);
    for (0..rows) |i| z.row(rng, picks[i * slots ..][0..slots]);
    for (wts[0 .. rows * slots]) |*w| w.* = rng.float(f32) * 0.2;
}

// ---------------------------------------------------------------------------------------------------- gate 2

fn verifyPool(gpu: Gpu, pool: *fpool.Pool, ld: *const LayerData) !usize {
    const gpa = gpu.gpa;
    try pool.lines.drain();
    const lb = pool.lines.layout.line_bytes;
    const got = try gpa.alloc(u8, lb);
    defer gpa.free(got);
    // tables
    const tl = pool.lines.tlay;
    const tab = try gpa.alloc(u8, tl.bytes);
    defer gpa.free(tab);
    try pool.lines.tables.download(0, tab);
    var checked: usize = 0;
    var resident: [routed_n]?usize = @splat(null);
    for (pool.slots_tab, pool.line_tab, 0..) |id, state, i| {
        if (state == .empty) continue;
        resident[id] = i;
        try pool.lines.mem.download(i * lb, got);
        for (0..3) |p| {
            const want = ld.routed[id][p];
            try check.sameBytes("pool line vs mapped checkpoint", got[pool.lines.layout.off[p]..][0..want.len], want);
        }
        checked += 1;
    }
    for (0..table_n) |e| for (0..3) |p| {
        const ptr = std.mem.readInt(u64, tab[tl.ptr[p] + e * 8 ..][0..8], .little);
        const k2 = std.mem.readInt(i32, tab[tl.k2[p] + e * 4 ..][0..4], .little);
        if (e == 512) {
            try check.expect(ptr != 0 and k2 > 0, "shared expert entry missing", .{});
        } else if (resident[e]) |line| {
            try check.expect(ptr == pool.lines.at(line, p) and k2 == pool.k2[e][p], "expert {d} proj {d}: table points at {x}, line {d} is at {x}", .{ e, p, ptr, line, pool.lines.at(line, p) });
        } else {
            try check.expect(ptr == 0 and k2 == 0, "absent expert {d} has a stale table entry", .{e});
        }
    };
    return checked;
}

const Script = struct { rows: usize, combine: bool };

fn equalRun(gpu: Gpu, ex: *const exl3e.Experts, s: cuda.Stream, ld: *const LayerData, lines: usize, prefetch_lines: usize, seed: u64) !void {
    const gpa = gpu.gpa;
    var pool = try fpool.Pool.init(gpa, gpu.io, gpu.d, ld.source(), ld.scales(), .{ .layer = @intCast(ld.layer), .lines = lines, .prefetch_bytes = 0 }, null);
    defer pool.deinit();
    pool.opt.prefetch_bytes = prefetch_lines * pool.lineBytes();
    var vp = try Io.init(gpu, pool.tables());
    defer vp.free();
    var vr = try Io.init(gpu, ld.res);
    defer vr.free();
    const z = Zipf.init(0x5eed + ld.layer, 1.1);
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    // decode rows, two 16-row prompt chunks, a 4-row chunk; runSlots and run (with combine) alternate
    var script: [40]Script = undefined;
    for (&script, 0..) |*sc, i| sc.* = .{ .rows = 1, .combine = i % 2 == 1 };
    script[6] = .{ .rows = max_rows, .combine = false };
    script[17] = .{ .rows = max_rows, .combine = true };
    script[25] = .{ .rows = 4, .combine = false };
    var x: [max_rows * dims]u16 = undefined;
    var picks: [max_rows * slots]i32 = undefined;
    var wts: [max_rows * slots]f32 = undefined;
    var ledger_staged: u64 = 0;
    var ledger_h2d: u64 = 0;
    var ledger_copies: u64 = 0;
    var split_rounds: usize = 0;
    var max_passes: usize = 0;
    var slot_bytes: usize = 0;
    for (script, 0..) |sc, ri| {
        genRound(&z, rng, sc.rows, &x, &picks, &wts);
        const n = sc.rows * slots;
        try vp.set(x[0 .. sc.rows * dims], picks[0..n], wts[0..n]);
        try vr.set(x[0 .. sc.rows * dims], picks[0..n], wts[0..n]);
        try vp.sbuf.fill8(0xff, s.handle); // a slot no pass wrote shows up
        const r = try poolRound(&pool, ex, s, vp, picks[0..n], sc.rows, sc.combine);
        for (r.copies[0..r.n_copies]) |c| {
            const b = ld.routed[c.id];
            ledger_staged += b[0].len + b[1].len + b[2].len;
            ledger_h2d += pool.lines.layout.off[2] + b[2].len;
            ledger_copies += 1;
        }
        if (r.masked()) split_rounds += 1;
        max_passes = @max(max_passes, r.n_passes);
        try call(ex, s, ld.res, vr, vr.picks.ptr, sc.rows, sc.combine);
        try s.synchronize();
        const yp = try vp.y(gpa, sc.rows);
        defer gpa.free(yp);
        const yr = try vr.y(gpa, sc.rows);
        defer gpa.free(yr);
        check.sameBytes("pool y vs resident y", yp, yr) catch |e| {
            std.debug.print("    round {d} ({d} rows, {d} passes) differs\n", .{ ri, sc.rows, r.n_passes });
            return e;
        };
        slot_bytes += yp.len;
        if (sc.combine) {
            const op = try gpa.alloc(u8, sc.rows * dims * 4);
            defer gpa.free(op);
            const or_ = try gpa.alloc(u8, sc.rows * dims * 4);
            defer gpa.free(or_);
            try vp.out.download(0, op);
            try vr.out.download(0, or_);
            try check.sameBytes("pool combined out vs resident", op, or_);
        }
    }
    const lines_checked = try verifyPool(gpu, &pool, ld);
    const ps = pool.stats;
    const ls = pool.lines.stats;
    const hit = @as(f64, @floatFromInt(ps.hit_slots)) / @as(f64, @floatFromInt(ps.hit_slots + ps.miss_slots));
    std.debug.print("  [{d} lines, prefetch {d}/swap round] {d} rounds ({d} split into passes, up to {d}): routed slots hit {d} miss {d} ({d:.3} hit); {d} copies ({d} misses + {d} prefetch), {d} evictions, {d} protected rounds; H2D {d:.1} MiB, staged {d:.1} MiB (host memcpy {d:.2} ms/expert), staging waits {d}, release waits {d}, gate waits {d}; y {d} bytes compared\n", .{
        lines,                                         prefetch_lines,                                       ps.rounds,                                                                                split_rounds,     max_passes,       ps.hit_slots,  ps.miss_slots, hit, ls.copies, ps.misses, ps.prefetches, ps.evictions, ps.protected,
        @as(f64, @floatFromInt(ls.bytes)) / 1048576.0, @as(f64, @floatFromInt(ls.staged_bytes)) / 1048576.0, @as(f64, @floatFromInt(ls.stage_ns)) / 1e6 / @as(f64, @floatFromInt(@max(ls.copies, 1))), ls.staging_waits, ls.release_waits, ls.gate_waits, slot_bytes,
    });
    try check.expect(ls.copies == ledger_copies and ls.copies == ps.misses + ps.prefetches, "copy count: device {d}, plans {d}, misses+prefetch {d}", .{ ls.copies, ledger_copies, ps.misses + ps.prefetches });
    try check.expect(ls.staged_bytes == ledger_staged, "staged bytes {d} vs plan ledger {d}", .{ ls.staged_bytes, ledger_staged });
    try check.expect(ls.bytes == ledger_h2d, "H2D bytes {d} vs plan ledger {d}", .{ ls.bytes, ledger_h2d });
    try check.expect(ps.evictions > 0 and ps.miss_slots > 0 and ps.hit_slots > 0 and split_rounds > 0, "a path was not exercised (evictions {d}, misses {d}, hits {d}, split {d})", .{ ps.evictions, ps.miss_slots, ps.hit_slots, split_rounds });
    if (prefetch_lines > 0) try check.expect(ps.prefetches > 0, "no prefetch happened", .{});
    check.pass("layer {d}, {d} lines: every round's runSlots/run output is byte-identical to the fully resident layer; {d} lines and 513 table entries match the checkpoint; staged bytes match the plans", .{ ld.layer, lines, lines_checked });
}

// ---------------------------------------------------------------------------------------------------- gate 3

fn graphRun(gpu: Gpu, ex: *const exl3e.Experts, s: cuda.Stream, ld: *const LayerData) !void {
    const gpa = gpu.gpa;
    const lines = 10;
    var pool = try fpool.Pool.init(gpa, gpu.io, gpu.d, ld.source(), ld.scales(), .{ .layer = @intCast(ld.layer), .lines = lines }, null);
    defer pool.deinit();
    var vp = try Io.init(gpu, pool.tables());
    defer vp.free();
    var vr = try Io.init(gpu, ld.res);
    defer vr.free();
    const z = Zipf.init(0xc0ffee + ld.layer, 0.6);
    var prng = std.Random.DefaultPrng.init(0x96a9 + ld.layer);
    const rng = prng.random();
    var x: [dims]u16 = undefined;
    var picks: [slots]i32 = undefined;
    var wts: [slots]f32 = undefined;
    var exec: ?cuda.graph.Exec = null;
    defer if (exec) |*e| e.deinit();
    var g: ?cuda.graph.Graph = null;
    defer if (g) |*gg| gg.deinit();
    var swaps: usize = 0;
    var last: ?[]u8 = null;
    defer if (last) |b| gpa.free(b);
    var distinct: usize = 0;
    const states = 8;
    for (0..states) |si| {
        genRound(&z, rng, 1, &x, &picks, &wts);
        if (si == 5) for (picks[0..5], 0..) |*p, i| {
            p.* = @intCast(pool.slots_tab[i]); // half the row from what the lines hold: a partial hit
        };
        try vp.set(&x, &picks, &wts);
        try vr.set(&x, &picks, &wts);
        const r = try pool.plan(&picks, 1);
        try check.expect(!r.masked(), "one row must fit 10 lines", .{});
        try pool.stage(r, 0, s);
        swaps += r.n_copies;
        const l = pool.tables();
        if (exec == null) {
            // capture once over the tables, with the outputs poisoned so a capture that ran shows up
            try s.synchronize();
            try vp.sbuf.fill8(0xff, s.handle);
            try s.synchronize();
            try cuda.graph.beginCapture(s, .thread_local);
            try call(ex, s, l, vp, vp.picks.ptr, 1, false);
            g = try cuda.graph.endCapture(s);
            exec = try g.?.instantiate();
            try exec.?.upload(s);
            const still = try vp.y(gpa, 1);
            defer gpa.free(still);
            try check.expect(std.mem.allEqual(u8, still, 0xff), "capture executed the kernels", .{});
            std.debug.print("  [graph] runSlots captured once: {d} nodes over the pool's fixed table addresses\n", .{try g.?.nodeCount()});
        }
        try vp.sbuf.fill8(0xff, s.handle);
        try exec.?.launchOn(s);
        try pool.release(r, 0, s);
        try s.synchronize();
        const yg = try vp.y(gpa, 1);
        defer gpa.free(yg);
        // eager over the same tables, then the fully resident layer
        try vp.sbuf.fill8(0xff, s.handle);
        try call(ex, s, l, vp, vp.picks.ptr, 1, false);
        try call(ex, s, ld.res, vr, vr.picks.ptr, 1, false);
        try s.synchronize();
        const ye = try vp.y(gpa, 1);
        defer gpa.free(ye);
        const yr = try vr.y(gpa, 1);
        defer gpa.free(yr);
        try check.sameBytes("graph replay vs eager", yg, ye);
        try check.sameBytes("graph replay vs fully resident", yg, yr);
        if (last) |b| {
            if (!std.mem.eql(u8, b, yg)) distinct += 1;
            gpa.free(b);
        }
        last = try gpa.dupe(u8, yg);
        std.debug.print("  [graph] state {d}: {d} lines swapped by H2D before the replay ({d} held); replay == eager == resident, {d} slot rows byte for byte\n", .{ si, r.n_copies, r.held, slots });
    }
    try check.expect(swaps > lines and distinct == states - 1, "the states did not exercise swaps ({d}) or distinct outputs ({d})", .{ swaps, distinct });
    check.pass("layer {d}: one captured runSlots graph replayed {d} states after {d} line swaps (no recapture), byte-identical to eager and to the resident layer", .{ ld.layer, states, swaps });
}

// ---------------------------------------------------------------------------------------------------- gate 4

fn ms(t0: i96, t1: i96) f64 {
    return @as(f64, @floatFromInt(t1 - t0)) / 1e6;
}

fn median(xs: []f64) f64 {
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    return xs[xs.len / 2];
}

/// A row whose first `k` routed picks are not held by the pool and the rest are.
fn rowWithMisses(pool: *const fpool.Pool, rng: std.Random, k: usize, out: []i32) void {
    var n: usize = 0;
    // held experts first (up to 10 - k)
    for (pool.slots_tab, pool.line_tab) |id, state| {
        if (n == top_k - k) break;
        if (state == .resident) {
            out[k + n] = id;
            n += 1;
        }
    }
    var m: usize = 0;
    while (m < k) {
        const id: u16 = rng.intRangeLessThan(u16, 0, routed_n);
        if (policy.lineOf(pool.v, pool.opt.layer, id) != null) continue;
        if (std.mem.indexOfScalar(i32, out[0..m], id) != null) continue;
        out[m] = id;
        m += 1;
    }
    out[top_k] = 512;
}

fn missCost(gpu: Gpu, ex: *const exl3e.Experts, s: cuda.Stream, ld: *const LayerData, src: fpool.Source, direct: bool, label: []const u8) !void {
    const gpa = gpu.gpa;
    var pool = try fpool.Pool.init(gpa, gpu.io, gpu.d, src, ld.scales(), .{ .layer = @intCast(ld.layer), .lines = 64, .direct = direct, .double_buffer = false }, null);
    defer pool.deinit();
    var vp = try Io.init(gpu, pool.tables());
    defer vp.free();
    var prng = std.Random.DefaultPrng.init(0xbe7c4);
    const rng = prng.random();
    var x: [dims]u16 = undefined;
    for (&x) |*b| b.* = bf16Bits((rng.float(f32) - 0.5) * 2.0);
    const wts: [slots]f32 = @splat(0.1);
    var picks: [slots]i32 = undefined;
    // warm: fill the pool with 64 experts
    for (0..7) |_| {
        rowWithMisses(&pool, rng, top_k, &picks);
        try vp.set(&x, &picks, &wts);
        _ = try poolRound(&pool, ex, s, vp, &picks, 1, false);
        try s.synchronize();
    }
    var c0 = try cuda.Event.init(gpu.d, true);
    defer c0.deinit();
    var c1 = try cuda.Event.init(gpu.d, true);
    defer c1.deinit();
    var k0 = try cuda.Event.init(gpu.d, true);
    defer k0.deinit();
    var k1 = try cuda.Event.init(gpu.d, true);
    defer k1.deinit();
    std.debug.print("  [{s}] one decode row (10 routed + shared), 64 lines, rounds of k misses (median of 9; idle card):\n", .{label});
    var base: f64 = 0;
    for ([_]usize{ 0, 1, 2, 4, 10 }) |k| {
        var wall: [9]f64 = undefined;
        var copy: [9]f64 = undefined;
        var comp: [9]f64 = undefined;
        var stage_host: [9]f64 = undefined;
        for (0..9) |it| {
            rowWithMisses(&pool, rng, k, &picks);
            try vp.set(&x, &picks, &wts);
            try s.synchronize();
            const t0 = check.now(gpu.io);
            const r = try pool.plan(&picks, 1);
            const t1 = check.now(gpu.io);
            try c0.record(pool.lines.copy);
            try pool.stage(r, 0, s);
            try c1.record(pool.lines.copy);
            const t2 = check.now(gpu.io);
            try k0.record(s);
            try call(ex, s, pool.tables(), vp, pool.picksFor(r, 0, vp.picks.ptr), 1, false);
            try k1.record(s);
            try pool.release(r, 0, s);
            try s.synchronize();
            try pool.lines.copy.synchronize();
            const t3 = check.now(gpu.io);
            wall[it] = ms(t0, t3);
            copy[it] = try c0.elapsedMs(c1);
            comp[it] = try k0.elapsedMs(k1);
            stage_host[it] = ms(t1, t2);
        }
        const w = median(&wall);
        if (k == 0) base = w;
        std.debug.print("    k={d:>2}: round wall {d:.3} ms (+{d:.3} vs all-hit), copy stream {d:.3} ms ({d:.3} ms/expert), experts compute {d:.3} ms (measured after the stream's wait on the copies), host stage {d:.3} ms\n", .{
            k, w, w - base, median(&copy), if (k > 0) median(&copy) / @as(f64, @floatFromInt(k)) else 0, median(&comp), median(&stage_host),
        });
    }
    const ls = pool.lines.stats;
    if (!direct) std.debug.print("    host staging memcpy (mmap -> pinned): {d:.3} ms/expert average over {d} copies ({d:.2} GB/s)\n", .{ @as(f64, @floatFromInt(ls.stage_ns)) / 1e6 / @as(f64, @floatFromInt(ls.copies)), ls.copies, @as(f64, @floatFromInt(ls.staged_bytes)) / @as(f64, @floatFromInt(ls.stage_ns)) });
}

/// A 16-row chunk through a 24-line pool (several passes): copies of pass j+1 overlap compute of pass j or not.
fn overlapChunk(gpu: Gpu, ex: *const exl3e.Experts, s: cuda.Stream, ld: *const LayerData, src: fpool.Source, serial: bool) !struct { wall: f64, copy: f64, comp: f64, passes: usize, copies: usize } {
    const gpa = gpu.gpa;
    var pool = try fpool.Pool.init(gpa, gpu.io, gpu.d, src, ld.scales(), .{ .layer = @intCast(ld.layer), .lines = 24, .direct = true }, null);
    defer pool.deinit();
    var vp = try Io.init(gpu, pool.tables());
    defer vp.free();
    const z = Zipf.init(0x0e1a9 + ld.layer, 0.8);
    var prng = std.Random.DefaultPrng.init(0x0e1a9);
    const rng = prng.random();
    var x: [max_rows * dims]u16 = undefined;
    var picks: [max_rows * slots]i32 = undefined;
    var wts: [max_rows * slots]f32 = undefined;
    genRound(&z, rng, max_rows, &x, &picks, &wts);
    try vp.set(&x, &picks, &wts);
    var cs: [64]cuda.Event = undefined;
    for (&cs) |*e| e.* = try cuda.Event.init(gpu.d, true);
    defer for (&cs) |*e| e.deinit();
    var ks: [64]cuda.Event = undefined;
    for (&ks) |*e| e.* = try cuda.Event.init(gpu.d, true);
    defer for (&ks) |*e| e.deinit();
    try s.synchronize();
    const t0 = check.now(gpu.io);
    const r = try pool.plan(&picks, max_rows);
    if (r.n_passes > 32) return error.TooManyPasses;
    const l = pool.tables();
    for (0..r.n_passes) |j| {
        try cs[2 * j].record(pool.lines.copy);
        try pool.stage(r, j, s);
        try cs[2 * j + 1].record(pool.lines.copy);
        if (serial) try pool.lines.copy.synchronize();
        try ks[2 * j].record(s);
        try call(ex, s, l, vp, pool.picksFor(r, j, vp.picks.ptr), max_rows, false);
        try ks[2 * j + 1].record(s);
        try pool.release(r, j, s);
        if (serial) try s.synchronize();
    }
    try s.synchronize();
    try pool.lines.copy.synchronize();
    const wall = ms(t0, check.now(gpu.io));
    var copy: f64 = 0;
    var comp: f64 = 0;
    for (0..r.n_passes) |j| {
        copy += try cs[2 * j].elapsedMs(cs[2 * j + 1]);
        comp += try ks[2 * j].elapsedMs(ks[2 * j + 1]);
    }
    return .{ .wall = wall, .copy = copy, .comp = comp, .passes = r.n_passes, .copies = r.n_copies };
}

/// 60 decode rounds (one row each, skewed picks known ahead), sequential (sync every round) vs pipelined (the host
/// stages round N+1 while round N computes), with and without double buffering.
fn decodePipe(gpu: Gpu, ex: *const exl3e.Experts, s: cuda.Stream, ld: *const LayerData, src: fpool.Source, lines: usize, pipelined: bool, double: bool) !struct { wall: f64, hit: f64, release_waits: u64, protected: u64 } {
    const gpa = gpu.gpa;
    var pool = try fpool.Pool.init(gpa, gpu.io, gpu.d, src, ld.scales(), .{ .layer = @intCast(ld.layer), .lines = lines, .direct = true, .double_buffer = double }, null);
    defer pool.deinit();
    const rounds = 60;
    var ios: [rounds]Io = undefined;
    var made: usize = 0;
    defer for (ios[0..made]) |*v| v.free();
    const z = Zipf.init(0xdec0 + ld.layer, 1.0);
    var prng = std.Random.DefaultPrng.init(0xdec0);
    const rng = prng.random();
    var picks: [rounds][slots]i32 = undefined;
    var x: [dims]u16 = undefined;
    var wts: [slots]f32 = undefined;
    for (0..rounds) |i| {
        ios[i] = try Io.init(gpu, pool.tables());
        made += 1;
        genRound(&z, rng, 1, &x, &picks[i], &wts);
        try ios[i].set(&x, &picks[i], &wts);
    }
    try s.synchronize();
    const t0 = check.now(gpu.io);
    for (0..rounds) |i| {
        _ = try poolRound(&pool, ex, s, ios[i], &picks[i], 1, false);
        if (!pipelined) try s.synchronize();
    }
    try s.synchronize();
    const wall = ms(t0, check.now(gpu.io));
    const ps = pool.stats;
    return .{ .wall = wall, .hit = @as(f64, @floatFromInt(ps.hit_slots)) / @as(f64, @floatFromInt(ps.hit_slots + ps.miss_slots)), .release_waits = pool.lines.stats.release_waits, .protected = ps.protected };
}

fn benchRun(gpu: Gpu, ex: *const exl3e.Experts, s: cuda.Stream, ld: *const LayerData) !void {
    const gpa = gpu.gpa;
    // (a) miss cost, staged from the mapped checkpoint
    try missCost(gpu, ex, s, ld, ld.source(), false, "staged: mmap -> pinned staging -> H2D");
    // the same experts in one pinned bank (what keeping the routed trellis in pinned host RAM gives): direct H2D
    var total: usize = 0;
    for (ld.routed) |e| for (e) |b| {
        total += b.len;
    };
    var bank = try cuda.HostBuffer.alloc(gpu.d, total);
    defer bank.free();
    const pinned = try gpa.alloc([3][]const u8, routed_n);
    defer gpa.free(pinned);
    var at: usize = 0;
    for (ld.routed, pinned) |e, *o| for (e, 0..) |b, p| {
        @memcpy(bank.bytes[at..][0..b.len], b);
        o[p] = bank.bytes[at..][0..b.len];
        at += b.len;
    };
    std.debug.print("  pinned bank for layer {d}: {d:.0} MiB of routed trellis copied into page-locked host memory\n", .{ ld.layer, @as(f64, @floatFromInt(total)) / 1048576.0 });
    const psrc: fpool.Source = .{ .routed = pinned, .shared = ld.shared };
    try missCost(gpu, ex, s, ld, psrc, true, "pinned bank: direct H2D");

    // (b) a 16-row chunk split into passes: serial vs overlapped copies
    const ser = try overlapChunk(gpu, ex, s, ld, psrc, true);
    const ovl = try overlapChunk(gpu, ex, s, ld, psrc, false);
    const hidden_ms = (ser.wall - ovl.wall);
    std.debug.print("  [16-row chunk, 24 lines, pinned bank] {d} passes, {d} copies: serial {d:.3} ms (copy {d:.3} + compute {d:.3}), overlapped {d:.3} ms; overlap hides {d:.3} ms = {d:.0}% of the smaller of copy/compute\n", .{
        ser.passes, ser.copies, ser.wall, ser.copy, ser.comp, ovl.wall, hidden_ms, 100.0 * hidden_ms / @min(ser.copy, ser.comp),
    });
    // (c) decode rounds: pipelined staging, double buffering on/off
    for ([_]usize{ 24, 48 }) |lines| {
        const seq = try decodePipe(gpu, ex, s, ld, psrc, lines, false, true);
        const pipe1 = try decodePipe(gpu, ex, s, ld, psrc, lines, true, false);
        const pipe2 = try decodePipe(gpu, ex, s, ld, psrc, lines, true, true);
        std.debug.print("  [60 decode rounds, {d} lines, pinned bank, hit {d:.3}] sequential {d:.2} ms; pipelined {d:.2} ms single-buffered ({d} copies waited for the previous round's compute), {d:.2} ms double-buffered ({d} waited, {d} rounds spared the previous round's lines)\n", .{
            lines, seq.hit, seq.wall, pipe1.wall, pipe1.release_waits, pipe2.wall, pipe2.release_waits, pipe2.protected,
        });
    }
    check.pass("layer {d}: miss cost, chunk overlap and decode pipelining measured (see lines above)", .{ld.layer});
}

// ---------------------------------------------------------------------------------------------------- hit rates

const line_counts = [_]usize{ 10, 24, 48, 96, 160 };

const Sim = struct {
    hit: u64 = 0,
    miss: u64 = 0,
    dec_hit: u64 = 0,
    dec_miss: u64 = 0,
    copies: u64 = 0,
};

/// Host-only replay of a pick sequence through one layer's policy (the same planner the pool runs).
/// picks [tokens][10] routed ids; the first `prompt` tokens go in 16-row rounds, the rest one row a round.
fn simulate(gpa: std.mem.Allocator, layer: u8, picks: []const u16, tokens: usize, prompt: usize, lines: usize, prefetch_lines: usize, counts: *[policy.shape.layers][policy.shape.experts]u32) !Sim {
    const slots_tab = try gpa.alloc(u16, lines);
    defer gpa.free(slots_tab);
    const line_tab = try gpa.alloc(policy.Line, lines);
    defer gpa.free(line_tab);
    const stamp = try gpa.alloc(u32, lines);
    defer gpa.free(stamp);
    const lay = try gpa.alloc(u8, lines);
    defer gpa.free(lay);
    var v = policy.Vram.init(slots_tab, line_tab, stamp, lay);
    var r = try fpool.Round.init(gpa, lines, max_rows * slots);
    defer r.deinit(gpa);
    var buf: [max_rows * slots]i32 = undefined;
    var sim: Sim = .{};
    var prev: [routed_n]u16 = undefined;
    var prev_n: usize = 0;
    var t: usize = 0;
    var round: u32 = 0;
    while (t < tokens) {
        const rows: usize = if (t < prompt) @min(max_rows, prompt - t) else 1;
        for (0..rows) |i| {
            for (0..top_k) |k| buf[i * slots + k] = picks[(t + i) * top_k + k];
            buf[i * slots + top_k] = 512;
        }
        round += 1;
        try fpool.round.plan(&v, layer, buf[0 .. rows * slots], counts, round, round, .{ .prefetch_bytes = prefetch_lines, .line_bytes = 1, .protect = prev[0..prev_n] }, &r);
        @memcpy(prev[0..r.n_ids], r.ids[0..r.n_ids]);
        prev_n = r.n_ids;
        // prefetch lines land before the next round (the device has a whole layer stack of time)
        for (r.prefetch()) |c| policy.reapAt(&v, layer, c.id, round) catch {};
        sim.hit += r.hit_slots;
        sim.miss += r.miss_slots;
        sim.copies += r.n_copies;
        if (t >= prompt) {
            sim.dec_hit += r.hit_slots;
            sim.dec_miss += r.miss_slots;
        }
        t += rows;
    }
    return sim;
}

fn rate(h: u64, m: u64) f64 {
    return if (h + m == 0) 0 else @as(f64, @floatFromInt(h)) / @as(f64, @floatFromInt(h + m));
}

fn hitRun(gpu: Gpu, files: []const [:0]const u8) !void {
    const gpa = gpu.gpa;
    const counts = try gpa.create([policy.shape.layers][policy.shape.experts]u32);
    defer gpa.destroy(counts);
    // synthetic: 200 tokens of Zipf(1.1) picks, decode rows only
    {
        const z = Zipf.init(0x200, 1.1);
        var prng = std.Random.DefaultPrng.init(0x200);
        var row: [slots]i32 = undefined;
        var picks: [200 * top_k]u16 = undefined;
        for (0..200) |t| {
            z.row(prng.random(), &row);
            for (0..top_k) |k| picks[t * top_k + k] = @intCast(row[k]);
        }
        std.debug.print("  synthetic Zipf(1.1) over 512 experts, 200 decode tokens, one layer:\n", .{});
        for (line_counts) |lines| {
            counts.* = @splat(@splat(0));
            const a = try simulate(gpa, 0, &picks, 200, 0, lines, 0, counts);
            counts.* = @splat(@splat(0));
            const b = try simulate(gpa, 0, &picks, 200, 0, lines, 4, counts);
            std.debug.print("    {d:>3} lines ({d:>5.1} MiB/layer at 1.76 MiB a line): hit {d:.3} (with 4-line prefetch every 4 rounds {d:.3})\n", .{ lines, @as(f64, @floatFromInt(lines)) * 1.7578, rate(a.hit, a.miss), rate(b.hit, b.miss) });
        }
    }
    for (files) |path| {
        const blob = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpa, .limited(1 << 30));
        defer gpa.free(blob);
        const hdr = std.mem.bytesAsSlice(u32, blob[0..12]);
        const tokens: usize = hdr[0];
        const prompt: usize = hdr[1];
        const layers: usize = hdr[2];
        const raw = try gpa.alloc(u16, tokens * layers * top_k);
        defer gpa.free(raw);
        @memcpy(std.mem.sliceAsBytes(raw), blob[12..][0 .. raw.len * 2]);
        std.debug.print("  real router picks {s}: {d} tokens ({d} prompt as 16-row chunks, {d} decode rows) x {d} layers\n", .{ path, tokens, prompt, tokens - prompt, layers });
        const per = try gpa.alloc(u16, tokens * top_k);
        defer gpa.free(per);
        // distinct experts a layer over the decode part (an upper bound on what any pool must hold)
        var distinct_sum: usize = 0;
        for (0..layers) |l| {
            var seen: [routed_n]bool = @splat(false);
            for (prompt..tokens) |t| for (0..top_k) |k| {
                seen[raw[(t * layers + l) * top_k + k]] = true;
            };
            for (seen) |b| distinct_sum += @intFromBool(b);
        }
        std.debug.print("    distinct experts a layer over the {d} decode tokens: {d:.1} on average\n", .{ tokens - prompt, @as(f64, @floatFromInt(distinct_sum)) / @as(f64, @floatFromInt(layers)) });
        for (line_counts) |lines| {
            for ([_]usize{ 0, 4 }) |pf| {
                counts.* = @splat(@splat(0));
                var tot: Sim = .{};
                var lo: f64 = 1;
                var hi: f64 = 0;
                var copies: u64 = 0;
                for (0..layers) |l| {
                    for (0..tokens) |t| @memcpy(per[t * top_k ..][0..top_k], raw[(t * layers + l) * top_k ..][0..top_k]);
                    const s = try simulate(gpa, @intCast(l), per, tokens, prompt, lines, pf, counts);
                    tot.hit += s.hit;
                    tot.miss += s.miss;
                    tot.dec_hit += s.dec_hit;
                    tot.dec_miss += s.dec_miss;
                    copies += s.copies;
                    const dr = rate(s.dec_hit, s.dec_miss);
                    lo = @min(lo, dr);
                    hi = @max(hi, dr);
                }
                const dec_tokens = tokens - prompt;
                const miss_per_tok = @as(f64, @floatFromInt(tot.dec_miss)) / @as(f64, @floatFromInt(dec_tokens));
                std.debug.print("    {d:>3} lines{s}: decode hit {d:.3} (layers {d:.3}..{d:.3}), {d:.1} routed misses a decode token over 48 layers; all rows hit {d:.3}; pool {d:.2} GiB for 48 layers\n", .{
                    lines, if (pf > 0) " +prefetch" else "          ", rate(tot.dec_hit, tot.dec_miss), lo, hi, miss_per_tok, rate(tot.hit, tot.miss), @as(f64, @floatFromInt(lines * 48)) * 1.7578 / 1024.0,
                });
            }
        }
    }
    check.pass("hit rates printed (host simulation of the pool's own planner)", .{});
}

// ---------------------------------------------------------------------------------------------------- driver

fn parseLayers(spec: []const u8, out: []usize) !usize {
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, spec, ',');
    while (it.next()) |tok| {
        if (n == out.len) return error.TooManyLayers;
        out[n] = try std.fmt.parseInt(usize, tok, 10);
        n += 1;
    }
    return n;
}

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 1) return error.MissingArgument;
    const dir = rest[0];
    var mode: []const u8 = "all";
    var layer_spec: []const u8 = "0,17";
    if (rest.len > 1) {
        if (std.ascii.isDigit(rest[1][0])) layer_spec = rest[1] else mode = rest[1];
    }
    if (std.mem.eql(u8, mode, "hit")) return hitRun(gpu, rest[2..]);
    if (rest.len > 2) layer_spec = rest[2];
    var layers: [8]usize = undefined;
    const n_layers = try parseLayers(layer_spec, &layers);
    var pack: Pack = .{};
    defer pack.close(gpu.io);
    try pack.open(gpu, dir);
    var ex = try exl3e.Experts.load(gpu.d);
    defer ex.unload();
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    for (layers[0..n_layers]) |layer| {
        std.debug.print("== layer {d}\n", .{layer});
        const ld = try LayerData.load(gpu, &pack, layer);
        defer ld.free(gpu.gpa);
        const all = std.mem.eql(u8, mode, "all");
        if (all or std.mem.eql(u8, mode, "equal")) {
            try equalRun(gpu, &ex, s, ld, 10, 0, 0x10 + layer);
            try equalRun(gpu, &ex, s, ld, 24, 4, 0x24 + layer);
            try equalRun(gpu, &ex, s, ld, 64, 0, 0x64 + layer);
        }
        if (all or std.mem.eql(u8, mode, "graph")) try graphRun(gpu, &ex, s, ld);
        if (std.mem.eql(u8, mode, "bench")) try benchRun(gpu, &ex, s, ld);
    }
}
