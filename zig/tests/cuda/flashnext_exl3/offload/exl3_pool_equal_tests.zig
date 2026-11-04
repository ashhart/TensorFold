//! exl3-pool: gate 2 and 3, the pool against the resident experts and the graph replay.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const check = @import("../../check.zig");

const st = core.safetensors;
const exl3e = cuda.exl3_experts;
const fpool = flash.exl3_pool;
const policy = fpool.offload;
const Gpu = check.Gpu;

const data = @import("exl3_pool_data.zig");
const dims = data.dims;
const width = data.width;
const routed_n = data.routed_n;
const table_n = data.table_n;
const top_k = data.top_k;
const slots = data.slots;
const max_rows = data.max_rows;
const shard_count = data.shard_count;
const projs = data.projs;
const Pack = data.Pack;
const LayerData = data.LayerData;
const Zipf = data.Zipf;
const bf16Bits = data.bf16Bits;
const Io = data.Io;
const call = data.call;
const poolRound = data.poolRound;
const genRound = data.genRound;

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

pub fn equalRun(gpu: Gpu, ex: *const exl3e.Experts, s: cuda.Stream, ld: *const LayerData, lines: usize, prefetch_lines: usize, seed: u64) !void {
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

pub fn graphRun(gpu: Gpu, ex: *const exl3e.Experts, s: cuda.Stream, ld: *const LayerData) !void {
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
