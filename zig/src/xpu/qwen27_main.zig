//! Development runner of the Qwen3.8-27B XPU forward: greedy tokens and timings, to check against a fp32 reference.
const std = @import("std");
const q = @import("qwen27.zig");
const dr_mod = @import("drafter.zig");
const Drafter = dr_mod.Drafter;
const fp64_ref = @import("fp64_ref.zig");

const alloc = std.heap.c_allocator;
extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn lseek(fd: c_int, off: i64, whence: c_int) i64;
extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) ?*anyopaque;

pub fn main(init: std.process.Init) !void {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len < 3) {
        std.debug.print("usage: xpu-qwen27 MODEL_DIR IDS_FILE [STEPS]\n", .{});
        return;
    }
    const steps: usize = if (argv.len > 3) try std.fmt.parseInt(usize, argv[3], 10) else 32;
    const ctx: u32 = if (std.c.getenv("XPU_CTX")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 4096;
    var ids: std.ArrayList(u32) = .empty;
    if (std.mem.startsWith(u8, argv[2], "--bench=")) {
        const n = try std.fmt.parseInt(usize, argv[2][8..], 10);
        for (0..n) |i| try ids.append(alloc, @intCast((i * 7919 + 13) % 200000));
    } else {
        const path = try std.fmt.allocPrintSentinel(alloc, "{s}", .{argv[2]}, 0);
        const fd = open(path.ptr, 0);
        if (fd < 0) return error.FileNotFound;
        const size: usize = @intCast(lseek(fd, 0, 2));
        const text: [*]const u8 = @ptrCast(mmap(null, size, 1, 2, fd, 0) orelse return error.MmapFailed);
        var it = std.mem.tokenizeAny(u8, text[0..size], " \n,[]");
        while (it.next()) |w| try ids.append(alloc, try std.fmt.parseInt(u32, w, 10));
    }
    q.verbose = true;
    const t_load = q.now();
    const r = try q.Runner.init(argv[1], ctx);
    std.debug.print("loaded in {d:.1} s\n", .{q.now() - t_load});
    if (std.c.getenv("XPU_FIDELITY") != null) return fp64_ref.fidelity(r, ids.items);
    if (std.c.getenv("XPU_FP64_REF") != null) fp64_ref.arm(r);
    const t0 = q.now();
    try r.prompt(ids.items, true);
    const t1 = q.now();
    fp64_ref.disarm();
    q.profReport("prompt");
    var out: std.ArrayList(u32) = .empty;
    try out.append(alloc, r.last);
    for (1..steps) |_| {
        try r.step(r.last);
        try out.append(alloc, r.last);
    }
    const t2 = q.now();
    q.profReport("decode");
    if (std.c.getenv("XPU_OVERLAP_PROBE") != null) try overlapProbe(r);
    if (std.c.getenv("XPU_ROWS_PROBE") != null) try rowsProbe(r);
    if (std.c.getenv("XPU_EXL3_PROBE") != null) try exl3Probe(r);
    if (std.c.getenv("XPU_MULTI_TEST")) |v| try multiTest(r, ids.items, try std.fmt.parseInt(u32, std.mem.span(v), 10));
    if (std.c.getenv("XPU_GEMM_ZEBIN")) |path| try gemmLoop(r, std.mem.span(path));
    if (std.c.getenv("XPU_A8_ZEBIN")) |path| try a8Loop(r, std.mem.span(path));
    if (std.c.getenv("XPU_VERIFY_TEST") != null) try verifyTest(r, ids.items, out.items);
    if (std.c.getenv("XPU_DRAFTER")) |dir| try drafted(r, std.mem.span(dir), ids.items, out.items);
    std.debug.print("prompt: {d} tokens in {d:.3} s ({d:.0} tok/s)\n", .{ ids.items.len, t1 - t0, @as(f64, @floatFromInt(ids.items.len)) / (t1 - t0) });
    std.debug.print("decode: {d} tokens in {d:.3} s ({d:.1} tok/s)\n", .{ steps - 1, t2 - t1, @as(f64, @floatFromInt(steps - 1)) / (t2 - t1) });
    std.debug.print("tokens:", .{});
    for (out.items) |t| std.debug.print(" {d}", .{t});
    std.debug.print("\n", .{});
}

/// The drafted-equals-serial contract with stand-in drafts: blocks of 8 rows, perfect and wrong drafts equal serial.
fn verifyTest(r: *q.Runner, prompt: []const u32, serial: []const u32) !void {
    for ([_]bool{ true, false }) |perfect| {
        try r.reset();
        try r.prompt(prompt, true);
        var got: std.ArrayList(u32) = .empty;
        try got.append(alloc, r.last);
        var rounds: usize = 0;
        while (got.items.len < serial.len) {
            var block: [8]u32 = undefined;
            block[0] = got.items[got.items.len - 1];
            for (1..8) |j| {
                const at = got.items.len - 1 + j;
                block[j] = if (perfect and at < serial.len) serial[at] else 0;
            }
            const greedy = try r.verify(&block);
            var n: u32 = 0; // accepted drafts
            while (n + 1 < 8 and greedy[n] == block[n + 1]) n += 1;
            try r.commit(n + 1);
            for (greedy[0 .. n + 1]) |t| try got.append(alloc, t);
            rounds += 1;
        }
        const same = std.mem.eql(u32, got.items[0..serial.len], serial);
        std.debug.print("verify test ({s} drafts): {d} rounds, equal to serial: {}\n", .{ if (perfect) "perfect" else "wrong", rounds, same });
        if (!same) {
            std.debug.print("  got:", .{});
            for (got.items[0..serial.len]) |t| std.debug.print(" {d}", .{t});
            std.debug.print("\n", .{});
        }
    }
}

fn onTaps(ctx: *anyopaque, taps: *anyopaque, n: u32) !void {
    const dr: *Drafter = @ptrCast(@alignCast(ctx));
    try dr.addTaps(taps, n);
}

/// Greedy decode with DFlash2 drafts (blocks of XPU_BLOCK rows, 8 by default): must equal the serial reply.
fn drafted(r: *q.Runner, dir: []const u8, prompt: []const u32, serial: []const u32) !void {
    const t_load = q.now();
    const dr = try Drafter.init(r, dir);
    std.debug.print("drafter loaded in {d:.1} s\n", .{q.now() - t_load});
    const L: u32 = if (std.c.getenv("XPU_BLOCK")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 8;
    try r.reset();
    dr.reset();
    r.taps_on = true;
    r.on_taps = .{ .ctx = dr, .f = onTaps };
    try r.prompt(prompt, true);
    var got: std.ArrayList(u32) = .empty;
    try got.append(alloc, r.last);
    var rounds: usize = 0;
    q.profReport("drafter prompt (discarded)");
    var t_draft: f64 = 0;
    var t_verify: f64 = 0;
    const t0 = q.now();
    const nodes: u32 = if (std.c.getenv("XPU_TREE")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 0;
    var accepted: usize = 0;
    var drafted_n: usize = 0;
    while (got.items.len < serial.len) {
        if (nodes > 0) { // tree rounds: drafting and verification timed together
            const a = q.now();
            var out: [16]u32 = undefined;
            const round = try dr_mod.treeRound(dr, r.last, L - 1, nodes, null, &out);
            t_verify += q.now() - a;
            for (round.tokens[1..]) |t| try got.append(alloc, t);
            try got.append(alloc, round.terminal);
            accepted += round.accepted;
            drafted_n += round.drafted;
            rounds += 1;
            continue;
        }
        var block: [16]u32 = undefined;
        block[0] = r.last;
        const a = q.now();
        try dr.propose(r.last, L - 1, block[1..L]);
        const b = q.now();
        const greedy = try r.verify(block[0..L]);
        var n: u32 = 0;
        while (n + 1 < L and greedy[n] == block[n + 1]) n += 1;
        try r.commit(n + 1);
        try dr.addTaps(r.acts.taps, n + 1);
        t_verify += q.now() - b;
        t_draft += b - a;
        for (greedy[0 .. n + 1]) |t| try got.append(alloc, t);
        rounds += 1;
    }
    const dt = q.now() - t0;
    q.profReport("drafted rounds");
    if (nodes > 0) std.debug.print("tree: {d} nodes a round, {d:.2} accepted of {d:.1} drafted\n", .{ nodes, @as(f64, @floatFromInt(accepted)) / @as(f64, @floatFromInt(rounds)), @as(f64, @floatFromInt(drafted_n)) / @as(f64, @floatFromInt(rounds)) });
    const len = @min(got.items.len, serial.len);
    const same = std.mem.eql(u32, got.items[0..len], serial[0..len]);
    std.debug.print("drafted: {d} tokens in {d} rounds ({d:.2} a round), {d:.1} tok/s, draft {d:.1} ms + verify {d:.1} ms a round, equal to serial: {}\n", .{ got.items.len - 1, rounds, @as(f64, @floatFromInt(got.items.len - 1)) / @as(f64, @floatFromInt(rounds)), @as(f64, @floatFromInt(got.items.len - 1)) / dt, t_draft * 1e3 / @as(f64, @floatFromInt(rounds)), t_verify * 1e3 / @as(f64, @floatFromInt(rounds)), same });
}

/// Can a prompt GEMM and the chunked GDN of another chunk share the card? Times each alone, chained, and unordered.
fn overlapProbe(r: *q.Runner) !void {
    const d = &r.d;
    const a = &r.acts;
    const T: u32 = q.CHUNK;
    const ordered = d.list;
    const free = try d.g.immediateUnordered(0);
    var gdn_layer: usize = 0;
    while (r.model.layers[gdn_layer].attn) gdn_layer += 1;
    const lg = &r.model.layers[gdn_layer];
    const l0 = &r.model.layers[0];
    const nc: i32 = @intCast(T / 64);
    const ncu: u32 = T / 64;
    const hk: i32 = 16;
    const hv: i32 = 48;
    const runA = struct {
        fn f(dd: *q.Dev, aa: *q.Acts, l: anytype, t: u32) !void {
            try q.gemm(dd, aa.pscr, aa.h, aa.ahi, aa.alo, l.gu, aa.p, t);
        }
    }.f;
    const runB = struct {
        fn f(dd: *q.Dev, aa: *q.Acts, l: anytype, ncv: i32, ncuv: u32, hkv: i32, hvv: i32) !void {
            try dd.launch(dd.k.gdn_prep, .{ .x = ncuv, .y = 48 }, .{ aa.k, aa.v, aa.g, aa.beta, aa.la, aa.wb, aa.u0, hkv, hvv, ncv, @as(i32, 0) });
            try dd.launch(dd.k.gdn_state, .{ .x = 48 }, .{ aa.k, aa.wb, aa.u0, aa.la, l.rec.?, l.rec.?, aa.hs, aa.vn, hkv, hvv, ncv });
            try dd.launch(dd.k.gdn_out, .{ .x = ncuv, .y = 48 }, .{ aa.q, aa.k, aa.la, aa.hs, aa.vn, aa.y, hkv, hvv, ncv });
        }
    }.f;
    var best = [5]f64{ 1e9, 1e9, 1e9, 1e9, 1e9 };
    for (0..6) |_| {
        for (0..5) |mode| {
            d.list = if (mode >= 3) free else ordered;
            try d.sync();
            const t0 = q.now();
            if (mode == 4) try runB(d, a, lg, nc, ncu, hk, hv);
            if (mode == 0 or mode >= 2) try runA(d, a, l0, T);
            if (mode >= 1 and mode <= 3) try runB(d, a, lg, nc, ncu, hk, hv);
            try d.sync();
            best[mode] = @min(best[mode], q.now() - t0);
        }
    }
    d.list = ordered;
    std.debug.print("overlap probe ({d} rows): GEMM gate|up {d:.2} ms, GDN chunk {d:.2} ms, chained {d:.2} ms, unordered A,B {d:.2} ms, unordered B,A {d:.2} ms\n", .{ T, best[0] * 1e3, best[1] * 1e3, best[2] * 1e3, best[3] * 1e3, best[4] * 1e3 });
}

/// Power study: the gate|up prompt GEMM of a zebin given by path, in a loop of CHUNK-row calls; time and TFLOPS.
fn gemmLoop(r: *q.Runner, path: []const u8) !void {
    const d = &r.d;
    const a = &r.acts;
    const fd = open(try std.fmt.allocPrintSentinel(alloc, "{s}", .{path}, 0), 0);
    if (fd < 0) return error.FileNotFound;
    const size: usize = @intCast(lseek(fd, 0, 2));
    const bin: [*]const u8 = @ptrCast(mmap(null, size, 1, 2, fd, 0) orelse return error.MmapFailed);
    const copy = try alloc.alignedAlloc(u8, .@"8", size);
    @memcpy(copy, bin[0..size]);
    const k = try d.kernel(try d.g.module(copy), "w4g_dpb", 256);
    const w = r.model.layers[0].gu;
    const T = q.CHUNK;
    const tiles = (T + 63) / 64;
    const args = .{ a.h, w.w, w.s, w.bg, a.p, @as(i32, @intCast(T)), @as(i32, @intCast(w.n)), @as(i32, @intCast(w.k)), a.ahi, a.alo, @as(i32, @intCast((T + 7) / 8 * 8)), @as(i32, @intCast(w.k / w.gs)) };
    for (0..20) |_| try d.launch(k, .{ .x = 8, .y = w.n / 256, .z = (tiles + 7) / 8 }, args);
    try d.sync();
    const t0 = q.now();
    for (0..4000) |_| try d.launch(k, .{ .x = 8, .y = w.n / 256, .z = (tiles + 7) / 8 }, args);
    try d.sync();
    const dt = (q.now() - t0) / 4000;
    const flops = 2.0 * @as(f64, @floatFromInt(T)) * @as(f64, @floatFromInt(w.n)) * @as(f64, @floatFromInt(w.k));
    std.debug.print("gemm loop {s}: {d:.3} ms a call, {d:.1} TFLOPS\n", .{ path, dt * 1e3, flops / dt / 1e12 });
}

/// The W4A8 gate|up GEMM of a zebin given by path in a loop of CHUNK-row calls: time and TOPS per call.
fn a8Loop(r: *q.Runner, path: []const u8) !void {
    const d = &r.d;
    const a = &r.acts;
    const fd = open(try std.fmt.allocPrintSentinel(alloc, "{s}", .{path}, 0), 0);
    if (fd < 0) return error.FileNotFound;
    const size: usize = @intCast(lseek(fd, 0, 2));
    const bin: [*]const u8 = @ptrCast(mmap(null, size, 1, 2, fd, 0) orelse return error.MmapFailed);
    const copy = try alloc.alignedAlloc(u8, .@"8", size);
    @memcpy(copy, bin[0..size]);
    const k = try d.kernel(try d.g.module(copy), "w4a8", 256);
    // XPU_A8_PROJ: gu (default), in (gdn.in), out (gdn.out), down
    const which = if (std.c.getenv("XPU_A8_PROJ")) |v| std.mem.span(v) else "gu";
    const l0 = &r.model.layers[0];
    const w = if (std.mem.eql(u8, which, "in")) l0.mixer_in else if (std.mem.eql(u8, which, "out")) l0.mixer_out else if (std.mem.eql(u8, which, "down")) l0.down else l0.gu;
    const T = q.CHUNK;
    const tiles = (T + 63) / 64;
    try d.launch(d.k.quant_a8, .{ .x = T }, .{ a.h, a.a8, a.sa, a.ahi, a.alo, @as(i32, @intCast(w.k)), @as(i32, @intCast(T + 32)), @as(i32, 0) });
    const args = .{ a.a8, w.w, w.s, w.bg, a.p, @as(i32, @intCast(T)), @as(i32, @intCast(w.n)), @as(i32, @intCast(w.k)), a.ahi, a.alo, @as(i32, @intCast(T + 32)), @as(i32, @intCast(w.k / 128)), a.sa };
    const groups: q.ze.GroupCount = .{ .x = 8, .y = w.n / 256, .z = (tiles + 7) / 8 };
    for (0..20) |_| try d.launch(k, groups, args);
    try d.sync();
    const t0 = q.now();
    for (0..2000) |_| try d.launch(k, groups, args);
    try d.sync();
    const dt = (q.now() - t0) / 2000;
    const ops = 2.0 * @as(f64, @floatFromInt(T)) * @as(f64, @floatFromInt(w.n)) * @as(f64, @floatFromInt(w.k));
    std.debug.print("a8 loop {s} {s} (N {d}, K {d}): {d:.3} ms a call, {d:.1} TOPS\n", .{ path, which, w.n, w.k, dt * 1e3, ops / dt / 1e12 });
}

/// Multi-stream estimate: the cost of one forward of T rows (1 to 16) as a verification block; the state is left as is.
fn rowsProbe(r: *q.Runner) !void {
    var ids: [16]u32 = undefined;
    for (&ids, 0..) |*x, i| x.* = @intCast(1000 + i);
    for ([_]u32{ 1, 2, 4, 8, 12, 16 }) |T| {
        var best: f64 = 1e9;
        for (0..12) |_| {
            const t0 = q.now();
            _ = try r.verify(ids[0..T]);
            best = @min(best, q.now() - t0);
            r.block_rows = 0;
        }
        std.debug.print("rows probe: {d:2} rows {d:6.2} ms ({d:5.1} rows/s at one row a stream)\n", .{ T, best * 1e3, @as(f64, @floatFromInt(T)) / best });
    }
}

/// Multi-stream contract: `n` slots at different positions decode serially, then in shared stepMulti rounds; equal.
fn multiTest(r: *q.Runner, prompt: []const u32, n: u32) !void {
    const steps = 24;
    var serial: [q.MAX_SLOTS][steps]u32 = undefined;
    for (0..n) |k| {
        try r.select(@intCast(k));
        try r.reset();
        try r.prompt(prompt[k % 8 ..], true);
        for (0..steps) |j| {
            serial[k][j] = r.last;
            try r.step(r.last);
        }
    }
    var shared: [q.MAX_SLOTS][steps]u32 = undefined;
    var slots: [q.MAX_SLOTS]u32 = undefined;
    var toks: [q.MAX_SLOTS]u32 = undefined;
    for (0..n) |k| {
        try r.select(@intCast(k));
        try r.reset();
        try r.prompt(prompt[k % 8 ..], true);
        slots[k] = @intCast(k);
    }
    for (0..steps) |j| {
        for (0..n) |k| {
            toks[k] = r.seqs[k].last;
            if (k == r.slot) toks[k] = r.last;
            shared[k][j] = toks[k];
        }
        try r.stepMulti(slots[0..n], toks[0..n]);
    }
    var equal = true;
    for (0..n) |k| equal = equal and std.mem.eql(u32, &serial[k], &shared[k]);
    std.debug.print("multi test: {d} slots, {d} tokens each, shared rounds equal to serial: {}\n", .{ n, steps, equal });
    if (!equal) for (0..n) |k| {
        std.debug.print("  slot {d} serial:", .{k});
        for (serial[k]) |t| std.debug.print(" {d}", .{t});
        std.debug.print("\n  slot {d} shared:", .{k});
        for (shared[k]) |t| std.debug.print(" {d}", .{t});
        std.debug.print("\n", .{});
    };
    // shared windows of 4 rows: perfect drafts (the serial reply) for even slots, wrong ones for odd slots
    var got: [q.MAX_SLOTS][steps + 8]u32 = undefined;
    var have: [q.MAX_SLOTS]usize = @splat(0);
    for (0..n) |k| {
        try r.select(@intCast(k));
        try r.reset();
        try r.prompt(prompt[k % 8 ..], true);
    }
    var rounds: usize = 0;
    while (true) {
        var wins: [16]q.Runner.Win = undefined;
        var ids: [16][4]u32 = undefined;
        var nw: usize = 0;
        var rows: usize = 0;
        for (0..n) |k| {
            if (have[k] >= steps or rows + 4 > 16) continue;
            const pending = if (k == r.slot) r.last else r.seqs[k].last;
            ids[k][0] = pending;
            for (1..4) |j| ids[k][j] = if (k % 2 == 0 and have[k] + j < steps) serial[k][have[k] + j] else 0;
            wins[nw] = .{ .slot = @intCast(k), .ids = &ids[k] };
            nw += 1;
            rows += 4;
        }
        if (nw == 0) break;
        try r.verifyMulti(wins[0..nw]);
        var keep: [16]u32 = undefined;
        var row: usize = 0;
        for (wins[0..nw], 0..) |w, i| {
            var a: u32 = 0; // accepted drafts
            while (a + 1 < 4 and r.block[row + a] == w.ids[a + 1]) a += 1;
            keep[i] = a + 1;
            const k = w.slot;
            for (w.ids[0 .. a + 1]) |t| {
                got[k][have[k]] = t;
                have[k] += 1;
            }
            row += 4;
        }
        try r.keepMulti(keep[0..nw]);
        rounds += 1;
    }
    var equal4 = true;
    for (0..n) |k| equal4 = equal4 and std.mem.eql(u32, got[k][0..steps], &serial[k]);
    std.debug.print("multi test: windows of 4 rows (perfect / wrong drafts), {d} rounds, equal to serial: {}\n", .{ rounds, equal4 });
    if (!equal4) for (0..n) |k| {
        const at = std.mem.indexOfDiff(u32, got[k][0..steps], &serial[k]);
        if (at) |x| std.debug.print("  slot {d} differs at token {d}\n", .{ k, x });
    };
    // shared tree windows of 7 rows (parents -1 0 0 2 1 3 3); the walk keeps a root path and keepPaths replays it
    @memset(&have, 0);
    for (0..n) |k| {
        try r.select(@intCast(k));
        try r.reset();
        try r.prompt(prompt[k % 8 ..], true);
    }
    const par = [7]i32{ -1, 0, 0, 2, 1, 3, 3 };
    var trounds: usize = 0;
    while (true) {
        var wins: [16]q.Runner.Win = undefined;
        var tids: [16][7]u32 = undefined;
        var nw: usize = 0;
        var rows: usize = 0;
        for (0..n) |k| {
            if (have[k] >= steps or rows + 7 > 16) continue;
            const x = &tids[k];
            x.* = .{ if (k == r.slot) r.last else r.seqs[k].last, 7, 0, 0, 11, 0, 13 };
            const at = have[k];
            const sv = struct {
                fn f(row: []const u32, i: usize) u32 {
                    return if (i < row.len) row[i] else 0;
                }
            }.f;
            if (k % 2 == 0) {
                x[2] = sv(&serial[k], at + 1);
                x[3] = sv(&serial[k], at + 2);
                x[5] = sv(&serial[k], at + 3);
            } else x[1] = sv(&serial[k], at + 1);
            wins[nw] = .{ .slot = @intCast(k), .ids = x, .parents = &par };
            nw += 1;
            rows += 7;
        }
        if (nw == 0) break;
        try r.verifyMulti(wins[0..nw]);
        var pbuf: [16][7]u32 = undefined;
        var paths: [16][]const u32 = undefined;
        var row0: usize = 0;
        for (wins[0..nw], 0..) |w, i| {
            var len: usize = 1;
            pbuf[i][0] = 0;
            walk: while (true) {
                const cur = pbuf[i][len - 1];
                for (1..7) |c| if (par[c] == @as(i32, @intCast(cur)) and w.ids[c] == r.block[row0 + cur]) {
                    pbuf[i][len] = @intCast(c);
                    len += 1;
                    continue :walk;
                };
                break;
            }
            paths[i] = pbuf[i][0..len];
            const k = w.slot;
            for (paths[i]) |c| if (have[k] < steps + 8) {
                got[k][have[k]] = w.ids[c];
                have[k] += 1;
            };
            row0 += 7;
        }
        try r.keepPaths(paths[0..nw]);
        trounds += 1;
    }
    var equalt = true;
    for (0..n) |k| equalt = equalt and std.mem.eql(u32, got[k][0..steps], &serial[k]);
    std.debug.print("multi test: tree windows of 7 rows, {d} rounds, equal to serial: {}\n", .{ trounds, equalt });
    if (!equalt) for (0..n) |k| {
        const at = std.mem.indexOfDiff(u32, got[k][0..steps], &serial[k]);
        if (at) |x| std.debug.print("  slot {d} differs at token {d}\n", .{ k, x });
    };
    var m: u32 = 1;
    while (m <= n) : (m *= 2) {
        var best: f64 = 1e9;
        for (0..8) |_| {
            for (0..m) |k| toks[k] = r.seqs[k].last;
            toks[r.slot] = r.last;
            const t0 = q.now();
            try r.stepMulti(slots[0..m], toks[0..m]);
            best = @min(best, q.now() - t0);
        }
        std.debug.print("multi rounds: {d:2} slots {d:6.2} ms a round, {d:6.1} tok/s together\n", .{ m, best * 1e3, @as(f64, @floatFromInt(m)) / best });
    }
}

/// EXL3 speed probe (no weights, random trellis bits): the mul1 decode GEMV at 4, 3 and 2 bits against our int4 GEMM.
fn exl3Probe(r: *q.Runner) !void {
    const d = &r.d;
    const a = &r.acts;
    const l0 = &r.model.layers[0];
    for ([_]q.Proj{ l0.gu, l0.down }) |w| {
        const K = w.k;
        const N = if (w.ilv) w.n else w.n;
        for ([_]u32{ 1, 8 }) |T| {
            var best: f64 = 1e9;
            for (0..200) |it| {
                const t0 = q.now();
                try q.gemm(d, a.pscr, a.h, a.ahi, a.alo, w, a.p, T);
                try d.sync();
                if (it > 10) best = @min(best, q.now() - t0);
            }
            const bytes4 = @as(f64, @floatFromInt(K)) * @as(f64, @floatFromInt(N)) / 2.0;
            std.debug.print("int4 (ours)  K {d:5} N {d:5} rows {d:2}: {d:6.3} ms, {d:5.0} GB/s\n", .{ K, N, T, best * 1e3, bytes4 / best / 1e9 });
            for ([_]u32{ 4, 3, 2 }) |bits| {
                const bytes: usize = @as(usize, K / 16) * (N / 16) * 16 * bits * 2;
                const tr = try d.alloc_(bytes);
                const host = try alloc.alloc(u8, bytes);
                defer alloc.free(host);
                var prng = std.Random.DefaultPrng.init(bits);
                prng.random().bytes(host);
                try d.copyIn(tr, host);
                const ks: u32 = if (N / 16 >= 2048) 1 else @max(1, 2048 / (N / 16));
                var kslices = ks;
                while ((K / 16) % kslices != 0) kslices -= 1;
                var b2: f64 = 1e9;
                for (0..100) |it| {
                    const t0 = q.now();
                    try d.launch(d.k.exl3_gemv, .{ .x = N / 16, .y = kslices }, .{ tr, a.h, a.pscr, @as(i32, @intCast(K)), @as(i32, @intCast(N)), @as(i32, @intCast(T)), @as(i32, @intCast(bits)), @as(i32, @intCast(kslices)) });
                    try d.sync();
                    if (it > 5) b2 = @min(b2, q.now() - t0);
                }
                std.debug.print("exl3 {d} bits  K {d:5} N {d:5} rows {d:2}: {d:6.3} ms, {d:5.0} GB/s ({d} K slices)\n", .{ bits, K, N, T, b2 * 1e3, @as(f64, @floatFromInt(bytes)) / b2 / 1e9, kslices });
                d.g.free(tr);
            }
        }
    }
}
