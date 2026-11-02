//! qwen35's pending kept rows: the slot scans and tf_q35k_apply_pend bit-equal to tf_q35k_apply, then the scan.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const dk = 128;
const nk = 16; // the 27B's linear key heads
const nv = 48; // and value heads
const nl = 2; // linear layers: the scans run on each, so the second one's offsets count
const taps = 4;
const most = 64; // a window's rows at most, and the log's rows (qwen35.cuh MAXW)
const rec = nv * dk * dk; // floats of one layer's recurrence
const head = nv * 4 + nv * dk * 4 + nk * dk * 2; // a row's decay, write and key in one layer (qwen35.cuh pend_row)
const log_row = head + (taps - 1) * (2 * nk + nv) * dk * 2; // the same with its conv tail (Geo.log_bytes)

// qwen35.cuh's Geo, Seg and Path (contract_qwen_pages.zig mirrors them too); SlotStep is core's tree_slots.Step
const Geo = extern struct { hid: c_int, nk: c_int, nv: c_int, qh: c_int, kvh: c_int, nl: c_int };
const Seg = extern struct { rec: u64 = 0, conv: u64 = 0, kv: u64 = 0, tap: u64 = 0, log: u64 = 0, before: u64 = 0, cap: c_int = 0, len: c_int = 0, row0: c_int = 0, rows: c_int = 0, tap_rows: c_int = 0, taps: c_int = 0, delta: c_int = 0, pt: u64 = 0, ps: u64 = 0, ptok: c_int = 0, pn: c_int = 0, plog: u64 = 0 };
const Path = extern struct { n: c_int, rows: [most]c_int };
const Step = core.tree_slots.Step;
const geo: Geo = .{ .hid = 0, .nk = nk, .nv = nv, .qh = 0, .kvh = 0, .nl = nl };

fn chain(comptime n: usize) [n]i32 {
    var p: [n]i32 = undefined;
    for (&p, 0..) |*x, i| x.* = @as(i32, @intCast(i)) - 1;
    return p;
}

/// A drafted tree: each true row's children are a junk leaf and the next true row.
fn drafted(comptime n: usize) [n]i32 {
    var p: [n]i32 = undefined;
    for (&p, 0..) |*x, i| x.* = if (i == 0) -1 else @intCast((i - 1) / 2 * 2);
    return p;
}

/// A binary tree: row i's parent is (i - 1) / 2, so more register slots hold states at once.
fn binary(comptime n: usize) [n]i32 {
    var p: [n]i32 = undefined;
    for (&p, 0..) |*x, i| x.* = if (i == 0) -1 else @intCast((i - 1) / 2);
    return p;
}

const decode_row = chain(1);
const chain4 = chain(4);
const drafted16 = drafted(16);
const binary16 = binary(16);
const chain64 = chain(most);
const Window = struct { what: []const u8, parents: []const i32 };
const windows = [_]Window{
    .{ .what = "a decode row", .parents = &decode_row },
    .{ .what = "a 4-row chain", .parents = &chain4 },
    .{ .what = "a 16-row drafted tree", .parents = &drafted16 },
    .{ .what = "a 16-row binary tree", .parents = &binary16 },
    .{ .what = "a 64-row chain", .parents = &chain64 },
};

/// Pending rows: one, the scan's 8-row load batches around 8, and cuda_cache.zig's pend_commit and pend_cap.
const counts = [_]usize{ 1, 7, 8, 9, 24, 48 };

/// qwen35_slots.cuh's scans: register slots, and value rows a warp (cuda_engine.zig deltaScan's grid).
const Scan = struct { slots: usize, r: usize };
const scans = [_]Scan{ .{ .slots = 1, .r = 2 }, .{ .slots = 2, .r = 2 }, .{ .slots = 4, .r = 2 }, .{ .slots = 8, .r = 1 }, .{ .slots = 16, .r = 1 } };

fn bf16(x: f32) u16 {
    return @truncate(@as(u32, @bitCast(x)) >> 16);
}

/// bf16 in [-1, 1).
fn randoms(a: std.mem.Allocator, rng: std.Random, n: usize) ![]u16 {
    const out = try a.alloc(u16, n);
    for (out) |*x| x.* = bf16(rng.float(f32) * 2 - 1);
    return out;
}

/// One case's host data: two streams' states, the logs their pending rows were kept from, and a window each.
const Fixture = struct {
    rows: usize, // a stream's window rows
    n: [2]usize, // pending rows a stream
    state: [2][]f32, // nl layers' recurrences
    src: [2][]u8, // `most` rows of nl layers' log records
    path: [2]Path, // the kept rows, in order
    pend: [2][]u8, // their heads in pend_row's layout, as Cache.pendRows copies them
    q: []u16,
    k: []u16,
    v: []u16,
    beta: []u16,
    g: []f32,
    steps: []Step,
    peak: usize, // register slots the steps hold at once

    fn make(a: std.mem.Allocator, parents: []const i32, n: usize, seed: u64) !Fixture {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const rows = parents.len;
        var f: Fixture = undefined;
        f.rows = rows;
        f.n = .{ n, if (n % 3 == 0) 0 else n * 7 % 11 }; // the second stream: none pending, or another count
        for (0..2) |st| {
            f.state[st] = try a.alloc(f32, nl * rec);
            for (f.state[st]) |*x| x.* = rng.float(f32) * 2 - 1;
            f.src[st] = try a.alloc(u8, most * nl * log_row);
            @memset(f.src[st], 0);
            for (0..most * nl) |i| {
                const r = f.src[st][i * log_row ..][0..head];
                for (std.mem.bytesAsSlice(f32, r[0 .. nv * 4 + nv * dk * 4]), 0..) |*x, j| x.* = if (j < nv) 0.9 + rng.float(f32) / 10 else rng.float(f32) * 2 - 1;
                for (std.mem.bytesAsSlice(u16, r[nv * 4 + nv * dk * 4 ..])) |*x| x.* = bf16(rng.float(f32) * 2 - 1);
            }
            f.path[st] = .{ .n = @intCast(f.n[st]), .rows = @splat(0) };
            for (0..f.n[st]) |j| f.path[st].rows[j] = @intCast((j * 37 + st * 11) % most);
            f.pend[st] = try a.alloc(u8, f.n[st] * nl * head);
            for (0..f.n[st]) |j| for (0..nl) |l| {
                const at = (@as(usize, @intCast(f.path[st].rows[j])) * nl + l) * log_row;
                @memcpy(f.pend[st][(j * nl + l) * head ..][0..head], f.src[st][at..][0..head]);
            };
        }
        const total = 2 * rows;
        f.q = try randoms(a, rng, total * nk * dk);
        f.k = try randoms(a, rng, total * nk * dk);
        f.v = try randoms(a, rng, total * nv * dk);
        f.beta = try a.alloc(u16, total * nv);
        for (f.beta) |*x| x.* = bf16(rng.float(f32));
        f.g = try a.alloc(f32, total * nv);
        for (f.g) |*x| x.* = 0.9 + rng.float(f32) / 10;
        f.steps = try a.alloc(Step, total);
        f.peak = 0;
        for (0..2) |st| f.peak = @max(f.peak, try core.tree_slots.build(a, parents, f.steps[st * rows ..][0..rows], @intCast(st * rows), core.tree_slots.max_slots));
        return f;
    }
};

/// Pending rows i and j of a buffer swapped, every layer's heads.
fn swapRows(pend: []u8, i: usize, j: usize) void {
    for (pend[i * nl * head ..][0 .. nl * head], pend[j * nl * head ..][0 .. nl * head]) |*x, *y| std.mem.swap(u8, x, y);
}

/// One launch set's bytes: each layer's rows out, and each stream's state, window log and pre-round state.
const Out = struct { y: [nl][]const u8, state: [2][]const u8, log: [2][]const u8, before: [2][]const u8 };

/// The scan on every layer after tf_q35k_apply (`apply` given), or with the rows pending (`sign` > 0: written back).
fn launch(gpu: check.Gpu, s: cuda.Stream, apply: ?cuda.Function, scan: cuda.Function, r: usize, f: Fixture, sign: c_int, pend0: []const u8, drop: usize, a: std.mem.Allocator) !Out {
    var rig = rig_.Rig{ .gpu = gpu, .s = s };
    defer rig.deinit();
    const total = 2 * f.rows;
    var segs: [2]Seg = undefined;
    var src: [2]u64 = .{ 0, 0 };
    for (0..2, &segs) |st, *sg| {
        const state = try rig.scratch(nl * rec * 4);
        try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = state, .len = nl * rec * 4 }).upload(0, std.mem.sliceAsBytes(f.state[st]));
        sg.* = .{ .rec = state, .log = try rig.scratch(f.rows * nl * log_row), .row0 = @intCast(st * f.rows), .rows = @intCast(f.rows) };
        if (f.rows == 1) sg.before = try rig.scratch(nl * rec * 4);
        const n = f.n[st] - (if (st == 0) drop else 0);
        if (apply != null) {
            if (f.n[st] > 0) src[st] = try rig.input(f.src[st]);
        } else if (n > 0) {
            sg.pn = sign * @as(c_int, @intCast(n));
            sg.plog = try rig.input(if (st == 0) pend0 else f.pend[st]);
        }
    }
    const params = [5]c_int{ 0, 0, @intCast(total), 2, 1 }; // P: the forward's rows, its windows, recorded
    const segs_d = try rig.input(std.mem.sliceAsBytes(&segs));
    const q = try rig.input(std.mem.sliceAsBytes(f.q));
    const k = try rig.input(std.mem.sliceAsBytes(f.k));
    const v = try rig.input(std.mem.sliceAsBytes(f.v));
    const g = try rig.input(std.mem.sliceAsBytes(f.g));
    const beta = try rig.input(std.mem.sliceAsBytes(f.beta));
    const steps = try rig.input(std.mem.sliceAsBytes(f.steps));
    const p = try rig.input(std.mem.sliceAsBytes(&params));
    var y: [nl]u64 = undefined;
    for (&y) |*x| x.* = try rig.output(total * nv * dk * 2);
    const seen = try a.dupe(u8, try rig.inputBytes());
    if (apply) |ap| {
        for (0..2) |st| if (src[st] != 0) {
            var args: cuda.Args = .{};
            inline for (.{ geo, Seg{ .rec = segs[st].rec, .log = src[st] }, f.path[st] }) |x| args.add(x);
            try cuda.launch.launch(ap, .{ .grid = .{ .x = 1, .y = dk / 4, .z = nv * nl }, .block = .{ .x = 32, .y = 4 } }, s, &args);
        };
    }
    for (0..nl) |layer| {
        var args: cuda.Args = .{};
        inline for (.{ geo, q, k, v, g, beta, segs_d, steps, @as(c_int, @intCast(layer)), y[layer], p }) |x| args.add(x);
        try cuda.launch.launch(scan, .{ .grid = .{ .x = 1, .y = @intCast(dk / (4 * r)), .z = nv * 2 }, .block = .{ .x = 32, .y = 4 } }, s, &args);
    }
    var out: Out = undefined;
    for (0..nl) |l| out.y[l] = try a.dupe(u8, try rig.bytes(y[l], total * nv * dk * 2));
    for (0..2) |st| {
        out.state[st] = try a.dupe(u8, try rig.bytes(segs[st].rec, nl * rec * 4));
        out.log[st] = try a.dupe(u8, try rig.bytes(segs[st].log, f.rows * nl * log_row));
        out.before[st] = if (f.rows == 1) try a.dupe(u8, try rig.bytes(segs[st].before, nl * rec * 4)) else &.{};
    }
    try check.expect(try rig.intact() and !rig.overlap() and std.mem.eql(u8, seen, try rig.inputBytes()), "qwen35 pending: a launch changed an input or a guard", .{});
    return out;
}

/// The scan with rows pending against tf_q35k_apply then the scan: rows out, window logs, pre-round state, states.
fn same(f: Fixture, sign: c_int, got: Out, want: Out) !void {
    for (got.y, want.y) |x, y| try check.sameBytes("qwen35 pending: rows out", x, y);
    for (0..2) |st| {
        try check.sameBytes("qwen35 pending: window log", got.log[st], want.log[st]);
        try check.sameBytes("qwen35 pending: pre-round state", got.before[st], want.before[st]);
        // written back: the state tf_q35k_apply left; only replayed: the state as it was
        try check.sameBytes("qwen35 pending: state", got.state[st], if (sign > 0) want.state[st] else std.mem.sliceAsBytes(f.state[st]));
    }
}

/// Stream 0's state after tf_q35k_apply on its path, or tf_q35k_apply_pend on its pending rows (`drop` short).
fn landed(gpu: check.Gpu, s: cuda.Stream, fun: cuda.Function, from_pend: bool, f: Fixture, pend: []const u8, drop: usize, a: std.mem.Allocator) ![]u8 {
    var rig = rig_.Rig{ .gpu = gpu, .s = s };
    defer rig.deinit();
    const state = try rig.scratch(nl * rec * 4);
    try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = state, .len = nl * rec * 4 }).upload(0, std.mem.sliceAsBytes(f.state[0]));
    const n = f.n[0] - drop;
    const rows = if (from_pend) (if (n > 0) try rig.input(pend) else 0) else try rig.input(f.src[0]);
    const seen = try a.dupe(u8, try rig.inputBytes());
    var args: cuda.Args = .{};
    if (from_pend) {
        inline for (.{ geo, Seg{ .rec = state, .plog = rows }, @as(c_int, @intCast(n)) }) |x| args.add(x);
    } else inline for (.{ geo, Seg{ .rec = state, .log = rows }, f.path[0] }) |x| args.add(x);
    try cuda.launch.launch(fun, .{ .grid = .{ .x = 1, .y = dk / 4, .z = nv * nl }, .block = .{ .x = 32, .y = 4 } }, s, &args);
    const out = try a.dupe(u8, try rig.bytes(state, nl * rec * 4));
    try check.expect(try rig.intact() and !rig.overlap() and std.mem.eql(u8, seen, try rig.inputBytes()), "qwen35 pending: a keep launch changed an input or a guard", .{});
    return out;
}

pub fn run(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.qwen35);
    defer mod.unload();
    var keep = try cuda.Module.load(gpu.d, cuda.kernels.qwen35_keep);
    defer keep.unload();
    const apply = try keep.function("tf_q35k_apply");
    const apply_pend = try keep.function("tf_q35k_apply_pend");
    var fns: [scans.len]cuda.Function = undefined;
    inline for (scans, 0..) |sc, i| fns[i] = try mod.function(std.fmt.comptimePrint("tf_q35_gdn_slots_{d}", .{sc.slots}));
    var red: [scans.len + 1][2]bool = @splat(@splat(false)); // each scan's, then tf_q35k_apply_pend's: dropped, swapped
    var cases: [scans.len]usize = @splat(0);
    var scratch = std.heap.ArenaAllocator.init(gpu.gpa);
    defer scratch.deinit();
    for (counts, 0..) |n, ci| {
        // tf_q35k_apply_pend: the rows land as tf_q35k_apply leaves the path (the window plays no part)
        {
            var arena = std.heap.ArenaAllocator.init(gpu.gpa);
            defer arena.deinit();
            const a = arena.allocator();
            const f = try Fixture.make(a, &decode_row, n, 0x9e1d + ci);
            const want = try landed(gpu, s, apply, false, f, f.pend[0], 0, a);
            for (0..5) |_| try check.sameBytes("qwen35 pending: tf_q35k_apply_pend against tf_q35k_apply", try landed(gpu, s, apply_pend, true, f, f.pend[0], 0, a), want);
            const swapped = try a.dupe(u8, f.pend[0]);
            if (n >= 2) swapRows(swapped, n - 2, n - 1);
            for (0..2) |c| if (c == 0 or n >= 2) {
                const moved = !std.mem.eql(u8, try landed(gpu, s, apply_pend, true, f, if (c == 0) f.pend[0] else swapped, 1 - c, a), want);
                red[scans.len][c] = red[scans.len][c] or moved;
                try check.expect(moved, "qwen35 pending tf_q35k_apply_pend: {s} still landed tf_q35k_apply's state (control missed) at {d} rows", .{ if (c == 0) "a dropped row" else "two swapped rows", n });
            };
            std.debug.print("PASS qwen35 pending tf_q35k_apply_pend: {d} rows equal tf_q35k_apply's path, 5 repeats, guards\n", .{n});
        }
        for (windows, 0..) |win, wi| {
            var arena = std.heap.ArenaAllocator.init(gpu.gpa);
            defer arena.deinit();
            const a = arena.allocator();
            const f = try Fixture.make(a, win.parents, n, 0x5ca9 + 64 * ci + wi);
            const swapped = try a.dupe(u8, f.pend[0]);
            if (n >= 2) swapRows(swapped, n - 2, n - 1);
            var ran: usize = 0;
            for (scans, fns, 0..) |sc, scan, si| {
                if (sc.slots < f.peak) continue;
                const want = try launch(gpu, s, apply, scan, sc.r, f, 0, f.pend[0], 0, a);
                for ([_]c_int{ 1, -1 }) |sign| {
                    if (sign < 0 and f.rows == 1) continue; // a one-row window always writes its state back
                    for (0..5) |_| {
                        defer _ = scratch.reset(.retain_capacity);
                        const got = try launch(gpu, s, null, scan, sc.r, f, sign, f.pend[0], 0, scratch.allocator());
                        same(f, sign, got, want) catch |err| {
                            std.debug.print("  at tf_q35_gdn_slots_{d}, {s}, {d} rows pending, sign {d}\n", .{ sc.slots, win.what, n, sign });
                            return err;
                        };
                    }
                    // controls: stream 0's last pending row dropped, then its last two swapped
                    for (0..2) |c| if (c == 0 or n >= 2) {
                        defer _ = scratch.reset(.retain_capacity);
                        const bad = try launch(gpu, s, null, scan, sc.r, f, sign, if (c == 0) f.pend[0] else swapped, 1 - c, scratch.allocator());
                        const moved = !std.mem.eql(u8, bad.y[0], want.y[0]) or !std.mem.eql(u8, bad.y[1], want.y[1]);
                        red[si][c] = red[si][c] or moved;
                        try check.expect(moved, "qwen35 pending tf_q35_gdn_slots_{d}: {s} still gave the bytes of tf_q35k_apply then the scan (control missed) at {s}, {d} rows", .{ sc.slots, if (c == 0) "a dropped row" else "two swapped rows", win.what, n });
                    };
                }
                cases[si] += 1;
                ran += 1;
            }
            std.debug.print("PASS qwen35 pending: {s}, {d} rows pending, {d} scans (slot peak {d}): rows out, logs and states equal tf_q35k_apply then the scan, {s}, 5 repeats, guards\n", .{ win.what, n, ran, f.peak, if (f.rows == 1) "written back" else "written back and not" });
        }
    }
    for (scans, cases, 0..) |sc, nc, si| try check.expect(nc > 0 and red[si][0] and red[si][1], "qwen35 pending tf_q35_gdn_slots_{d}: no case, or a control never fired", .{sc.slots});
    try check.expect(red[scans.len][0] and red[scans.len][1], "qwen35 pending tf_q35k_apply_pend: a control never fired", .{});
    std.debug.print("PASS qwen35 pending: controls RED as planted (a dropped pending row, two swapped) on every scan and tf_q35k_apply_pend\n", .{});
}
