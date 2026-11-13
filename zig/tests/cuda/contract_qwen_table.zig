//! qwen35's gdn_pre, conv_commit, gdn_chain and tf_q35k_conv: _table on pieces == main's on blocks, red controls.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const dk = 128;
const nk = 16; // the 27B's linear key heads
const nv = 48; // and value heads
const nl = 2; // linear layers: every launch runs layer 1, so a table entry's layer counts
const layer = 1;
const taps = 4;
const most = 64; // qwen35.cuh MAXW: a table's log rows
const logs = 8; // log rows the fixture holds (its windows' rows at most)
const width = (2 * nk + nv) * dk; // Geo.conv(): the channels the convolution sees
const tail = (taps - 1) * width; // Geo.conv_rows(): a layer's conv tail
const rec = nv * dk * dk; // floats of one layer's recurrence
const log_bytes = nv * 4 + nv * dk * 4 + nk * dk * 2 + tail * 2; // Geo.log_bytes()
const streams = 2; // a forward's segments: the plants sit in segment 1's table, so its offset counts
const seg_states = 232480; // qwen35.cuh SEG_STATES
const entries = 4 * nl + most; // cuda_cache.zig States: recurrences, tails, both copies, log rows

// qwen35.cuh's Geo and Seg (contract_qwen_pending.zig mirrors them too)
const Geo = extern struct { hid: c_int, nk: c_int, nv: c_int, qh: c_int, kvh: c_int, nl: c_int };
const Seg = extern struct { rec: u64 = 0, conv: u64 = 0, kv: u64 = 0, tap: u64 = 0, log: u64 = 0, before: u64 = 0, cap: c_int = 0, len: c_int = 0, row0: c_int = 0, rows: c_int = 0, tap_rows: c_int = 0, taps: c_int = 0, delta: c_int = 0, pt: u64 = 0, ps: u64 = 0, ptok: c_int = 0, pn: c_int = 0, plog: u64 = 0 };
const geo: Geo = .{ .hid = 0, .nk = nk, .nv = nv, .qh = 0, .kvh = 0, .nl = nl };

// A stream's state in main's block layout: recurrences, tails, pre-round recurrences then tails, log rows.
const at_conv = nl * rec * 4;
const at_before = at_conv + nl * tail * 2;
const at_log = at_before + nl * rec * 4 + nl * tail * 2;
const state_bytes = at_log + logs * nl * log_bytes;

const Kernel = enum { pre, commit, chain, conv };
const names = [_][:0]const u8{ "tf_q35_gdn_pre", "tf_q35_conv_commit", "tf_q35_gdn_chain", "tf_q35k_conv" };
const tables = [_][:0]const u8{ "tf_q35_gdn_pre_table", "tf_q35_conv_commit_table", "tf_q35_gdn_chain_table", "tf_q35k_conv_table" };
/// Each kernel's planted fault in segment 1's table: layer 1's entry pointing at layer 0's piece, or another log row.
const plants = [_][]const u8{ "the tail entry", "the tail entry", "the recurrence entry", "the log row entry" };

const decode_row = [_]i32{-1};
const chain4 = [_]i32{ -1, 0, 1, 2 };
const tree8 = [_]i32{ -1, 0, 0, 2, 2, 4, 4, 6 }; // a drafted tree: a true row has a junk leaf and the next
const Window = struct { what: []const u8, parents: []const i32 };
const windows = [_]Window{
    .{ .what = "a decode row", .parents = &decode_row },
    .{ .what = "a 4-row chain", .parents = &chain4 },
    .{ .what = "an 8-row drafted tree", .parents = &tree8 },
};

fn bf16(x: f32) u16 {
    return @truncate(@as(u32, @bitCast(x)) >> 16);
}

fn fillBf16(rng: std.Random, out: []align(1) u16) void {
    for (out) |*x| x.* = bf16(rng.float(f32) * 2 - 1);
}

/// A stream's starting state: random recurrences, tails and copies; log rows with decays near 1.
fn startState(a: std.mem.Allocator, rng: std.Random) ![]u8 {
    const s = try a.alignedAlloc(u8, .@"8", state_bytes);
    for (std.mem.bytesAsSlice(f32, s[0..at_conv])) |*x| x.* = rng.float(f32) * 2 - 1;
    fillBf16(rng, std.mem.bytesAsSlice(u16, s[at_conv..at_before]));
    for (std.mem.bytesAsSlice(f32, s[at_before..][0 .. nl * rec * 4])) |*x| x.* = rng.float(f32) * 2 - 1;
    fillBf16(rng, std.mem.bytesAsSlice(u16, s[at_before + nl * rec * 4 .. at_log]));
    for (0..logs * nl) |i| {
        const r = s[at_log + i * log_bytes ..][0..log_bytes];
        for (std.mem.bytesAsSlice(f32, r[0 .. nv * 4 + nv * dk * 4]), 0..) |*x, j| x.* = if (j < nv) 0.9 + rng.float(f32) / 10 else rng.float(f32) * 2 - 1;
        fillBf16(rng, std.mem.bytesAsSlice(u16, r[nv * 4 + nv * dk * 4 ..]));
    }
    return s;
}

/// One device allocation holding bytes [off, off + len) of a stream's block-layout state.
const Piece = struct { off: usize, len: usize, ptr: u64 };

/// A stream's state on the device: main's four blocks, or one piece a layer's region and a log row (in reverse).
const Placed = struct {
    pieces: [4 * nl + logs]Piece = undefined,
    n: usize = 0,
    seg: Seg = .{}, // main: the blocks
    table: [entries]u64 = @splat(0), // _table: every piece's address

    fn add(p: *Placed, rig: *rig_.Rig, state: []const u8, off: usize, len: usize) !u64 {
        const ptr = try rig.scratch(len);
        try (cuda.DeviceBuffer{ .d = rig.gpu.d, .ptr = ptr, .len = len }).upload(0, state[off..][0..len]);
        p.pieces[p.n] = .{ .off = off, .len = len, .ptr = ptr };
        p.n += 1;
        return ptr;
    }

    fn make(rig: *rig_.Rig, state: []const u8, tab: bool) !Placed {
        var p: Placed = .{};
        if (!tab) {
            p.seg.rec = try p.add(rig, state, 0, at_conv);
            p.seg.conv = try p.add(rig, state, at_conv, nl * tail * 2);
            p.seg.before = try p.add(rig, state, at_before, at_log - at_before);
            p.seg.log = try p.add(rig, state, at_log, logs * nl * log_bytes);
            return p;
        }
        var l: usize = nl;
        while (l > 0) {
            l -= 1;
            p.table[l] = try p.add(rig, state, l * rec * 4, rec * 4);
            p.table[nl + l] = try p.add(rig, state, at_conv + l * tail * 2, tail * 2);
            p.table[2 * nl + l] = try p.add(rig, state, at_before + l * rec * 4, rec * 4);
            p.table[3 * nl + l] = try p.add(rig, state, at_before + nl * rec * 4 + l * tail * 2, tail * 2);
        }
        var r: usize = logs;
        while (r > 0) {
            r -= 1;
            p.table[4 * nl + r] = try p.add(rig, state, at_log + r * nl * log_bytes, nl * log_bytes);
        }
        return p;
    }

    /// The state back in block layout.
    fn gather(p: Placed, rig: *rig_.Rig, a: std.mem.Allocator) ![]u8 {
        const out = try a.alloc(u8, state_bytes);
        @memset(out, 0);
        for (p.pieces[0..p.n]) |pc| @memcpy(out[pc.off..][0..pc.len], try rig.bytes(pc.ptr, pc.len));
        return out;
    }
};

/// The planted fault in a table: `k`'s entry of layer 1 now names layer 0's piece, or log row `last` another row.
fn plant(t: *[entries]u64, k: Kernel, last: usize) void {
    switch (k) {
        .pre, .commit => t[nl + layer] = t[nl],
        .chain => t[layer] = t[0],
        .conv => t[4 * nl + last] = t[4 * nl + (last + 1) % logs],
    }
}

/// One case's inputs: a window per stream, the forward's rows and the inputs its launches read.
const Case = struct {
    rows: usize,
    record: bool,
    last: usize, // tf_q35k_conv's kept row
    rowseg: []i32,
    rowpar: []i32,
    state: [streams][]u8,
    qkv: []u16,
    cw: []u16,
    ain: []u16,
    bin: []u16,
    alog: []f32,
    dt: []u16,
    q: []u16,
    k: []u16,
    v: []u16,
    g: []f32,
    beta: []u16,

    fn make(a: std.mem.Allocator, parents: []const i32, record: bool, last: usize, seed: u64) !Case {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const rows = parents.len;
        const total = streams * rows;
        var c: Case = undefined;
        c.rows = rows;
        c.record = record;
        c.last = last;
        c.rowseg = try a.alloc(i32, total);
        c.rowpar = try a.alloc(i32, total);
        for (0..streams) |st| for (parents, 0..) |p, i| {
            c.rowseg[st * rows + i] = @intCast(st);
            c.rowpar[st * rows + i] = if (p < 0) -1 else @as(i32, @intCast(st * rows)) + p;
        };
        for (&c.state) |*s| s.* = try startState(a, rng);
        inline for (.{ .{ "qkv", total * width }, .{ "cw", width * taps }, .{ "ain", total * nv }, .{ "bin", total * nv }, .{ "dt", nv }, .{ "q", total * nk * dk }, .{ "k", total * nk * dk }, .{ "v", total * nv * dk }, .{ "beta", total * nv } }) |f| {
            @field(c, f[0]) = try a.alloc(u16, f[1]);
            fillBf16(rng, @field(c, f[0]));
        }
        c.alog = try a.alloc(f32, nv);
        for (c.alog) |*x| x.* = rng.float(f32) - 0.5;
        c.g = try a.alloc(f32, total * nv);
        for (c.g) |*x| x.* = 0.9 + rng.float(f32) / 10;
        return c;
    }
};

/// One launch of `fun` (main's or _table) on the case: its outputs, then each stream's state in block layout.
fn launch(gpu: check.Gpu, s: cuda.Stream, k: Kernel, fun: cuda.Function, tab: bool, planted: bool, c: Case, a: std.mem.Allocator) ![]const []const u8 {
    var rig = rig_.Rig{ .gpu = gpu, .s = s };
    defer rig.deinit();
    const total = streams * c.rows;
    var placed: [streams]Placed = undefined;
    for (&placed, c.state) |*p, st| p.* = try Placed.make(&rig, st, tab);
    if (planted) plant(&placed[1].table, k, c.last + 1); // segment 1 keeps row last + 1
    var outs: std.ArrayList([]const u8) = .empty;
    var y: [5]u64 = @splat(0);
    if (k == .conv) {
        var segs: [streams]Seg = undefined;
        for (&segs, placed) |*sg, p| sg.* = if (tab) .{ .rec = try rig.input(std.mem.sliceAsBytes(&p.table)) } else p.seg;
        const seen = try a.dupe(u8, try rig.inputBytes());
        for (segs, 0..) |sg, st| {
            var args: cuda.Args = .{};
            inline for (.{ geo, sg, @as(c_int, @intCast(c.last + st)) }) |x| args.add(x);
            try cuda.launch.launch(fun, .{ .grid = .{ .x = nl, .y = (tail + 255) / 256, .z = 1 }, .block = .{ .x = 256 } }, s, &args);
        }
        try check.expect(std.mem.eql(u8, seen, try rig.inputBytes()), "qwen35 table tf_q35k_conv: a launch changed an input", .{});
    } else {
        // the forward's segments, then (_table) each one's table where forward_states finds it; Seg's blocks 0 there
        const block = try a.alignedAlloc(u8, .@"8", seg_states + streams * entries * 8);
        @memset(block, 0);
        var segs: [streams]Seg = undefined;
        for (&segs, placed, 0..) |*sg, p, st| {
            sg.* = if (tab) .{} else p.seg;
            sg.row0 = @intCast(st * c.rows);
            sg.rows = @intCast(c.rows);
            if (tab) @memcpy(block[seg_states + st * entries * 8 ..][0 .. entries * 8], std.mem.sliceAsBytes(&p.table));
        }
        @memcpy(block[0 .. streams * @sizeOf(Seg)], std.mem.sliceAsBytes(&segs));
        const segs_d = try rig.input(if (tab) block else block[0 .. streams * @sizeOf(Seg)]);
        const params = [5]c_int{ 0, 0, @intCast(total), streams, @intFromBool(c.record) }; // P: rows, windows, recorded
        const p_d = try rig.input(std.mem.sliceAsBytes(&params));
        const par = try rig.input(std.mem.sliceAsBytes(c.rowpar));
        var in: [11]u64 = undefined; // qkv, rowseg, cw, ain, bin, alog, dt, q, k, v, g; beta follows
        inline for (.{ c.qkv, c.rowseg, c.cw, c.ain, c.bin, c.alog, c.dt, c.q, c.k, c.v, c.g }, 0..) |x, i| in[i] = try rig.input(std.mem.sliceAsBytes(x));
        const beta = try rig.input(std.mem.sliceAsBytes(c.beta));
        const sizes = [5]usize{ total * nk * dk * 2, total * nk * dk * 2, total * nv * dk * 2, total * nv * 4, total * nv * 2 };
        if (k == .pre) {
            for (&y, sizes) |*o, n| o.* = try rig.output(n);
        } else if (k == .chain) y[0] = try rig.output(total * nv * dk * 2);
        const seen = try a.dupe(u8, try rig.inputBytes());
        var args: cuda.Args = .{};
        switch (k) {
            .pre => {
                inline for (.{ geo, in[0], segs_d, in[1], par, @as(c_int, layer), in[2], in[3], in[4], in[5], in[6], y[0], y[1], y[2], y[3], y[4], p_d }) |x| args.add(x);
                try cuda.launch.launch(fun, .{ .grid = .{ .x = 1, .y = 2 * nk + nv, .z = @intCast(total) }, .block = .{ .x = 32 } }, s, &args);
                for (y, sizes) |o, n| try outs.append(a, try a.dupe(u8, try rig.bytes(o, n)));
            },
            .commit => {
                inline for (.{ geo, in[0], segs_d, par, @as(c_int, layer), p_d }) |x| args.add(x);
                try cuda.launch.launch(fun, .{ .grid = .{ .x = (width + 255) / 256, .y = streams, .z = 1 }, .block = .{ .x = 256 } }, s, &args);
            },
            .chain => {
                inline for (.{ geo, in[7], in[8], in[9], in[10], beta, segs_d, par, @as(c_int, layer), y[0], p_d }) |x| args.add(x);
                try cuda.launch.launch(fun, .{ .grid = .{ .x = 1, .y = dk / 4, .z = nv * streams }, .block = .{ .x = 32, .y = 4 } }, s, &args);
                try outs.append(a, try a.dupe(u8, try rig.bytes(y[0], total * nv * dk * 2)));
            },
            .conv => unreachable,
        }
        try check.expect(std.mem.eql(u8, seen, try rig.inputBytes()), "qwen35 table {s}: a launch changed an input", .{names[@backingInt(k)]});
    }
    for (placed) |p| try outs.append(a, try p.gather(&rig, a));
    try check.expect(try rig.intact() and !rig.overlap(), "qwen35 table {s}: a launch wrote past a buffer", .{names[@backingInt(k)]});
    return outs.items;
}

fn equal(x: []const []const u8, y: []const []const u8) bool {
    if (x.len != y.len) return false;
    for (x, y) |p, q| if (!std.mem.eql(u8, p, q)) return false;
    return true;
}

pub fn run(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.qwen35);
    defer mod.unload();
    var keep = try cuda.Module.load(gpu.d, cuda.kernels.qwen35_keep);
    defer keep.unload();
    var red: [names.len]bool = @splat(false);
    for (names, 0..) |name, ki| {
        const k: Kernel = @fromBackingInt(@intCast(ki));
        const m = if (k == .conv) &keep else &mod;
        const main_fn = try m.function(name);
        const tab_fn = try m.function(tables[ki]);
        for (windows, 0..) |win, wi| for ([_]bool{ false, true }) |record| {
            var arena = std.heap.ArenaAllocator.init(gpu.gpa);
            defer arena.deinit();
            const a = arena.allocator();
            const last = (wi * 3) % (logs - 1); // tf_q35k_conv: stream 0 keeps row last, stream 1 row last + 1
            const c = try Case.make(a, win.parents, record, last, 0x7ab1 + 16 * ki + 2 * wi + @intFromBool(record));
            const want = try launch(gpu, s, k, main_fn, false, false, c, a);
            for (0..3) |_| {
                const got = try launch(gpu, s, k, tab_fn, true, false, c, a);
                for (got, want, 0..) |x, w, i| check.sameBytes("qwen35 table: _table on pieces against main on blocks", x, w) catch |err| {
                    std.debug.print("  at {s}_table, {s}, recorded {}, output {d}\n", .{ name, win.what, record, i });
                    return err;
                };
            }
            // a recorded window of 2+ rows: conv_commit returns early (gdn_pre logged the tails), no plant can show
            const live = !(k == .commit and record and c.rows != 1);
            const moved = !equal(try launch(gpu, s, k, tab_fn, true, true, c, a), want);
            red[ki] = red[ki] or moved;
            try check.expect(moved == live, "qwen35 table {s}_table: {s} planted in segment 1's table {s} (control) at {s}, recorded {}", .{ name, plants[ki], if (live) "still gave main's bytes" else "changed a launch that returns early", win.what, record });
            std.debug.print("PASS qwen35 table {s}_table: {s}, recorded {}: outputs and both streams' state on pieces equal {s} on blocks, 3 repeats, guards\n", .{ name, win.what, record, name });
        };
    }
    for (names, red, plants) |name, r, p| try check.expect(r, "qwen35 table {s}_table: the control ({s}) never fired", .{ name, p });
    std.debug.print("PASS qwen35 table: controls RED as planted (gdn_pre and conv_commit: tail entry; gdn_chain: recurrence entry; tf_q35k_conv: log row entry)\n", .{});
}
