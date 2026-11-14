//! tf-cuda-test dspine-table: DSpine's exact-sender second pass (sender_message, table) against float64, with controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const D = 512; // hidden
const R = 128; // the bridge's rank
const K = 16; // candidates a row, senders a row
const B = 16; // block rows: one stream's anchor, then 15 head rows
const rows = B - 1;
const P = rows * K; // senders
const V = 256; // vocab of the table and the logits rows
const lp_words = 43;
const eps: f32 = 1e-6;

/// What a reference drops: nothing, or one DSpine step a kernel could lose.
const Drop = enum { none, sender_gate, message, write, sender };

const Fixture = struct {
    q: []u16, // (B, R): the block rows' own gate terms
    sv: []u16, // (P, 2R): each sender's [predecessor gate | value]
    gb: []u16, // (R)
    raw: []u16, // (B, D)
    rms: []f32, // (B)
    wm: []u16, // (P, D): W_write m per sender
    tg: []u16, // (V, R)
    logits: []u16, // (2 rows, V): the masked rows (unused here), then the base rows
    words: []u32, // (rows, lp_words): candidate ids at [2..2+K]
};

fn bf(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}

fn fill(r: std.Random, out: []u16, scale: f32) void {
    for (out) |*h| h.* = core.bf16_round.bits(scale * r.floatNorm(f32));
}

fn make(a: std.mem.Allocator, seed: u64) !Fixture {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const f: Fixture = .{ .q = try a.alloc(u16, B * R), .sv = try a.alloc(u16, P * 2 * R), .gb = try a.alloc(u16, R), .raw = try a.alloc(u16, B * D), .rms = try a.alloc(f32, B), .wm = try a.alloc(u16, P * D), .tg = try a.alloc(u16, V * R), .logits = try a.alloc(u16, 2 * rows * V), .words = try a.alloc(u32, rows * lp_words) };
    fill(r, f.q, 1);
    fill(r, f.sv, 1);
    fill(r, f.gb, 0.5);
    fill(r, f.raw, 1);
    for (f.rms) |*x| x.* = 0.5 + 1.5 * r.float(f32);
    fill(r, f.wm, 0.3);
    fill(r, f.tg, 0.3);
    fill(r, f.logits, 3);
    @memset(f.words, 0);
    for (0..rows) |row| {
        var ids: [V]u32 = undefined;
        for (&ids, 0..) |*x, i| x.* = @intCast(i);
        r.shuffle(u32, &ids);
        @memcpy(f.words[row * lp_words + 2 ..][0..K], ids[0..K]);
    }
    return f;
}

fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

/// The message of sender p at rank i in float64, and the magnitude of its terms.
fn message(f: Fixture, p: usize, i: usize, drop: Drop) f64 {
    const t = 1 + p / K;
    const s = if (drop == .sender_gate) 0 else bf(f.sv[p * 2 * R + i]);
    return sigmoid(bf(f.q[t * R + i]) + s + bf(f.gb[i])) * bf(f.sv[p * 2 * R + R + i]);
}

/// Table entry (p, j) in float64 over the kernel's own messages `m`, and the magnitude its fp32 sums carry.
fn entry(f: Fixture, m: []const u16, p: usize, j: usize, drop: Drop) [2]f64 {
    const row = p / K;
    const t = 1 + row;
    const src = if (drop == .sender) row * K else p; // sender-blind: every sender takes the row's first
    const rms: f64 = f.rms[t];
    var ss: f64 = 0;
    for (0..D) |i| {
        const y = bf(f.raw[t * D + i]) + (if (drop == .write) 0 else rms * bf(f.wm[src * D + i]));
        ss += y * y;
    }
    const inv = 1 / @sqrt(ss / D + eps);
    const id = f.words[row * lp_words + 2 + j];
    var dot: f64 = 0;
    var mag: f64 = 0;
    for (0..R) |k| {
        const v = bf(m[src * R + k]) * bf(f.tg[id * R + k]);
        dot += v;
        mag += @abs(v);
    }
    if (drop == .message) dot = 0;
    const base = bf(f.logits[(rows + row) * V + id]);
    return .{ (base + rms * dot) * inv, (@abs(base) + rms * mag) * inv };
}

const Fns = struct { message: cuda.Function, table: cuda.Function };

/// One launch of each kernel: the messages (P, R) bf16 and the table (P, K) f32, inputs and guards checked.
fn run(gpu: check.Gpu, s: cuda.Stream, fns: Fns, f: Fixture, a: std.mem.Allocator) !struct { m: []u16, table: []f32 } {
    var rig = rig_.Rig{ .gpu = gpu, .s = s };
    defer rig.deinit();
    const q = try rig.input(std.mem.sliceAsBytes(f.q));
    const sv = try rig.input(std.mem.sliceAsBytes(f.sv));
    const gb = try rig.input(std.mem.sliceAsBytes(f.gb));
    const raw = try rig.input(std.mem.sliceAsBytes(f.raw));
    const rms = try rig.input(std.mem.sliceAsBytes(f.rms));
    const wm = try rig.input(std.mem.sliceAsBytes(f.wm));
    const tg = try rig.input(std.mem.sliceAsBytes(f.tg));
    const logits = try rig.input(std.mem.sliceAsBytes(f.logits));
    const words = try rig.input(std.mem.sliceAsBytes(f.words));
    const m = try rig.output(P * R * 2);
    const out = try rig.output(P * K * 4);
    const seen = try a.dupe(u8, try rig.inputBytes());
    const int = struct {
        fn of(x: usize) c_int {
            return @intCast(x);
        }
    }.of;
    var am: cuda.Args = .{};
    inline for (.{ q, sv, gb, m, int(R), int(K), int(B) }) |x| am.add(x);
    try cuda.launch.launch(fns.message, .{ .grid = .{ .x = P }, .block = .{ .x = 256 } }, s, &am);
    var at: cuda.Args = .{};
    inline for (.{ raw, rms, m, wm, tg, logits, words, out, int(D), int(R), int(K), int(B), int(rows), int(V), int(lp_words), int(0), int(0), eps }) |x| at.add(x);
    try cuda.launch.launch(fns.table, .{ .grid = .{ .x = P }, .block = .{ .x = 256 } }, s, &at);
    try s.synchronize();
    const got_m = try a.alloc(u16, P * R);
    @memcpy(std.mem.sliceAsBytes(got_m), try rig.bytes(m, P * R * 2));
    const got_t = try a.alloc(f32, P * K);
    @memcpy(std.mem.sliceAsBytes(got_t), try rig.bytes(out, P * K * 4));
    try check.expect(try rig.intact() and !rig.overlap() and std.mem.eql(u8, seen, try rig.inputBytes()), "dspine-table: a launch changed an input or a guard", .{});
    return .{ .m = got_m, .table = got_t };
}

/// Outputs past the reference without `drop`: messages beyond a bf16 ulp, table entries beyond 2^-16 of magnitude.
fn misses(f: Fixture, m: []const u16, table: []const f32, drop: Drop) usize {
    var n: usize = 0;
    for (0..P) |p| for (0..R) |i| {
        const want = message(f, p, i, drop);
        if (@abs(bf(m[p * R + i]) - want) > @abs(want) * 0x1p-7 + 1e-6) n += 1;
    };
    for (0..P) |p| for (0..K) |j| {
        const e = entry(f, m, p, j, drop);
        if (@abs(table[p * K + j] - e[0]) > e[1] * 0x1p-16 + 1e-6) n += 1;
    };
    return n;
}

pub fn runAll(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.qwen35_draft);
    defer mod.unload();
    const fns: Fns = .{ .message = try mod.function("tf_q35d_sender_message"), .table = try mod.function("tf_q35d_table") };
    for ([_]u64{ 0xd5b1, 0x5e4d, 0x7ab1 }) |seed| {
        var arena = std.heap.ArenaAllocator.init(gpu.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const f = try make(a, seed);
        const got = try run(gpu, s, fns, f, a);
        for (0..2) |_| {
            const again = try run(gpu, s, fns, f, a);
            try check.sameBytes("dspine-table: repeat messages", std.mem.sliceAsBytes(again.m), std.mem.sliceAsBytes(got.m));
            try check.sameBytes("dspine-table: repeat table", std.mem.sliceAsBytes(again.table), std.mem.sliceAsBytes(got.table));
        }
        const bad = misses(f, got.m, got.table, .none);
        try check.expect(bad == 0, "dspine-table seed {x}: {d} outputs past float64's tolerance", .{ seed, bad });
        check.pass("dspine-table seed {x}: {d} messages and {d} table entries within tolerance, 3 runs same bytes, guards", .{ seed, P * R, P * K });
        for ([_]Drop{ .sender_gate, .message, .write, .sender }) |drop| {
            const red = misses(f, got.m, got.table, drop);
            try check.expect(red > 0, "dspine-table control {t}: a reference without it still matches (control missed)", .{drop});
            std.debug.print("RED  dspine-table control {t}: {d} outputs miss once it is dropped\n", .{ drop, red });
        }
    }
}
