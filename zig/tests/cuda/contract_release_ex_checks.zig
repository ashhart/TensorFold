//! DeepSeek's release experts end to end (group, gate/up, down, combine): one token's bytes in any batch and position.
const std = @import("std");
const cuda = @import("cuda");
const kr = @import("dsv41_release");
const check = @import("check.zig");
const kx = kr.ex;

const hidden = 5120;
const width = 576;
const Config = struct { name: []const u8, topk: usize, nids: usize, nslots: usize };
const configs = [_]Config{
    .{ .name = "top-6 of 384, 96 held", .topk = 6, .nids = 384, .nslots = 96 },
    .{ .name = "top-3 of 128, 32 held", .topk = 3, .nids = 128, .nslots = 32 },
};
const most_rows = 2048;

fn buffer(d: *const cuda.Driver, len: usize) !cuda.DeviceBuffer {
    return cuda.DeviceBuffer.alloc(d, @max(len, 16));
}

/// One expert layer's weights and a batch's buffers, sized for most_rows rows.
const Layer = struct {
    cfg: Config,
    s: cuda.Stream,
    ops: kx.Ops,
    bufs: [18]cuda.DeviceBuffer = undefined,
    made: usize = 0,

    const W1 = 0;
    const S1 = 1;
    const W3 = 2;
    const S3 = 3;
    const W2 = 4;
    const S2 = 5;
    const X = 6;
    const XS = 7;
    const IDS = 8;
    const WTS = 9;
    const SHARED = 10;
    const SLOT_OF = 11;
    const GROUP = 12;
    const CODES = 13;
    const SC = 14;
    const PART = 15;
    const Y = 16;
    const NACTIVE = 17;

    fn init(gpu: check.Gpu, cfg: Config, ops: kx.Ops, rng: std.Random) !Layer {
        var l: Layer = .{ .cfg = cfg, .s = ops.s, .ops = ops };
        errdefer l.deinit();
        const p = most_rows * cfg.topk;
        const sizes = [_]usize{ cfg.nslots * width * hidden / 2, cfg.nslots * width * hidden / 32, cfg.nslots * width * hidden / 2, cfg.nslots * width * hidden / 32, cfg.nslots * hidden * width / 2, cfg.nslots * hidden * width / 32, most_rows * hidden, most_rows * hidden / 32, 4 * p, 4 * p, 2 * most_rows * hidden, 4 * cfg.nids, 4 * (2 * cfg.nslots + 2 + 2 * p), p * width, p * width / 32, 2 * p * hidden, 2 * most_rows * hidden, 4 };
        for (sizes, 0..) |len, i| {
            l.bufs[i] = try buffer(gpu.d, len);
            l.made += 1;
        }
        const a = gpu.gpa;
        for ([_]usize{ W1, W3, W2 }) |i| try fillRandom(a, l.bufs[i], rng, false);
        for ([_]usize{ S1, S3, S2 }) |i| try fillRandom(a, l.bufs[i], rng, true);
        const slot_of = try a.alloc(i32, cfg.nids);
        defer a.free(slot_of);
        const lo = (cfg.nids - cfg.nslots) / 2;
        for (slot_of, 0..) |*s, id| s.* = if (id >= lo and id < lo + cfg.nslots) @intCast(id - lo) else -1;
        try l.bufs[SLOT_OF].upload(0, std.mem.sliceAsBytes(slot_of));
        return l;
    }

    fn deinit(l: *Layer) void {
        for (l.bufs[0..l.made]) |*b| b.free();
    }

    /// Random bytes; scales in [2^-11, 2^0] and FP8 codes without NaN.
    fn fillRandom(a: std.mem.Allocator, b: cuda.DeviceBuffer, rng: std.Random, scales: bool) !void {
        const host = try a.alloc(u8, b.len);
        defer a.free(host);
        rng.bytes(host);
        if (scales) for (host) |*v| {
            v.* = 116 + v.* % 12;
        };
        try b.upload(0, host);
    }

    fn word(l: Layer, i: usize) u64 {
        const p = most_rows * l.cfg.topk;
        const g = l.bufs[GROUP].ptr;
        return switch (i) {
            0 => g,
            1 => g + 4 * (l.cfg.nslots + 1),
            2 => g + 4 * (l.cfg.nslots + 1 + p),
            else => g + 4 * (l.cfg.nslots + 1 + 2 * p),
        };
    }

    const Launch = enum { served, one_cta, warps, one_warp_cta, drop_slot };

    /// The routed layer over m rows already uploaded; every output poisoned first.
    fn run(l: Layer, m: usize, how: Launch) !void {
        const s = l.s.handle;
        for ([_]usize{ GROUP, CODES, SC, PART, Y, NACTIVE }) |i| try l.bufs[i].fill8(0xff, s);
        const topk = l.cfg.topk;
        try l.ops.group(.{ .ids = l.bufs[IDS].ptr, .slot_of = l.bufs[SLOT_OF].ptr, .offs = l.word(0), .pair_tok = l.word(1), .pair_of = l.word(2), .active = l.word(3), .nactive = l.bufs[NACTIVE].ptr, .m = @intCast(m), .topk = @intCast(topk), .nslots = @intCast(l.cfg.nslots), .nids = @intCast(l.cfg.nids) });
        if (how == .drop_slot) {
            // control: the last active slot's tiles never run, so its pairs keep the poison
            try l.s.synchronize();
            var n: i32 = 0;
            try l.bufs[NACTIVE].download(0, std.mem.asBytes(&n));
            n -= 1;
            try l.bufs[NACTIVE].upload(0, std.mem.asBytes(&n));
        }
        const warps: u32 = @intCast((kx.tileWarps(m, topk, l.cfg.nslots, hidden) + 3) / 4);
        const grid: kx.Launch = switch (how) {
            .served, .drop_slot => .served,
            .one_cta => .{ .ctas = 1 },
            .warps => .{ .warps = warps },
            .one_warp_cta => .{ .warps = 1 },
        };
        try l.ops.gateUp(.{ .x = l.bufs[X].ptr, .x_ld = hidden, .xs = l.bufs[XS].ptr, .xs_ld = hidden / 32, .w1 = l.bufs[W1].ptr, .s1 = l.bufs[S1].ptr, .w3 = l.bufs[W3].ptr, .s3 = l.bufs[S3].ptr, .wts = l.bufs[WTS].ptr, .pair_tok = l.word(1), .offs = l.word(0), .active = l.word(3), .nactive = l.bufs[NACTIVE].ptr, .codes = l.bufs[CODES].ptr, .scales = l.bufs[SC].ptr, .n = width, .k = hidden, .topk = @intCast(topk) }, m, l.cfg.nslots, grid);
        try l.ops.down(.{ .x = l.bufs[CODES].ptr, .xs = l.bufs[SC].ptr, .w2 = l.bufs[W2].ptr, .s2 = l.bufs[S2].ptr, .offs = l.word(0), .active = l.word(3), .nactive = l.bufs[NACTIVE].ptr, .part = l.bufs[PART].ptr, .n = hidden, .k = width }, m, topk, l.cfg.nslots, grid);
        try l.ops.combine(.{ .part = l.bufs[PART].ptr, .pair_of = l.word(2), .shared = l.bufs[SHARED].ptr, .shared_ld = hidden, .y = l.bufs[Y].ptr, .y_ld = hidden, .m = @intCast(m), .n = hidden, .topk = @intCast(topk) });
        try l.s.synchronize();
    }

    fn row(l: Layer, r: usize, out: []u8) !void {
        try l.bufs[Y].download(2 * r * hidden, out[0 .. 2 * hidden]);
    }
};

/// A batch's host rows: codes, scales, ids, route weights and shared rows, row by row.
const Rows = struct {
    x: []u8,
    xs: []u8,
    ids: []i32,
    wts: []f32,
    shared: []u16,

    fn init(a: std.mem.Allocator, topk: usize) !Rows {
        return .{ .x = try a.alloc(u8, most_rows * hidden), .xs = try a.alloc(u8, most_rows * hidden / 32), .ids = try a.alloc(i32, most_rows * topk), .wts = try a.alloc(f32, most_rows * topk), .shared = try a.alloc(u16, most_rows * hidden) };
    }

    fn deinit(r: Rows, a: std.mem.Allocator) void {
        a.free(r.x);
        a.free(r.xs);
        a.free(r.ids);
        a.free(r.wts);
        a.free(r.shared);
    }

    /// Row r random: codes without NaN, scales in [2^-9, 2^6], topk distinct ids (one in eight unheld), bf16 shared.
    fn random(r: Rows, i: usize, cfg: Config, rng: std.Random) void {
        rng.bytes(r.x[i * hidden ..][0..hidden]);
        for (r.x[i * hidden ..][0..hidden]) |*v| v.* = if (v.* & 0x7f == 0x7f) v.* ^ 1 else v.*;
        for (r.xs[i * hidden / 32 ..][0 .. hidden / 32]) |*v| v.* = 118 + rng.uintLessThan(u8, 16);
        const lo = (cfg.nids - cfg.nslots) / 2;
        var k: usize = 0;
        while (k < cfg.topk) {
            const id: i32 = @intCast(if (rng.uintLessThan(u32, 8) != 0) lo + rng.uintLessThan(usize, cfg.nslots) else rng.uintLessThan(usize, cfg.nids));
            if (std.mem.indexOfScalar(i32, r.ids[i * cfg.topk ..][0..k], id) != null) continue;
            r.ids[i * cfg.topk + k] = id;
            r.wts[i * cfg.topk + k] = 0.25 + rng.float(f32);
            k += 1;
        }
        for (r.shared[i * hidden ..][0..hidden]) |*v| v.* = @import("core").contract.spec.toBf16(rng.float(f32) - 0.5);
    }

    fn copy(r: Rows, to: usize, from: usize, topk: usize) void {
        @memcpy(r.x[to * hidden ..][0..hidden], r.x[from * hidden ..][0..hidden]);
        @memcpy(r.xs[to * hidden / 32 ..][0 .. hidden / 32], r.xs[from * hidden / 32 ..][0 .. hidden / 32]);
        @memcpy(r.ids[to * topk ..][0..topk], r.ids[from * topk ..][0..topk]);
        @memcpy(r.wts[to * topk ..][0..topk], r.wts[from * topk ..][0..topk]);
        @memcpy(r.shared[to * hidden ..][0..hidden], r.shared[from * hidden ..][0..hidden]);
    }

    fn upload(r: Rows, l: Layer, m: usize) !void {
        const topk = l.cfg.topk;
        try l.bufs[Layer.X].upload(0, r.x[0 .. m * hidden]);
        try l.bufs[Layer.XS].upload(0, r.xs[0 .. m * hidden / 32]);
        try l.bufs[Layer.IDS].upload(0, std.mem.sliceAsBytes(r.ids[0 .. m * topk]));
        try l.bufs[Layer.WTS].upload(0, std.mem.sliceAsBytes(r.wts[0 .. m * topk]));
        try l.bufs[Layer.SHARED].upload(0, std.mem.sliceAsBytes(r.shared[0 .. m * hidden]));
    }
};

/// The probe row (row `probe` of a batch of m random rows) gives its one-row bytes; `same` first rows share its ids.
fn probeBytes(l: Layer, r: Rows, probe_row: Rows, m: usize, at: usize, same: usize, rng: std.Random, out: []u8) !void {
    const topk = l.cfg.topk;
    for (0..m) |i| r.random(i, l.cfg, rng);
    for (0..same) |i| @memcpy(r.ids[i * topk ..][0..topk], probe_row.ids[0..topk]);
    @memcpy(r.x[at * hidden ..][0..hidden], probe_row.x[0..hidden]);
    @memcpy(r.xs[at * hidden / 32 ..][0 .. hidden / 32], probe_row.xs[0 .. hidden / 32]);
    @memcpy(r.ids[at * topk ..][0..topk], probe_row.ids[0..topk]);
    @memcpy(r.wts[at * topk ..][0..topk], probe_row.wts[0..topk]);
    @memcpy(r.shared[at * hidden ..][0..hidden], probe_row.shared[0..hidden]);
    try r.upload(l, m);
    try l.run(m, .served);
    try l.row(at, out);
}

fn experts(gpu: check.Gpu, ops: kx.Ops, cfg: Config) !void {
    const a = gpu.gpa;
    var prng = std.Random.DefaultPrng.init(0x7e2 + cfg.topk);
    const rng = prng.random();
    var l = try Layer.init(gpu, cfg, ops, rng);
    defer l.deinit();
    const r = try Rows.init(a, cfg.topk);
    defer r.deinit(a);
    const probe = try Rows.init(a, cfg.topk);
    defer probe.deinit(a);
    probe.random(0, cfg, rng);
    const want = try a.alloc(u8, 2 * hidden);
    defer a.free(want);
    const got = try a.alloc(u8, 2 * hidden);
    defer a.free(got);
    try probeBytes(l, r, probe, 1, 0, 0, rng, want);
    var bad: usize = 0;
    var runs: usize = 0;
    // prompt == decode: the probe in batches of 1 to 2048 rows, first, and at 15, 16 and 2047 of the largest
    for ([_][2]usize{ .{ 5, 0 }, .{ 16, 0 }, .{ 17, 0 }, .{ 100, 0 }, .{ 2048, 0 }, .{ 2048, 15 }, .{ 2048, 16 }, .{ 2048, 2047 } }) |b| {
        try probeBytes(l, r, probe, b[0], b[1], 0, rng, got);
        runs += 1;
        if (!std.mem.eql(u8, want, got)) bad += 1;
    }
    try check.expect(bad == 0, "dsv41_rel_ex prompt == decode {s}: {d} of {d} batches differ", .{ cfg.name, bad, runs });
    check.pass("dsv41_rel_ex prompt == decode {s}: the probe row at 5, 16, 17, 100, 2048 rows and offsets 15, 16, 2047 equals its one-row bytes", .{cfg.name});
    // every tile position: p earlier rows share the probe's ids, so its pairs sit at position p of their slots' tiles
    runs = 0;
    for (0..16) |p| for (0..5) |_| {
        try probeBytes(l, r, probe, p + 1 + rng.uintLessThan(usize, 8), p, p, rng, got);
        runs += 1;
        if (!std.mem.eql(u8, want, got)) bad += 1;
    };
    // the prompt tile: 400 rows (64-pair tiles), the probe's pairs at every position 0-63, 2 neighbour sets each
    for (0..64) |p| for (0..2) |_| {
        try probeBytes(l, r, probe, 400, p, p, rng, got);
        runs += 1;
        if (!std.mem.eql(u8, want, got)) bad += 1;
    };
    try check.expect(bad == 0, "dsv41_rel_ex positions {s}: {d} of {d} batches differ", .{ cfg.name, bad, runs });
    check.pass("dsv41_rel_ex positions {s}: the probe's pairs at every position of 16-pair (5 neighbour sets) and 64-pair (2 sets) tiles ({d} batches)", .{ cfg.name, runs });
    // one CTA over every tile (gate/up and down) equals the served launch; a launch missing a slot must differ
    const all = try a.alloc(u8, 2 * 400 * hidden);
    defer a.free(all);
    const one = try a.alloc(u8, 2 * 400 * hidden);
    defer a.free(one);
    var caught: usize = 0;
    for ([_]usize{ 5, 100, 400 }) |m| {
        for (0..m) |i| r.random(i, cfg, rng);
        try r.upload(l, m);
        try l.run(m, .served);
        try l.bufs[Layer.Y].download(0, all[0 .. 2 * m * hidden]);
        for ([_]Layer.Launch{ .one_cta, .warps, .one_warp_cta }) |how| {
            try l.run(m, how);
            try l.bufs[Layer.Y].download(0, one[0 .. 2 * m * hidden]);
            if (!std.mem.eql(u8, all[0 .. 2 * m * hidden], one[0 .. 2 * m * hidden])) bad += 1;
        }
        try l.run(m, .drop_slot);
        try l.bufs[Layer.Y].download(0, one[0 .. 2 * m * hidden]);
        if (!std.mem.eql(u8, all[0 .. 2 * m * hidden], one[0 .. 2 * m * hidden])) caught += 1;
    }
    try check.expect(bad == 0 and caught == 3, "dsv41_rel_ex launch shapes {s}: {d} runs differ, control red {d} of 3", .{ cfg.name, bad, caught });
    check.pass("dsv41_rel_ex launch shapes {s}: one CTA over every tile, the warp form and the warp form in one CTA equal the served bytes at 5, 100 and 400 rows on poisoned buffers; a launch missing a slot differs", .{cfg.name});
}

/// Every end-to-end check at both expert counts (sm_121 only).
pub fn run(gpu: check.Gpu, ops: kx.Ops) !void {
    for (configs) |cfg| try experts(gpu, ops, cfg);
}
