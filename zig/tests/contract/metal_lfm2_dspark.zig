//! LFM2's DSpark kernels: the Markov step (scan then pick), and held drafts into a verify's ids.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
const controls = @import("metal_lfm2_controls.zig");

pub const Which = enum { markov_step, draft_ids, markov_step_bf16 };
const rank = 256;
const vocab = 128000;
const span = 512; // tokens a scan threadgroup takes (the kernel's MARKOV_SPAN)
const groups = (vocab + span - 1) / span;
const table = 1000; // W1's rows in the check: ids are inside it or past the vocabulary (a zero row)
const names = [_][]const u8{ "metal lfm2 markov step", "metal lfm2 draft ids", "metal lfm2 markov step bf16" };
const stride = 64; // words between a stream's picks: a hold pass's 32 blocks, a token and a flag each

const ids_shapes = [_]c.Shape{ .{ .rows = 1, .n = 1, .k = 1 }, .{ .rows = 4, .n = 1, .k = 1 }, .{ .rows = 15, .n = 1, .k = 1 } };
const step_shapes = [_]c.Shape{ .{ .rows = 1, .n = vocab, .k = rank }, .{ .rows = 4, .n = vocab, .k = rank }, .{ .rows = 9, .n = vocab, .k = rank } };

fn none(_: c.Shape) usize {
    return 0;
}

fn depth(_: c.Shape) usize {
    return rank;
}

pub fn spec(w: Which) c.Spec {
    return switch (w) {
        .draft_ids => .{ .name = names[1], .accum = .none, .depth = none, .out = .raw, .served = &ids_shapes, .sums = .{ .f16 = 0, .bf16 = 0, .why = "a copy moves tokens: no sums" } },
        .markov_step, .markov_step_bf16 => .{ .name = if (w == .markov_step) names[0] else names[2], .accum = .none, .depth = depth, .out = .raw, .served = &step_shapes, .nan = .{ .stated = "a row with a NaN score is flagged nonfinite" }, .sums = .{ .f16 = 0, .bf16 = 0, .why = "picks are tokens: the check bounds each pick's score by gamma(256), not sum bits" } },
    };
}

/// A bf16 value for element `i` over signs and binades; the nonfinite fill plants NaN and infinities.
fn value(i: usize, seed: u64, fill: g.Fill) u16 {
    if (fill == .nonfinite) switch ((@as(u64, i) *% 2654435761 +% seed) % 9973) {
        0 => return 0x7fc0,
        1 => return 0x7f80,
        2 => return 0xff80,
        else => {},
    };
    const h: u64 = (@as(u64, i) *% 0x9e3779b97f4a7c15 +% seed *% 0xbf58476d1ce4e5b9) >> 40;
    const mant: f32 = @as(f32, @floatFromInt(h % 1024)) / 1024 + 1;
    const exp: i32 = @as(i32, @intCast((h >> 10) % 8)) - 4;
    const v = std.math.ldexp(mant, exp) * @as(f32, if (((h >> 15) & 1) == 0) 1 else -1);
    return c.toBf16(v);
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    which: Which,
    mutation: controls.Mutation,
    pipes: [2]mtl.Pipeline,
    rows: usize = 0,
    src: []u16 = &.{}, // the base logits rows
    w1: []u16 = &.{},
    codes: []u8 = &.{},
    scales: []u16 = &.{},
    biases: []u16 = &.{},
    w2: []u16 = &.{}, // the bf16 step's W2
    ids: []u32 = &.{},
    args: [8]rig_.Ref = undefined,
    y: rig_.Ref = undefined,
    bad: rig_.Ref = undefined, // draft_ids: the flag word it sets for a flagged or invalid pick, zeroed each run

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, w: Which, mutation: controls.Mutation) !Probe {
        const text = try controls.source(a, mutation);
        defer if (mutation != .none) a.free(text);
        const kernels: [2][:0]const u8 = switch (w) {
            .markov_step => .{ "lfm2_markov_scan", "lfm2_markov_pick" },
            .markov_step_bf16 => .{ "lfm2_markov_scan_bf16", "lfm2_markov_pick" },
            .draft_ids => .{ "lfm2_draft_ids", "lfm2_draft_ids" },
        };
        const first = try rig_.pipeline(device, text, kernels[0]);
        errdefer first.deinit();
        const second = try rig_.pipeline(device, text, kernels[1]);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .which = w, .mutation = mutation, .pipes = .{ first, second } };
    }

    fn drop(p: *Probe) void {
        inline for (.{ "src", "w1", "codes", "scales", "biases", "w2", "ids" }) |f| {
            p.a.free(@field(p, f));
            @field(p, f) = &.{};
        }
        p.rig.reset();
    }

    pub fn deinit(p: *Probe) void {
        p.drop();
        p.rig.deinit();
        for (p.pipes) |x| x.deinit();
    }

    const vt = g.hooks(Probe);
    pub fn kernel(p: *Probe) g.Kernel {
        var s = spec(p.which);
        if (p.mutation != .none) s.name = p.mutation.name();
        return .{ .ctx = p, .vt = &vt, .spec = s, .control = p.mutation.gate() };
    }

    fn fill(out: []u16, seed: u64, f: g.Fill) void {
        for (out, 0..) |*v, i| v.* = value(i, seed, f);
    }

    pub fn prepare(p: *Probe, s: c.Shape, f: g.Fill, seed: u64) !bool {
        p.drop();
        p.rows = s.rows;
        if (p.which == .draft_ids) { // picks a stride apart: tokens inside the vocabulary, its last, and past it
            if (s.n != 1 or s.rows == 0 or s.rows > 16) return false;
            p.ids = try p.a.alloc(u32, s.rows * stride);
            for (p.ids, 0..) |*id, i| id.* = switch ((i / stride) % 3) {
                0 => @intCast((@as(u64, i) * 7919 + seed) % vocab),
                1 => vocab - 1,
                else => vocab + @as(u32, @intCast(i % 5)),
            };
            // the flag word after each pick: zero, but planted nonzero on the last of every four streams
            for (0..s.rows) |r| p.ids[r * stride + 1] = if (r % 4 == 3) 1 else 0;
            p.args[0] = try p.rig.input(std.mem.sliceAsBytes(p.ids));
            const at = [4]u32{ stride, @intCast(s.rows), vocab, 0 };
            p.args[1] = try p.rig.input(std.mem.asBytes(&at));
            p.y = try p.rig.output(p.count());
            return true;
        }
        if (s.n != vocab or s.k != rank or s.rows == 0) return false;
        // each stream's step row sits a block (two rows here) apart, as a hold's rows of one step do
        p.src = try p.a.alloc(u16, 2 * s.rows * vocab);
        fill(p.src, seed, f);
        p.w1 = try p.a.alloc(u16, table * rank);
        fill(p.w1, seed +% 5, .random);
        p.codes = try p.a.alloc(u8, vocab * rank);
        for (p.codes, 0..) |*q, i| q.* = @truncate((@as(u64, i) *% 2654435761 +% seed) >> 11);
        p.scales = try p.a.alloc(u16, vocab * rank / 64);
        p.biases = try p.a.alloc(u16, vocab * rank / 64);
        for (p.scales, p.biases, 0..) |*sc, *bi, i| {
            sc.* = c.toBf16(@as(f32, @floatFromInt(i % 7 + 1)) / 4096);
            bi.* = c.toBf16(@as(f32, @floatFromInt(i % 5)) / 64 - 0.03);
        }
        if (p.which == .markov_step_bf16) {
            p.w2 = try p.a.alloc(u16, vocab * rank);
            fill(p.w2, seed +% 9, .random);
        }
        // the previous token every second word (the picks' layout): inside W1, its last row, or past the vocabulary
        p.ids = try p.a.alloc(u32, 2 * s.rows);
        for (p.ids, 0..) |*id, i| id.* = if (i % 2 == 1) 0xdead else switch ((i / 2) % 3) {
            0 => @intCast((@as(u64, i) * 37 + seed) % table),
            1 => table - 1,
            else => vocab + @as(u32, @intCast(i)),
        };
        p.args[0] = try p.rig.input(std.mem.sliceAsBytes(p.src));
        p.args[1] = try p.rig.input(std.mem.sliceAsBytes(p.w1));
        p.args[2] = if (p.which == .markov_step_bf16) try p.rig.input(std.mem.sliceAsBytes(p.w2)) else try p.rig.input(p.codes);
        p.args[3] = try p.rig.input(std.mem.sliceAsBytes(p.ids));
        const args = [4]u32{ 2 * vocab, rank, vocab, 2 };
        p.args[4] = try p.rig.input(std.mem.asBytes(&args));
        p.args[5] = try p.rig.input(std.mem.sliceAsBytes(p.scales));
        p.args[6] = try p.rig.input(std.mem.sliceAsBytes(p.biases));
        const n: u32 = groups;
        p.args[7] = try p.rig.input(std.mem.asBytes(&n));
        p.y = try p.rig.output(p.count());
        return true;
    }

    pub fn count(p: *Probe) usize {
        return switch (p.which) {
            .markov_step, .markov_step_bf16 => p.rows * 8,
            .draft_ids => p.rows * 4,
        };
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        if (p.which == .draft_ids) {
            e.setPipeline(p.pipes[0]);
            e.setBuffer(p.args[0].buf, p.args[0].off, 0);
            e.setBuffer(p.args[1].buf, p.args[1].off, 1);
            e.setBuffer(p.y.buf, p.y.off, 2);
            p.bad = try p.rig.scratch(4);
            @memset(p.rig.bytes(p.bad, 4), 0);
            e.setBuffer(p.bad.buf, p.bad.off, 3);
            e.dispatchThreads(mtl.Size.of(p.rows, 1, 1), mtl.Size.of(32, 1, 1));
            return p.rig.finish(pass);
        }
        const best = try p.rig.scratch(p.rows * groups * 4);
        const at = try p.rig.scratch(p.rows * groups * 4);
        e.setPipeline(p.pipes[0]);
        for (p.args[0..5], 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setBuffer(best.buf, best.off, 5);
        e.setBuffer(at.buf, at.off, 6);
        if (p.which == .markov_step) {
            e.setBuffer(p.args[5].buf, p.args[5].off, 7);
            e.setBuffer(p.args[6].buf, p.args[6].off, 8);
        }
        e.dispatchGroups(mtl.Size.of(groups, p.rows, 1), mtl.Size.of(256, 1, 1));
        e.setPipeline(p.pipes[1]);
        e.setBuffer(best.buf, best.off, 0);
        e.setBuffer(at.buf, at.off, 1);
        e.setBuffer(p.args[7].buf, p.args[7].off, 2);
        e.setBuffer(p.y.buf, p.y.off, 3);
        e.dispatchGroups(mtl.Size.of(p.rows, 1, 1), mtl.Size.of(256, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn reference(_: *Probe, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        @memset(want, 0); // the step and the ids check themselves (check)
    }

    /// Each stream's pick against float64 scores with the kernel's bf16 roundings: the best, but for near ties.
    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        const out = p.rig.bytes(p.y, p.count());
        if (p.which == .draft_ids) { // exact: each stream's first pick, or 0 for a token past the vocabulary or a flag
            var any = false;
            for (0..p.rows) |r| {
                const token = p.ids[r * stride];
                const ok = token < vocab and p.ids[r * stride + 1] == 0;
                any = any or !ok;
                if (std.mem.readInt(u32, out[4 * r ..][0..4], .little) != (if (ok) token else 0)) return "an id differs from its pick";
            }
            const raised = std.mem.readInt(u32, p.rig.bytes(p.bad, 4)[0..4], .little);
            if ((raised != 0) != any) return "the bad flag does not follow the flagged and invalid picks";
            return null;
        }
        for (0..p.rows) |s| {
            const token = std.mem.readInt(u32, out[8 * s ..][0..4], .little);
            const flag = std.mem.readInt(u32, out[8 * s + 4 ..][0..4], .little);
            const id = p.ids[2 * s];
            var top: f64 = -std.math.inf(f64);
            var mine: ?f64 = null;
            var nan = false;
            var bound: f64 = 0;
            for (0..vocab) |v| {
                var dot: f64 = 0;
                var size: f64 = 0;
                if (id < table) for (0..rank) |i| {
                    const k = (v * rank + i) / 64;
                    const w: f64 = if (p.which == .markov_step_bf16) c.fromBf16(p.w2[v * rank + i]) else @as(f64, c.fromBf16(p.scales[k])) * @as(f64, @floatFromInt(p.codes[v * rank + i])) + c.fromBf16(p.biases[k]);
                    dot += w * c.fromBf16(p.w1[id * rank + i]);
                    size += @abs(w * c.fromBf16(p.w1[id * rank + i]));
                };
                const base: f32 = c.fromBf16(p.src[2 * s * vocab + v]);
                const score: f64 = c.fromBf16(c.toBf16(base + c.fromBf16(c.toBf16(@floatCast(dot)))));
                if (std.math.isNan(score)) nan = true;
                if (score > top) top = score;
                if (v == token) mine = score;
                // two bf16 steps of the score and the fp32 sum's own error
                bound = @max(bound, @abs(score) * 0x1p-7 + c.gamma(rank, 0x1p-24) * size);
            }
            if (nan) {
                if (flag != 1) return "a row with a NaN score is not flagged";
                continue;
            }
            if (flag != 0) return "a finite row is flagged nonfinite";
            const got = mine orelse return "the pick is past the vocabulary";
            if (got < top - bound) return "the pick's score is below the row's best by more than its bound";
        }
        return null;
    }
};
