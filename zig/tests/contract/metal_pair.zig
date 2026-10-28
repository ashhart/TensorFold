//! The pair seam's four kernels (core/pair_rows.metal), and the float64 rules the gates check.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const controls = @import("metal_pair_controls.zig");

const c = contract.spec;
const g = contract.gates;

/// The seam's four kernels; the tail of each registry name is the MSL symbol launchOf hands the source.
pub const Which = enum { stage, gather, argmax, pick };

pub const all = [_]Which{ .stage, .gather, .argmax, .pick };

/// What pair_rows.zig dispatches per kernel, so `init` can compare it against Metal's own reflection.
pub const Launched = struct { threads: u32, tg_memory: u32 };

pub fn launch(w: Which) Launched {
    return switch (w) {
        .stage => .{ .threads = 256, .tg_memory = 0 },
        .gather => .{ .threads = 256, .tg_memory = 0 },
        .argmax => .{ .threads = 1024, .tg_memory = 384 },
        .pick => .{ .threads = 32, .tg_memory = 0 },
    };
}

/// The MSL symbol pair_rows.metal defines.
pub fn symbol(w: Which) []const u8 {
    return switch (w) {
        .stage => "pair_stage",
        .gather => "pair_gather",
        .argmax => "pair_argmax",
        .pick => "pair_pick",
    };
}

/// The spec's name: `metal <what> <symbol>`, so launchOf's last word is the symbol and its prefix picks core_pair_rows.
pub fn name(w: Which) []const u8 {
    return switch (w) {
        .stage => "metal pair stage pair_stage",
        .gather => "metal pair gather pair_gather",
        .argmax => "metal pair argmax pair_argmax",
        .pick => "metal pair pick pair_pick",
    };
}

/// Qwen3.8-27B's rank-local slice: the seam reduces ids [124160, 248320) on rank 1.
const logit_lo: u32 = 124160;

/// An argmax id word's bit 31: the row had a logit that is not finite, so pair_pick sends the row to the id sentinel.
const flag: u32 = 0x80000000;

/// The id a flagged row, and a broken link, read as: no token, visibly wrong.
const sentinel: u32 = 0xFFFFFFFF;

/// A thread's chain over every 1024th logit, five shuffle levels, then the 32 simdgroup winners.
fn chainLogits(s: c.Shape) usize {
    return (s.k + 1023) / 1024 + 5 + 5;
}

/// Only moves and compares values: no sums to carry an accumulator's error.
fn noSums(_: c.Shape) usize {
    return 0;
}

const move_why = "a move writes each bf16 element as it read it: no sum, no rounding, so no accumulator's error can enter";

const argmax_rule = "a winner is (value bits, id): NaN loses every comparison, a row of only NaNs reports the -inf sentinel with id 0xFFFFFFFF, and a row with any non-finite logit reports NaN bits and id | 0x80000000";
const pick_rule = "the pick copies one rank's (value, id) word pair; an id with bit 31 set, and the broken-link word, draw 0xFFFFFFFF: the same sentinel, so one downstream check covers both";

/// A kernel's spec. The raw kernels check themselves; the two moves' bound is exact, so gate 3 keeps every edge.
pub fn spec(w: Which) c.Spec {
    return switch (w) {
        // One spec for both moves; a gather's `n` is its frame row stride, and the 17,408 shape is seam B's.
        .stage, .gather => .{
            .name = name(w),
            .accum = .none,
            .depth = noSums,
            .out = .bf16,
            .tiles = .{ .{}, .{}, .{ .size = 256, .step = 256, .min = 256 } },
            .served = if (w == .stage) &.{.{ .rows = 8, .n = 5120, .k = 5120 }} else &.{
                .{ .rows = 8, .n = 10240, .k = 5120 }, .{ .rows = 8, .n = 17408, .k = 8704 },
                .{ .rows = 8, .n = 10496, .k = 5120 },
            },
            .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = move_why },
        },
        .argmax => .{
            .name = name(w),
            .accum = .none,
            .depth = chainLogits,
            .out = .raw,
            .nan = .{ .stated = argmax_rule },
            .tiles = .{ .{}, .{}, .{ .size = 2048, .step = 64, .min = 64 } },
            .served = &.{.{ .rows = 8, .n = 1, .k = 124160 }},
        },
        .pick => .{
            .name = name(w),
            .accum = .none,
            .depth = noSums,
            .out = .raw,
            .nan = .{ .stated = pick_rule },
            .served = &.{ .{ .rows = 32, .n = 1, .k = 1 }, .{ .rows = 128, .n = 1, .k = 1 } },
        },
    };
}

/// `constant PairRows&` at buffer 1: `width`, the frame row `stride` and which rank's half this side writes.
const PairRows = extern struct {
    width: u32,
    stride: u32,
    rank: u32,
    x: u32,
};

fn rowsArgs(width: usize, stride: usize, rank: u32, x: u32) PairRows {
    return .{ .width = @intCast(width), .stride = @intCast(stride), .rank = rank, .x = x };
}

fn bit(v: f32) u32 {
    return @bitCast(v);
}

fn floatOf(v: u32) f32 {
    return @bitCast(v);
}

/// One prepared call: the buffers the kernel is bound to, the host values its reference reads, and its own check.
pub const Pair = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    which: Which,
    mutation: controls.Mutation,
    pipe: mtl.Pipeline,
    /// The hooks this probe exposes: the raw kernels check their own rule, the two computing ones use the reference.
    v: g.VTable,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    rows: usize = 0,
    width: usize = 0,
    /// The rank this prepare writes first: the gather's `mine` half is its payload.
    rank: u32 = 0,
    /// The row stride in PairRows, in bf16 elements: the stage's source frame row, the gather's destination row.
    stride: usize = 0,

    // Host inputs of the last prepare, as the reference and the check read them (freed by `drop`).
    sh: ?[]u16 = null, // stage: the src payload, bf16 (rows, each a frame row of `stride`)
    mh: ?[]u16 = null, // gather: this rank's half (mine), bf16
    nh: ?[]u16 = null, // gather: the peer's half (recv), bf16
    kh: ?[]u16 = null, // argmax: the bf16 logits
    pw: ?[]u32 = null, // pick: this rank's (value bits, id) words
    qw: ?[]u32 = null, // pick: the peer's words

    msg: [192]u8 = undefined,
    sync: rig_.Ref = undefined,
    mine: rig_.Ref = undefined,
    peer: rig_.Ref = undefined,
    logits: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, which: Which, mutation: controls.Mutation) !Pair {
        const src = if (mutation == .none) sources.core_pair_rows else try controls.source(a, mutation);
        defer if (mutation != .none) a.free(src);
        const pipe = try rig_.pipeline(device, src, symbol(which));
        errdefer pipe.deinit();
        const want = launch(which);
        const tg = rig_.tgMemory(pipe);
        if (tg != want.tg_memory) {
            std.debug.print("metal pair: {s} takes {d} bytes of threadgroup memory, its record says {d}\n", .{ symbol(which), tg, want.tg_memory });
            return error.ThreadgroupMemoryMismatch;
        }
        var v = g.hooks(Pair);
        if (which != .argmax and which != .pick) v.check = null;
        if (which != .stage and which != .gather) v.bounds = null;
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .which = which, .mutation = mutation, .pipe = pipe, .v = v };
    }

    pub fn deinit(self: *Pair) void {
        self.drop();
        self.rig.deinit();
        self.pipe.deinit();
    }

    pub fn kernel(self: *Pair) g.Kernel {
        var s = spec(self.which);
        if (self.mutation != .none) {
            s.name = self.mutation.name();
            return .{ .spec = s, .ctx = self, .vt = &self.v, .control = self.mutation.gate() };
        }
        return .{ .spec = s, .ctx = self, .vt = &self.v };
    }

    fn drop(self: *Pair) void {
        if (self.sh) |s| self.a.free(s);
        if (self.mh) |s| self.a.free(s);
        if (self.nh) |s| self.a.free(s);
        if (self.kh) |s| self.a.free(s);
        if (self.pw) |s| self.a.free(s);
        if (self.qw) |s| self.a.free(s);
        self.sh = null;
        self.mh = null;
        self.nh = null;
        self.kh = null;
        self.pw = null;
        self.qw = null;
    }

    /// Ordinary values: what the seam carries in production, symmetric about zero.
    fn ordinary(rng: std.Random) f32 {
        return rng.float(f32) * 2 - 1;
    }

    /// A bf16 payload: what the two moves carry, with gate 1's non-finite plants where the reference looks for them.
    fn payload(self: *Pair, len: usize, fill: g.Fill, rng: std.Random) ![]u16 {
        const s = try self.a.alloc(u16, len);
        for (s) |*v| v.* = c.toBf16(ordinary(rng));
        if (fill == .nonfinite) {
            const nan = c.toBf16(std.math.nan(f32));
            const inf = c.toBf16(std.math.inf(f32));
            const ninf = c.toBf16(-std.math.inf(f32));
            for ([_]u16{ nan, inf, ninf, nan, ninf }, 0..) |plant, p| {
                const at = (p + 1) * (len / 8 + 1) + p * 3;
                if (at < len) s[at] = plant;
            }
        }
        return s;
    }

    /// Inputs for `shape` as `fill` asks, outputs poisoned and guarded; false when the kernel does not serve it.
    pub fn prepare(self: *Pair, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 512) return false;
        self.drop();
        self.rig.reset();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        switch (self.which) {
            .stage, .gather => {
                if (shape.k == 0 or shape.k > 32768 or shape.k % 256 != 0) return false;
                // The gather writes this rank's half first; alternating the rank with the seed covers both orders.
                self.rank = @intCast(seed & 1);
                self.sync = try self.rig.scratch(128);
                if (self.which == .stage) {
                    // The source is a frame row: the stage reads this rank's `width` columns out of its `stride`.
                    self.stride = 2 * shape.k;
                    const len = shape.rows * self.stride;
                    const mine = try self.payload(len, fill, rng);
                    self.sh = mine;
                    self.mine = try self.rig.input(std.mem.sliceAsBytes(mine));
                    self.out = try self.rig.output(shape.rows * shape.k * 2);
                } else {
                    const n = shape.rows * shape.k;
                    // A frame row holds both halves, so its stride is at least 2*width: `n` sets a wider one.
                    self.stride = if (shape.n > 2 * shape.k) shape.n else 2 * shape.k;
                    const mine = try self.payload(n, fill, rng);
                    const peer = try self.payload(n, fill, rng);
                    self.mh = mine;
                    self.nh = peer;
                    self.mine = try self.rig.input(std.mem.sliceAsBytes(mine));
                    self.peer = try self.rig.input(std.mem.sliceAsBytes(peer));
                    self.out = try self.rig.output(2 * ((shape.rows - 1) * self.stride + 2 * shape.k));
                }
            },
            .argmax => {
                if (shape.k == 0 or shape.k > 262144) return false;
                const n = shape.rows * shape.k;
                const k = try self.a.alloc(u16, n);
                self.kh = k;
                for (k) |*v| v.* = c.toBf16(ordinary(rng));
                // Two in-range ids at +2.0: the lower id sits in the first half, so the tie rule's pick is fixed.
                for (0..shape.rows) |r| {
                    const row = k[r * shape.k ..][0..shape.k];
                    row[0] = c.toBf16(2.0);
                    row[shape.k / 2] = c.toBf16(2.0);
                }
                if (fill == .nonfinite) {
                    const nan = c.toBf16(std.math.nan(f32));
                    const inf = c.toBf16(std.math.inf(f32));
                    const ninf = c.toBf16(-std.math.inf(f32));
                    for (0..shape.rows) |r| {
                        const row = k[r * shape.k ..][0..shape.k];
                        row[shape.k / 3] = switch (r % 4) {
                            0 => nan,
                            1 => inf,
                            2 => ninf,
                            else => row[shape.k / 3],
                        };
                        if (r + 1 == shape.rows) for (row) |*v| {
                            v.* = nan;
                        };
                    }
                }
                self.logits = try self.rig.input(std.mem.sliceAsBytes(k));
                self.out = try self.rig.output(shape.rows * 8);
            },
            .pick => {
                if (shape.n != 1 or shape.k != 1) return false;
                self.pw = try self.a.alloc(u32, 2 * shape.rows);
                self.qw = try self.a.alloc(u32, 2 * shape.rows);
                for (0..shape.rows) |r| {
                    self.pw.?[2 * r] = bit(ordinary(rng));
                    self.qw.?[2 * r] = bit(ordinary(rng));
                    self.pw.?[2 * r + 1] = rng.uintLessThan(u32, 1 << 20);
                    self.qw.?[2 * r + 1] = rng.uintLessThan(u32, 1 << 20);
                }
                // Equal values on both ranks: the tie rule decides, and its direction alternates row by row.
                for (0..shape.rows) |r| {
                    if (r % 4 != 2 and r % 4 != 3) continue;
                    self.qw.?[2 * r] = self.pw.?[2 * r];
                    self.pw.?[2 * r + 1] = 300 + @as(u32, @intCast(r));
                    self.qw.?[2 * r + 1] = if (r % 8 == 2) 700 + @as(u32, @intCast(r)) else 100 + @as(u32, @intCast(r));
                }
                if (fill == .nonfinite) {
                    // Two plants prove the id flag is the rule: a flagged row draws the sentinel, a plain one picks.
                    for (0..shape.rows) |r| {
                        switch (r % 4) {
                            0 => self.pw.?[2 * r + 1] |= flag,
                            1 => self.qw.?[2 * r + 1] |= flag,
                            2 => self.pw.?[2 * r] = bit(-std.math.inf(f32)),
                            else => self.qw.?[2 * r] = bit(std.math.inf(f32)),
                        }
                    }
                }
                self.sync = try self.rig.scratch(128);
                self.mine = try self.rig.input(std.mem.sliceAsBytes(self.pw.?));
                self.peer = try self.rig.input(std.mem.sliceAsBytes(self.qw.?));
                self.out = try self.rig.output(shape.rows * 4);
            },
        }
        self.shape = shape;
        self.rows = shape.rows;
        self.width = shape.k;
        return true;
    }

    /// One launch on the prepared buffers, waited for.
    pub fn run(self: *Pair) !void {
        const p = self.rig.begin();
        const e = p.e;
        const groups = mtl.Size.of(self.rows, 1, 1);
        const threads = mtl.Size.of(launch(self.which).threads, 1, 1);
        switch (self.which) {
            .stage => {
                @memset(self.rig.bytes(self.sync, 128), 0);
                e.setPipeline(self.pipe);
                e.setBuffer(self.sync.buf, self.sync.off, 0);
                e.setValue(rowsArgs(self.width, self.stride, self.rank, 1), 1);
                e.setBuffer(self.mine.buf, self.mine.off, 2);
                e.setBuffer(self.out.buf, self.out.off, 3);
                e.dispatchGroups(groups, threads);
            },
            .gather => {
                @memset(self.rig.bytes(self.sync, 128), 0);
                const frame = self.rig.bytes(self.out, 2 * self.count());
                @memset(frame, 0); // the frame's gaps: no launch writes them, so their planted value is zero
                e.setPipeline(self.pipe);
                e.setBuffer(self.sync.buf, self.sync.off, 0);
                e.setValue(rowsArgs(self.width, self.stride, self.rank, 1), 1);
                e.setBuffer(self.mine.buf, self.mine.off, 2);
                e.setBuffer(self.peer.buf, self.peer.off, 3);
                e.setBuffer(self.out.buf, self.out.off, 4);
                e.dispatchGroups(groups, threads);
            },
            .argmax => {
                e.setPipeline(self.pipe);
                e.setBuffer(self.logits.buf, self.logits.off, 0);
                const dims = [4]u32{ @intCast(self.width), logit_lo, @intCast(self.width), 1 };
                e.setValue(dims, 1);
                e.setBuffer(self.out.buf, self.out.off, 2);
                e.dispatchGroups(groups, threads);
            },
            .pick => {
                @memset(self.rig.bytes(self.sync, 128), 0);
                e.setPipeline(self.pipe);
                e.setBuffer(self.sync.buf, self.sync.off, 0);
                e.setValue([2]u32{ @intCast(self.rows), 1 }, 1);
                e.setBuffer(self.mine.buf, self.mine.off, 2);
                e.setBuffer(self.peer.buf, self.peer.off, 3);
                e.setBuffer(self.out.buf, self.out.off, 4);
                e.dispatchGroups(mtl.Size.of(1, 1, 1), threads);
            },
        }
        try self.rig.finish(p);
    }

    /// Output elements of the prepared call: bf16 elements for the two moves, the two raw kernels' bytes.
    pub fn count(self: *Pair) usize {
        return switch (self.which) {
            .stage => self.rows * self.width,
            .gather => (self.rows - 1) * self.stride + 2 * self.width,
            .argmax => self.rows * 8,
            .pick => self.rows * 4,
        };
    }

    /// Float64 outputs of the prepared inputs and each one's sums' term magnitude (a move's term is the value itself).
    pub fn reference(self: *Pair, want: []f64, mag: []f64) !void {
        const width = self.width;
        switch (self.which) {
            .stage => {
                const sh = self.sh orelse return error.NoReference;
                for (0..self.rows) |r| {
                    const row = sh[r * self.stride ..][0..width];
                    for (0..width) |i| {
                        const v = c.fromBf16(row[i]);
                        want[r * width + i] = v;
                        mag[r * width + i] = @abs(v);
                    }
                }
            },
            .gather => {
                const mh = self.mh orelse return error.NoReference;
                const nh = self.nh orelse return error.NoReference;
                @memset(want, 0); // the frame's gaps: no launch writes them, so their reference value is zero
                for (0..self.rows) |row| {
                    const at = row * self.stride;
                    for (0..width) |i| {
                        want[at + i] = c.fromBf16(if (self.rank == 0) mh[row * width + i] else nh[row * width + i]);
                        want[at + width + i] = c.fromBf16(if (self.rank == 0) nh[row * width + i] else mh[row * width + i]);
                    }
                }
                for (want, mag) |w, *m| m.* = @abs(w);
            },
            else => return error.NoReference,
        }
    }

    /// The moves are exact: every output element must equal the reference's bf16 bit for bit.
    pub fn bounds(self: *Pair, want: []const f64, mag: []const f64, allow: []f64) !void {
        _ = want;
        _ = mag;
        if (self.which != .stage and self.which != .gather) return error.NotMove;
        for (allow) |*l| l.* = 0;
    }

    /// The seam's own checks: the raw kernels' rule at gate 1 (non-finite inputs) and gate 3 (every served edge).
    pub fn check(self: *Pair, _: g.Gate) !?[]const u8 {
        const out = try self.rig.outputBytes();
        switch (self.which) {
            .argmax => {
                const k = self.kh orelse return error.NoReference;
                const n = self.width;
                for (0..self.rows) |r| {
                    var best: f32 = -std.math.inf(f32);
                    var at: u32 = sentinel;
                    var bad = false;
                    for (0..n) |i| {
                        const v = c.fromBf16(k[r * n + i]);
                        if (!std.math.isFinite(v)) {
                            bad = true;
                            continue;
                        }
                        if (v > best) {
                            best = v;
                            at = logit_lo + @as(u32, @intCast(i));
                        }
                    }
                    const got = std.mem.readInt(u32, out[r * 8 ..][0..4], .little);
                    const id = std.mem.readInt(u32, out[r * 8 + 4 ..][0..4], .little);
                    if (bad) {
                        if (!std.math.isNan(floatOf(got))) return try std.fmt.bufPrint(&self.msg, "row {d}: value {e} where a non-finite logit must write NaN", .{ r, floatOf(got) });
                        if (id != at | flag) return try std.fmt.bufPrint(&self.msg, "row {d}: id {d} where flagging gives {d}", .{ r, id, at | flag });
                    } else if (got != bit(best) or id != at) {
                        return try std.fmt.bufPrint(&self.msg, "row {d}: ({e} {d}) where its own rule gives ({e} {d})", .{ r, floatOf(got), id, best, at });
                    }
                }
            },
            .pick => {
                const pw = self.pw orelse return error.NoReference;
                const qw = self.qw orelse return error.NoReference;
                for (0..self.rows) |r| {
                    const vm = floatOf(pw[2 * r]);
                    const am = pw[2 * r + 1];
                    const vp = floatOf(qw[2 * r]);
                    const ap = qw[2 * r + 1];
                    const flagged = ((am | ap) & flag) != 0;
                    const want = if (flagged) sentinel else if (vp > vm or (vp == vm and ap < am)) ap else am;
                    const got = std.mem.readInt(u32, out[r * 4 ..][0..4], .little);
                    if (got != want) return try std.fmt.bufPrint(&self.msg, "row {d} picks {d} where its own rule gives {d}", .{ r, got, want });
                }
            },
            else => return error.NotRaw,
        }
        return null;
    }
};
