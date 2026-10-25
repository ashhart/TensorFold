//! sample.cu's tf_penalize_rows under the contract: entry i takes SUB[i] from logit TOK[i] of row ROW[i], in place.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const bfr = core.bf16_round;

/// Logits a row: entries are distinct (row, token) cells, so a shape's rows x vocab holds its n entries.
const vocab = 1024;

fn none(_: c.Shape) usize {
    return 0;
}

/// Shape: logits rows x entries (n); each entry is one fp32 subtraction rounded once to bf16, no sums.
pub const spec: c.Spec = .{
    .name = "cuda sample tf_penalize_rows",
    .accum = .none,
    .depth = none,
    .ops = 1,
    .out = .bf16,
    .in_place = true,
    .tiles = .{ .{}, .{ .size = 256, .step = 1, .min = 1 }, .{} }, // 256 entries a block
    .served = &.{ .{ .rows = 1, .n = 1, .k = 1 }, .{ .rows = 128, .n = 4096, .k = 1 }, .{ .rows = 128, .n = 128 * vocab, .k = 1 } },
};

pub const Penalize = struct {
    rig: rig_.Rig,
    draws: *const cuda.sample.Draws,
    arena: ?std.heap.ArenaAllocator = null,
    rows: usize = 0,
    n: usize = 0,
    logits: []u16 = &.{}, // the rows as the launch finds them
    row: []u32 = &.{},
    tok: []u32 = &.{},
    sub: []f32 = &.{},
    out: u64 = 0,
    entries: u64 = 0,

    pub fn deinit(self: *Penalize) void {
        if (self.arena) |*a| a.deinit();
        self.rig.deinit();
    }

    const vt = g.hooks(Penalize);

    pub fn kernel(self: *Penalize) g.Kernel {
        return .{ .spec = spec, .ctx = self, .vt = &vt };
    }

    pub fn prepare(self: *Penalize, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n > shape.rows * vocab) return false;
        self.rig.reset();
        if (self.arena) |*a| a.deinit();
        self.arena = std.heap.ArenaAllocator.init(self.rig.gpu.gpa);
        const a = self.arena.?.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.rows = shape.rows;
        self.n = shape.n;
        self.logits = try a.alloc(u16, shape.rows * vocab);
        for (self.logits) |*x| x.* = bfr.bits(rng.float(f32) * 40 - 20);
        // distinct cells: a permutation's first n, so no two entries write one logit
        const cells = try a.alloc(u32, shape.rows * vocab);
        for (cells, 0..) |*x, i| x.* = @intCast(i);
        rng.shuffle(u32, cells);
        self.row = try a.alloc(u32, shape.n);
        self.tok = try a.alloc(u32, shape.n);
        self.sub = try a.alloc(f32, shape.n);
        for (self.row, self.tok, self.sub, cells[0..shape.n]) |*r, *t, *s, cell| {
            r.* = cell / vocab;
            t.* = cell % vocab;
            s.* = rng.float(f32) * 4 - 1; // presence + frequency * count, in OpenAI's -2..2 range and a little past
        }
        if (fill == .nonfinite) {
            const planted = [_]u16{ bfr.bits(std.math.nan(f32)), bfr.bits(std.math.inf(f32)), bfr.bits(-std.math.inf(f32)) };
            for (planted, 0..) |bits, i| if (i < shape.n) {
                self.logits[self.row[i] * vocab + self.tok[i]] = bits;
            };
            if (shape.n > 3) self.sub[3] = std.math.inf(f32);
            if (shape.n > 4) self.sub[4] = std.math.nan(f32);
        }
        if (fill == .stress) for (self.sub) |*s| { // bf16's largest values: the difference rounds, or overflows
            s.* = 0x1p127;
        };
        self.out = try self.rig.output(self.logits.len * 2);
        const bytes = try a.alloc(u8, shape.n * 12);
        @memcpy(bytes[0 .. shape.n * 4], std.mem.sliceAsBytes(self.row));
        @memcpy(bytes[shape.n * 4 .. shape.n * 8], std.mem.sliceAsBytes(self.tok));
        @memcpy(bytes[shape.n * 8 ..], std.mem.sliceAsBytes(self.sub));
        self.entries = try self.rig.input(bytes);
        return true;
    }

    /// The rows go in fresh each run (the launch writes them in place), then one tf_penalize_rows over the entries.
    pub fn run(self: *Penalize) !void {
        const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.out, .len = self.logits.len * 2 };
        try b.upload(0, std.mem.sliceAsBytes(self.logits));
        try self.draws.penalize(self.rig.s, self.out, vocab, self.entries, self.n);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Penalize) usize {
        return self.logits.len;
    }

    /// Every logit as it was, except each entry's, less its amount; no sums, so no magnitude.
    pub fn reference(self: *Penalize, want: []f64, mag: []f64) !void {
        for (want, self.logits) |*w, x| w.* = bfr.value(x);
        for (self.row, self.tok, self.sub) |r, t, s| want[r * vocab + t] -= s;
        @memset(mag, 0);
    }
};

test "a shape's entries fit its rows, and the served shapes do" {
    for (spec.served) |s| try std.testing.expect(s.n <= s.rows * vocab);
    try std.testing.expect(spec.in_place and spec.ops == 1 and spec.depth(spec.served[0]) == 0);
}
