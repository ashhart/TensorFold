//! The glue's embedding gather on XPU: MLX affine 4-bit words with bf16 scales and biases, as embed stores them.
const std = @import("std");
const xpu = @import("xpu");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const c = core.spec;
const g = core.gates;
const ref = core.xpu_ref;

const VOCAB = 4099; // the table's rows: an id's first and last, and the ones between
const GS = 64; // the group a scale and a bias cover
const D = 5120; // Q27_D, the widest row the gather takes

fn none(_: c.Shape) usize {
    return 0;
}

const rule = "a NaN stays a NaN, an infinity stays an infinity, and a finite value rounds once to bf16, to nearest " ++
    "even, as glue.cl's embed stores it";

const served = [_]c.Shape{
    .{ .rows = 1, .n = D, .k = 0 },
    .{ .rows = 7, .n = D, .k = 0 },
    .{ .rows = 512, .n = D, .k = 0 },
    .{ .rows = 1152, .n = D, .k = 0 },
};

pub fn spec() c.Spec {
    return .{
        .name = "xpu embed (the gathered row, fp32)",
        .accum = .none,
        .depth = none,
        .out = .f32,
        .nan = .{ .stated = rule },
        .tiles = .{ .{ .size = 16, .min = 1 }, .{ .size = D, .step = GS, .min = GS }, .{} },
        .served = &served,
    };
}

/// shape.rows ids into a table of VOCAB rows of shape.n: the 4-bit words and their bf16 scales and biases.
pub const Embed = struct {
    rig: rig_.Rig,
    k: xpu.Kernel,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    ids: []i32 = &.{},
    q: []u8 = &.{},
    s: []f64 = &.{},
    b: []f64 = &.{},
    ids_dev: ?*anyopaque = null,
    w_dev: ?*anyopaque = null,
    s_dev: ?*anyopaque = null,
    b_dev: ?*anyopaque = null,
    out: ?*anyopaque = null,
    msg: [192]u8 = undefined,

    pub fn init(a: std.mem.Allocator, dev: *rig_.Dev) !Embed {
        return .{ .rig = .{ .a = a, .dev = dev }, .k = try dev.kernelFor(.glue, "embed", a), .arena = .init(a) };
    }

    pub fn deinit(self: *Embed) void {
        self.k.deinit();
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Embed) g.Kernel {
        return .{ .spec = spec(), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Embed);

    fn nibble(w: u32, e: usize) u8 {
        return @intCast((w >> @intCast(4 * (e % 8))) & 0xf);
    }

    pub fn prepare(self: *Embed, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % GS != 0 or shape.n > D) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = shape.n;
        const words = try a.alloc(u32, VOCAB * (n / 8));
        for (words) |*w| w.* = rng.int(u32);
        self.q = try a.alloc(u8, VOCAB * n);
        for (words, 0..) |w, i| {
            for (0..8) |j| self.q[i * 8 + j] = nibble(w, j);
        }
        self.s = try a.alloc(f64, VOCAB * (n / GS));
        self.b = try a.alloc(f64, VOCAB * (n / GS));
        for (self.s) |*v| v.* = kit.round(.bf16, 0.001 + rng.float(f64) * 0.02);
        for (self.b) |*v| v.* = kit.round(.bf16, -0.05 + rng.float(f64) * 0.1);
        self.ids = try a.alloc(i32, shape.rows);
        for (self.ids, 0..) |*id, r| id.* = switch (r) {
            0 => VOCAB - 1,
            1 => 0,
            else => rng.intRangeLessThan(i32, 0, VOCAB),
        };
        if (fill == .nonfinite) {
            const planted = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), 60000 };
            for (planted, 0..) |v, j| {
                const row: usize = @intCast(self.ids[j % shape.rows]);
                self.s[row * (n / GS) + (j * 37) % (n / GS)] = kit.round(.bf16, v);
                self.b[row * (n / GS) + (j * 11 + 1) % (n / GS)] = kit.round(.bf16, v);
            }
        }
        const sb = try kit.bytes(a, self.s, .bf16);
        const bb = try kit.bytes(a, self.b, .bf16);
        self.ids_dev = try self.rig.input(std.mem.sliceAsBytes(self.ids));
        self.w_dev = try self.rig.input(std.mem.sliceAsBytes(words));
        self.s_dev = try self.rig.input(sb);
        self.b_dev = try self.rig.input(bb);
        self.out = try self.rig.output(shape.rows * n * 4);
        return true;
    }

    pub fn run(self: *Embed) !void {
        const s = self.shape;
        try self.rig.launch(self.k, .{ @intCast(s.rows), 1, 1 }, .{
            self.ids_dev, self.w_dev, self.s_dev, self.b_dev, self.out, @as(u32, @intCast(s.n)),
        });
    }

    pub fn count(self: *Embed) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn reference(_: *Embed, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// Every output element against its table value rounded once to bf16, by the stated rule.
    pub fn check(self: *Embed, _: g.Gate) !?[]const u8 {
        const s = self.shape;
        const n = s.n;
        const got = try self.rig.bytes(self.out, self.count() * 4);
        for (0..s.rows) |r| for (0..n) |col| {
            const id: usize = @intCast(self.ids[r]);
            const q: f64 = @floatFromInt(self.q[id * n + col]);
            const want = ref.embedValue(q, self.s[id * (n / GS) + col / GS], self.b[id * (n / GS) + col / GS]);
            const out = c.read(.f32, got, r * n + col);
            const same = if (std.math.isNan(want)) std.math.isNan(out) else out == want;
            if (!same) return try std.fmt.bufPrint(&self.msg, "row {d} (id {d}) column {d} is {e}, the table's rounds to {e}", .{ r, self.ids[r], col, out, want });
        };
        return null;
    }
};
