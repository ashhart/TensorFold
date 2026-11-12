//! The Qwen3.8-27B glue's argmax under the contract on XPU: the partial maxima, then their maximum, by its stated rule.
const std = @import("std");
const xpu = @import("xpu");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const c = core.spec;
const g = core.gates;
const ref = core.xpu_ref;

const PARTS = 256; // argmax_part's grid width, one part a work-group
const VOCAB = 248320; // Qwen3.8-27B's logits a row
const MAX_ROWS = 16; // decode and verification rows

fn none(_: c.Shape) usize {
    return 0;
}

const rule = "a NaN compares false, so it is skipped; the pick is the largest value above -Inf, the lowest index on a " ++
    "tie, and 0x7fffffff when no value compares above -Inf (a row of NaNs, or of -Inf)";

const served = [_]c.Shape{
    .{ .rows = 1, .n = VOCAB, .k = 0 },
    .{ .rows = 16, .n = VOCAB, .k = 0 },
};

pub fn spec() c.Spec {
    return .{
        .name = "xpu argmax (the pick)",
        .accum = .none,
        .depth = none,
        .out = .raw,
        .nan = .{ .stated = rule },
        .tiles = .{ .{ .size = 16, .min = 1 }, .{ .size = 131072, .step = 256, .min = 256 }, .{} },
        .served = &served,
    };
}

/// shape.rows rows of shape.n logits of fp16: each row's pick as an int32.
pub const Argmax = struct {
    rig: rig_.Rig,
    part: xpu.Kernel,
    k: xpu.Kernel,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    x: []f64 = &.{},
    inb: []u8 = &.{},
    xin: ?*anyopaque = null,
    pv: ?*anyopaque = null,
    pi: ?*anyopaque = null,
    out: ?*anyopaque = null,
    msg: [160]u8 = undefined,

    pub fn init(a: std.mem.Allocator, dev: *rig_.Dev) !Argmax {
        return .{
            .rig = .{ .a = a, .dev = dev },
            .part = try dev.kernel(.glue, "argmax_part", .{ PARTS, 1, 1 }),
            .k = try dev.kernelFor(.glue, "argmax", a),
            .arena = .init(a),
        };
    }

    pub fn deinit(self: *Argmax) void {
        self.k.deinit();
        self.part.deinit();
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Argmax) g.Kernel {
        return .{ .spec = spec(), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Argmax);

    pub fn prepare(self: *Argmax, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > MAX_ROWS or shape.n == 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = shape.rows * shape.n;
        self.x = try a.alloc(f64, n);
        for (self.x) |*v| v.* = kit.round(.f16, @as(f64, @floatFromInt(rng.uintLessThan(u32, 256))) / 8 - 16);
        if (fill == .nonfinite) {
            const at = [_][2]usize{
                .{ 0, shape.n / 3 },
                .{ 0, shape.n - 1 },
                .{ 1, shape.n / 2 },
                .{ 2, 0 },
                .{ 3, 7 },
            };
            const put = [_]f64{ std.math.nan(f64), std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), std.math.inf(f64) };
            for (at, put) |p, v| if (p[0] < shape.rows and p[1] < shape.n) {
                self.x[p[0] * shape.n + p[1]] = v;
            };
        }
        self.inb = try kit.bytes(a, self.x, .f16);
        self.xin = try self.rig.input(self.inb);
        const parts = shape.rows * PARTS;
        self.pv = try self.rig.scratch(parts * 4);
        self.pi = try self.rig.scratch(parts * 4);
        self.out = try self.rig.output(shape.rows * 4);
        return true;
    }

    pub fn run(self: *Argmax) !void {
        const s = self.shape;
        try self.rig.launch(self.part, .{ PARTS, @intCast(s.rows), 1 }, .{
            self.xin, @as(u32, @intCast(s.n)), self.pv, self.pi,
        });
        try self.rig.launch(self.k, .{ @intCast(s.rows), 1, 1 }, .{ self.pv, self.pi, self.out });
    }

    pub fn count(self: *Argmax) usize {
        return self.shape.rows * 4;
    }

    pub fn reference(_: *Argmax, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// Each row's pick against the stated rule over the row's values as fp16 holds them.
    pub fn check(self: *Argmax, _: g.Gate) !?[]const u8 {
        const s = self.shape;
        const got = try self.rig.bytes(self.out, s.rows * 4);
        for (0..s.rows) |r| {
            const want = ref.argmaxPick(self.x[r * s.n ..][0..s.n]);
            const id = std.mem.readInt(i32, got[4 * r ..][0..4], .little);
            if (id != want) return try std.fmt.bufPrint(&self.msg, "row {d} picks {d}, the rule gives {d}", .{ r, id, want });
        }
        return null;
    }
};
