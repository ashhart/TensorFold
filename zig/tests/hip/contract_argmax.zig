//! The argmax of a wide logits row in parts under the contract on HIP: tf_argmax's picks, by its stated order.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

const per = 1024; // a part's values

fn none(_: c.Shape) usize {
    return 0;
}

const names = [2][]const u8{
    "hip tf_argmax_part and tf_argmax_join (fp16)",
    "hip tf_argmax_part and tf_argmax_join (bf16)",
};
const rule = "torch's argmax order: a NaN ahead of every value, then the larger value, then the lower index";

pub fn spec(k: kit.Kind) c.Spec {
    return .{
        .name = names[@intFromBool(k == .bf16)],
        .accum = .none,
        .depth = none,
        .out = .raw,
        .replaces = "tf_argmax",
        .nan = .{ .stated = rule },
        .tiles = .{ .{}, .{ .size = per }, .{} },
        // a greedy row and a draft window over Nano's 131,072 logits, and the narrowest row the parts take
        .served = &.{
            .{ .rows = 1, .n = 131072, .k = 0 },
            .{ .rows = 4, .n = 131072, .k = 0 },
            .{ .rows = 1, .n = 2048, .k = 0 },
        },
    };
}

/// shape.rows rows of shape.n logits of the kind: each row's pick as an int32.
pub const Argmax = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    k: kit.Kind,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    x: []f64 = &.{},
    in: u64 = 0,
    parts: [2]u64 = @splat(0),
    out: u64 = 0,
    base: u64 = 0,
    msg: [160]u8 = undefined,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher) Argmax {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .k = kit.act(), .arena = .init(a) };
    }

    pub fn deinit(self: *Argmax) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Argmax) g.Kernel {
        return .{ .spec = spec(self.k), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Argmax);

    fn count_parts(self: *const Argmax) usize {
        return (self.shape.n + per - 1) / per;
    }

    /// Values in eighths, so ties are common; non-finite: a NaN row, an infinite row, and -Inf among numbers.
    pub fn prepare(self: *Argmax, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = shape.rows * shape.n;
        self.x = try a.alloc(f64, n);
        for (self.x) |*v| v.* = @as(f64, @floatFromInt(rng.uintLessThan(u32, 256))) / 8 - 16;
        if (fill == .nonfinite) {
            const at = [_][2]usize{
                .{ 0, shape.n / 3 },
                .{ 0, shape.n - 1 },
                .{ 1, shape.n / 2 },
                .{ 2, 0 },
                .{ 3, 7 },
            };
            const put = [_]f64{
                std.math.nan(f64),
                std.math.nan(f64),
                std.math.inf(f64),
                -std.math.inf(f64),
                std.math.inf(f64),
            };
            for (at, put) |p, v| if (p[0] < shape.rows and p[1] < shape.n) {
                self.x[p[0] * shape.n + p[1]] = v;
            };
        }
        const b = try a.alloc(u8, n * 2);
        for (self.x, 0..) |v, i| kit.put(self.k, b, i, v);
        self.in = try self.rig.input(b);
        const parts = shape.rows * self.count_parts();
        self.parts = .{ try self.rig.scratch(parts * 4), try self.rig.scratch(parts * 4) };
        self.out = try self.rig.output(shape.rows * 4);
        self.base = try self.rig.scratch(shape.rows * 4);
        return true;
    }

    pub fn run(self: *Argmax) !void {
        const s = self.shape;
        try self.l.tf_argmax_parts(
            @ptrFromInt(self.in),
            @backingInt(self.k),
            @intCast(s.rows),
            @intCast(s.n),
            per,
            @intCast(self.count_parts()),
            @ptrFromInt(self.parts[0]),
            @ptrFromInt(self.parts[1]),
            @ptrFromInt(self.out),
            self.rig.s.handle,
        );
        try self.rig.s.synchronize();
    }

    /// tf_argmax's one block a row on the same rows.
    pub fn replaced(self: *Argmax) ![]const u8 {
        const s = self.shape;
        try self.l.tf_argmax_rows(
            @ptrFromInt(self.in),
            @backingInt(self.k),
            @intCast(s.rows),
            @intCast(s.n),
            @ptrFromInt(self.base),
            self.rig.s.handle,
        );
        return self.rig.bytes(self.base, s.rows * 4);
    }

    pub fn count(self: *Argmax) usize {
        return self.shape.rows * 4;
    }

    pub fn reference(_: *Argmax, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// Each row's pick against the stated order over the row's values as the kind holds them.
    pub fn check(self: *Argmax, _: g.Gate) !?[]const u8 {
        const s = self.shape;
        const got = try self.rig.bytes(self.out, s.rows * 4);
        for (0..s.rows) |r| {
            const row = self.x[r * s.n ..][0..s.n];
            var at: usize = 0;
            for (row, 0..) |v, i| {
                const top = row[at];
                const nan = std.math.isNan(v);
                if (!std.math.isNan(top) and (nan or v > top)) at = i;
            }
            const id = std.mem.readInt(i32, got[4 * r ..][0..4], .little);
            if (id != @as(i32, @intCast(at))) return try std.fmt.bufPrint(
                &self.msg,
                "row {d} picks {d}, the order gives {d}",
                .{ r, id, at },
            );
        }
        return null;
    }
};
