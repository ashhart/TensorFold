//! The MoE down codes' prefetch under the contract on HIP: it reads the live items' experts and writes nothing.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

const held = 16; // the experts the codes buffer holds; every live id is below it

fn none(_: c.Shape) usize {
    return 0;
}

const rule = "no output: every buffer as prepared and every guard intact after each run; with no output it has no " ++
    "float class or bound, so gates 1 and 3 check this, and no sums, so gate 4 has none";

/// rows: items; n: experts an id may name; k: 16-byte words of one expert's codes.
pub fn spec() c.Spec {
    return .{
        .name = "hip tf_moe_prefetch",
        .accum = .none,
        .depth = none,
        .out = .raw,
        .nan = .{ .stated = rule },
        // a decode row's 11 items over Flash Next's 513 experts, each a 4-bit 2560 x 640 down of 51,200 words
        .served = &.{.{ .rows = 11, .n = 513, .k = 51200 }},
    };
}

/// shape.rows items naming experts below 16 (a dead item and an id past shape.n among them), and their codes.
pub const Prefetch = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    in: [2]u64 = @splat(0),
    before: []const u8 = &.{},
    msg: [120]u8 = undefined,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher) Prefetch {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .arena = .init(a) };
    }

    pub fn deinit(self: *Prefetch) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Prefetch) g.Kernel {
        return .{ .spec = spec(), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Prefetch);

    pub fn prepare(self: *Prefetch, shape: c.Shape, _: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const items = try a.alloc(i32, shape.rows * 3);
        for (0..shape.rows) |i| {
            const id: i32 = @intCast(if (i == 2) shape.n else rng.uintLessThan(usize, @min(held, shape.n)));
            items[3 * i ..][0..3].* = .{ id, @intCast(i), @intFromBool(i % 5 != 4) };
        }
        const words = try a.alloc(u8, @min(held, shape.n) * shape.k * 16);
        rng.bytes(words);
        self.in = .{ try self.rig.input(std.mem.sliceAsBytes(items)), try self.rig.input(words) };
        self.before = try a.dupe(u8, try self.rig.inputBytes());
        return true;
    }

    /// The engine's cap: every item's expert in full.
    pub fn run(self: *Prefetch) !void {
        const s = self.shape;
        const bytes: i64 = @intCast(s.k * 16);
        const items: c_int = @intCast(s.rows);
        try self.l.tf_moe_prefetch(@ptrFromInt(self.in[0]), items, self.in[1], bytes, @intCast(s.n), @as(i64, items) * bytes, self.rig.s.handle);
        try self.rig.s.synchronize();
    }

    pub fn count(_: *Prefetch) usize {
        return 0;
    }

    pub fn reference(_: *Prefetch, _: []f64, _: []f64) !void {}

    /// Every input byte as prepared and every guard intact.
    pub fn check(self: *Prefetch, _: g.Gate) !?[]const u8 {
        const now = try self.rig.inputBytes();
        if (std.mem.indexOfDiff(u8, self.before, now)) |at|
            return try std.fmt.bufPrint(&self.msg, "input byte {d} changed", .{at});
        if (!try self.rig.intact()) return "a run wrote outside its buffers";
        return null;
    }
};
