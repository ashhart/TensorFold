//! The top-k of a wide logits row in parts under the contract on HIP: tf_topk's ids, values and order.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

const part_least = 4096; // ops/draw.zig part_topk: the values a part takes at least
const most_parts = 64; // ops/draw.zig max_topk_parts
const join_most = 4096; // draw.hpp kJoinMost: the candidates a row's join holds

fn none(_: c.Shape) usize {
    return 0;
}

const names = [2][]const u8{
    "hip tf_topk_part and tf_topk_join (fp16)",
    "hip tf_topk_part and tf_topk_join (bf16)",
};
const rule = "each row's ks[r] largest by a 16-bit key of the value's bits (-0 equal to +0, a positive NaN above " ++
    "+Inf, a negative NaN below -Inf), ties to the lower id, as ids and the values' bits; no slot past ks[r] written";

/// rows: logits rows; n: values a row; k: the stride, the most any row takes.
pub fn spec(k: kit.Kind) c.Spec {
    return .{
        .name = names[@intFromBool(k == .bf16)],
        .accum = .none,
        .depth = none,
        .out = .raw,
        .replaces = "tf_topk",
        .nan = .{ .stated = rule },
        // the narrowest row ops.topkRows takes in parts
        .tiles = .{ .{}, .{ .size = 2 * part_least, .min = 2 * part_least }, .{} },
        // a row and a window of rows over Flash Next's 248,320 logits, and the narrowest row the parts take
        .served = &.{
            .{ .rows = 1, .n = 248320, .k = 64 },
            .{ .rows = 4, .n = 248320, .k = 64 },
            .{ .rows = 4, .n = 2 * part_least, .k = 64 },
        },
    };
}

/// draw.hpp order_key: monotone in the value, -0 and +0 one key, NaNs past the infinities by their sign.
fn key(bits: u16) u32 {
    const b: u32 = if (bits == 0x8000) 0 else bits;
    return if (b & 0x8000 != 0) ~b & 0xffff else b | 0x8000;
}

/// shape.rows rows of shape.n 16-bit logits: each row's ks[r] largest as ids and values, shape.k slots apart.
pub const Topk = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    k: kit.Kind,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    bits: []u16 = &.{},
    ks: []i32 = &.{},
    parts: usize = 0,
    per: usize = 0,
    in: [2]u64 = @splat(0),
    cand: [2]u64 = @splat(0),
    out: [2]u64 = @splat(0),
    base: [2]u64 = @splat(0),
    msg: [160]u8 = undefined,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher) Topk {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .k = kit.act(), .arena = .init(a) };
    }

    pub fn deinit(self: *Topk) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Topk) g.Kernel {
        return .{ .spec = spec(self.k), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Topk);

    /// Values in eighths, so ties are common; ks cycle 28, 1, 0 and the stride; non-finite: NaNs, infinities, zeros.
    pub fn prepare(self: *Topk, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n < 16 or shape.k == 0) return false;
        // ops.topkRows' parts of a row
        const parts = @min(most_parts, (shape.n + part_least - 1) / part_least);
        if (parts * shape.k > join_most) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        self.parts = parts;
        self.per = (shape.n + parts - 1) / parts;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = shape.rows * shape.n;
        const b = try a.alloc(u8, n * 2);
        for (0..n) |i| kit.put(self.k, b, i, @as(f64, @floatFromInt(rng.uintLessThan(u32, 256))) / 8 - 16);
        self.bits = try a.alloc(u16, n);
        for (self.bits, 0..) |*v, i| v.* = std.mem.readInt(u16, b[2 * i ..][0..2], .little);
        if (fill == .nonfinite) {
            const nan: u16 = if (self.k == .bf16) 0x7fc0 else 0x7e00;
            const inf: u16 = if (self.k == .bf16) 0x7f80 else 0x7c00;
            const at = [_][2]usize{
                .{ 0, shape.n / 3 }, .{ 0, shape.n - 1 }, .{ 1, shape.n / 2 }, .{ 1, 5 }, .{ 2, 0 },
                .{ 2, 9 },           .{ 2, 10 },          .{ 3, 7 },           .{ 3, 8 }, .{ 4, 1 },
                .{ 4, 2 },
            };
            const put = [_]u16{ nan, nan, inf, nan | 0x8000, inf | 0x8000, 0x8000, 0, inf, inf, nan | 0x8000, nan };
            for (at, put) |p, v| if (p[0] < shape.rows) {
                self.bits[p[0] * shape.n + p[1]] = v;
            };
        }
        self.ks = try a.alloc(i32, shape.rows);
        const cycle = [_]usize{ 28, 1, 0, shape.k };
        for (self.ks, 0..) |*v, r| v.* = @intCast(@min(cycle[r % 4], shape.k, shape.n));
        self.in = .{
            try self.rig.input(std.mem.sliceAsBytes(self.bits)),
            try self.rig.input(std.mem.sliceAsBytes(self.ks)),
        };
        const cands = shape.rows * parts * shape.k;
        self.cand = .{ try self.rig.scratch(cands * 4), try self.rig.scratch(cands * 2) };
        const lens = [2]usize{ shape.rows * shape.k * 4, shape.rows * shape.k * 2 };
        for (&self.out, &self.base, lens) |*o, *base, len| {
            o.* = try self.rig.output(len);
            base.* = try self.rig.scratch(len);
            const poison = try a.alloc(u8, len);
            @memset(poison, 0xff); // as the outputs start, so a slot neither writes compares equal
            try self.rig.put(base.*, poison);
        }
        return true;
    }

    pub fn run(self: *Topk) !void {
        const s = self.shape;
        try self.l.tf_topk_parts(
            @ptrFromInt(self.in[0]),
            @intCast(s.rows),
            @intCast(s.n),
            @ptrFromInt(self.in[1]),
            @intCast(s.k),
            @intCast(self.per),
            @intCast(self.parts),
            @ptrFromInt(self.cand[0]),
            @ptrFromInt(self.cand[1]),
            @ptrFromInt(self.out[0]),
            @ptrFromInt(self.out[1]),
            self.rig.s.handle,
        );
        try self.rig.s.synchronize();
    }

    /// Ids then values at two device addresses, on the host.
    fn both(self: *Topk, at: [2]u64) ![]const u8 {
        const a = self.arena.allocator();
        const s = self.shape;
        const ids = try a.dupe(u8, try self.rig.bytes(at[0], s.rows * s.k * 4));
        return std.mem.concat(a, u8, &.{ ids, try self.rig.bytes(at[1], s.rows * s.k * 2) });
    }

    /// tf_topk's one block a row on the same rows.
    pub fn replaced(self: *Topk) ![]const u8 {
        const s = self.shape;
        try self.l.tf_topk_rows(
            @ptrFromInt(self.in[0]),
            @intCast(s.rows),
            @intCast(s.n),
            @ptrFromInt(self.in[1]),
            @intCast(s.k),
            @ptrFromInt(self.base[0]),
            @ptrFromInt(self.base[1]),
            self.rig.s.handle,
        );
        return self.both(self.base);
    }

    pub fn count(self: *Topk) usize {
        return self.shape.rows * self.shape.k * 6;
    }

    pub fn reference(_: *Topk, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// Each row's slots against the rule over the row's bits; the slots past ks[r] keep their poison.
    pub fn check(self: *Topk, _: g.Gate) !?[]const u8 {
        const a = self.arena.allocator();
        const s = self.shape;
        const got = try self.both(self.out);
        const vals = got[s.rows * s.k * 4 ..];
        const order = try a.alloc(u32, s.n);
        const By = struct {
            fn ahead(row: []const u16, x: u32, y: u32) bool {
                const kx = key(row[x]);
                const ky = key(row[y]);
                return kx > ky or (kx == ky and x < y);
            }
        };
        for (0..s.rows) |r| {
            const row: []const u16 = self.bits[r * s.n ..][0..s.n];
            for (order, 0..) |*v, i| v.* = @intCast(i);
            std.sort.pdq(u32, order, row, By.ahead);
            const k: usize = @intCast(self.ks[r]);
            for (0..s.k) |j| {
                const at = r * s.k + j;
                const id = std.mem.readInt(i32, got[4 * at ..][0..4], .little);
                const v = std.mem.readInt(u16, vals[2 * at ..][0..2], .little);
                const want_id: i32 = if (j < k) @intCast(order[j]) else -1;
                const want_v: u16 = if (j < k) row[order[j]] else 0xffff;
                if (id != want_id or v != want_v) return try std.fmt.bufPrint(
                    &self.msg,
                    "row {d} slot {d} holds {d} (0x{x:0>4}), the rule gives {d} (0x{x:0>4})",
                    .{ r, j, id, v, want_id, want_v },
                );
            }
        }
        return null;
    }
};
