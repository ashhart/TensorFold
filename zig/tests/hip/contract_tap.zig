//! A drafter's taps under the contract on HIP: tf_plan_tap and tf_tap_rows fill a tap ring; a planted control for each.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

// Qwen3.8-27B: 64 layers of hidden 5120, five tapped layers, a ring of the 2048-row window and a 1024-row chunk
const layers = 64;
const ring_rows = 2048 + 1024;
const words = 2 + 2 * layers + 1; // a descriptor's words before its page table: the ring is the last
const first = ring_rows - 2; // a round's first position: its rows wrap the ring
const pos0 = 2 * 1024 + 552; // a prompt chunk's first position: its 1024 rows wrap the ring

fn none(_: c.Shape) usize {
    return 0;
}

/// A round's rows over their slots' descriptors, or one stream's prompt rows from a position; and a control of each.
pub const Mode = enum { plan, rows, wrong_row, wrong_tap };

const rule = "every ring word is its row's residual word, NaN and infinity bits included, or keeps its bytes";

pub fn spec(m: Mode) c.Spec {
    return switch (m) {
        .plan, .wrong_row => .{
            .name = if (m == .plan) "hip tf_plan_tap" else "control: tf_plan_tap writes each row one ring row on",
            .accum = .none,
            .depth = none,
            .out = .raw,
            .nan = .{ .stated = rule },
            // a decode row, a draft window and a 16-row verify over 27B's five taps
            .served = &.{ .{ .rows = 1, .n = 5, .k = 5120 }, .{ .rows = 4, .n = 5, .k = 5120 }, .{ .rows = 16, .n = 5, .k = 5120 } },
        },
        .rows, .wrong_tap => .{
            .name = if (m == .rows) "hip tf_tap_rows" else "control: tf_tap_rows writes the next tap",
            .accum = .none,
            .depth = none,
            .out = .raw,
            .nan = .{ .stated = rule },
            // a prompt's short tail and a full prompt chunk
            .served = &.{ .{ .rows = 16, .n = 5, .k = 5120 }, .{ .rows = 1024, .n = 5, .k = 5120 } },
        },
    };
}

/// shape.rows residual rows of shape.k 16-bit values as tap `tap` of shape.n into a ring of `ring_rows` rows.
pub const Tap = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    m: Mode,
    k: kit.Kind,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    x: []u32 = &.{},
    pos: []i32 = &.{},
    slot: []i32 = &.{},
    in: u64 = 0,
    ring: u64 = 0,
    plan: hip.launches.PlanRef = undefined,
    msg: [160]u8 = undefined,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, m: Mode) Tap {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .m = m, .k = kit.act(), .arena = .init(a) };
    }

    pub fn deinit(self: *Tap) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Tap) g.Kernel {
        const control: ?g.Gate = if (self.m == .wrong_row or self.m == .wrong_tap) .shapes else null;
        return .{ .spec = spec(self.m), .ctx = self, .vt = &vt, .control = control };
    }

    const vt = g.hooks(Tap);

    fn round(self: *const Tap) bool {
        return self.m == .plan or self.m == .wrong_row;
    }

    fn tap(self: *const Tap) usize {
        return if (self.round()) self.shape.n - 1 else 1;
    }

    fn ringWords(self: *const Tap) usize {
        return ring_rows * self.shape.n * self.shape.k / 2;
    }

    /// Random residuals (non-finite: NaN and infinities) over slot 0 (ring), 1 (ring word 0) and 2 (no descriptor).
    pub fn prepare(self: *Tap, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k % 2 != 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = shape.rows * shape.k;
        const b = try a.alloc(u8, n * 2);
        for (0..n) |i| kit.put(self.k, b, i, rng.float(f64) * 8 - 4);
        if (fill == .nonfinite) for ([_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) }, 0..) |v, i| {
            kit.put(self.k, b, (i * 2711 + 3) % n, v);
        };
        self.x = try a.alloc(u32, n / 2);
        @memcpy(std.mem.sliceAsBytes(self.x), b);
        self.in = try self.rig.input(b);
        self.ring = try self.rig.output(self.ringWords() * 4);
        self.pos = try a.alloc(i32, shape.rows);
        self.slot = try a.alloc(i32, shape.rows);
        const start: usize = if (self.round()) first else pos0;
        for (self.pos, self.slot, 0..) |*p, *s, r| {
            p.* = @intCast(start + r);
            s.* = if (self.round() and r % 4 != 0) @intCast(r % 4 % 3) else 0;
        }
        if (!self.round()) return true;
        // the control's plan names each row's next position
        const pos = try a.dupe(i32, self.pos);
        if (self.m == .wrong_row) for (pos) |*p| {
            p.* += 1;
        };
        var descs: [3]u64 = @splat(0);
        for (descs[0..2], [_]u64{ self.ring, 0 }) |*d, at| {
            var desc: [words]u64 = @splat(0);
            desc[words - 1] = at;
            d.* = try self.rig.input(std.mem.asBytes(&desc));
        }
        const one = [1]i32{@intCast(shape.rows)};
        self.plan = .{ .rows = shape.rows, .slots = descs.len, .args = .{
            .pos = try self.rig.input(std.mem.sliceAsBytes(pos)),
            .slot = try self.rig.input(std.mem.sliceAsBytes(self.slot)),
            .count = try self.rig.input(std.mem.asBytes(&one)),
            .desc = try self.rig.input(std.mem.asBytes(&descs)),
            .pages = words,
        } };
        return true;
    }

    pub fn run(self: *Tap) !void {
        const s = self.shape;
        const st = self.rig.s.handle;
        switch (self.m) {
            .plan, .wrong_row => try self.l.planTap(self.in, self.plan, s.k, self.tap(), s.n, ring_rows, st),
            .rows => try self.l.tapRows(self.in, self.ring, s.rows, s.k, self.tap(), s.n, ring_rows, pos0, st),
            .wrong_tap => try self.l.tapRows(self.in, self.ring, s.rows, s.k, self.tap() + 1, s.n, ring_rows, pos0, st),
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Tap) usize {
        return self.ringWords() * 4;
    }

    pub fn reference(_: *Tap, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// The ring word for word: a slot-0 row's words at (its position % ring rows, tap), the poison elsewhere.
    pub fn check(self: *Tap, _: g.Gate) !?[]const u8 {
        const s = self.shape;
        const pairs = s.k / 2;
        const want = try self.arena.allocator().alloc(u32, self.ringWords());
        @memset(want, 0xffffffff);
        for (self.pos, self.slot, 0..) |p, slot, r| if (slot == 0) {
            const row: usize = @intCast(@mod(p, ring_rows));
            @memcpy(want[(row * s.n + self.tap()) * pairs ..][0..pairs], self.x[r * pairs ..][0..pairs]);
        };
        const got = try self.rig.bytes(self.ring, want.len * 4);
        const at = std.mem.indexOfDiff(u8, std.mem.sliceAsBytes(want), got) orelse return null;
        const w = at / 4;
        return try std.fmt.bufPrint(&self.msg, "ring word {d} (row {d}, tap {d}) is 0x{x:0>8}, want 0x{x:0>8}", .{
            w,
            w / pairs / s.n,
            w / pairs % s.n,
            std.mem.readInt(u32, got[w * 4 ..][0..4], .little),
            want[w],
        });
    }
};
