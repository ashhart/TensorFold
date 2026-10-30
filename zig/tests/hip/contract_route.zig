//! A prompt's MoE plan in three launches under the contract on HIP: tf_moe_route's members and items.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

const seg = 512; // moe.hpp kRouteSeg: the pairs a block of tf_route_count and tf_route_scatter takes

fn none(_: c.Shape) usize {
    return 0;
}

const rule = "members: the pairs stably sorted by expert; items: each expert's pairs in tiles, experts in order, " ++
    "then dead items, as tf_moe_route cuts them";

/// rows: token rows; n: the plan's experts, the shared one last; k: slots a row.
pub fn spec() c.Spec {
    return .{
        .name = "hip tf_route_count, tf_route_scan and tf_route_scatter",
        .accum = .none,
        .depth = none,
        .out = .raw,
        .replaces = "tf_moe_route",
        .nan = .{ .stated = rule },
        // the rows where the plan's tile goes from 8 pairs to 128
        .tiles = .{ .{ .size = 64 }, .{}, .{} },
        // Flash Next's 512 experts and shared one, top 10: prompt chunks, and the fewest rows the launcher plans here
        .served = &.{
            .{ .rows = 2048, .n = 513, .k = 11 },
            .{ .rows = 8192, .n = 513, .k = 11 },
            .{ .rows = 187, .n = 513, .k = 11 },
        },
    };
}

/// tf_moe_route's plan of `picks` (each below `experts`) at `tile` pairs an item: members [pairs], items [cap][3].
pub fn plan(a: std.mem.Allocator, picks: []const i32, experts: usize, tile: usize, cap: usize) !struct { []i32, []i32 } {
    const counts = try a.alloc(usize, 3 * experts);
    defer a.free(counts);
    const starts = counts[experts..][0..experts];
    const ends = counts[2 * experts ..][0..experts];
    @memset(counts[0..experts], 0);
    for (picks) |p| counts[@intCast(p)] += 1;
    var start: usize = 0;
    var end: usize = 0;
    for (counts[0..experts], starts, ends) |n, *s, *e| {
        s.* = start;
        start += n;
        end += (n + tile - 1) / tile;
        e.* = end;
    }
    const members = try a.alloc(i32, picks.len);
    const placed = try a.alloc(usize, experts);
    defer a.free(placed);
    @memset(placed, 0);
    for (picks, 0..) |p, i| {
        const e: usize = @intCast(p);
        members[starts[e] + placed[e]] = @intCast(i);
        placed[e] += 1;
    }
    const items = try a.alloc(i32, cap * 3);
    const t: i64 = @intCast(tile);
    for (0..cap) |slot| {
        // searchsorted(ends, slot, right=True); the slots past the live items take the last expert
        var lo: usize = 0;
        var hi: usize = experts;
        while (lo < hi) {
            const mid = (lo + hi) / 2;
            if (ends[mid] <= slot) lo = mid + 1 else hi = mid;
        }
        const e = @min(lo, experts - 1);
        const reps = (counts[e] + tile - 1) / tile;
        const within = @as(i64, @intCast(slot)) - @as(i64, @intCast(ends[e] - reps));
        const left = @as(i64, @intCast(counts[e])) - within * t;
        items[3 * slot] = @intCast(e);
        items[3 * slot + 1] = @intCast(@as(i64, @intCast(starts[e])) + within * t);
        items[3 * slot + 2] = @intCast(std.math.clamp(left, 0, t));
    }
    return .{ members, items };
}

/// shape.rows rows of shape.k picks over shape.n experts: the plan's members and items in three launches.
pub const Route = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    picks: []i32 = &.{},
    tile: usize = 0,
    cap: usize = 0,
    segs: usize = 0,
    in: u64 = 0,
    offs: u64 = 0,
    out: [2]u64 = @splat(0),
    base: [2]u64 = @splat(0),
    msg: [160]u8 = undefined,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher) Route {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .arena = .init(a) };
    }

    pub fn deinit(self: *Route) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Route) g.Kernel {
        return .{ .spec = spec(), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Route);

    fn pairs(self: *const Route) usize {
        return self.shape.rows * self.shape.k;
    }

    /// Routed picks half from 8 hot experts, then the shared expert; the nonfinite gate's from 3, so tiles fill.
    pub fn prepare(self: *Route, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k == 0 or shape.n < 2 or shape.n > 1024) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const experts = shape.n;
        const n = self.pairs();
        // moe.zig tileOf: the decode tile below 64 rows, the GEMM tile from there; capacity at the call's pairs
        self.tile = if (shape.rows < 64) 8 else 128;
        self.cap = @min(n, experts) + n / 8;
        self.segs = (n + seg - 1) / seg;
        self.picks = try a.alloc(i32, n);
        const hot = @min(if (fill == .nonfinite) @as(usize, 3) else 8, experts - 1);
        for (self.picks, 0..) |*p, i| {
            const from = if (fill == .nonfinite or rng.boolean()) hot else experts - 1;
            p.* = @intCast(if (i % shape.k == shape.k - 1) experts - 1 else rng.uintLessThan(usize, from));
        }
        self.in = try self.rig.input(std.mem.sliceAsBytes(self.picks));
        self.offs = try self.rig.scratch((self.segs + 1) * experts * 4);
        const lens = [2]usize{ n * 4, self.cap * 12 };
        for (&self.out, &self.base, lens) |*o, *b, len| {
            o.* = try self.rig.output(len);
            b.* = try self.rig.scratch(len);
            const poison = try a.alloc(u8, len);
            @memset(poison, 0xff); // as the outputs start
            try self.rig.put(b.*, poison);
        }
        return true;
    }

    pub fn run(self: *Route) !void {
        const op = &self.l.op;
        const experts: c_int = @intCast(self.shape.n);
        const segs: c_int = @intCast(self.segs);
        const n: c_int = @intCast(self.pairs());
        const blocks: hip.launch.Dim3 = .{ .x = @intCast(self.segs) };
        var a: hip.launch.Args = .{};
        try a.add(self.in);
        for ([_]c_int{ n, experts }) |v| try a.add(v);
        try a.add(self.offs);
        try hip.launch.launch(op.route_count, .{ .grid = blocks, .block = .{ .x = 256 } }, self.rig.s, &a);
        var b: hip.launch.Args = .{};
        try b.add(self.offs);
        for ([_]c_int{ segs, experts, @intCast(self.tile) }) |v| try b.add(v);
        try b.add(self.out[1]);
        try b.add(@as(c_int, @intCast(self.cap)));
        try hip.launch.launch(op.route_scan, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, self.rig.s, &b);
        var d: hip.launch.Args = .{};
        try d.add(self.in);
        for ([_]c_int{ n, segs, experts }) |v| try d.add(v);
        try d.add(self.offs);
        try d.add(self.out[0]);
        try hip.launch.launch(op.route_scatter, .{ .grid = blocks, .block = .{ .x = seg } }, self.rig.s, &d);
        try self.rig.s.synchronize();
    }

    /// Members then items at two device addresses, on the host.
    fn both(self: *Route, at: [2]u64) ![]const u8 {
        const a = self.arena.allocator();
        const members = try a.dupe(u8, try self.rig.bytes(at[0], self.pairs() * 4));
        return std.mem.concat(a, u8, &.{ members, try self.rig.bytes(at[1], self.cap * 12) });
    }

    /// tf_moe_route's one block on the same picks.
    pub fn replaced(self: *Route) ![]const u8 {
        const s = self.shape;
        try self.l.tf_moe_route(
            @ptrFromInt(self.in),
            @intCast(self.pairs()),
            @intCast(s.n),
            @intCast(self.tile),
            @ptrFromInt(self.base[0]),
            @ptrFromInt(self.base[1]),
            @intCast(self.cap),
            self.rig.s.handle,
        );
        return self.both(self.base);
    }

    pub fn count(self: *Route) usize {
        return self.pairs() * 4 + self.cap * 12;
    }

    pub fn reference(_: *Route, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// The members and items against the rule's plan of the same picks, byte for byte.
    pub fn check(self: *Route, _: g.Gate) !?[]const u8 {
        const a = self.arena.allocator();
        const members, const items = try plan(a, self.picks, self.shape.n, self.tile, self.cap);
        const want = try std.mem.concat(a, u8, &.{ std.mem.sliceAsBytes(members), std.mem.sliceAsBytes(items) });
        const got = try self.both(self.out);
        const at = std.mem.indexOfDiff(u8, got, want) orelse return null;
        const m = members.len * 4;
        const what = if (at < m) "member" else "item field";
        const i = if (at < m) at / 4 else (at - m) / 4;
        return try std.fmt.bufPrint(&self.msg, "{s} {d} differs from the rule's plan", .{ what, i });
    }
};
