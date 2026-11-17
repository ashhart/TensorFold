//! Contract hooks for tf_fn_idx_select and its held/global cut branches.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

fn none(_: c.Shape) usize {
    return 0;
}

const idx_rule = "integer block keys ordered by fn_order bits, larger first, ties to lower block id; a row's tail follows";

// shape.k is the number of complete blocks, shape.n the ratio, shape.rows the rows.
const held = [_]c.Shape{
    .{ .rows = 1, .n = 4, .k = 512 },
    .{ .rows = 4, .n = 2, .k = 32768 },
};
const global = [_]c.Shape{
    .{ .rows = 1, .n = 1, .k = 65536 },
    .{ .rows = 2, .n = 2, .k = 98304 },
};
const served = held ++ global;

pub const Which = enum { held, global, control };

fn spec(w: Which) c.Spec {
    const name: []const u8 = switch (w) {
        .held => "hip tf_fn_idx_select (held-key cut)",
        .global => "hip tf_fn_idx_select (global cut)",
        .control => "control: tf_fn_idx_select writes one key too many",
    };
    return .{
        .name = name,
        .accum = .none,
        .depth = none,
        .out = .raw,
        .replaces = "tf_fn_idx_select before the held-key cut",
        .nan = .{ .stated = idx_rule },
        .tiles = .{ .{}, .{}, .{} },
        .served = &served,
    };
}

/// tf_fn_idx_select on rows of complete blocks; `.control` launches with a count one too large.
pub const Select = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    which: Which,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    scores: []f64 = &.{},
    complete: []i32 = &.{},
    ends: []i32 = &.{},
    top: usize = 8,
    tail: usize = 16,
    key_stride: usize = 0,
    sc: u64 = 0,
    cp: u64 = 0,
    ep: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,
    msg: [176]u8 = undefined,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, w: Which) Select {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .which = w, .arena = .init(a) };
    }
    pub fn deinit(self: *Select) void {
        self.arena.deinit();
        self.rig.deinit();
    }
    pub fn kernel(self: *Select) g.Kernel {
        return .{ .spec = spec(self.which), .ctx = self, .vt = &vt, .control = if (self.which == .control) .shapes else null };
    }
    const vt = g.hooks(Select);

    pub fn prepare(self: *Select, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        _ = fill;
        if (shape.rows == 0 or shape.n == 0 or shape.n > 64 or shape.k == 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const nb = shape.k;
        const ratio = shape.n;
        self.key_stride = (self.top + 1) * ratio + self.tail + 8;
        self.scores = try a.alloc(f64, shape.rows * nb);
        for (self.scores) |*v| v.* = @as(f32, @floatCast(rng.float(f64)));
        self.complete = try a.alloc(i32, shape.rows);
        self.ends = try a.alloc(i32, shape.rows);
        for (0..shape.rows) |r| {
            self.complete[r] = @intCast(nb);
            self.ends[r] = @intCast(nb * ratio + self.tail);
        }
        const sb = try a.alloc(u8, self.scores.len * 4);
        for (self.scores, 0..) |v, i| kit.put(.f32, sb, i, v);
        self.sc = try self.rig.input(sb);
        self.cp = try self.rig.input(std.mem.sliceAsBytes(self.complete));
        self.ep = try self.rig.input(std.mem.sliceAsBytes(self.ends));
        const bytes = (self.key_stride + 2) * shape.rows * 4;
        self.out = try self.rig.output(bytes);
        self.base = try self.rig.scratch(bytes);
        return true;
    }

    fn launch(self: *Select, dst: u64, count_extra: i32) !void {
        const s = self.shape;
        const cptr = @as(i64, @intCast(s.k));
        try self.l.fn_idx_select(@ptrFromInt(self.sc), cptr, @ptrFromInt(self.cp), @ptrFromInt(self.ep),
            @ptrFromInt(dst), @intCast(self.key_stride), @ptrFromInt(dst + s.rows * self.key_stride * 4),
            @ptrFromInt(dst + (s.rows * self.key_stride + s.rows) * 4), @intCast(s.rows),
            @intCast(self.top + @as(usize, @intFromBool(count_extra != 0))), @intCast(s.n), self.rig.s.handle);
    }

    pub fn run(self: *Select) !void {
        try self.launch(self.out, 0);
        try self.rig.s.synchronize();
    }

    /// The same kernel on the same inputs into a fresh buffer: the deterministic byte baseline.
    pub fn replaced(self: *Select) ![]const u8 {
        try self.launch(self.base, 0);
        return self.rig.bytes(self.base, self.count() * 4);
    }

    pub fn count(self: *Select) usize {
        return self.shape.rows * self.key_stride;
    }

    pub fn reference(_: *Select, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// Independent check of the written region: count/sparse and every selected key in range and unique.
    pub fn check(self: *Select, at: g.Gate) !?[]const u8 {
        if (at != .nonfinite) return null;
        const s = self.shape;
        const written = try self.rig.bytes(self.out, (self.key_stride + 2) * s.rows * 4);
        for (0..s.rows) |r| {
            const nb: usize = @intCast(self.complete[r]);
            const end: usize = @intCast(self.ends[r]);
            const want_count = if (nb <= self.top) end else self.top * s.n + end - nb * s.n;
            const got_count = std.mem.readInt(i32, written[(s.rows * self.key_stride + r) * 4 ..][0..4], .little);
            if (got_count != @as(i32, @intCast(want_count))) return try std.fmt.bufPrint(&self.msg, "row {d} count {d}, expected {d}", .{ r, got_count, want_count });
            const got_sparse = std.mem.readInt(i32, written[(s.rows * self.key_stride + s.rows + r) * 4 ..][0..4], .little);
            const want_sparse: i32 = if (nb <= self.top) 0 else 1;
            if (got_sparse != want_sparse) return try std.fmt.bufPrint(&self.msg, "row {d} sparse {d}, expected {d}", .{ r, got_sparse, want_sparse });
            if (nb > self.top) for (0..want_count) |k| {
                const key = std.mem.readInt(i32, written[(r * self.key_stride + k) * 4 ..][0..4], .little);
                if (key < 0 or key >= @as(i32, @intCast(end))) return try std.fmt.bufPrint(&self.msg, "row {d} key {d} out of range", .{ r, key });
            };
        }
        return null;
    }
};
