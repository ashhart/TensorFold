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
    // Each branch entry runs its own cut shapes; the control fails at its first held shape.
    const shapes: []const c.Shape = switch (w) {
        .held => &held,
        .global => &global,
        .control => &served,
    };
    return .{
        .name = name,
        .accum = .none,
        .depth = none,
        .out = .raw,
        .replaces = "tf_fn_idx_select before the held-key cut",
        .nan = .{ .stated = idx_rule },
        .tiles = .{ .{}, .{}, .{} },
        .served = shapes,
    };
}

// The kernel's order-keeping key of a float: larger floats, larger unsigned keys.
fn fnOrder(u: u32) u32 {
    return if ((u & 0x80000000) != 0) ~u else (u | 0x80000000);
}

fn scoreKey(v: f64) u32 {
    const f: f32 = @floatCast(v);
    return fnOrder(@as(u32, @bitCast(f)));
}

fn putI32(buf: []u8, i: usize, v: i32) void {
    std.mem.writeInt(i32, buf[4 * i ..][0..4], v, .little);
}

fn getI32(buf: []const u8, i: usize) i32 {
    return std.mem.readInt(i32, buf[4 * i ..][0..4], .little);
}

fn higher(_: void, x: u32, y: u32) bool {
    return x > y;
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
        for (0..shape.rows) |r| {
            const first = r * nb;
            if (nb <= self.top + 1) continue;
            if (fill == .nonfinite) {
                const nan: f64 = @bitCast(@as(u64, 0x7ff8000000000000));
                const nnan: f64 = @bitCast(@as(u64, 0xfff8000000000000));
                self.scores[first + 0] = std.math.inf(f64);
                self.scores[first + 1] = -std.math.inf(f64);
                self.scores[first + 2] = nan;
                self.scores[first + 3] = nnan;
            } else {
                self.scores[first] = 8.0;
                for (1..self.top + 1) |b| self.scores[first + b] = 4.0;
            }
        }
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
        return true;
    }

    fn launch(self: *Select, dst: u64, count_extra: i32) !void {
        const s = self.shape;
        const cptr = @as(i64, @intCast(s.k));
        try self.l.fn_idx_select(
            @ptrFromInt(self.sc),
            cptr,
            @ptrFromInt(self.cp),
            @ptrFromInt(self.ep),
            @ptrFromInt(dst),
            @intCast(self.key_stride),
            @ptrFromInt(dst + s.rows * self.key_stride * 4),
            @ptrFromInt(dst + (s.rows * self.key_stride + s.rows) * 4),
            @intCast(s.rows),
            @intCast(self.top + @as(usize, @intFromBool(count_extra != 0))),
            @intCast(s.n),
            self.rig.s.handle,
        );
    }

    /// The fault is real: the control launches top+1 while an ordinary entry launches top.
    pub fn run(self: *Select) !void {
        try self.launch(self.out, if (self.which == .control) 1 else 0);
        try self.rig.s.synchronize();
    }

    /// Independent host oracle: rank by fn_order (ties to lower block id), then append the row's tail.
    fn oracle(self: *Select) ![]const u8 {
        const s = self.shape;
        const stride = self.key_stride;
        const rows = s.rows;
        const a = self.arena.allocator();
        const buf = try a.alloc(u8, (stride + 2) * rows * 4);
        @memset(buf, 0xff); // the output slot's poison, so unwritten padding matches byte for byte
        for (0..rows) |r| {
            const nb: usize = @intCast(self.complete[r]);
            const end: usize = @intCast(self.ends[r]);
            if (nb <= self.top) {
                putI32(buf, rows * stride + r, @intCast(end));
                putI32(buf, rows * stride + rows + r, 0);
                continue;
            }
            const sorted = try a.alloc(u32, nb);
            for (0..nb) |b| sorted[b] = scoreKey(self.scores[r * nb + b]);
            std.mem.sort(u32, sorted, {}, higher);
            const prefix = sorted[self.top - 1];
            var need: usize = self.top;
            var out: usize = 0;
            for (0..nb) |b| {
                const kb = scoreKey(self.scores[r * nb + b]);
                var take = kb > prefix;
                if (!take and kb == prefix and need > 0) {
                    need -= 1;
                    take = true;
                }
                if (!take) continue;
                for (0..s.n) |j| putI32(buf, r * stride + out * s.n + j, @intCast(b * s.n + j));
                out += 1;
            }
            var kk = nb * s.n;
            while (kk < end) : (kk += 1) putI32(buf, r * stride + self.top * s.n + (kk - nb * s.n), @intCast(kk));
            putI32(buf, rows * stride + r, @intCast(self.top * s.n + end - nb * s.n));
            putI32(buf, rows * stride + rows + r, 1);
        }
        return buf;
    }

    /// Gate 3's baseline: the oracle's bytes on the same poisoned, equal-length output layout.
    pub fn replaced(self: *Select) ![]const u8 {
        return try self.oracle();
    }

    pub fn count(self: *Select) usize {
        return self.shape.rows * self.key_stride;
    }

    pub fn reference(self: *Select, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        const buf = try self.oracle();
        for (want, 0..) |*v, i| v.* = @floatFromInt(getI32(buf, i));
    }

    /// The stated rule holds when every byte of the written result equals the independent oracle.
    pub fn check(self: *Select, at: g.Gate) !?[]const u8 {
        if (at != .nonfinite) return null;
        const got = try self.rig.outputBytes();
        const want = try self.oracle();
        if (got.len != want.len) return try std.fmt.bufPrint(&self.msg, "length {d}, expected {d}", .{ got.len, want.len });
        if (std.mem.indexOfDiff(u8, got, want)) |i| return try std.fmt.bufPrint(&self.msg, "byte {d}: {d}, expected {d}", .{ i, got[i], want[i] });
        return null;
    }
};
