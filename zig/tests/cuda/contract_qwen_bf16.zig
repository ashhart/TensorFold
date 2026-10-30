//! Qwen's BF16 embedding copies and small dense projections through every numerical gate and two controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const Rig = @import("contract_rig.zig").Rig;
const ref = @import("qwen_bf16_ref.zig");
const c = core.contract.spec;
const g = core.contract.gates;
pub const Kind = enum { copy, copy_unaligned, dense };
pub const Fault = enum { none, tail, half };
pub const Entry = struct { kind: Kind, fault: Fault = .none };
pub const entries = [_]Entry{ .{ .kind = .copy }, .{ .kind = .copy_unaligned }, .{ .kind = .dense }, .{ .kind = .copy, .fault = .tail }, .{ .kind = .dense, .fault = .half } };
fn depth(s: c.Shape) usize {
    return (s.k + 2047) / 2048 * 8 + 13;
}
fn noDepth(_: c.Shape) usize {
    return 0;
}
pub fn spec(e: Entry) c.Spec {
    if (e.kind == .dense) return .{ .name = if (e.fault == .half) "control: Qwen BF16 dense uses fp16 sums" else "cuda qwen35_bf16_rows tf_q35_dense_bf16", .accum = .f32, .depth = depth, .out = .bf16, .tiles = .{ .{ .size = 128 }, .{ .size = 32 }, .{ .size = 2048, .step = 8, .min = 8 } }, .served = &.{ .{ .rows = 4, .n = 32, .k = 2048 }, .{ .rows = 128, .n = 64, .k = 3072 }, .{ .rows = 17, .n = 32, .k = 5120 }, .{ .rows = 1, .n = 64, .k = 8192 } } };
    return .{ .name = if (e.fault == .tail) "control: Qwen BF16 copy omits a real row" else if (e.kind == .copy_unaligned) "cuda qwen35_bf16_rows tf_q35_embed_bf16 unaligned" else "cuda qwen35_bf16_rows tf_q35_embed_bf16 aligned", .accum = .none, .depth = noDepth, .out = .bf16, .nan = .{ .stated = "every live word copied unchanged; device-count padding and prefix left poisoned" }, .tiles = .{ .{ .size = 16 }, .{ .size = 8 }, .{} }, .served = &.{ .{ .rows = 4, .n = 2048, .k = 1 }, .{ .rows = 17, .n = 3072, .k = 1 }, .{ .rows = 16, .n = 5120, .k = 1 }, .{ .rows = 3, .n = 7, .k = 1 } }, .sums = .{ .f16 = 0, .bf16 = 0, .why = "copy contains no sum; each stored bit is checked" } };
}

/// Kernels is the production family's cuda_bf16_rows.Kernels; the host metadata tests need no device module.
pub fn Case(comptime Kernels: type) type {
    return struct {
        const Self = @This();
        rig: Rig,
        kernels: *const Kernels,
        entry: Entry,
        ctl: ?cuda.Function = null,
        arena: ?std.heap.ArenaAllocator = null,
        shape: c.Shape = undefined,
        live: usize = 0,
        shift: usize = 0,
        x: []u16 = &.{},
        w: []u16 = &.{},
        want: []u16 = &.{},
        xd: u64 = 0,
        wd: u64 = 0,
        ids: u64 = 0,
        params: u64 = 0,
        out: u64 = 0,
        pub fn deinit(self: *Self) void {
            if (self.arena) |*a| a.deinit();
            self.rig.deinit();
        }
        const vt = g.hooks(Self);
        const dense_vt: g.VTable = blk: {
            var v = vt;
            v.check = null; // Dense gate 3 must compare every value against FP64, never the copy checker.
            break :blk v;
        };
        pub fn kernel(self: *Self) g.Kernel {
            return .{ .spec = spec(self.entry), .ctx = self, .vt = if (self.entry.kind == .dense) &dense_vt else &vt, .control = switch (self.entry.fault) {
                .none => null,
                .tail => .shapes,
                .half => .sums,
            } };
        }
        pub fn prepare(self: *Self, s: c.Shape, fill: g.Fill, seed: u64) !bool {
            if (s.rows == 0 or s.n == 0 or s.k == 0) return false;
            if (self.entry.kind == .dense and (s.rows > 128 or s.k % 8 != 0)) return false;
            if (self.entry.kind != .dense and s.k != 1) return false;
            self.rig.reset();
            if (self.arena) |*a| a.deinit();
            self.arena = std.heap.ArenaAllocator.init(self.rig.gpu.gpa);
            const a = self.arena.?.allocator();
            self.shape = s;
            var random = std.Random.DefaultPrng.init(seed);
            const rng = random.random();
            if (self.entry.kind == .dense) {
                self.x = try a.alloc(u16, s.rows * s.k);
                self.w = try a.alloc(u16, s.n * s.k);
                for (self.x) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
                for (self.w) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
                if (fill == .stress) {
                    ref.stress(self.x);
                    @memset(self.w, 0x3f80);
                }
                if (fill == .nonfinite) {
                    @memset(self.w, 0x3f80);
                    for (0..@min(s.rows, 3)) |r| self.x[r * s.k] = ([_]u16{ 0x7fc0, 0x7f80, 0xff80 })[r];
                }
                self.xd = try self.rig.input(std.mem.sliceAsBytes(self.x));
                self.wd = try self.rig.input(std.mem.sliceAsBytes(self.w));
                self.out = try self.rig.output(s.rows * s.n * 2);
            } else {
                self.shift = @intFromBool(self.entry.kind == .copy_unaligned);
                const words = try a.alloc(u16, 97 * s.n + self.shift);
                for (words, 0..) |*v, i| v.* = if (fill == .nonfinite) ([_]u16{ 0x7fc1, 0x7f80, 0xff80, 0x8000, 0x3f80 })[i % 5] else c.toBf16(rng.float(f32) * 4 - 2);
                const ids = try a.alloc(i32, s.rows);
                for (ids, 0..) |*id, i| id.* = @intCast(if (i == 0) 96 else i % 97);
                self.live = s.rows - @intFromBool(s.rows > 1);
                self.want = try a.alloc(u16, s.rows * s.n + self.shift);
                @memset(self.want, 0xffff);
                ref.copy(words[self.shift..], ids, s.n, self.live, self.want[self.shift..]);
                self.xd = try self.rig.input(std.mem.sliceAsBytes(words));
                self.ids = try self.rig.input(std.mem.sliceAsBytes(ids));
                const params = [_]i32{ 0, 0, @intCast(self.live) };
                self.params = try self.rig.input(std.mem.sliceAsBytes(&params));
                self.out = try self.rig.output(self.want.len * 2);
            }
            return true;
        }
        pub fn run(self: *Self) !void {
            const s = self.shape;
            if (self.entry.kind == .dense) {
                if (self.ctl) |f| {
                    var args: cuda.Args = .{};
                    for ([_]u64{ self.xd, self.wd, self.out }) |v| args.add(v);
                    args.add(@as(c_int, @intCast(s.n)));
                    args.add(@as(c_int, @intCast(s.k)));
                    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(s.n), .y = @intCast(s.rows) }, .block = .{ .x = 32 } }, self.rig.s, &args);
                } else try self.kernels.denseRows(self.rig.s, self.xd, self.wd, self.out, s.rows, s.n, s.k);
            } else {
                const rows = if (self.entry.fault == .tail) self.live - 1 else s.rows;
                if (rows > 0) try self.kernels.embedRows(self.rig.s, self.ids, self.xd + self.shift * 2, self.out + self.shift * 2, rows, s.n, self.params);
            }
            try self.rig.s.synchronize();
        }
        pub fn count(self: *Self) usize {
            return if (self.entry.kind == .dense) self.shape.rows * self.shape.n else self.want.len;
        }
        pub fn reference(self: *Self, out: []f64, mag: []f64) !void {
            const s = self.shape;
            if (self.entry.kind != .dense) {
                for (out, self.want) |*v, word| v.* = c.fromBf16(word);
                @memset(mag, 0);
                return;
            }
            for (out, mag, 0..) |*v, *m, i| {
                const row = self.x[(i / s.n) * s.k ..][0..s.k];
                const weight = self.w[(i % s.n) * s.k ..][0..s.k];
                v.* = ref.dot(row, weight);
                m.* = 0;
                for (row, weight) |x, w| m.* += @abs(@as(f64, c.fromBf16(x)) * c.fromBf16(w));
            }
        }
        pub fn simulate(self: *Self, acc: c.Accum, order: g.Order, out: []f64) !void {
            if (self.entry.kind != .dense) return error.NoSums;
            const s = self.shape;
            const terms = try self.arena.?.allocator().alloc(f64, s.k);
            for (out, 0..) |*v, i| {
                for (terms, 0..) |*term, j| term.* = @as(f64, c.fromBf16(self.x[(i / s.n) * s.k + j])) * c.fromBf16(self.w[(i % s.n) * s.k + j]);
                v.* = core.contract.rows.sumIn(acc, order, terms);
            }
        }
        pub fn check(self: *Self, _: g.Gate) !?[]const u8 {
            if (self.entry.kind == .dense) return null;
            const got = try self.rig.outputBytes();
            return if (std.mem.eql(u8, got, std.mem.sliceAsBytes(self.want))) null else "embedding word, masked row or unaligned prefix differs";
        }
    };
}

test "Qwen BF16 contract: two production kernels include vector scalar and misaligned copies and two controls" {
    var normal: usize = 0;
    var controls: usize = 0;
    for (entries) |e| if (e.fault == .none) {
        normal += 1;
    } else {
        controls += 1;
    };
    try std.testing.expectEqual(@as(usize, 3), normal);
    try std.testing.expectEqual(@as(usize, 2), controls);
    try std.testing.expectEqual(c.Accum.f32, spec(.{ .kind = .dense }).accum);
    try std.testing.expectEqual(c.Accum.none, spec(.{ .kind = .copy }).accum);
}
test {
    _ = ref;
}
