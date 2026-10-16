//! Hyper-connection writeback: guarded hidden rows, exact fused FP32 rounding and independent FP64 square sums.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const streams = 4;
const slots = 11;

pub const Mode = enum { sums, moe, slices };

fn depth(_: c.Shape) usize {
    return 13;
}

fn spec(mode: Mode) c.Spec {
    return .{
        .name = switch (mode) {
            .sums => "cuda Flash Next fn_hc_writeback0",
            .moe => "cuda Flash Next fn_hc_writeback_moe",
            .slices => "cuda Flash Next fn_hc_writeback_slices",
        },
        .accum = .f32,
        .depth = depth,
        .out = .f32,
        .tiles = .{ .{ .size = 16 }, .{ .size = 256, .step = 256, .min = 256 }, .{} },
        .served = &.{ .{ .rows = 17, .n = 2560, .k = 2 }, .{ .rows = 3, .n = 512, .k = 8 } },
    };
}

/// PSS is first in the output byte stream; the hidden output follows and remains part of the repeat and guard gates.
pub const Writeback = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    mode: Mode,
    arena: ?std.heap.ArenaAllocator = null,
    shape: c.Shape = undefined,
    h: []u16 = &.{},
    inj: []u16 = &.{},
    y: []f32 = &.{},
    wts: []f32 = &.{},
    br: []f32 = &.{},
    dh: u64 = 0,
    dinj: u64 = 0,
    dy: u64 = 0,
    dwts: u64 = 0,
    dbr: u64 = 0,
    pss: u64 = 0,
    hout: u64 = 0,
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Writeback) void {
        if (self.arena) |*arena| arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Writeback) g.Kernel {
        return .{ .spec = spec(self.mode), .ctx = self, .vt = &g.hooks(Writeback) };
    }

    pub fn prepare(self: *Writeback, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 33 or shape.n == 0 or shape.n > 2560 or shape.n % 256 != 0 or shape.k == 0 or shape.k > 8) return false;
        self.rig.reset();
        if (self.arena) |*arena| arena.deinit();
        self.arena = std.heap.ArenaAllocator.init(self.rig.gpu.gpa);
        const a = self.arena.?.allocator();
        self.shape = shape;
        const wide = shape.n * streams;
        self.h = try a.alloc(u16, shape.rows * wide);
        self.inj = try a.alloc(u16, shape.rows * streams);
        self.y = try a.alloc(f32, shape.rows * slots * shape.n);
        self.wts = try a.alloc(f32, shape.rows * slots);
        self.br = try a.alloc(f32, shape.k * shape.rows * shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.h) |*b| b.* = c.toBf16(if (fill == .stress) @floatFromInt(1 + rng.uintLessThan(u32, 64)) else rng.float(f32) * 2 - 1);
        for (self.inj) |*b| b.* = c.toBf16(if (fill == .stress) 1 else rng.float(f32) * 4 - 2);
        for (self.y) |*v| v.* = if (fill == .stress) 0 else rng.float(f32) * 2 - 1;
        for (self.wts) |*v| v.* = rng.float(f32) * 0.2;
        for (self.br) |*v| v.* = if (fill == .stress) 0 else rng.float(f32) - 0.5;
        if (fill != .stress and self.mode != .sums) {
            self.h[0] = 0x3b34;
            self.inj[0] = 0x3c9a;
            for (0..slots) |k| self.y[k * shape.n] = 0;
            self.y[0] = c.fromBf16(0xbe16);
            self.wts[0] = 1;
            for (0..shape.k) |k| self.br[k * shape.rows * shape.n] = 0;
            self.br[0] = c.fromBf16(0xbe16);
        }
        if (fill == .nonfinite) for (0..shape.rows) |row| {
            self.h[row * wide] = switch (row % 5) {
                0 => 0x7fc1,
                1 => 0x7f80,
                2 => 0xff80,
                3 => 0x8000,
                else => 0x0001,
            };
        };
        self.dh = try self.rig.input(std.mem.sliceAsBytes(self.h));
        self.dinj = try self.rig.input(std.mem.sliceAsBytes(self.inj));
        self.dy = try self.rig.input(std.mem.sliceAsBytes(self.y));
        self.dwts = try self.rig.input(std.mem.sliceAsBytes(self.wts));
        self.dbr = try self.rig.input(std.mem.sliceAsBytes(self.br));
        self.pss = try self.rig.output(self.count() * 4);
        self.hout = if (self.mode == .sums) 0 else try self.rig.output(self.h.len * 2);
        return true;
    }

    pub fn run(self: *Writeback) !void {
        const sh = self.shape;
        var args: cuda.Args = .{};
        args.add(self.dh);
        if (self.mode != .sums) args.add(self.hout);
        args.add(self.pss);
        switch (self.mode) {
            .sums => {},
            .moe => {
                args.add(self.dinj);
                args.add(self.dy);
                args.add(self.dwts);
            },
            .slices => {
                args.add(self.dbr);
                args.add(self.dinj);
                args.add(@as(c_longlong, @intCast(sh.rows * sh.n)));
            },
        }
        args.add(@as(c_int, @intCast(sh.n)));
        args.add(@as(c_int, streams));
        if (self.mode == .moe) {
            args.add(@as(c_int, slots - 1));
            args.add(@as(c_int, slots));
        }
        if (self.mode == .slices) args.add(@as(c_int, @intCast(sh.k)));
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(sh.rows), .y = @intCast(sh.n / 256) }, .block = .{ .x = 256 } }, self.rig.s, &args);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Writeback) usize {
        return self.shape.rows * (self.shape.n / 256) * streams;
    }

    fn hidden(self: *Writeback, row: usize, stream: usize, col: usize) f32 {
        const sh = self.shape;
        const h = c.fromBf16(self.h[(row * streams + stream) * sh.n + col]);
        if (self.mode == .sums) return h;
        var branch: f32 = 0;
        switch (self.mode) {
            .moe => for (0..slots) |k| {
                const at = row * slots + k;
                branch = @mulAdd(f32, self.y[at * sh.n + col], self.wts[at], branch);
            },
            .slices => {
                branch = self.br[row * sh.n + col];
                for (1..sh.k) |k| branch += self.br[(k * sh.rows + row) * sh.n + col];
            },
            .sums => unreachable,
        }
        branch = c.fromBf16(c.toBf16(branch));
        return c.fromBf16(c.toBf16(@mulAdd(f32, branch, c.fromBf16(self.inj[row * streams + stream]), h)));
    }

    pub fn reference(self: *Writeback, want: []f64, mag: []f64) !void {
        for (0..self.shape.rows) |row| for (0..self.shape.n / 256) |chunk| for (0..streams) |stream| {
            var sum: f64 = 0;
            for (0..256) |i| {
                const v: f64 = self.hidden(row, stream, chunk * 256 + i);
                sum += v * v;
            }
            const at = (row * (self.shape.n / 256) + chunk) * streams + stream;
            want[at] = sum;
            mag[at] = sum;
        };
    }

    pub fn simulate(self: *Writeback, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (0..self.shape.rows) |row| for (0..self.shape.n / 256) |chunk| for (0..streams) |stream| {
            var terms: [256]f64 = undefined;
            for (&terms, 0..) |*term, i| {
                const v: f64 = self.hidden(row, stream, chunk * 256 + i);
                term.* = v * v;
            }
            out[(row * (self.shape.n / 256) + chunk) * streams + stream] = core.contract.rows.sumIn(acc, order, &terms);
        };
    }

    pub fn check(self: *Writeback, _: g.Gate) !?[]const u8 {
        const a = self.arena.?.allocator();
        const pss = try a.dupe(u8, try self.rig.bytes(self.pss, self.count() * 4));
        if (self.mode != .sums) {
            const hout = try self.rig.bytes(self.hout, self.h.len * 2);
            for (0..self.shape.rows) |row| for (0..streams) |stream| for (0..self.shape.n) |col| {
                const at = (row * streams + stream) * self.shape.n + col;
                const got = std.mem.readInt(u16, hout[at * 2 ..][0..2], .little);
                const want = c.toBf16(self.hidden(row, stream, col));
                if (got != want and !(std.math.isNan(c.fromBf16(got)) and std.math.isNan(c.fromBf16(want))))
                    return try std.fmt.bufPrint(&self.msg, "hidden row {d} stream {d} column {d}: {x} != fused FP32 reference {x}", .{ row, stream, col, got, want });
            };
        }
        const want = try a.alloc(f64, self.count());
        const mag = try a.alloc(f64, self.count());
        try self.reference(want, mag);
        for (want, mag, 0..) |w, m, i| {
            const got = c.read(.f32, pss, i);
            const ok = if (std.math.isFinite(w)) @abs(got - w) <= c.bound(spec(self.mode), self.shape, w, m) else c.classOf(got) == c.classOf(w);
            if (!ok) return try std.fmt.bufPrint(&self.msg, "PSS {d}: {d} differs from independent FP64 sum {d}", .{ i, got, w });
        }
        return null;
    }
};

test "integer square stress requires FP32 sums in every simulated order" {
    var prng = std.Random.DefaultPrng.init(0x46e3);
    const rng = prng.random();
    var terms: [4][256]f64 = undefined;
    var want: [4]f64 = @splat(0);
    for (&terms, &want) |*row, *sum| for (row) |*term| {
        const v: f64 = @floatFromInt(1 + rng.uintLessThan(u32, 64));
        term.* = v * v;
        sum.* += term.*;
    };
    for (std.enums.values(g.Order)) |order| {
        for (terms, want) |row, w| try std.testing.expectEqual(w, core.contract.rows.sumIn(.f32, order, &row));
        for ([_]c.Accum{ .f16, .bf16 }) |acc| {
            var miss: f64 = 0;
            for (terms, want) |row, w| {
                const got = core.contract.rows.sumIn(acc, order, &row);
                const bound = c.boundExact(spec(.sums), w);
                miss = @max(miss, if (std.math.isFinite(got)) @abs(got - w) / bound else std.math.inf(f64));
            }
            try std.testing.expect(miss >= 100);
        }
    }
}

test "the cancellation fixture distinguishes fused update from rounded product" {
    const h = c.fromBf16(0x3b34);
    const branch = c.fromBf16(0xbe16);
    const inject = c.fromBf16(0x3c9a);
    const fused = c.toBf16(@mulAdd(f32, branch, inject, h));
    const old = c.toBf16(h + c.fromBf16(c.toBf16(branch * inject)));
    const exact: f64 = @as(f64, h) + @as(f64, branch) * @as(f64, inject);
    try std.testing.expect(fused != old);
    try std.testing.expectEqual(c.toBf16(@floatCast(exact)), fused);
}
