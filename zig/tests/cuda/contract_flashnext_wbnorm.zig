//! The fused writeback and normalize under the contract: Glue.hcWritebackNorm against hcWriteback then hcNormalize.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;

pub const Mode = enum { sums, moe, slices };

const dims = 2560;
const streams = 4;
const wide = dims * streams;
const nc = dims / 256;
const slots = 11; // the routed experts' slots and the shared one, as the pass binds them
const eps: f32 = 1e-6;

fn noSums(_: c.Shape) usize {
    return 0;
}

pub fn spec(mode: Mode) c.Spec {
    const rule = "the written rows, their 256-column square sums and the normalized rows bit-equal to the two launches";
    return .{
        .name = switch (mode) {
            .sums => "cuda flashnext fn_hc_writeback_norm0 (square sums, then normalize)",
            .moe => "cuda flashnext fn_hc_writeback_norm_moe (MoE slots written back, then normalize)",
            .slices => "cuda flashnext fn_hc_writeback_norm_slices (K slices written back, then normalize)",
        },
        .accum = .none,
        .depth = noSums,
        .out = .raw,
        .nan = .{ .stated = rule },
        .replaces = "fn_hc_writeback0 / _moe / _slices, then fn_hc_normalize",
        .served = &.{ .{ .rows = 1, .n = dims, .k = 1 }, .{ .rows = 17, .n = dims, .k = 2 }, .{ .rows = 300, .n = dims, .k = 3 } },
    };
}

pub const WbNorm = struct {
    rig: rig_.Rig,
    mode: Mode,
    glue: *const flash.Glue,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    ins: [6]u64 = @splat(0),
    lens: [3]usize = @splat(0),
    outs: [3]u64 = @splat(0),
    nout: usize = 0,
    base: u64 = 0,
    seen: std.ArrayList(u8) = .empty,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *WbNorm) void {
        self.seen.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    const vt = g.hooks(WbNorm);

    pub fn kernel(self: *WbNorm) g.Kernel {
        return .{ .spec = spec(self.mode), .ctx = self, .vt = &vt };
    }

    fn modeNumber(self: *const WbNorm) usize {
        return switch (self.mode) {
            .sums => 0,
            .moe => 2,
            .slices => 4,
        };
    }

    pub fn prepare(self: *WbNorm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        self.nout = 0;
        self.shape = shape;
        const m = shape.rows;
        if (m == 0 or m > 300 or shape.n != dims or shape.k == 0 or shape.k > 4) return false;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed ^ 0x77626e);
        const r = prng.random();
        const h = try a.alloc(u16, m * wide);
        defer a.free(h);
        for (h) |*x| x.* = c.toBf16(r.float(f32) * 2 - 1);
        const inj = try a.alloc(u16, m * streams);
        defer a.free(inj);
        for (inj) |*x| x.* = c.toBf16(r.float(f32) * 4 - 2);
        const y = try a.alloc(f32, m * slots * dims);
        defer a.free(y);
        for (y) |*x| x.* = r.float(f32) * 2 - 1;
        const wts = try a.alloc(f32, m * slots);
        defer a.free(wts);
        for (wts) |*x| x.* = r.float(f32) * 0.2;
        const br = try a.alloc(f32, shape.k * m * dims);
        defer a.free(br);
        for (br) |*x| x.* = r.float(f32) - 0.5;
        const scale = try a.alloc(f32, wide);
        defer a.free(scale);
        for (scale) |*x| x.* = 0.5 + r.float(f32);
        if (fill == .nonfinite) for (0..m) |row| {
            h[row * wide + (row % streams) * dims + 3] = switch (row % 4) {
                0 => 0x7fc1,
                1 => 0x7f80,
                2 => 0xff80,
                else => 0x8000,
            };
        };
        for ([_][]const u8{ std.mem.sliceAsBytes(h), std.mem.sliceAsBytes(inj), std.mem.sliceAsBytes(y), std.mem.sliceAsBytes(wts), std.mem.sliceAsBytes(br), std.mem.sliceAsBytes(scale) }, 0..) |b, i| self.ins[i] = try self.rig.input(b);
        if (self.mode != .sums) try self.out(m * wide * 2); // the written-back rows
        try self.out(m * nc * streams * 4); // square sums a 256-column slice and stream
        try self.out(m * wide * 2); // the normalized rows
        var total: usize = 0;
        for (self.lens[0..self.nout]) |l| total += l;
        self.base = try self.rig.scratch(total);
        return true;
    }

    fn out(self: *WbNorm, len: usize) !void {
        self.outs[self.nout] = try self.rig.output(len);
        self.lens[self.nout] = len;
        self.nout += 1;
    }

    /// The written rows, the sums and the normalized rows in `o`: the first is absent for plain sums.
    fn split(self: *const WbNorm, o: []const u64) struct { hout: u64, pss: u64, norm: u64 } {
        return if (self.mode == .sums) .{ .hout = 0, .pss = o[0], .norm = o[1] } else .{ .hout = o[0], .pss = o[1], .norm = o[2] };
    }

    fn ours(self: *WbNorm, o: []const u64) !void {
        const p = self.split(o);
        const m = self.shape.rows;
        try self.glue.hcWritebackNorm(self.rig.s, self.ins[0], p.hout, p.pss, self.ins[4], self.ins[1], self.ins[2], self.ins[3], self.ins[5], p.norm, eps, m * dims, m, dims, streams, self.modeNumber(), slots - 1, slots, self.shape.k);
    }

    fn theirs(self: *WbNorm, o: []const u64) !void {
        const p = self.split(o);
        const m = self.shape.rows;
        try self.glue.hcWriteback(self.rig.s, self.ins[0], p.hout, p.pss, self.ins[4], self.ins[1], self.ins[2], self.ins[3], m * dims, m, dims, streams, self.modeNumber(), slots - 1, slots, self.shape.k);
        try self.glue.hcNormalize(self.rig.s, if (self.mode == .sums) self.ins[0] else p.hout, p.pss, self.ins[5], p.norm, eps, m, dims, streams);
    }

    pub fn run(self: *WbNorm) !void {
        try self.ours(self.outs[0..self.nout]);
        try self.rig.s.synchronize();
    }

    /// The scratch the two launches write: the outputs' layout back to back, poisoned like them (0xff).
    fn baseline(self: *WbNorm) ![]const u8 {
        var at = self.base;
        var o: [3]u64 = undefined;
        var total: usize = 0;
        for (self.lens[0..self.nout], 0..) |l, i| {
            o[i] = at;
            at += l;
            total += l;
        }
        const ff = try self.rig.gpu.gpa.alloc(u8, total);
        defer self.rig.gpu.gpa.free(ff);
        @memset(ff, 0xff);
        try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = self.base, .len = total }).upload(0, ff);
        try self.theirs(o[0..self.nout]);
        try self.rig.s.synchronize();
        return self.rig.bytes(self.base, total);
    }

    pub fn replaced(self: *WbNorm) ![]const u8 {
        return self.baseline();
    }

    /// Gate 1's stated rule: the outputs on the planted inputs equal the two launches' bytes.
    pub fn check(self: *WbNorm, gate: g.Gate) !?[]const u8 {
        _ = gate;
        self.seen.clearRetainingCapacity();
        try self.seen.appendSlice(self.rig.gpu.gpa, try self.rig.outputBytes());
        const want = try self.baseline();
        if (std.mem.eql(u8, self.seen.items, want)) return null;
        return try std.fmt.bufPrint(&self.msg, "bytes differ from the two launches' from byte {d}", .{std.mem.indexOfDiff(u8, self.seen.items, want).?});
    }

    pub fn count(self: *WbNorm) usize {
        var total: usize = 0;
        for (self.lens[0..self.nout]) |l| total += l;
        return total;
    }

    pub fn reference(_: *WbNorm, _: []f64, _: []f64) !void {
        return error.NoFloat64Reference;
    }
};
