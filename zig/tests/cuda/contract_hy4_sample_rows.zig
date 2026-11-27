//! Contract suite for `hy4_rows_f32_to_bf16` and `hy4_rows_interleave_bf16`: RNE, transpose, and controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hr = @import("hy4_route_rows_cuda");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;

fn zeroDepth(_: c.Shape) usize {
    return 0;
}

pub const Mode = enum { f32_to_bf16, interleave_bf16 };
pub const Fault = enum { none, bad_nan };

const f32_rule = "exact RNE f32->bf16 on every element; sets *nonfinite = 1 if any input element is NaN or Inf";
const ilv_rule = "exact [world, rows, local_cols] -> [rows, world * local_cols] bf16 transpose; sets *nonfinite = 1 on Inf/NaN";

const served_f32 = [_]c.Shape{
    .{ .rows = 1, .n = 60416, .k = 1 },
    .{ .rows = 6, .n = 30208, .k = 1 },
    .{ .rows = 17, .n = 4103, .k = 1 },
};

const served_ilv = [_]c.Shape{
    .{ .rows = 1, .n = 60416, .k = 2 },
    .{ .rows = 6, .n = 30208, .k = 4 },
    .{ .rows = 17, .n = 2048, .k = 2 },
};

pub const Case = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    mode: Mode,
    fault: Fault,
    name: []const u8,
    shape: c.Shape = undefined,
    f32_in: std.ArrayList(f32) = .empty,
    bf16_in: std.ArrayList(u16) = .empty,
    dev_in: u64 = 0,
    dev_out: u64 = 0,
    dev_flag: u64 = 0,

    pub fn deinit(self: *Case) void {
        self.f32_in.deinit(self.rig.gpu.gpa);
        self.bf16_in.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    pub fn kernel(self: *Case) g.Kernel {
        const ctl: ?g.Gate = if (self.fault == .none) null else .nonfinite;
        return .{
            .spec = .{
                .name = self.name,
                .accum = .none,
                .depth = zeroDepth,
                .out = .raw,
                .nan = .{ .stated = if (self.mode == .f32_to_bf16) f32_rule else ilv_rule },
                .tiles = if (self.mode == .f32_to_bf16)
                    .{ .{}, .{ .size = 4, .min = 1 }, .{} }
                else
                    .{ .{}, .{ .size = 8, .step = 8, .min = 8 }, .{ .size = 2, .step = 2, .min = 2 } },
                .served = if (self.mode == .f32_to_bf16) &served_f32 else &served_ilv,
            },
            .ctx = self,
            .vt = &vt,
            .control = ctl,
        };
    }

    const vt = g.hooks(Case);

    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        if (self.mode == .interleave_bf16 and shape.n % 8 != 0) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        self.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed ^ 0x5341);
        const rng = prng.random();
        switch (self.mode) {
            .f32_to_bf16 => {
                const total = shape.rows * shape.n;
                try self.f32_in.resize(a, total);
                for (self.f32_in.items) |*v| v.* = rng.float(f32) * 16.0 - 8.0;
                if (fill == .nonfinite) {
                    self.f32_in.items[total - 1] = std.math.nan(f32);
                    if (total > 2) self.f32_in.items[total / 2] = -std.math.inf(f32);
                }
                self.dev_in = try self.rig.input(std.mem.sliceAsBytes(self.f32_in.items));
                self.dev_out = try self.rig.output(total * 2);
            },
            .interleave_bf16 => {
                const total = shape.k * shape.rows * shape.n;
                try self.bf16_in.resize(a, total);
                for (self.bf16_in.items) |*v| {
                    var h: u16 = rng.int(u16);
                    if ((h & 0x7f80) == 0x7f80) h &= ~@as(u16, 0x0080);
                    v.* = h;
                }
                if (fill == .nonfinite) {
                    self.bf16_in.items[total - 1] = 0x7fc0;
                    if (total > 2) self.bf16_in.items[total / 2] = 0xff80;
                }
                self.dev_in = try self.rig.input(std.mem.sliceAsBytes(self.bf16_in.items));
                self.dev_out = try self.rig.output(total * 2);
            },
        }
        self.dev_flag = try self.rig.scratch(4);
        return true;
    }

    pub fn run(self: *Case) !void {
        const zero: u32 = 0;
        try cuda.copy_source.enqueue(self.rig.gpu.d, self.dev_flag, std.mem.asBytes(&zero), self.rig.s);
        var a: cuda.Args = .{};
        switch (self.mode) {
            .f32_to_bf16 => {
                const total = self.shape.rows * self.shape.n;
                const n4 = (total + 3) / 4;
                a.add(self.dev_in);
                a.add(self.dev_out);
                a.add(@as(i32, @intCast(total)));
                a.add(self.dev_flag);
                try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast((n4 + 255) / 256) }, .block = .{ .x = hr.threads } }, self.rig.s, &a);
            },
            .interleave_bf16 => {
                const local_u4 = self.shape.n / 8;
                const total = self.shape.rows * self.shape.k * local_u4;
                a.add(self.dev_in);
                a.add(self.dev_out);
                a.add(@as(i32, @intCast(local_u4)));
                a.add(@as(i32, @intCast(self.shape.rows)));
                a.add(@as(i32, @intCast(self.shape.k)));
                a.add(self.dev_flag);
                try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = hr.threads } }, self.rig.s, &a);
            },
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *const Case) usize {
        return switch (self.mode) {
            .f32_to_bf16 => self.shape.rows * self.shape.n * 2,
            .interleave_bf16 => self.shape.k * self.shape.rows * self.shape.n * 2,
        };
    }

    pub fn reference(_: *Case, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    pub fn check(self: *Case, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const raw = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(raw);
        const flag_b = try self.rig.bytes(self.dev_flag, 4);
        const got_bad = std.mem.readInt(u32, flag_b[0..4], .little);
        const got_u16: []align(1) const u16 = std.mem.bytesAsSlice(u16, raw);
        var want_bad: u32 = 0;
        switch (self.mode) {
            .f32_to_bf16 => {
                for (self.f32_in.items, 0..) |v, i| {
                    if (!std.math.isFinite(v)) {
                        want_bad = 1;
                        if ((got_u16[i] & 0x7f80) != 0x7f80) return "nonfinite f32 did not convert to bf16 Inf/NaN";
                    } else if (got_u16[i] != core.bf16_round.bits(v)) {
                        return "f32_to_bf16 bits differ from exact RNE host reference";
                    }
                }
            },
            .interleave_bf16 => {
                const rows = self.shape.rows;
                const local = self.shape.n;
                const world = self.shape.k;
                for (0..rows) |r| {
                    for (0..world) |rk| {
                        const src_off = (rk * rows + r) * local;
                        const dst_off = (r * world + rk) * local;
                        for (0..local) |c_idx| {
                            const h = self.bf16_in.items[src_off + c_idx];
                            if ((h & 0x7f80) == 0x7f80) want_bad = 1;
                            if (got_u16[dst_off + c_idx] != h) return "interleave_bf16 output differs from transpose reference";
                        }
                    }
                }
            },
        }
        if (got_bad != want_bad) return "nonfinite flag mismatch on hy4_rows sampling kernel";
        if (!try self.rig.intact()) return "hy4_rows sampling kernel overwrote a guard region";
        return null;
    }
};
