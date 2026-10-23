//! Compare the fused epilogue against the original launches on the same guarded inputs.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const c = core.contract.spec;
const g = core.contract.gates;
const Rig = @import("contract_rig.zig").Rig;

fn depth(_: c.Shape) usize {
    return 2;
}

fn spec(act: cuda.vit.Act, residual: bool, in_place: bool) c.Spec {
    const names = comptime blk: {
        var result: [3][4][]const u8 = undefined;
        for (.{ "no residual", "residual", "in-place residual" }, 0..) |mode, i| {
            for (.{ "none", "gelu_tanh", "gelu", "relu2" }, 0..) |activation, j| {
                result[i][j] = "cuda vit tf_vit_bias_act " ++ activation ++ " " ++ mode;
            }
        }
        break :blk result;
    };
    return .{
        .name = names[if (in_place) @as(usize, 2) else if (residual) 1 else 0][@intCast(@backingInt(act))],
        .accum = .f32,
        .depth = depth,
        .out = .bf16,
        .in_place = in_place,
        .replaces = "tf_vit_bias then tf_vit_pointwise",
        .nan = .{ .stated = "byte-identical to bias then pointwise, including non-finite inputs" },
        .tiles = .{ .{}, .{ .size = 256 }, .{} },
        .served = &.{ .{ .rows = 13322, .n = 1280, .k = 1 }, .{ .rows = 13322, .n = 5120, .k = 1 } },
        .sums = .{ .f16 = 0, .bf16 = 0, .why = "pointwise rounding boundaries are checked against the replaced kernels" },
    };
}

pub const Glue = struct {
    rig: Rig,
    k: *const cuda.vit.Kernels,
    act: cuda.vit.Act,
    residual: bool,
    in_place: bool,
    wrong_bias_round: bool = false,
    ctl: ?cuda.Function = null,
    shape: c.Shape = undefined,
    acc: u64 = 0,
    bias: u64 = 0,
    r: u64 = 0,
    out: u64 = 0,
    mid: u64 = 0,
    base: u64 = 0,
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Glue) void {
        self.rig.deinit();
    }

    const vt = g.hooks(Glue);

    pub fn kernel(self: *Glue) g.Kernel {
        var declared = spec(self.act, self.residual, self.in_place);
        if (self.wrong_bias_round) declared.name = "control: RADIO omitted bias BF16 round";
        return .{ .spec = declared, .ctx = self, .vt = &vt, .control = if (self.wrong_bias_round) .shapes else null };
    }

    pub fn prepare(self: *Glue, sh: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (sh.k != 1) return false;
        self.rig.reset();
        self.shape = sh;
        const a = self.rig.gpu.gpa;
        const acc = try a.alloc(f32, self.count());
        defer a.free(acc);
        const bias = try a.alloc(u16, sh.n);
        defer a.free(bias);
        const residual = try a.alloc(u16, self.count());
        defer a.free(residual);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (acc, residual, 0..) |*v, *r, i| {
            v.* = rng.float(f32) * 16 - 8;
            r.* = c.toBf16(rng.float(f32) * 8 - 4);
            if (fill == .nonfinite) v.* = switch (i % 7) {
                0 => std.math.nan(f32),
                1 => std.math.inf(f32),
                2 => -std.math.inf(f32),
                3 => -0.0,
                else => v.*,
            };
        }
        for (bias) |*b| b.* = c.toBf16(rng.float(f32) * 2 - 1);
        if (fill == .nonfinite) {
            bias[0] = c.toBf16(std.math.nan(f32));
            if (sh.n > 1) bias[1] = c.toBf16(std.math.inf(f32));
            for (residual, 0..) |*r, i| {
                if (i % 11 == 4) r.* = c.toBf16(std.math.inf(f32));
                if (i % 11 == 5) r.* = c.toBf16(-std.math.inf(f32));
            }
        }
        if (fill == .stress) {
            @memset(acc, 8);
            @memset(bias, c.toBf16(0.25));
            @memset(residual, c.toBf16(-8));
        }
        // The same finite tie in the real variants and control proves the intermediate round matters.
        if (fill == .random and self.act == .none and self.residual and acc.len > 0) {
            acc[0] = 1.00390625;
            bias[0] = 0;
            residual[0] = c.toBf16(-1);
        }
        self.acc = try self.rig.input(std.mem.sliceAsBytes(acc));
        self.bias = try self.rig.input(std.mem.sliceAsBytes(bias));
        self.r = try self.rig.input(std.mem.sliceAsBytes(residual));
        self.out = try self.rig.output(self.count() * 2);
        self.mid = try self.rig.scratch(self.count() * 2);
        self.base = try self.rig.scratch(self.count() * 2);
        return true;
    }

    pub fn count(self: *Glue) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn run(self: *Glue) !void {
        if (self.in_place) try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = self.out, .len = self.count() * 2 }).copyFrom(0, self.r, self.count() * 2, self.rig.s.handle);
        const residual: ?u64 = if (self.in_place) self.out else if (self.residual) self.r else null;
        if (self.ctl) |f| {
            var args: cuda.Args = .{};
            for ([_]u64{ self.acc, self.bias, residual orelse 0, self.out }) |v| args.add(v);
            args.add(@as(c_int, @intCast(self.shape.n)));
            try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = @intCast((self.shape.n + 255) / 256) }, .block = .{ .x = 256 } }, self.rig.s, &args);
        } else try self.k.biasAct(self.rig.s, self.acc, self.bias, residual, self.out, self.shape.rows, self.shape.n, self.act);
        try self.rig.s.synchronize();
    }

    pub fn replaced(self: *Glue) ![]const u8 {
        try self.k.bias(self.rig.s, self.acc, self.bias, self.mid, self.shape.rows, self.shape.n);
        try self.k.pointwise(self.rig.s, self.mid, if (self.residual) self.r else null, self.base, self.count(), self.act);
        return self.rig.bytes(self.base, self.count() * 2);
    }

    pub fn reference(self: *Glue, want: []f64, mag: []f64) !void {
        const bytes = try self.replaced();
        for (want, mag, 0..) |*w, *m, i| {
            w.* = c.read(.bf16, bytes, i);
            m.* = @abs(w.*);
        }
    }

    pub fn simulate(self: *Glue, acc: c.Accum, order: g.Order, out: []f64) !void {
        const biased = core.contract.rows.sumIn(acc, order, &.{ 8, 0.25 });
        var v = c.fromBf16(c.toBf16(@floatCast(biased)));
        // Both GELU forms round to the identity on these positive stress inputs.
        if (self.act == .relu2) v = c.fromBf16(c.toBf16(v * v));
        const result = if (self.residual) core.contract.rows.sumIn(acc, order, &.{ -8, v }) else v;
        @memset(out, c.fromBf16(c.toBf16(@floatCast(result))));
    }

    pub fn check(self: *Glue, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        const want = try self.replaced();
        if (std.mem.indexOfDiff(u8, got, want)) |at| return try std.fmt.bufPrint(&self.msg, "byte {d} of {d} differs from tf_vit_bias then tf_vit_pointwise", .{ at, got.len });
        for (0..self.count()) |i| if (c.classOf(c.read(.bf16, got, i)) != .finite) return null;
        return "no output is non-finite: the planted values prove nothing";
    }
};
