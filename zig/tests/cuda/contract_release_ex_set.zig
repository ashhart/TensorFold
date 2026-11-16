//! DeepSeek V4.1's release experts on CUDA: the shared expert's SwiGLU round trip and the experts' registrations.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const kr = @import("dsv41_release");
const xr = @import("contract_release_ex_ref.zig");
const ex = @import("contract_release_ex.zig");
const more = @import("contract_release_ex_more.zig");
const glue = @import("contract_release_glue.zig");
const Rig = @import("contract_rig.zig").Rig;
const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const kx = kr.ex;
const servedOf = @import("contract_release.zig").servedOf;

fn none(_: c.Shape) usize {
    return 0;
}

/// tf_dsv41_rel_swiglu_actq: shape (rows, n, 1) over bf16 gate and up views; `v` within its range, or its codes.
pub const Swiglu = struct {
    rig: Rig,
    ops: kx.Ops,
    name: []const u8,
    codes: bool,
    served: []const c.Shape,
    gate: []u16 = &.{},
    up: []u16 = &.{},
    args: kx.Swiglu = undefined,
    shape: c.Shape = undefined,
    want: std.ArrayList(u8) = .empty,
    vbits: std.ArrayList(u16) = .empty,
    msg: [160]u8 = undefined,

    pub fn kernel(self: *Swiglu) g.Kernel {
        return .{ .spec = .{ .name = self.name, .accum = .none, .depth = none, .out = .raw, .nan = .{ .stated = "v takes its FP32 steps' class; a non-finite v makes its 32-column block NaN" }, .tiles = .{ .{ .size = 16 }, .{ .size = 64, .step = 32, .min = 32 }, .{} }, .served = self.served }, .ctx = self, .vt = &vt };
    }

    const vt = blk: {
        var v = g.hooks(Swiglu);
        v.release = release;
        break :blk v;
    };

    fn release(ctx: *anyopaque) void {
        const self: *Swiglu = @ptrCast(@alignCast(ctx));
        const a = self.rig.gpu.gpa;
        a.free(self.gate);
        a.free(self.up);
        self.gate = &.{};
        self.up = &.{};
        self.want.deinit(a);
        self.want = .empty;
        self.vbits.deinit(a);
        self.vbits = .empty;
        self.rig.finishGates();
    }

    pub fn deinit(self: *Swiglu) void {
        release(self);
    }

    pub fn prepare(self: *Swiglu, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const a = self.rig.gpu.gpa;
        a.free(self.gate);
        a.free(self.up);
        self.gate = &.{};
        self.up = &.{};
        self.rig.reset();
        if (shape.rows == 0 or shape.n == 0 or shape.n % 32 != 0) return false;
        self.shape = shape;
        const m = shape.rows;
        const n = shape.n;
        const ld = n + 32;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.gate = try a.alloc(u16, m * ld);
        self.up = try a.alloc(u16, m * ld);
        for (self.gate, self.up, 0..) |*gv, *uv, i| {
            const edge = i % 17 == 3;
            gv.* = c.toBf16(if (edge) 10 else (rng.float(f32) * 2 - 1) * std.math.ldexp(@as(f32, 1), rng.intRangeAtMost(i32, -6, 4)));
            uv.* = c.toBf16(if (edge) -10.5 else (rng.float(f32) * 2 - 1) * std.math.ldexp(@as(f32, 1), rng.intRangeAtMost(i32, -6, 4)));
        }
        if (fill == .nonfinite) {
            const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (put, 0..) |v, r| if (r < m) {
                self.gate[r * ld + (r * 5) % n] = c.toBf16(v);
            };
            if (m > 3) self.up[3 * ld + n / 2] = c.toBf16(-std.math.inf(f32));
        }
        const pn = m * n;
        self.args = .{
            .gate = try self.rig.input(std.mem.sliceAsBytes(self.gate)),
            .gate_ld = @intCast(ld),
            .up = try self.rig.input(std.mem.sliceAsBytes(self.up)),
            .up_ld = @intCast(ld),
            .codes = if (self.codes) try self.rig.output(pn) else try self.rig.scratch(pn),
            .scales = if (self.codes) try self.rig.output(pn / 32) else try self.rig.scratch(pn / 32),
            .v = if (self.codes) try self.rig.scratch(2 * pn) else try self.rig.output(2 * pn),
            .m = @intCast(m),
            .n = @intCast(n),
        };
        return true;
    }

    pub fn run(self: *Swiglu) !void {
        try self.ops.swigluActq(self.args);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Swiglu) usize {
        const pn = self.shape.rows * self.shape.n;
        return if (self.codes) pn + pn / 32 else 2 * pn;
    }

    pub fn reference(_: *Swiglu, _: []f64, _: []f64) !void {
        return error.CheckedByBytes;
    }

    pub fn check(self: *Swiglu, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const m = self.shape.rows;
        const n = self.shape.n;
        try self.vbits.resize(a, m * n);
        @memcpy(std.mem.sliceAsBytes(self.vbits.items), try self.rig.bytes(self.args.v, 2 * m * n));
        if (self.codes) {
            const want = try glue.expected(a, self.vbits.items, m, n, false, &self.want);
            return glue.verdict(&self.msg, try self.rig.outputBytes(), want, m * n);
        }
        const ld = n + 32;
        for (0..m) |r| for (0..n) |col| {
            const gv = c.fromBf16(self.gate[r * ld + col]);
            const uv = c.fromBf16(self.up[r * ld + col]);
            const got = self.vbits.items[r * n + col];
            const ok = if (xr.swigluRange(gv, uv, 1, false)) |range| xr.inSpan(got, range) else c.classOf(c.fromBf16(got)) == c.classOf(xr.swigluIeee(gv, uv, 1, false));
            if (!ok) return std.fmt.bufPrint(&self.msg, "row {d} column {d}: v {e} outside the SwiGLU range of g {e}, u {e}", .{ r, col, c.fromBf16(got), gv, uv }) catch "v outside its range";
        };
        return null;
    }
};

const rows = [_]usize{ 1, 5, 16, 17, 64, 2048 };
const group_6 = servedOf(&.{.{ 6, 384 }}, &rows);
const group_3 = servedOf(&.{.{ 3, 128 }}, &rows);
const gateup_served = servedOf(&.{.{ 5120, 576 }}, &rows);
const down_served = servedOf(&.{.{ 576, 5120 }}, &rows);
const combine_served = servedOf(&.{.{ 6, 5120 }}, &rows) ++ servedOf(&.{.{ 3, 5120 }}, &.{ 1, 17 });
const swiglu_served = servedOf(&.{.{ 1, 576 }}, &rows);

/// The expert registrations, sm_121 only; the image from the contract's own embedded set.
pub const Set = struct {
    gpa: std.mem.Allocator,
    on: bool = false,
    mod: cuda.Module = undefined,
    f: kx.Functions = undefined,
    groups: [4]ex.Group = undefined,
    gateup: [3]ex.GateUp = undefined,
    down: more.Down = undefined,
    combine: [3]more.Combine = undefined,
    swiglu: [2]Swiglu = undefined,

    pub fn load(d: *const cuda.Driver, mod: *cuda.Module) !?kx.Functions {
        const image = @import("contract_deepseek_images.zig").image("dsv41_rel_ex").?;
        if (image.len == 0) return null;
        mod.* = try cuda.Module.load(d, image);
        errdefer mod.unload();
        return try kx.Functions.resolve(mod.*);
    }

    pub fn init(gpu: @import("check.zig").Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        set.* = .{ .gpa = gpu.gpa };
        if (try gpu.ctx.capability() != 121) return set;
        set.f = try load(gpu.d, &set.mod) orelse return error.ReleaseImagesNotBuilt;
        const ops: kx.Ops = .{ .f = &set.f, .s = s };
        const rig: Rig = .{ .gpu = gpu, .s = s };
        set.groups = .{
            .{ .rig = rig, .ops = ops, .name = "dsv41_rel_ex group, top-6 of 384 held", .held = .all, .served = &group_6 },
            .{ .rig = rig, .ops = ops, .name = "dsv41_rel_ex group, top-6 of 384, a quarter held", .held = .quarter, .served = &group_6 },
            .{ .rig = rig, .ops = ops, .name = "dsv41_rel_ex group, top-3 of 128 held", .held = .all, .served = &group_3 },
            .{ .rig = rig, .ops = ops, .name = "dsv41_rel_ex group, top-3 of 128, a quarter held", .held = .quarter, .served = &group_3 },
        };
        set.gateup = .{
            .{ .rig = rig, .ops = ops, .name = "dsv41_rel_ex gateup FP32 sums", .part = .sums, .topk = 6, .nids = 384, .nslots = 24, .served = &gateup_served },
            .{ .rig = rig, .ops = ops, .name = "dsv41_rel_ex gateup SwiGLU v", .part = .v, .topk = 6, .nids = 384, .nslots = 24, .served = &gateup_served },
            .{ .rig = rig, .ops = ops, .name = "dsv41_rel_ex gateup round trip", .part = .codes, .topk = 3, .nids = 128, .nslots = 24, .served = &gateup_served },
        };
        set.down = .{ .rig = rig, .ops = ops, .topk = 6, .nids = 384, .nslots = 24, .served = &down_served };
        set.combine = .{
            .{ .rig = rig, .ops = ops, .name = "dsv41_rel_ex combine with the shared expert, bf16 out", .f32_out = false, .shared = true, .nids = 384, .served = &combine_served },
            .{ .rig = rig, .ops = ops, .name = "dsv41_rel_ex combine, fp32 out", .f32_out = true, .shared = false, .nids = 384, .served = &combine_served },
            .{ .rig = rig, .ops = ops, .name = "dsv41_rel_ex combine in the FP32 order, fp32 out", .f32_out = true, .shared = true, .exact = true, .nids = 384, .served = &combine_served },
        };
        set.swiglu = .{
            .{ .rig = rig, .ops = ops, .name = "dsv41_rel_ex shared SwiGLU v", .codes = false, .served = &swiglu_served },
            .{ .rig = rig, .ops = ops, .name = "dsv41_rel_ex shared SwiGLU round trip", .codes = true, .served = &swiglu_served },
        };
        set.on = true;
        for (&set.groups) |*x| try list.append(a, x.kernel());
        for (&set.gateup) |*x| try list.append(a, x.kernel());
        try list.append(a, set.down.kernel());
        for (&set.combine) |*x| try list.append(a, x.kernel());
        for (&set.swiglu) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        if (set.on) {
            for (&set.groups) |*x| x.deinit();
            for (&set.gateup) |*x| x.deinit();
            set.down.deinit();
            for (&set.combine) |*x| x.deinit();
            for (&set.swiglu) |*x| x.deinit();
            set.mod.unload();
        }
        set.gpa.destroy(set);
    }
};
