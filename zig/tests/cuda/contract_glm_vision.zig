//! GLM's image tower on vit's full attention under the contract: glm_vis_qk's rows, then vision_kernel<64, 8>.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm_cuda");
const check_ = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;

const D = 64;
const H = 16; // heads
const W = H * D;

const controls = struct {
    pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
};

/// Shape: heads (rows), head width (n), a frame group's patches (k): two heads at an image's sizes, 16 at small ones.
pub const Full = @import("contract_vit_full.zig").Full(D, "cuda glm_vision vit vision_kernel<64, 8> (full attention)", &.{ .{ .rows = 2, .n = D, .k = 6528 }, .{ .rows = 2, .n = D, .k = 1024 }, .{ .rows = H, .n = D, .k = 196 }, .{ .rows = H, .n = D, .k = 36 }, .{ .rows = H, .n = D, .k = 4 } });

/// A lane's two squares, then the warp's 5 shuffles.
fn qkDepth(_: c.Shape) usize {
    return 2 + 5;
}

/// Shape: patches (rows), heads (n), head width (k).
fn qkSpec(control: bool) c.Spec {
    return .{
        .name = if (control) "control: glm_vis_qk rotates a head before its bf16 rounding" else "cuda glm_vision glm_vis_qk (q and k head norm and 2D rotation, in place)",
        .accum = .f32,
        .depth = qkDepth,
        .out = .bf16,
        .in_place = true,
        .replaces = "glm_vis_qkv at segments of one row",
        .nan = .{ .stated = "byte-identical to glm_vis_qkv's rows, including non-finite inputs" },
        .served = &.{ .{ .rows = 6528, .n = H, .k = D }, .{ .rows = 1024, .n = H, .k = D }, .{ .rows = 196, .n = H, .k = D }, .{ .rows = 4, .n = H, .k = D } },
        .sums = .{ .f16 = 0, .bf16 = 0, .why = "a head's sum of squares is checked against the replaced kernel's bytes" },
    };
}

/// glm_vis_qk through cuda_vision_attn.heads, on a copy of the rows [rows, 3 x 1024], against glm_vis_qkv.
pub const Qk = struct {
    rig: rig_.Rig,
    run_: glm.vision.Run,
    att: *cuda.vit.Attention,
    control: bool = false,
    shape: c.Shape = undefined,
    qkv: u64 = 0,
    norms: u64 = 0,
    pos: u64 = 0,
    freq: u64 = 0,
    out: u64 = 0,
    /// glm_vis_qkv's q, k and v at segments of one row: [rows, 1024] each
    old: [3]u64 = @splat(0),
    base: []u8 = &.{},
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Qk) void {
        self.rig.gpu.gpa.free(self.base);
        self.rig.deinit();
    }

    const vt = g.hooks(Qk);

    pub fn kernel(self: *Qk) g.Kernel {
        return .{ .spec = qkSpec(self.control), .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null };
    }

    pub fn prepare(self: *Qk, sh: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (sh.rows == 0 or sh.n != H or sh.k != D) return false;
        self.rig.reset();
        self.shape = sh;
        const a = self.rig.gpu.gpa;
        const x = try a.alloc(u16, self.count());
        defer a.free(x);
        const pos = try a.alloc(i32, 2 * sh.rows);
        defer a.free(pos);
        var norms: [2 * D]u16 = undefined;
        var freq: [16]f32 = undefined;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (x) |*v| v.* = c.toBf16(rng.float(f32) * 8 - 4);
        for (&norms) |*v| v.* = c.toBf16(0.25 + rng.float(f32) * 1.75);
        for (pos) |*p| p.* = rng.intRangeLessThan(i32, 0, 275);
        for (&freq, 0..) |*v, i| v.* = 1.0 / std.math.pow(f32, 10000.0, @as(f32, @floatFromInt(2 * i)) / 32.0);
        if (fill == .nonfinite) for (x, 0..) |*v, i| switch (i % (3 * W + 7)) {
            0 => v.* = c.toBf16(std.math.nan(f32)),
            1 => v.* = c.toBf16(std.math.inf(f32)),
            2 => v.* = c.toBf16(-std.math.inf(f32)),
            else => {},
        };
        if (fill == .stress) { // every q and k element 8 under unit weights at the origin: each head norms to ones
            for (0..sh.rows) |r| @memset(x[r * 3 * W ..][0 .. 2 * W], c.toBf16(8));
            @memset(&norms, c.toBf16(1));
            @memset(pos, 0);
        }
        self.qkv = try self.rig.input(std.mem.sliceAsBytes(x));
        self.norms = try self.rig.input(std.mem.sliceAsBytes(&norms));
        self.pos = try self.rig.input(std.mem.sliceAsBytes(pos));
        self.freq = try self.rig.input(std.mem.sliceAsBytes(&freq));
        self.out = try self.rig.output(self.count() * 2);
        for (&self.old) |*o| o.* = try self.rig.scratch(sh.rows * W * 2);
        a.free(self.base);
        self.base = &.{};
        self.base = try a.alloc(u8, self.count() * 2);
        return true;
    }

    pub fn count(self: *Qk) usize {
        return self.shape.rows * 3 * W;
    }

    fn heads(self: *Qk, rows: bool, qkv: u64) !void {
        var tower: glm.vision.Tower = undefined;
        tower.full = if (rows) self.att.* else null;
        tower.inv_freq.ptr = self.freq;
        const b: glm.vision_attn.Buffers = .{ .qkv = qkv, .qh = self.old[0], .kh = self.old[1], .vt = self.old[2], .att = 0, .sc = 0, .pr = 0, .oh = 0, .sums = 0 };
        try glm.vision_attn.heads(&tower, &self.run_, b, self.norms, self.norms + D * 2, self.pos, @intCast(self.shape.rows), 1);
    }

    pub fn run(self: *Qk) !void {
        try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = self.out, .len = self.count() * 2 }).copyFrom(0, self.qkv, self.count() * 2, self.rig.s.handle);
        try self.heads(true, self.out);
        try self.rig.s.synchronize();
    }

    /// glm_vis_qkv's bytes as rows of q, k, v: at segments of one row its head-major q and k are row-major.
    pub fn replaced(self: *Qk) ![]const u8 {
        try self.heads(false, self.qkv);
        const row = W * 2;
        for (self.old, 0..) |ptr, which| {
            const part = try self.rig.bytes(ptr, self.shape.rows * row);
            for (0..self.shape.rows) |r| @memcpy(self.base[(r * 3 + which) * row ..][0..row], part[r * row ..][0..row]);
        }
        return self.base;
    }

    pub fn reference(self: *Qk, want: []f64, mag: []f64) !void {
        const bytes = try self.replaced();
        for (want, mag, 0..) |*w, *m, i| {
            w.* = c.read(.bf16, bytes, i);
            m.* = @abs(w.*);
        }
    }

    /// The stress rows: 64 squares of 8 a head, then 8 / sqrt(their mean) as bf16; v's 8s as they are.
    pub fn simulate(self: *Qk, acc: c.Accum, order: g.Order, out: []f64) !void {
        const squares: [D]f64 = @splat(64);
        const ss = core.contract.rows.sumIn(acc, order, &squares);
        const one = c.fromBf16(c.toBf16(@floatCast(8 / @sqrt(ss / D + 1e-5))));
        for (0..self.shape.rows) |r| {
            @memset(out[r * 3 * W ..][0 .. 2 * W], one);
            for (out[(r * 3 + 2) * W ..][0..W], 0..) |*v, i| v.* = c.read(.bf16, self.base, (r * 3 + 2) * W + i);
        }
    }

    pub fn check(self: *Qk, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        const want = try self.replaced();
        if (std.mem.indexOfDiff(u8, got, want)) |at| return try std.fmt.bufPrint(&self.msg, "byte {d} of {d} differs from glm_vis_qkv's rows", .{ at, got.len });
        for (0..self.count()) |i| if (c.classOf(c.read(.bf16, got, i)) != .finite) return null;
        return "no output is non-finite: the planted values prove nothing";
    }
};

/// The tower's q and k rows with their control, and its attention, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    mod: cuda.Module,
    ctl: cuda.Module,
    att: cuda.vit.Attention,
    qk: [2]Qk = undefined,
    full: Full = undefined,

    /// The set, its kernels appended to `list` (the control after the kernel it breaks).
    pub fn init(gpu: check_.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_vision);
        errdefer mod.unload();
        var ctl = try cuda.Module.load(gpu.d, &controls.bytes);
        errdefer ctl.unload();
        set.* = .{ .gpa = gpu.gpa, .mod = mod, .ctl = ctl, .att = try cuda.vit.Attention.load(gpu.d) };
        errdefer set.att.unload();
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        const f = try glm.vision.Funcs.load(mod);
        var broken = f;
        broken.qk = try ctl.function("tf_ctl_glm_vis_qk_unrounded");
        set.qk[0] = .{ .rig = rig, .run_ = .{ .stream = s, .f = f }, .att = &set.att };
        set.qk[1] = .{ .rig = rig, .run_ = .{ .stream = s, .f = broken }, .att = &set.att, .control = true };
        set.full = .{ .rig = rig, .att = &set.att };
        for (&set.qk) |*x| try list.append(a, x.kernel());
        try list.append(a, set.full.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.qk) |*x| x.deinit();
        set.full.deinit();
        set.att.unload();
        set.ctl.unload();
        set.mod.unload();
        set.gpa.destroy(set);
    }
};
