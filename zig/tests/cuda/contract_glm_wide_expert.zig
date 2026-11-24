//! Actual GLM down dispatch: a hot expert with a 64-pair plan, compared to its unchanged waves path.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;
fn depth(shape: c.Shape) usize {
    return shape.k;
}
pub fn spec() c.Spec {
    const served = comptime blk: {
        var out: [8]c.Shape = undefined;
        var at: usize = 0;
        for (.{ 512, 1024 }) |k| for (.{ 16, 17, 24, 32 }) |rows| {
            out[at] = .{ .rows = rows, .n = 4096, .k = k };
            at += 1;
        };
        break :blk out;
    };
    return .{ .name = "cuda GLM wide expert fact down", .accum = .f32, .depth = depth, .out = .raw, .replaces = "GLM factored down waves", .nan = .{ .stated = "Exact original GLM down bytes including NaN/Inf payloads and untouched skipped slots" }, .served = &served, .sums = .{ .why = "Same per-group MMA from zero, scale FMA and global scale; replacement byte equality" } };
}
pub const Partial = struct {
    rig: rig_.Rig,
    ks: *glm.experts.Kernels,
    shape: c.Shape = undefined,
    plan: glm.experts.Plan = undefined,
    layer: cuda.experts.Layer = undefined,
    x: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,
    nonfinite: bool = false,
    corrupt: bool = false,
    pub fn kernel(self: *Partial) g.Kernel {
        var s = spec();
        if (self.nonfinite) s.name = "cuda GLM wide expert fact down nonfinite";
        if (self.corrupt) s.name = "control: GLM wide expert output byte changed";
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = if (self.corrupt) .shapes else null };
    }
    const vt = g.hooks(Partial);
    pub fn deinit(self: *Partial) void {
        self.rig.deinit();
    }
    pub fn count(self: *Partial) usize {
        return self.shape.rows * 2 * self.shape.n * 4;
    }
    pub fn prepare(self: *Partial, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows < 1 or shape.rows > 64 or shape.n != 4096 or (shape.k != 512 and shape.k != 1024)) return false;
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        const x = try a.alloc(u16, shape.rows * 2 * shape.k);
        defer a.free(x);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (x) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        if (self.nonfinite or fill == .nonfinite) for (0..shape.rows) |row| for ([_]u16{ 0x7fc1, 0xffc2, 0x7f80, 0xff80 }, 0..) |h, i| {
            x[row * 2 * shape.k + 31 + i * 32] = h;
        };
        const weights = try a.alloc(u8, (shape.n / 32) * (shape.k / 32) * 576);
        defer a.free(weights);
        var off: usize = 0;
        while (off < weights.len) : (off += 576) {
            rng.bytes(weights[off..][0..512]);
            @memset(weights[off + 512 ..][0..64], 0x20);
        }
        const global: f32 = 0.125;
        self.x = try self.rig.input(std.mem.sliceAsBytes(x));
        self.layer = .{ .format = .nvfp4, .up = 0, .down = try self.rig.input(weights), .down_scale = try self.rig.input(std.mem.asBytes(&global)), .dims = shape.n, .width = shape.k, .experts = 1 };
        const items = [_]i32{ 0, 0, @intCast(shape.rows) };
        const counts = [_]i32{1};
        const members = try a.alloc(i32, shape.rows);
        defer a.free(members);
        for (members, 0..) |*v, i| v.* = @intCast(i * 2);
        self.plan = .{ .route = .{ .items = try self.rig.input(std.mem.asBytes(&items)), .counts = try self.rig.input(std.mem.asBytes(&counts)), .members = try self.rig.input(std.mem.sliceAsBytes(members)), .rank = 0, .hist = 0 }, .slots = 2, .experts = 2, .prompt = false, .fact = true };
        self.out = try self.rig.output(self.count());
        self.base = try self.rig.scratch(self.count());
        const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.base, .len = self.count() };
        try b.fill8(0xff, self.rig.s.handle);
        return true;
    }
    fn launch(self: *Partial, wide: bool) !void {
        var ks = self.ks.*;
        ks.wide_tile = wide;
        try glm.experts.run(&ks, self.rig.s, .down_f32, self.layer, self.plan, self.x, self.shape.k, self.shape.rows, if (wide) self.out else self.base, 1);
        try self.rig.s.synchronize();
    }
    pub fn run(self: *Partial) !void {
        try self.launch(true);
        if (self.corrupt) {
            const bytes = try self.rig.outputBytes();
            var byte = [_]u8{bytes[0] ^ 1};
            const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.out, .len = self.count() };
            try b.upload(0, &byte);
        }
    }
    pub fn replaced(self: *Partial) ![]const u8 {
        try self.launch(false);
        return self.rig.bytes(self.base, self.count());
    }
    pub fn reference(_: *Partial, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }
    pub fn check(self: *Partial, _: g.Gate) !?[]const u8 {
        const got = try self.rig.gpu.gpa.dupe(u8, try self.rig.outputBytes());
        defer self.rig.gpu.gpa.free(got);
        return if (std.mem.eql(u8, got, try self.replaced())) null else "GLM wide expert output differs from original waves";
    }
};
pub const Suite = struct {
    ks: glm.experts.Kernels,
    cells: [3]Partial,
    pub fn create(gpu: @import("check.zig").Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !*Suite {
        const a = gpu.gpa;
        const self = try a.create(Suite);
        errdefer a.destroy(self);
        self.ks = .{ .nvfp4 = try cuda.experts.Nvfp4Experts.load(gpu.d, @intCast(try gpu.ctx.attribute(.multiprocessor_count))), .b32 = null, .router = undefined };
        errdefer self.ks.nvfp4.unload();
        for (&self.cells, 0..) |*p, i| {
            p.* = .{ .rig = .{ .gpu = gpu, .s = s }, .ks = &self.ks, .nonfinite = i == 1, .corrupt = i == 2 };
            try kernels.append(a, p.kernel());
        }
        return self;
    }
    pub fn destroy(self: *Suite) void {
        const a = self.cells[0].rig.gpu.gpa;
        for (&self.cells) |*p| p.deinit();
        self.ks.nvfp4.unload();
        a.destroy(self);
    }
};
