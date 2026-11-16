//! Routed half-product rotations against binary64; the existing widened product is the planted rounding control.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const oracle = @import("exl3_routed_half_reference.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const served = [_]c.Shape{
    .{ .rows = 8, .n = 4608, .k = 128 },
    .{ .rows = 16, .n = 2304, .k = 128 },
    .{ .rows = 2048, .n = 4608, .k = 128 },
};
fn depth(_: c.Shape) usize {
    return 7;
}
const untouched: f64 = @as(f16, @bitCast(@as(u16, 0x3c3c)));

const Case = struct {
    rig: rig_.Rig,
    shared: bool,
    experts: usize,
    control: bool,
    function: cuda.Function,
    wide: cuda.Function,
    shape: c.Shape = undefined,
    x: std.ArrayList(f16) = .empty,
    suh: std.ArrayList(f16) = .empty,
    picks: std.ArrayList(i32) = .empty,
    xd: u64 = 0,
    sd: u64 = 0,
    pd: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,

    fn kernel(self: *Case) g.Kernel {
        const name = if (self.control)
            (if (self.shared) "control exl3 routed half shared widened product" else "control exl3 routed half per-pair widened product")
        else if (self.experts == 384)
            (if (self.shared) "cuda exl3 routed half shared 384 experts" else "cuda exl3 routed half per-pair 384 experts")
        else
            (if (self.shared) "cuda exl3 routed half shared 192 experts" else "cuda exl3 routed half per-pair 192 experts");
        return .{ .spec = .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .ops = 1,
            .out = .f16,
            .served = &served,
            .tiles = .{ .{ .size = 8 }, .{ .size = 128, .step = 128, .min = 128 }, .{} },
            .replaces = if (self.control) "routed FP16 product before the FP32 rotation" else null,
        }, .ctx = self, .vt = &g.hooks(Case), .control = if (self.control) .shapes else null };
    }

    pub fn reset(self: *Case) void {
        self.rig.finishGates();
    }
    fn deinit(self: *Case) void {
        self.rig.deinit();
        self.x.deinit(self.rig.gpu.gpa);
        self.suh.deinit(self.rig.gpu.gpa);
        self.picks.deinit(self.rig.gpu.gpa);
    }
    fn slots(self: *const Case) usize {
        return if (self.shared) 8 else 0;
    }

    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 2048 or shape.n < 128 or shape.n > 4608 or shape.n % 128 != 0 or shape.k != 128) return false;
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        const rows = if (self.shared) (shape.rows + 7) / 8 else shape.rows;
        try self.x.resize(a, rows * shape.n);
        try self.suh.resize(a, self.experts * shape.n);
        try self.picks.resize(a, shape.rows);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.x.items) |*v| v.* = @floatCast((rng.float(f32) - 0.5) * 2);
        for (self.suh.items) |*v| v.* = @floatCast(0.5 + rng.float(f32));
        for (self.picks.items, 0..) |*p, i| p.* = switch (i % 8) {
            0 => 0,
            1 => @intCast(self.experts - 1),
            5 => -1,
            6 => @intCast(self.experts),
            else => @intCast(i % self.experts),
        };
        if (fill == .stress) {
            @memset(self.suh.items, 1);
            @memset(self.picks.items, 0);
            for (self.x.items, 0..) |*v, i| v.* = if (i % 128 == 0) 1025 else 1024;
        } else if (self.control) {
            @memset(self.x.items, 0);
            @memset(self.suh.items, 1);
            for (0..rows) |row| {
                self.x.items[row * shape.n] = 1 + 0x1p-10;
                self.x.items[row * shape.n + 1] = 1;
            }
            self.suh.items[0] = 1 + 0x1p-10;
            self.suh.items[1] = 1 + 0x1p-9;
        } else if (fill == .nonfinite) {
            self.suh.items[0] = std.math.nan(f16);
            self.suh.items[2 * shape.n] = std.math.inf(f16);
            self.suh.items[3 * shape.n] = -std.math.inf(f16);
            self.suh.items[4 * shape.n] = std.math.inf(f16);
            self.suh.items[4 * shape.n + 1] = -std.math.inf(f16);
            self.x.items[128] = std.math.nan(f16);
            self.x.items[256] = std.math.inf(f16);
            self.x.items[384] = -std.math.inf(f16);
        }
        self.xd = try self.rig.input(std.mem.sliceAsBytes(self.x.items));
        self.sd = try self.rig.input(std.mem.sliceAsBytes(self.suh.items));
        self.pd = try self.rig.input(std.mem.sliceAsBytes(self.picks.items));
        self.out = try self.rig.output(self.count() * 2);
        try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = self.out, .len = self.count() * 2 }).fill8(0x3c, self.rig.s.handle);
        self.base = try self.rig.scratch(self.count() * 2);
        return true;
    }

    pub fn count(self: *Case) usize {
        return self.shape.rows * self.shape.n;
    }
    fn launch(self: *Case, out: u64) !void {
        try cuda.exl3_routed_half.run(self.function, self.rig.s, self.xd, self.pd, self.sd, out, .{
            .pairs = self.shape.rows,
            .width = self.shape.n,
            .input_slots = self.slots(),
            .experts = self.experts,
        });
    }
    pub fn run(self: *Case) !void {
        if (!self.control) return self.launch(self.out);
        var args: cuda.Args = .{};
        args.add(self.xd);
        args.add(@as(i32, @intCast(self.shape.n)));
        args.add(@as(i32, 0));
        inline for (.{ self.pd, self.sd, self.sd, self.out, self.out }) |p| args.add(p);
        inline for (.{ self.shape.n, if (self.shared) @as(usize, 8) else 1, self.experts }) |n| args.add(@as(i32, @intCast(n)));
        try cuda.launch.launch(self.wide, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = @intCast(self.shape.n / 128) }, .block = .{ .x = 32 } }, self.rig.s, &args);
    }
    pub fn replaced(self: *Case) ![]const u8 {
        try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = self.base, .len = self.count() * 2 }).fill8(0x3c, self.rig.s.handle);
        try self.launch(self.base);
        return self.rig.bytes(self.base, self.count() * 2);
    }

    pub fn reference(self: *Case, want: []f64, mag: []f64) !void {
        const width = self.shape.n;
        for (self.picks.items, 0..) |pick, pair| {
            const at = pair * width;
            if (pick < 0 or @as(usize, @intCast(pick)) >= self.experts) {
                @memset(want[at..][0..width], untouched);
                @memset(mag[at..][0..width], 0);
                continue;
            }
            const row = if (self.shared) pair / 8 else pair;
            var offset: usize = 0;
            while (offset < width) : (offset += 128) oracle.block(
                self.x.items[row * width + offset ..][0..128],
                self.suh.items[@as(usize, @intCast(pick)) * width + offset ..][0..128],
                want[at + offset ..][0..128],
                mag[at + offset ..][0..128],
            );
        }
    }

    pub fn simulate(self: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        var terms: [128]f64 = undefined;
        var row: [128]f64 = undefined;
        for (&row, 0..) |*v, i| {
            for (&terms, 0..) |*term, j| {
                const x: f64 = if (j == 0) 1025 else 1024;
                term.* = if (@popCount(i & j) % 2 == 0) x else -x;
            }
            v.* = oracle.half(core.contract.rows.sumIn(acc, order, &terms) * oracle.scale);
        }
        for (out, 0..) |*v, i| v.* = row[i % 128];
        _ = self;
    }
};

pub const Images = struct { served: []const u8, wide: []const u8 };
pub const Set = struct {
    gpu: check.Gpu,
    module: cuda.Module,
    legacy: cuda.Module,
    cases: [6]Case,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !?*Set {
        if (!cuda.kernels.available) return null;
        return initWithImages(gpu, s, a, list, .{ .served = cuda.kernels.exl3_routed_half, .wide = cuda.kernels.exl3_experts });
    }
    pub fn initWithImages(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel), images: Images) !*Set {
        const set = try a.create(Set);
        errdefer a.destroy(set);
        var module = try cuda.Module.load(gpu.d, images.served);
        errdefer module.unload();
        var legacy = try cuda.Module.load(gpu.d, images.wide);
        errdefer legacy.unload();
        set.* = .{ .gpu = gpu, .module = module, .legacy = legacy, .cases = undefined };
        const f = try module.function("tf_exl3_routed_half_rotate");
        const w = try legacy.function("tf_exl3_experts_rot_in");
        for (&set.cases, 0..) |*item, i| item.* = .{
            .rig = .{ .gpu = gpu, .s = s },
            .shared = i % 2 == 0,
            .experts = if (i == 2 or i == 3) 384 else 192,
            .control = i >= 4,
            .function = f,
            .wide = w,
        };
        errdefer for (&set.cases) |*item| item.deinit();
        for (&set.cases) |*item| try list.append(a, item.kernel());
        return set;
    }
    pub fn deinit(set: *Set) void {
        for (&set.cases) |*item| item.deinit();
        set.legacy.unload();
        set.module.unload();
        set.gpu.gpa.destroy(set);
    }
};
