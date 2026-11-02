//! Expert-first Qwen router against the frozen row-first kernel and independent FP64 dots.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const c = core.contract.spec;
const g = core.contract.gates;
const Dense = core.contract.dense.Dense;
fn depth(s: c.Shape) usize {
    return s.k;
}
pub const served = [_]c.Shape{
    .{ .rows = 17, .n = 257, .k = 2048 },  .{ .rows = 33, .n = 257, .k = 4096 },
    .{ .rows = 128, .n = 257, .k = 2055 }, .{ .rows = 2048, .n = 257, .k = 2048 },
};

pub const Inputs = struct {
    d: Dense,
    weights: []u16,
    x: []u16,
    stride: usize,
    pub fn init(a: std.mem.Allocator, s: c.Shape, pad: usize, fill: g.Fill, seed: u64) !Inputs {
        const d = try Dense.init(a, s, 1);
        errdefer d.deinit();
        d.fillX(fill, seed, 4);
        const w = try a.alloc(u16, d.w.len);
        errdefer a.free(w);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x77);
        const r = prng.random();
        for (w, 0..) |*v, i| {
            const k = i % s.k;
            const z: f32 = if (fill == .stress) @floatFromInt(1 + ((k % (s.k / 2)) * 7 + i / s.k) % 7) else r.float(f32) * 2 - 1;
            v.* = c.toBf16(if (fill == .stress and k >= s.k / 2) -z else z);
        }
        for (d.w, w) |*v, b| v.* = c.fromBf16(b);
        const stride = s.k + pad;
        const x = try a.alloc(u16, s.rows * stride);
        @memset(x, 0x7fc1);
        for (0..s.rows) |row| @memcpy(x[row * stride ..][0..s.k], d.x[row * s.k ..][0..s.k]);
        return .{ .d = d, .weights = w, .x = x, .stride = stride };
    }
    pub fn deinit(t: Inputs) void {
        t.d.a.free(t.x);
        t.d.a.free(t.weights);
        t.d.deinit();
    }
};

pub const Case = struct {
    rig: @import("contract_rig.zig").Rig,
    f: cuda.moe.Fns,
    replacement: bool = true,
    live: bool = false,
    pad: usize = 0,
    control: bool = false,
    data: ?Inputs = null,
    p: [5]u64 = @splat(0),
    seen: std.ArrayList(u8) = .empty,
    pub fn deinit(t: *Case) void {
        if (t.data) |d| d.deinit();
        t.seen.deinit(t.rig.gpu.gpa);
        t.rig.deinit();
    }
    pub fn kernel(t: *Case) g.Kernel {
        return .{ .ctx = t, .vt = if (t.live) &raw else &numeric, .control = if (t.control) .shapes else null, .spec = .{
            .name = if (t.control) "control: MoE expert-first router reads row zero" else if (t.live) "cuda MoE expert-first live rows replacement" else if (!t.replacement) "cuda MoE expert-first router FP64" else if (t.pad != 0) "cuda MoE expert-first padded input replacement" else "cuda MoE expert-first router replacement",
            .out = if (t.live) .raw else .f32,
            .accum = if (t.live) .none else .f32,
            .depth = depth,
            .nan = if (t.live) .{ .stated = "all bytes equal the old router, including unwritten bucket padding" } else .propagate,
            .replaces = if (t.replacement) "tf_moe_router" else null,
            .tiles = .{ .{ .size = 16 }, .{ .size = 16 }, .{ .size = 256 } },
            .served = &served,
        } };
    }
    const raw = g.hooks(Case);
    const numeric = blk: {
        var vt = raw;
        vt.check = null;
        break :blk vt;
    };
    pub fn prepare(t: *Case, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.n == 0 or s.k == 0) return false;
        if (t.data) |d| d.deinit();
        t.data = null;
        t.rig.reset();
        t.data = try Inputs.init(t.rig.gpu.gpa, s, t.pad, fill, seed);
        const d = &t.data.?;
        t.p[0] = try t.rig.input(std.mem.sliceAsBytes(d.x));
        t.p[1] = try t.rig.input(std.mem.sliceAsBytes(d.weights));
        t.p[2] = try t.rig.output(d.d.outputs() * 4);
        t.p[3] = try t.rig.scratch(d.d.outputs() * 4);
        const params = [3]i32{ 0, 0, @intCast(s.rows - @as(usize, if (t.live) 1 else 0)) };
        t.p[4] = if (t.live) try t.rig.input(std.mem.sliceAsBytes(&params)) else 0;
        return true;
    }
    fn launch(t: *Case, old: bool, output: u64) !void {
        const d = t.data.?;
        const s = d.d.shape;
        var f = t.f;
        if (old) f.router_expert = null;
        try f.route(t.rig.s, t.p[0], d.stride, s.rows, t.p[1], output, s.k, s.n, t.p[4]);
        try t.rig.s.synchronize();
    }
    pub fn run(t: *Case) !void {
        return t.launch(false, t.p[2]);
    }
    pub fn replaced(t: *Case) ![]const u8 {
        const len = t.data.?.d.outputs() * 4;
        const b: cuda.DeviceBuffer = .{ .d = t.rig.gpu.d, .ptr = t.p[3], .len = len };
        try b.fill8(0xff, t.rig.s.handle);
        try t.launch(true, t.p[3]);
        return t.rig.bytes(t.p[3], len);
    }
    pub fn check(t: *Case, _: g.Gate) !?[]const u8 {
        if (!t.live) return t.classes();
        t.seen.clearRetainingCapacity();
        try t.seen.appendSlice(t.rig.gpu.gpa, try t.rig.outputBytes());
        return if (std.mem.eql(u8, t.seen.items, try t.replaced())) null else "router live/padding bytes differ from the old path";
    }
    fn classes(t: *Case) !?[]const u8 {
        const a = t.rig.gpu.gpa;
        const n = t.data.?.d.outputs();
        const values = try a.alloc(f64, n * 2);
        defer a.free(values);
        try t.reference(values[0..n], values[n..]);
        const out = try t.rig.outputBytes();
        for (values[0..n], 0..) |v, i| if (c.classOf(c.read(.f32, out, i)) != c.classOf(v)) return "router classes differ from FP64";
        return null;
    }
    pub fn count(t: *Case) usize {
        return t.data.?.d.outputs() * @as(usize, if (t.live) 4 else 1);
    }
    pub fn reference(t: *Case, want: []f64, mag: []f64) !void {
        return t.data.?.d.reference(want, mag);
    }
    pub fn simulate(t: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        return t.data.?.d.simulate(acc, order, out);
    }
};

pub const Suite = struct {
    mod: cuda.Module,
    cases: [5]Case,
    pub fn init(gpu: @import("check.zig").Gpu, s: cuda.Stream, controls: cuda.Module) !Suite {
        var mod = try cuda.Module.load(gpu.d, cuda.kernels.moe);
        errdefer mod.unload();
        const f = try cuda.moe.Fns.resolveMode(mod, true);
        var cases: [5]Case = undefined;
        for (&cases, 0..) |*case, i| {
            var fns = f;
            if (i == 4) fns.router_expert = try controls.function("tf_moe_router_expert_bad_rows");
            case.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = fns, .replacement = i != 1, .pad = if (i == 2 or i == 3) 16 else 0, .live = i == 3, .control = i == 4 };
        }
        return .{ .mod = mod, .cases = cases };
    }
    pub fn append(t: *Suite, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        for (&t.cases) |*case| try list.append(a, case.kernel());
    }
    pub fn deinit(t: *Suite) void {
        for (&t.cases) |*case| case.deinit();
        t.mod.unload();
    }
};
