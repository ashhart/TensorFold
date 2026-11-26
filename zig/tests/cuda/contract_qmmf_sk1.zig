//! Physical SK1 replaces main's SK2 launch byte for byte, with its logical slices and prompt seam unchanged.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const Base = @import("contract_projection_f32.zig").Product;
const Rig = @import("contract_rig.zig").Rig;
const q = cuda.qmmf;

const verify_shapes = blk: {
    var shapes: [40]c.Shape = undefined;
    for (0..32) |i| shapes[i] = .{ .rows = i + 1, .n = 6144, .k = 5120 };
    for ([_]usize{ 33, 48, 64, 65, 127, 128, 255, 256 }, 32..) |rows, i| shapes[i] = .{ .rows = rows, .n = 6144, .k = 5120 };
    break :blk shapes;
};
const prompt_shapes = blk: {
    var shapes: [11]c.Shape = undefined;
    for ([_]usize{ 1, 16, 17, 32, 64, 127, 128, 129, 512, 1536, 2048 }, 0..) |rows, i| shapes[i] = .{ .rows = rows, .n = 6144, .k = 5120 };
    break :blk shapes;
};
pub const Fault = enum { none, unsplit, scale };

pub const Product = struct {
    inner: Base,
    rig: *Rig = undefined,
    lane: q.Lane,
    set: q.sk1.Set,
    prompt: *const cuda.nvfp4.Prompt,
    prefill: bool,
    fault: Fault = .none,
    base: u64 = 0,

    pub fn deinit(p: *Product) void {
        p.inner.deinit();
    }
    fn depth(shape: c.Shape) usize {
        return shape.k / 16 + 24;
    }
    pub fn kernel(p: *Product) g.Kernel {
        return .{ .spec = .{
            .name = if (p.fault == .unsplit) "control: Qwen NVFP4 physical SK1 loses logical split" else if (p.fault == .scale) "control: Qwen NVFP4 physical SK1 global scale" else if (p.prefill) (if (p.inner.out_kind == .bf16) "cuda Qwen NVFP4 physical SK1 prompt seam bf16" else "cuda Qwen NVFP4 physical SK1 prompt seam fp32") else if (p.inner.out_kind == .bf16) "cuda Qwen NVFP4 physical SK1 verify bf16" else "cuda Qwen NVFP4 physical SK1 verify fp32",
            .accum = .f32,
            .depth = depth,
            .ops = 2,
            .out = if (p.inner.out_kind == .bf16) .bf16 else .f32,
            .replaces = "main Qwen NVFP4 SK2 on identical packed weights, rows and prompt selection",
            .served = if (p.prefill) &prompt_shapes else &verify_shapes,
            .tiles = .{ .{ .size = 16 }, .{ .size = 64, .min = 6144 }, .{ .size = 64, .step = 64, .min = 5120 } },
        }, .ctx = p, .vt = &vt, .control = if (p.fault == .none) null else .shapes };
    }
    const vt = g.hooks(Product);
    fn width(p: *const Product) usize {
        return if (p.inner.out_kind == .bf16) 2 else 4;
    }
    pub fn prepare(p: *Product, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 2048 or shape.n != 6144 or shape.k != 5120) return false;
        p.rig = &p.inner.rig;
        p.inner.lane = &p.lane;
        if (!try p.inner.prepare(shape, fill, seed)) return false;
        p.base = try p.rig.scratch(p.count() * p.width());
        return true;
    }
    fn dispatch(p: *Product, output: u64, candidate: bool) !void {
        const shape = p.inner.d.?.shape;
        var w = p.inner.weight;
        if (candidate and p.fault == .unsplit) w = try q.pinned(w, 1);
        if (candidate and p.fault == .scale) w.scale *= 1.5;
        if (p.prefill and shape.rows > 16) {
            try p.prompt.matmulOut(p.rig.s, p.inner.x, shape.k, shape.rows, w, output, p.inner.out_kind);
        } else if (!candidate or !try q.sk1.matmul(p.set, p.lane, p.rig.s, p.inner.x, shape.k, shape.rows, w, output, p.inner.out_kind)) {
            try p.lane.matmulOut(p.rig.s, p.inner.x, shape.k, shape.rows, w, output, p.inner.out_kind, p.inner.part);
        }
        try p.rig.s.synchronize();
    }
    pub fn run(p: *Product) !void {
        try p.dispatch(p.inner.out, true);
    }
    pub fn replaced(p: *Product) ![]const u8 {
        try p.dispatch(p.base, false);
        return p.rig.bytes(p.base, p.count() * p.width());
    }
    pub fn count(p: *Product) usize {
        return p.inner.count();
    }
    pub fn reference(p: *Product, want: []f64, mag: []f64) !void {
        try p.inner.reference(want, mag);
    }
    pub fn simulate(p: *Product, acc: c.Accum, order: g.Order, out: []f64) !void {
        try p.inner.simulate(acc, order, out);
    }
    pub fn check(p: *Product, gate: g.Gate) !?[]const u8 {
        if (gate != .nonfinite) return null;
        const bytes = try p.rig.gpu.gpa.dupe(u8, try p.rig.outputBytes());
        defer p.rig.gpu.gpa.free(bytes);
        if (!std.mem.eql(u8, bytes, try p.replaced())) return "nonfinite bytes differ from main SK2";
        const want = try p.rig.gpu.gpa.alloc(f64, p.count());
        defer p.rig.gpu.gpa.free(want);
        const mag = try p.rig.gpu.gpa.alloc(f64, p.count());
        defer p.rig.gpu.gpa.free(mag);
        try p.reference(want, mag);
        for (want, 0..) |v, i| if (c.classOf(v) != c.classOf(c.read(p.kernel().spec.out, bytes, i))) return "nonfinite class differs from independent FP64";
        return null;
    }
};

pub const Registrations = struct {
    products: [8]Product = undefined,
    made: usize = 0,
    pub fn init(gpu: check.Gpu, s: cuda.Stream, lane: q.Lane, prompt: *const cuda.nvfp4.Prompt) !Registrations {
        var r: Registrations = .{};
        errdefer r.deinit();
        const cap = try gpu.ctx.capability();
        if (cap != 120 and cap != 121) return r;
        const set = try q.sk1.load(&lane);
        for ([_]q.Out{ .bf16, .f32 }) |out| for ([_]bool{ false, true }) |prefill| {
            r.products[r.made] = .{ .inner = .{ .rig = .{ .gpu = gpu, .s = s }, .lane = undefined, .mode = .fp4, .clusters = true, .out_kind = out }, .lane = lane, .set = set, .prompt = prompt, .prefill = prefill };
            r.made += 1;
        };
        for ([_]Fault{ .unsplit, .scale }) |fault| for ([_]q.Out{ .bf16, .f32 }) |out| {
            r.products[r.made] = .{ .inner = .{ .rig = .{ .gpu = gpu, .s = s }, .lane = undefined, .mode = .fp4, .clusters = true, .out_kind = out }, .lane = lane, .set = set, .prompt = prompt, .prefill = false, .fault = fault };
            r.made += 1;
        };
        return r;
    }
    pub fn deinit(r: *Registrations) void {
        for (r.products[0..r.made]) |*p| p.deinit();
    }
    pub fn append(r: *Registrations, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        for (r.products[0..r.made]) |*p| try list.append(a, p.kernel());
    }
};

pub fn run(gpu: check.Gpu) !void {
    const cap = try gpu.ctx.capability();
    try check.expect(cap == 120 or cap == 121, "Qwen NVFP4 physical SK1 requires sm120/sm121", .{});
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var lane = try q.Lane.load(gpu.d, cap / 10);
    defer lane.unload();
    lane.half_n = null;
    lane.half_below = 0;
    var prompt = try cuda.nvfp4.Prompt.load(gpu.d);
    defer prompt.unload();
    var regs = try Registrations.init(gpu, s, lane, &prompt);
    defer regs.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(gpu.gpa);
    try regs.append(gpu.gpa, &list);
    var tally: g.Tally = .{};
    try g.runAll(gpu.gpa, list.items, 0x514d4d46, &tally);
    std.debug.print("SUMMARY Qwen NVFP4 physical SK1 gates1-4: {d} pass, {d} fail; {d} controls red, {d} missed\n", .{ tally.passed, tally.failed, tally.caught, tally.missed });
    try check.expect(tally.problems() == 0, "Qwen NVFP4 physical SK1 contract failed", .{});
}
