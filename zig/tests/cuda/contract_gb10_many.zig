//! GB10 grouping is a byte replacement of each member's own decode/verify launch.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const many = @import("contract_qmm_prefill_many.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const A = cuda.qlinear.Affine4Kernels;
const family = @import("gb10_family_shapes.zig");
const rule = "all member bytes equal separate affine-4 decode/verify launches";
const shapes = [_]c.Shape{
    .{ .rows = 1, .n = 10240, .k = 5120 }, .{ .rows = 16, .n = 12288, .k = 5120 }, .{ .rows = 17, .n = 17408, .k = 5120 },
    .{ .rows = 33, .n = 8192, .k = 2048 }, .{ .rows = 65, .n = 65, .k = 512 },     .{ .rows = 128, .n = 12288, .k = 5120 },
};
fn depth(_: c.Shape) usize {
    return 0;
}
const Mode = enum { good, swapped, short };
pub const Registrations = struct {
    products: [3 + family.cases.len]many.Product,
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: *A) Registrations {
        return .{ .products = @splat(.{ .rig = .{ .gpu = gpu, .s = s }, .affine = a }) };
    }
    pub fn deinit(self: *Registrations) void {
        for (&self.products) |*p| p.deinit();
    }
    pub fn append(self: *Registrations, cap: u32, a: std.mem.Allocator, out: *std.ArrayList(g.Kernel)) !void {
        if (cap != 121) return;
        for (std.enums.values(Mode), 0..) |mode, j| try out.append(a, kernel(&self.products[j], mode, &shapes));
        for (family.cases, 0..) |v, i| {
            const p = &self.products[3 + i];
            p.member_widths = v.widths;
            p.padded_n = v.padded_n;
            try out.append(a, kernel(p, .good, family.shapes[i .. i + 1]));
        }
    }
};
fn kernel(p: *many.Product, mode: Mode, served: []const c.Shape) g.Kernel {
    return .{ .spec = .{ .name = switch (mode) {
        .good => "cuda GB10 grouped affine-4 decode/verify",
        .swapped => "control: GB10 grouping swaps equal-width outputs",
        .short => "control: GB10 grouping omits the last row",
    }, .out = .raw, .accum = .none, .depth = depth, .replaces = if (mode == .good) "each member's separate decode/verify" else null, .nan = .{ .stated = rule }, .served = served }, .ctx = p, .vt = switch (mode) {
        .good => &good_vt,
        .swapped => &swapped_vt,
        .short => &short_vt,
    }, .control = if (mode == .good) null else .shapes };
}
fn of(ctx: *anyopaque) *many.Product {
    return @ptrCast(@alignCast(ctx));
}
fn launch(p: *many.Product, mode: Mode) !void {
    var outs: [4]A.Out = undefined;
    for (outs[0..p.parts], 0..) |*o, i| o.* = .{ .q = p.qs[i], .out = p.outs[i] };
    if (mode == .swapped) find: for (0..p.parts) |i| for (i + 1..p.parts) |j| if (p.qs[i].n == p.qs[j].n) {
        std.mem.swap(u64, &outs[i].out, &outs[j].out);
        break :find;
    };
    const rows = if (mode == .short) p.rows - @intFromBool(p.rows > 1) else p.rows;
    try p.affine.verifyMany(p.rig.s, p.x, p.xs, outs[0..p.parts], rows);
    try p.rig.s.synchronize();
}
fn run(ctx: *anyopaque) anyerror!void {
    return launch(of(ctx), .good);
}
fn swap(ctx: *anyopaque) anyerror!void {
    return launch(of(ctx), .swapped);
}
fn short(ctx: *anyopaque) anyerror!void {
    return launch(of(ctx), .short);
}
fn output(ctx: *anyopaque) anyerror![]const u8 {
    const p = of(ctx);
    p.rig.outs.clearRetainingCapacity();
    for (p.outs[0..p.parts], p.qs[0..p.parts]) |out, q| try p.rig.outs.appendSlice(p.rig.gpu.gpa, try p.rig.bytes(out, p.rows * q.n * 2));
    return p.rig.outs.items;
}
fn replaced(ctx: *anyopaque) anyerror![]const u8 {
    const p = of(ctx);
    var at: usize = 0;
    for (p.qs[0..p.parts]) |q| {
        try p.affine.verify(p.rig.s, p.x, p.xs, q, p.base + at, p.rows);
        at += p.rows * q.n * 2;
    }
    return p.rig.bytes(p.base, at);
}
fn checkBytes(ctx: *anyopaque, _: c.Gate) anyerror!?[]const u8 {
    const p = of(ctx);
    const got = try p.rig.gpu.gpa.dupe(u8, try output(ctx));
    defer p.rig.gpu.gpa.free(got);
    return if (std.mem.eql(u8, got, try replaced(ctx))) null else rule;
}
const good_vt: g.VTable = blk: {
    var v = g.hooks(many.Product);
    v.run = run;
    v.output = output;
    v.replaced = replaced;
    v.check = checkBytes;
    v.prepare = prepare;
    break :blk v;
};
const swapped_vt: g.VTable = blk: {
    var v = good_vt;
    v.run = swap;
    break :blk v;
};
const short_vt: g.VTable = blk: {
    var v = good_vt;
    v.run = short;
    break :blk v;
};
fn prepare(ctx: *anyopaque, s: c.Shape, f: g.Fill, seed: u64) anyerror!bool {
    const p = of(ctx);
    if (!try p.prepare(s, f, seed)) return false;
    const xs = try p.rig.gpu.gpa.alloc(f32, s.rows * (s.k / 64));
    defer p.rig.gpu.gpa.free(xs);
    for (xs, 0..) |*x, i| {
        var v: f32 = 0;
        for (p.fixtures[0].?.x[i * 64 ..][0..64]) |h| v += c.fromBf16(h);
        x.* = v;
    }
    p.xs = try p.rig.input(std.mem.sliceAsBytes(xs));
    return true;
}
pub fn runAll(gpu: check.Gpu) !void {
    try check.expect(try gpu.ctx.capability() == 121, "GB10 grouping contract needs sm_121", .{});
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    const kk = cuda.kernels;
    var mods: [4]cuda.Module = undefined;
    var loaded: usize = 0;
    defer for (mods[0..loaded]) |*m| m.unload();
    for ([_][]const u8{ kk.qmm_group, kk.qmm_prefill, kk.lane_gemv, kk.affine4_pack }, 0..) |img, i| {
        mods[i] = try cuda.Module.load(gpu.d, img);
        loaded += 1;
    }
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var a = try A.resolve(gpu.d, mods[0], mods[1], mods[2], mods[3], sms, true);
    defer a.deinit();
    var regs = Registrations.init(gpu, s, &a);
    defer regs.deinit();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(gpu.gpa);
    try regs.append(121, gpu.gpa, &kernels);
    var tally: g.Tally = .{};
    try g.runAll(gpu.gpa, kernels.items, 0x110, &tally);
    try check.expect(tally.problems() == 0, "GB10 grouped gate failed", .{});
    for ([_]c.Shape{ .{ .rows = 128, .n = 10240, .k = 5120 }, .{ .rows = 128, .n = 12288, .k = 5120 }, .{ .rows = 128, .n = 17408, .k = 5120 }, .{ .rows = 128, .n = 8192, .k = 2048 }, .{ .rows = 128, .n = 65, .k = 512 } }) |shape| try sweep(&regs.products[0], shape);
    for (family.shapes, 0..) |shape, i| try sweep(&regs.products[3 + i], shape);
    std.debug.print("SUMMARY gb10_grouped: {d} pass, {d} fail; {d} controls red, {d} missed; all member sets x all 128 row counts x five poisoned runs byte-equal\n", .{ tally.passed, tally.failed, tally.caught, tally.missed });
}

fn sweep(p: *many.Product, shape: c.Shape) !void {
    _ = try prepare(p, shape, .random, 0x1101234);
    const inputs = try p.rig.gpu.gpa.dupe(u8, try p.rig.inputBytes());
    defer p.rig.gpu.gpa.free(inputs);
    for (1..129) |rows| {
        p.rows = rows;
        p.bytes = 0;
        for (p.qs[0..p.parts]) |q| p.bytes += rows * q.n * 2;
        const want = try p.rig.gpu.gpa.dupe(u8, try replaced(p));
        defer p.rig.gpu.gpa.free(want);
        for (0..5) |_| {
            for (p.outs[0..p.parts], p.qs[0..p.parts]) |o, q| try (cuda.DeviceBuffer{ .d = p.rig.gpu.d, .ptr = o, .len = rows * q.n * 2 }).fill8(0xff, p.rig.s.handle);
            try run(p);
            try check.expect(std.mem.eql(u8, want, try output(p)), "GB10 grouping differs n={d} k={d} rows={d}", .{ shape.n, shape.k, rows });
        }
    }
    try check.expect(try p.rig.intact(), "GB10 grouping damaged guards", .{});
    try check.expect(std.mem.eql(u8, inputs, try p.rig.inputBytes()), "GB10 grouping changed inputs", .{});
}
