//! Grouped NVFP4 decode against separate launches: unequal parts, scales, row buckets, K slices and controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const Rig = @import("contract_rig.zig").Rig;
const q = cuda.qmmf;
const Many = q.Many;
const c = core.contract.spec;
const g = core.contract.gates;
const Dense = core.contract.dense.Dense;
const served = [_]c.Shape{
    .{ .rows = 1, .n = 129, .k = 4096 },  .{ .rows = 2, .n = 129, .k = 4096 },
    .{ .rows = 8, .n = 129, .k = 4096 },  .{ .rows = 16, .n = 129, .k = 4096 },
    .{ .rows = 64, .n = 129, .k = 4096 }, .{ .rows = 128, .n = 129, .k = 4096 },
};
fn depth(shape: c.Shape) usize {
    return shape.k / 16 + 24;
}

const Product = struct {
    rig: Rig,
    lane: *const q.Lane,
    sk: u32,
    members: usize,
    control: bool,
    name: []const u8,
    d: [4]?Dense = @splat(null),
    parts: [4]Many = undefined,
    bases: [4]u64 = undefined,
    x: u64 = 0,
    baseline: std.ArrayList(u8) = .empty,

    fn drop(p: *Product) void {
        for (&p.d) |*d| {
            if (d.*) |data| data.deinit();
            d.* = null;
        }
        p.rig.reset();
    }
    pub fn deinit(p: *Product) void {
        p.drop();
        p.rig.deinit();
        p.baseline.deinit(p.rig.gpu.gpa);
    }
    const vt = g.hooks(Product);
    pub fn kernel(p: *Product) g.Kernel {
        return .{ .ctx = p, .vt = &vt, .control = if (p.control) .shapes else null, .spec = .{
            .name = p.name,
            .accum = .f32,
            .depth = depth,
            .ops = 1,
            .out = .bf16,
            .replaces = "each member's separate NVFP4 matmulOut launch",
            .tiles = .{ .{ .size = 64 }, .{ .size = 64 }, .{ .size = 512, .step = 512, .min = 512 } },
            .served = &served,
        } };
    }
    pub fn prepare(p: *Product, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.drop();
        if (shape.rows == 0 or shape.rows > 128 or shape.n == 0 or shape.k % (64 * p.sk) != 0 or shape.k / (64 * p.sk) < 8) return false;
        const a = p.rig.gpu.gpa;
        for (0..p.members) |member| {
            const n = shape.n + member * 65;
            const data = try Dense.init(a, .{ .rows = shape.rows, .n = n, .k = shape.k }, 1);
            p.d[member] = data;
            data.fillX(fill, seed, 2);
            if (member == 0) p.x = try p.rig.input(std.mem.sliceAsBytes(data.x));
            const codes = try a.alloc(u8, n * shape.k / 2);
            defer a.free(codes);
            const scales = try a.alloc(u8, n * (shape.k / 16));
            defer a.free(scales);
            for (scales, 0..) |*v, i| v.* = if (fill == .stress or (i + member) % 2 == 0) 0x38 else 0x30;
            @memset(codes, 0);
            const scale: f32 = if (fill == .stress) 1 else 1 + @as(f32, @floatFromInt(member)) * 0.25;
            for (0..n) |j| for (0..shape.k) |k| {
                const first = if (fill == .stress) k % (shape.k / 2) else k;
                const mag: u8 = @intCast(1 + (first * 7 + j * 3 + member) % 7);
                const nib = mag | @as(u8, if (fill == .stress and k >= shape.k / 2) 8 else 0);
                codes[j * (shape.k / 2) + k / 2] |= nib << @as(u3, @intCast((k % 2) * 4));
                const value = ([_]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 })[mag];
                const group_scale: f64 = if (scales[j * (shape.k / 16) + k / 16] == 0x38) 1 else 0.5;
                data.w[j * shape.k + k] = (if (nib & 8 != 0) -value else value) * group_scale * scale;
            };
            const npad = cuda.nvfp4.padded(n);
            const words = try a.alloc(u32, npad / 64 * (shape.k / 64) * 512);
            defer a.free(words);
            const packed_scales = try a.alloc(u8, npad * (shape.k / 16));
            defer a.free(packed_scales);
            cuda.nvfp4.packWords(words, codes, n, shape.k);
            cuda.nvfp4.packScales(packed_scales, scales, n, shape.k);
            const weight = try q.pinned(cuda.nvfp4.weight(try p.rig.input(std.mem.sliceAsBytes(words)), try p.rig.input(packed_scales), scale, n, shape.k), p.sk);
            p.parts[member] = .{ .w = weight, .out = try p.rig.output(data.outputs() * 2) };
            p.bases[member] = try p.rig.scratch(data.outputs() * 2);
        }
        return true;
    }
    pub fn run(p: *Product) !void {
        var parts = p.parts;
        if (p.control) parts[0].w.scale *= 1.5;
        const shape = p.d[0].?.shape;
        // a fallback to separate launches would compare them with themselves
        try check.expect(q.manyEligible(p.lane.*, shape.rows, parts[0..p.members]), "{s}: the grouped launch refused its parts", .{p.name});
        try p.lane.matmulMany(p.rig.s, p.x, shape.k, shape.rows, parts[0..p.members], 0);
    }
    pub fn replaced(p: *Product) ![]const u8 {
        p.baseline.clearRetainingCapacity();
        const shape = p.d[0].?.shape;
        for (p.parts[0..p.members], 0..) |part, i| {
            try p.lane.matmulOut(p.rig.s, p.x, shape.k, shape.rows, part.w, p.bases[i], .bf16, 0);
            try p.baseline.appendSlice(p.rig.gpu.gpa, try p.rig.bytes(p.bases[i], p.d[i].?.outputs() * 2));
        }
        return p.baseline.items;
    }
    pub fn count(p: *Product) usize {
        var n: usize = 0;
        for (p.d[0..p.members]) |data| n += data.?.outputs();
        return n;
    }
    pub fn reference(p: *Product, want: []f64, mag: []f64) !void {
        var at: usize = 0;
        for (p.d[0..p.members]) |data| {
            const n = data.?.outputs();
            try data.?.reference(want[at..][0..n], mag[at..][0..n]);
            at += n;
        }
    }
    pub fn simulate(p: *Product, acc: c.Accum, order: g.Order, out: []f64) !void {
        var at: usize = 0;
        for (p.d[0..p.members]) |data| {
            const n = data.?.outputs();
            try data.?.simulate(acc, order, out[at..][0..n]);
            at += n;
        }
    }
};

pub const Registrations = struct {
    gpu: check.Gpu,
    lane: *const q.Lane,
    stream: cuda.Stream,
    products: std.ArrayList(*Product) = .empty,
    pub fn append(p: *Registrations, kernels: *std.ArrayList(g.Kernel)) !void {
        for ([_]u32{ 1, 2, 4 }) |sk| {
            if (sk > 1 and !p.lane.clusters) continue;
            for ([_]usize{ 2, 4 }) |members| for ([_]bool{ false, true }) |control| {
                if (control and members != 4) continue;
                const product = blk: {
                    const created = try p.gpu.gpa.create(Product);
                    errdefer p.gpu.gpa.destroy(created);
                    const name = try std.fmt.allocPrint(p.gpu.gpa, "{s}qmmf NVFP4 grouped {d} parts sk{d}", .{ if (control) "control: " else "cuda ", members, sk });
                    errdefer p.gpu.gpa.free(name);
                    created.* = .{ .rig = .{ .gpu = p.gpu, .s = p.stream }, .lane = p.lane, .sk = sk, .members = members, .control = control, .name = name };
                    try p.products.append(p.gpu.gpa, created);
                    break :blk created;
                };
                try kernels.append(p.gpu.gpa, product.kernel());
            };
        }
    }
    pub fn deinit(p: *Registrations) void {
        for (p.products.items) |product| {
            product.deinit();
            p.gpu.gpa.free(product.name);
            p.gpu.gpa.destroy(product);
        }
        p.products.deinit(p.gpu.gpa);
    }
};

pub fn run(gpu: check.Gpu) !void {
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var lane = try q.Lane.load(gpu.d, (try gpu.ctx.capability()) / 10);
    defer lane.unload();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(gpu.gpa);
    var registrations: Registrations = .{ .gpu = gpu, .lane = &lane, .stream = stream };
    defer registrations.deinit();
    try registrations.append(&kernels);
    var tally: g.Tally = .{};
    try g.runAll(gpu.gpa, kernels.items, 0x713, &tally);
    std.debug.print("SUMMARY NVFP4 grouped gates 1-4: {d} pass, {d} fail; {d} controls red, {d} missed\n", .{ tally.passed, tally.failed, tally.caught, tally.missed });
    try check.expect(tally.problems() == 0, "grouped NVFP4 contract failed", .{});
}
