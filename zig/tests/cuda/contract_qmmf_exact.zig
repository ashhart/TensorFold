//! The wide FP4 tile against the original lane's bytes, with a real reversed-K16 control.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const Rig = @import("contract_rig.zig").Rig;
const c = core.contract.spec;
const g = core.contract.gates;
const Dense = core.contract.dense.Dense;
const q = cuda.qmmf;
const wide = q.exact_tile;
pub const panels = @import("qmmf_exact_panels.zig");
const production = panels.production();

fn shapes() [39]c.Shape {
    var out: [39]c.Shape = undefined;
    for (0..16) |i| {
        out[i] = .{ .rows = i + 1, .n = 17408, .k = 5120 };
        out[16 + i] = .{ .rows = i + 1, .n = 5120, .k = 17408 };
    }
    for ([_]usize{ 1, 63, 65, 127, 129, 257 }, 0..) |n, i|
        out[32 + i] = .{ .rows = 15, .n = n, .k = @as(usize, 512) << @intCast(i % 4) };
    out[38] = .{ .rows = 15, .n = 129, .k = 4096 };
    return out;
}
const served = shapes();

fn depth(s: c.Shape) usize {
    return s.k / 16 + 24;
}

pub fn spec(out: q.Out, clusters: bool, control: bool) c.Spec {
    return .{
        .name = if (control) "control: qmmf exact tile reversed K16" else switch (out) {
            .bf16 => if (clusters) "cuda qmmf exact tile NVFP4 16x128 bf16 cluster" else "cuda qmmf exact tile NVFP4 16x128 bf16 reduce",
            .f32 => if (clusters) "cuda qmmf exact tile NVFP4 16x128 fp32 cluster" else "cuda qmmf exact tile NVFP4 16x128 fp32 reduce",
        },
        .accum = .f32,
        .depth = depth,
        .ops = 1,
        .out = if (out == .f32) .f32 else .bf16,
        .replaces = "main qmmf FP4 16x64, identical K slices and ordered merge",
        .tiles = .{ .{ .size = 16 }, .{ .size = 128, .min = 1 }, .{ .size = 512, .step = 64, .min = 512 } },
        .served = if (control) &.{.{ .rows = 1, .n = 129, .k = 512 }} else &served,
    };
}

pub const Product = struct {
    rig: Rig,
    lane: q.Lane,
    format: q.Out,
    control: ?cuda.Function = null,
    d: ?Dense = null,
    w: q.Weight = undefined,
    x: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,
    part: u64 = 0,
    base_part: u64 = 0,
    ldx: usize = 0,
    split: u32 = 0,
    directed: panels.Case = .natural,

    pub fn deinit(self: *Product) void {
        if (self.d) |d| d.deinit();
        self.rig.deinit();
    }
    const vt = g.hooks(Product);
    pub fn kernel(self: *Product) g.Kernel {
        var s = spec(self.format, self.lane.clusters, self.control != null);
        if (self.split != 0) {
            s.name = panels.pinName(self.format, self.lane.clusters, self.split);
            s.served = &production;
            s.tiles = .{ .{}, .{}, .{} };
        } else if (self.directed != .natural) {
            s.name = panels.directName(self.format, self.lane.clusters, self.directed);
            s.served = &panels.direct_shapes;
        }
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = if (self.control != null) .shapes else null };
    }
    fn width(self: *const Product) usize {
        return if (self.format == .f32) 4 else 2;
    }
    pub fn panelFill(self: *const Product, s: c.Shape, fill: g.Fill, x: []u16, codes: []u8, scales: []u8) f32 {
        const directed = if (fill == .random) self.directed else .natural;
        panels.fill(directed, x, codes, scales, s.rows, s.n, s.k);
        return if (directed != .natural or self.format == .f32 or fill == .stress or self.control != null) 1 else 0.0123;
    }
    pub fn prepare(self: *Product, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (self.d) |d| d.deinit();
        self.d = null;
        self.part = 0;
        self.base_part = 0;
        self.rig.reset();
        if (s.rows == 0 or s.rows > 16 or s.n == 0 or s.k < 512 or s.k % 64 != 0) return false;
        const a = self.rig.gpu.gpa;
        const d = try Dense.init(a, s, 1);
        self.d = d;
        d.fillX(fill, seed, 16);
        const codes = try a.alloc(u8, s.n * s.k / 2);
        defer a.free(codes);
        const scales = try a.alloc(u8, s.n * (s.k / 16));
        defer a.free(scales);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x128f4);
        const rng = prng.random();
        for (codes) |*v| v.* = rng.int(u8);
        if (fill == .stress) for (codes) |*v| {
            v.* &= 0x77;
        };
        for (scales) |*v| v.* = if (fill == .stress) 0x38 else 0x20 + rng.uintLessThan(u8, 0x28);
        if (fill == .random) {
            const special = [_]u16{ 0, 0x8000, 1, 0x8001 };
            @memcpy(d.x[0..4], &special);
            for ([_]u8{ 0, 1, 0x7e }, 0..) |scale, i| if (i < s.n) {
                scales[i * (s.k / 16)] = scale;
            };
        }
        if (fill == .stress) for (0..s.n) |n| for (s.k / 2..s.k) |k| {
            const from = n * s.k + k - s.k / 2;
            const nib = ((codes[from / 2] >> @as(u3, @intCast((from & 1) * 4))) & 15) | 8;
            const at = n * s.k + k;
            if (k % 2 == 0) codes[at / 2] = nib else codes[at / 2] |= nib << 4;
        };
        if (self.control != null) {
            @memset(codes, 0x22);
            @memset(scales, 0x38);
            @memset(d.x, 0);
            for (0..s.rows) |r| {
                for ([_]f32{ 16777216, 1, -16777216, -1 }, 0..) |v, i| d.x[r * s.k + i * 16] = c.toBf16(v);
            }
        }
        const global = self.panelFill(s, fill, d.x, codes, scales);
        for (0..s.n) |n| for (0..s.k) |k| {
            const at = n * s.k + k;
            const nib = (codes[at / 2] >> @as(u3, @intCast((at & 1) * 4))) & 15;
            d.w[at] = core.quant.nvfp4.e2m1(nib) * core.quant.fp8.e4m3(scales[n * (s.k / 16) + k / 16]) * global;
        };
        const npad = cuda.nvfp4.padded(s.n);
        const words = try a.alloc(u32, npad * s.k / 8);
        defer a.free(words);
        const packed_scales = try a.alloc(u8, npad * (s.k / 16));
        defer a.free(packed_scales);
        cuda.nvfp4.packWords(words, codes, s.n, s.k);
        cuda.nvfp4.packScales(packed_scales, scales, s.n, s.k);
        self.w = cuda.nvfp4.weight(try self.rig.input(std.mem.sliceAsBytes(words)), try self.rig.input(packed_scales), global, s.n, s.k);
        if (self.split != 0) self.w = try q.pinned(self.w, self.split);
        self.ldx = if (s.rows == 1) s.k else s.k + 8;
        const strided = try a.alloc(u16, s.rows * self.ldx);
        defer a.free(strided);
        @memset(strided, 0x7fc1);
        for (0..s.rows) |r| @memcpy(strided[r * self.ldx ..][0..s.k], d.x[r * s.k ..][0..s.k]);
        self.x = try self.rig.input(std.mem.sliceAsBytes(strided));
        self.out = try self.rig.output(self.count() * self.width());
        self.base = try self.rig.scratch(self.count() * self.width());
        const parts = self.lane.partBytes(s.rows, self.w);
        if (parts != 0) {
            self.part = try self.rig.scratch(parts);
            self.base_part = try self.rig.scratch(parts);
        }
        return true;
    }
    pub fn run(self: *Product) !void {
        const s = self.d.?.shape;
        if (self.control == null and wide.eligible(s.rows, self.w)) {
            if (wide.pick(self.lane, s.rows, self.w, self.format) == null) return error.WidePathNotSelected;
            try self.lane.matmulOut(self.rig.s, self.x, self.ldx, s.rows, self.w, self.out, self.format, self.part);
        } else try self.forced();
        try self.rig.s.synchronize();
    }
    fn forced(self: *Product) !void {
        const s = self.d.?.shape;
        const p = q.planFor(s.rows, self.w, self.lane.clusters);
        const f = self.control orelse self.lane.fp4_wide.?.functions[@backingInt(self.format)][@intFromBool(p.cluster)];
        var args: cuda.Args = .{};
        args.add(self.x);
        args.add(self.w.codes);
        args.add(self.w.scales);
        args.add(self.w.scale);
        args.add(self.out);
        args.add(if (p.reduce) self.part else @as(u64, 0));
        for ([_]usize{ s.rows, s.n, s.k, p.sk, self.w.npad, if (s.rows == 1) s.k else self.ldx, 1 }) |v|
            args.add(@as(i32, @intCast(v)));
        try cuda.launch.launch(f, .{
            .grid = .{ .x = @intCast((s.n + 127) / 128), .z = p.sk },
            .block = .{ .x = 128 },
            .shared = wide.shared,
            .cluster = if (p.cluster) .{ .x = 1, .y = 1, .z = p.sk } else null,
        }, self.rig.s, &args);
        if (p.reduce) {
            const total = self.count();
            var r: cuda.Args = .{};
            r.add(self.part);
            r.add(self.out);
            r.add(@as(i64, @intCast(total)));
            r.add(@as(i32, @intCast(p.sk)));
            r.add(self.w.scale);
            try cuda.launch.launch(self.lane.reduce[@backingInt(self.format)], .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, self.rig.s, &r);
        }
    }
    pub fn replaced(self: *Product) ![]const u8 {
        var baseline = self.lane;
        baseline.fp4_wide = null;
        const s = self.d.?.shape;
        try baseline.matmulOut(self.rig.s, self.x, self.ldx, s.rows, self.w, self.base, self.format, self.base_part);
        if (self.part != 0 and self.control == null) {
            const bytes = baseline.partBytes(s.rows, self.w);
            const a = self.rig.gpu.gpa;
            const candidate = try a.dupe(u8, try self.rig.bytes(self.part, bytes));
            defer a.free(candidate);
            if (!std.mem.eql(u8, candidate, try self.rig.bytes(self.base_part, bytes))) return error.WidePartialBitsDiffer;
        }
        return self.rig.bytes(self.base, self.count() * self.width());
    }
    pub fn count(self: *Product) usize {
        return self.d.?.outputs();
    }
    pub fn reference(self: *Product, want: []f64, magnitude: []f64) !void {
        try self.d.?.reference(want, magnitude);
    }
    pub fn simulate(self: *Product, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.d.?.simulate(acc, order, out);
    }
};

pub const Registrations = struct {
    products: [27]Product = undefined,
    len: usize = 0,
    pub fn load(gpu: check.Gpu, s: cuda.Stream, lane: q.Lane, sm: u32) !Registrations {
        if (!wide.supported(sm)) return .{};
        const set = try wide.load(lane.mod, sm) orelse return error.MissingWideSymbols;
        const control = try lane.mod.function(wide.control_symbol);
        var r: Registrations = .{};
        for ([_]q.Out{ .bf16, .f32 }, 0..) |out, oi| for ([_]bool{ false, true }, 0..) |cluster, ci| {
            var selected = lane;
            selected.fp4_wide = set;
            selected.clusters = cluster;
            selected.half_n = null;
            selected.half_below = 0;
            selected.pdl = false;
            r.products[oi * 2 + ci] = .{ .rig = .{ .gpu = gpu, .s = s }, .lane = selected, .format = out };
        };
        r.products[4] = r.products[2];
        r.products[4].control = control;
        r.len = 5;
        for ([_]q.Out{ .bf16, .f32 }, 0..) |out, oi| {
            for ([_]u32{ 1, 2, 4, 8 }) |sk| for ([_]bool{ false, true }, 0..) |cluster, ci| {
                if (sk == 1 and cluster) continue;
                r.products[r.len] = r.products[oi * 2 + ci];
                r.products[r.len].split = sk;
                r.len += 1;
            };
            for ([_]panels.Case{ .scales, .ties }) |which| for (0..2) |ci| {
                r.products[r.len] = r.products[oi * 2 + ci];
                r.products[r.len].directed = which;
                r.len += 1;
            };
            _ = out;
        }
        return r;
    }
    pub fn deinit(self: *Registrations) void {
        for (self.products[0..self.len]) |*p| p.deinit();
    }
    pub fn append(self: *Registrations, a: std.mem.Allocator, kernels: *std.ArrayList(g.Kernel)) !void {
        for (self.products[0..self.len]) |*p| try kernels.append(a, p.kernel());
    }
};

test "qmmf exact tile: registrations cover both output types, merge forms, K splits and the altered-order control" {
    for ([_]q.Out{ .bf16, .f32 }) |out| for ([_]bool{ false, true }) |clusters| {
        const s = spec(out, clusters, false);
        try std.testing.expect(s.replaces != null and Product.vt.replaced != null and Product.vt.simulate != null);
        var seen: [9]bool = @splat(false);
        for (s.served) |shape| seen[q.splitK(shape.n, shape.k)] = true;
        for ([_]usize{ 1, 2, 4, 8 }) |sk| try std.testing.expect(seen[sk]);
    };
    try std.testing.expectEqualStrings("main qmmf FP4 16x64, identical K slices and ordered merge", spec(.f32, false, true).replaces.?);
}
