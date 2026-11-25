//! Contract probes, red controls, and FailingAllocator OOM tests for DFlash 2 fast BF16 NAX/MMA kernels.
const std = @import("std");
const mtl = @import("metal");
const controls = @import("metal_inkling2_dspark_controls.zig");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const bf16 = @import("tensorfold").inkling2_dflash2_bf16;
const sources = @import("inkling2_metal");

const c = contract.spec;
const g = contract.gates;
const Dense = contract.dense.Dense;

fn depth(s: c.Shape) usize {
    return s.k + 16;
}

const served16 = [_]c.Shape{
    .{ .rows = 8, .n = 256, .k = 2304 },
    .{ .rows = 16, .n = 67, .k = 96 },
    .{ .rows = 7, .n = 129, .k = 256 },
    .{ .rows = 1, .n = 33, .k = 64 },
};

const served32 = [_]c.Shape{
    .{ .rows = 32, .n = 256, .k = 2304 },
    .{ .rows = 33, .n = 129, .k = 256 },
    .{ .rows = 17, .n = 67, .k = 96 },
    .{ .rows = 9, .n = 33, .k = 64 },
};

const round_from = "static inline bfloat df2_fast_round(float v) { return bfloat(v); }";
const round_to = "static inline bfloat df2_fast_round(float v) { return bfloat(0.5f * v); }";

pub fn controlSource(a: std.mem.Allocator, production: []const u8) ![]u8 {
    if (std.mem.count(u8, production, round_from) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, production, round_from, round_to);
}

pub fn spec(kind: bf16.Kind, name: []const u8) c.Spec {
    return switch (kind) {
        .bf16_nax_16 => .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .out = .bf16,
            .tiles = .{ .{ .size = 16, .step = 1, .min = 1 }, .{ .size = 32, .step = 1, .min = 1 }, .{ .size = 256, .step = 32, .min = 32 } },
            .served = &served16,
        },
        .bf16_nax_32 => .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .out = .bf16,
            .tiles = .{ .{ .size = 32, .step = 1, .min = 1 }, .{ .size = 64, .step = 1, .min = 1 }, .{ .size = 256, .step = 32, .min = 32 } },
            .served = &served32,
        },
    };
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    kind: bf16.Kind,
    name: []const u8,
    control: bool,
    d: ?Dense = null,
    w16: []u16 = &.{},
    x: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    y: rig_.Ref = undefined,

    pub fn initSource(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: bf16.Kind, text: []const u8, name: []const u8, control: bool) !Probe {
        const opts = mtl.CompileOptions{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise };
        const lib = try mtl.Library.fromSource(device, text, opts);
        defer lib.deinit();
        return .{
            .a = a,
            .rig = .{ .a = a, .device = device, .queue = queue },
            .pipe = try mtl.Pipeline.init(device, lib, kind.symbol(), false),
            .kind = kind,
            .name = name,
            .control = control,
        };
    }

    pub fn deinit(p: *Probe) void {
        if (p.d) |d| d.deinit();
        p.a.free(p.w16);
        p.rig.deinit();
        p.pipe.deinit();
    }

    const vt = g.hooks(Probe);
    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .spec = spec(p.kind, p.name), .ctx = p, .vt = &vt, .control = if (p.control) .shapes else null };
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k % 32 != 0) return false;
        if (p.d) |d| d.deinit();
        p.d = null;
        p.a.free(p.w16);
        p.w16 = &.{};
        p.rig.reset();
        const d = try Dense.init(p.a, shape, 1);
        errdefer d.deinit();
        d.fillX(fill, seed, 2);
        const w16 = try p.a.alloc(u16, shape.n * shape.k);
        errdefer p.a.free(w16);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x9e37);
        const half = shape.k / 2;
        for (0..shape.n) |j| {
            for (0..shape.k) |i| {
                const mag: f32 = @floatFromInt(16 + (i * 7 + j) % 6);
                const sv: f32 = if (i < half) mag else -@as(f32, @floatFromInt(16 + ((i - half) * 7 + j) % 6));
                const raw = if (fill != .stress) blk: {
                    const v = prng.random().float(f32) * 2.0 - 1.0;
                    break :blk if (v == 0) @as(f32, 0.25) else v;
                } else sv;
                w16[j * shape.k + i] = c.toBf16(raw);
                d.w[j * shape.k + i] = c.fromBf16(w16[j * shape.k + i]);
            }
        }
        p.x = try p.rig.input(std.mem.sliceAsBytes(d.x));
        p.w = try p.rig.input(std.mem.sliceAsBytes(w16));
        p.y = try p.rig.output(d.outputs() * 2);
        p.w16 = w16;
        p.d = d;
        return true;
    }

    fn bfRef(ref: rig_.Ref) bf16.Ref {
        return .{ .buf = ref.buf, .off = ref.off };
    }

    pub fn run(p: *Probe) !void {
        const s = p.d.?.shape;
        const pass = p.rig.begin();
        const tn: usize = if (p.kind == .bf16_nax_32) 64 else 32;
        const tm: usize = if (p.kind == .bf16_nax_32) 32 else 16;
        bf16.Pipelines.encodeNax(p.pipe, pass.e, bfRef(p.x), bfRef(p.w), bfRef(p.y), .{ .rows = @intCast(s.rows), .n = @intCast(s.n), .k = @intCast(s.k), .ldy = @intCast(s.n) }, tn, tm, p.kind.requiredThreads());
        try p.rig.finish(pass);
    }

    pub fn count(p: *Probe) usize {
        return p.d.?.outputs();
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        return p.d.?.reference(want, mag);
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        return p.d.?.simulate(acc, order, out);
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,

    fn controlName(kind: bf16.Kind) []const u8 {
        return switch (kind) {
            inline else => |k| "control: metal inkling2 dflash2_fast " ++ @tagName(k),
        };
    }

    fn add(s: *Suite, kind: bf16.Kind, control: bool, only: ?[]const u8) !void {
        const name = if (control) controlName(kind) else kind.label();
        if (only) |text| if (std.mem.indexOf(u8, name, text) == null) return;
        const probe = try s.a.create(Probe);
        errdefer s.a.destroy(probe);
        const raw = if (s.device.tensorUnits()) sources.inkling2_dflash2_fast else sources.inkling2_dflash2_fast_pre_m5;
        if (control) {
            const text = try controlSource(s.a, raw);
            defer s.a.free(text);
            probe.* = try Probe.initSource(s.a, s.device, s.queue, kind, text, name, true);
        } else {
            probe.* = try Probe.initSource(s.a, s.device, s.queue, kind, raw, name, false);
        }
        errdefer probe.deinit();
        try contract.register(s.a, &s.probes, s.kernels, probe, probe.kernel());
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (std.enums.values(bf16.Kind)) |kind| {
            try s.add(kind, false, only);
            try s.add(kind, true, only);
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |probe| {
            probe.deinit();
            s.a.destroy(probe);
        }
        s.probes.deinit(s.a);
    }
};

test "DFlash2 fast BF16 control anchor applies once and rejects duplicates" {
    const a = std.testing.allocator;
    const prod = sources.inkling2_dflash2_fast;
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, prod, round_from));
    const edited = try controlSource(a, prod);
    defer a.free(edited);
    try std.testing.expectError(error.ControlEditMissing, controlSource(a, edited));
    try std.testing.expectError(error.ControlEditMissing, controlSource(a, ""));
    const twice = try std.mem.concat(a, u8, &.{ prod, prod });
    defer a.free(twice);
    try std.testing.expectError(error.ControlEditMissing, controlSource(a, twice));
}
