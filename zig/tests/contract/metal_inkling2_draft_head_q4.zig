//! Contract probes, red controls, and FailingAllocator OOM tests for Inkling 2 4-bit/64 draft logits.
const std = @import("std");
const mtl = @import("metal");
const controls = @import("metal_inkling2_dspark_controls.zig");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const q4 = @import("tensorfold").inkling2_draft_head_q4;
const sources = @import("inkling2_metal");

const c = contract.spec;
const g = contract.gates;
const Affine = contract.affine.Affine;

pub const Mode = enum { dspark_q4, batch_q4 };

fn depth(s: c.Shape) usize {
    return 64 + s.k / 32 + 8;
}

const dspark_served = [_]c.Shape{
    .{ .rows = 7, .n = 129, .k = 2304 },
    .{ .rows = 9, .n = 17, .k = 256 },
    .{ .rows = 1, .n = 5, .k = 64 },
};

const batch_served = [_]c.Shape{
    .{ .rows = 32, .n = 129, .k = 2304 },
    .{ .rows = 17, .n = 67, .k = 256 },
    .{ .rows = 9, .n = 17, .k = 256 },
    .{ .rows = 1, .n = 5, .k = 64 },
};

const batch_from = "const uint q1 = (byte_val >> 4u) & 0xfu;";
const batch_to = "const uint q1 = (byte_val >> 4u) & 0x7u;";
const batch_label = "metal inkling2 draft_head_q4 batch tf_ink2_draft_head_q4_logits";
const batch_control = "control: metal inkling2 draft_head_q4 batch";

pub fn batchControlSource(a: std.mem.Allocator, production: []const u8) ![]u8 {
    if (std.mem.count(u8, production, batch_from) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, production, batch_from, batch_to);
}

pub fn spec(mode: Mode, name: []const u8) c.Spec {
    return switch (mode) {
        .dspark_q4 => .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .out = .bf16,
            .tiles = .{ .{ .size = 8, .step = 1, .min = 1 }, .{ .size = 8, .step = 1, .min = 1 }, .{ .size = 256, .step = 64, .min = 64 } },
            .served = &dspark_served,
        },
        .batch_q4 => .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .out = .bf16,
            .tiles = .{ .{ .size = 32, .step = 1, .min = 1 }, .{ .size = 64, .step = 1, .min = 1 }, .{ .size = 256, .step = 64, .min = 64 } },
            .served = &batch_served,
        },
    };
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    mode: Mode,
    name: []const u8,
    control: bool,
    f: ?Affine = null,
    w: rig_.Ref = undefined,
    s: rig_.Ref = undefined,
    b: rig_.Ref = undefined,
    y: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn initSource(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, mode: Mode, text: []const u8, name: []const u8, control: bool) !Probe {
        const opts: mtl.CompileOptions = if (mode == .dspark_q4) .mlx() else .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise };
        const lib = try mtl.Library.fromSource(device, text, opts);
        defer lib.deinit();
        const sym = switch (mode) {
            .dspark_q4 => q4.symbol,
            .batch_q4 => q4.batch_symbol,
        };
        return .{
            .a = a,
            .rig = .{ .a = a, .device = device, .queue = queue },
            .pipe = try mtl.Pipeline.init(device, lib, sym, false),
            .mode = mode,
            .name = name,
            .control = control,
        };
    }

    pub fn deinit(p: *Probe) void {
        if (p.f) |f| f.deinit();
        p.f = null;
        p.rig.deinit();
        p.pipe.deinit();
    }

    const vt = g.hooks(Probe);
    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .spec = spec(p.mode, p.name), .ctx = p, .vt = &vt, .control = if (p.control) .shapes else null };
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (p.f) |f| f.deinit();
        p.f = null;
        p.rig.reset();
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k % 64 != 0) return false;
        var f = try Affine.init(p.a, shape, 4, 64);
        f.form = .dequantized;
        errdefer f.deinit();
        try f.fill(fill, seed);
        if (fill == .random and shape.rows == 1 and shape.k == 64 and shape.n >= 4) {
            // Full-span BF16 group (b=-255*2^120, s=34*2^120): q=15 needs fma so s*15 stays finite.
            @memset(f.x, 0);
            f.x[0] = c.toBf16(1.0);
            for ([_][2]u8{ .{ 15, 0 }, .{ 11, 0 }, .{ 0, 0 }, .{ 0, 15 } }, 0..) |pair, j| {
                f.scales[j] = 0x7e08;
                f.biases[j] = 0xff7f;
                @memset(f.codes[j * 64 ..][0..64], 0);
                f.codes[j * 64] = pair[0];
                f.codes[j * 64 + 1] = pair[1];
            }
        }
        const words = try p.a.alloc(u32, shape.n * (shape.k / 8));
        defer p.a.free(words);
        f.words(words);
        p.w = try p.rig.input(std.mem.sliceAsBytes(words));
        p.s = try p.rig.input(std.mem.sliceAsBytes(f.scales));
        p.b = try p.rig.input(std.mem.sliceAsBytes(f.biases));
        p.y = try p.rig.input(std.mem.sliceAsBytes(f.x));
        p.out = try p.rig.output(f.outputs() * @sizeOf(u16));
        p.f = f;
        return true;
    }

    fn q4Ref(ref: rig_.Ref) q4.Ref {
        return .{ .buf = ref.buf, .off = ref.off };
    }

    pub fn run(p: *Probe) !void {
        const f = p.f.?;
        const pass = p.rig.begin();
        switch (p.mode) {
            .dspark_q4 => q4.dispatch(p.pipe, pass.e, q4Ref(p.w), q4Ref(p.s), q4Ref(p.b), q4Ref(p.y), q4Ref(p.out), @intCast(f.shape.rows), @intCast(f.shape.n), @intCast(f.shape.k), false),
            .batch_q4 => q4.dispatchBatch(p.pipe, pass.e, q4Ref(p.w), q4Ref(p.s), q4Ref(p.b), q4Ref(p.y), q4Ref(p.out), @intCast(f.shape.rows), @intCast(f.shape.n), @intCast(f.shape.k), false),
        }
        try p.rig.finish(pass);
    }

    pub fn count(p: *Probe) usize {
        return p.f.?.outputs();
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        return p.f.?.reference(want, mag);
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        return p.f.?.simulate(acc, order, out);
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,

    fn add(s: *Suite, mode: Mode, control: bool, only: ?[]const u8) !void {
        const name = switch (mode) {
            .dspark_q4 => if (control) controls.name(.logits_q4) else @import("tensorfold").inkling2_dspark_host.Kind.logits_q4.label(),
            .batch_q4 => if (control) batch_control else batch_label,
        };
        if (only) |text| if (std.mem.indexOf(u8, name, text) == null) return;
        const probe = try s.a.create(Probe);
        errdefer s.a.destroy(probe);
        switch (mode) {
            .dspark_q4 => {
                const raw = if (s.device.tensorUnits()) sources.inkling2_dspark else sources.inkling2_dspark_pre_m5;
                if (control) {
                    const text = try controls.source(s.a, raw, .logits_q4);
                    defer s.a.free(text);
                    probe.* = try Probe.initSource(s.a, s.device, s.queue, mode, text, name, true);
                } else {
                    probe.* = try Probe.initSource(s.a, s.device, s.queue, mode, raw, name, false);
                }
            },
            .batch_q4 => {
                const raw = if (s.device.tensorUnits()) sources.inkling2_dflash2_fast else sources.inkling2_dflash2_fast_pre_m5;
                if (control) {
                    const text = try batchControlSource(s.a, raw);
                    defer s.a.free(text);
                    probe.* = try Probe.initSource(s.a, s.device, s.queue, mode, text, name, true);
                } else {
                    probe.* = try Probe.initSource(s.a, s.device, s.queue, mode, raw, name, false);
                }
            },
        }
        errdefer probe.deinit();
        try contract.register(s.a, &s.probes, s.kernels, probe, probe.kernel());
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for ([_]Mode{ .dspark_q4, .batch_q4 }) |mode| {
            try s.add(mode, false, only);
            try s.add(mode, true, only);
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

test "32-row Q4 draft head and pre-M5 planted controls apply once and reject duplicates" {
    const a = std.testing.allocator;
    const prod = sources.inkling2_dflash2_fast;
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, prod, batch_from));
    const edited = try batchControlSource(a, prod);
    defer a.free(edited);
    try std.testing.expectError(error.ControlEditMissing, batchControlSource(a, edited));
    try std.testing.expectError(error.ControlEditMissing, batchControlSource(a, ""));
    const twice = try std.mem.concat(a, u8, &.{ prod, prod });
    defer a.free(twice);
    try std.testing.expectError(error.ControlEditMissing, batchControlSource(a, twice));
    const pre_batch = try batchControlSource(a, sources.inkling2_dflash2_fast_pre_m5);
    defer a.free(pre_batch);
    try std.testing.expect(std.mem.startsWith(u8, pre_batch, "#define TF_SIMD_FRAGS 1\n"));
    const pre_dspark = try controls.source(a, sources.inkling2_dspark_pre_m5, .logits_q4);
    defer a.free(pre_dspark);
    try std.testing.expect(std.mem.startsWith(u8, pre_dspark, "#define TF_SIMD_FRAGS 1\n"));
}
