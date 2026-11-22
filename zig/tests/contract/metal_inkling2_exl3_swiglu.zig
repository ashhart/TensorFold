//! Inkling 2 Mini's EXL3 SwiGLU activation on Metal: one elementwise BF16 kernel against a float64 oracle.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const u = @import("inkling2_util.zig");

const c = contract.spec;
const g = contract.gates;

/// Distinct caps so a swapped pair is a fault; the store reads them from mlp_gate_softcap/mlp_up_softcap.
const gate_cap: f64 = 7.0;
const up_cap: f64 = 30.0;

const Args = extern struct { pairs: u32, inter: u32, gate_cap: f32, up_cap: f32 };

fn noDepth(_: c.Shape) usize {
    return 0;
}

pub fn spec(name: []const u8) c.Spec {
    return .{
        .name = name,
        .accum = .none,
        .depth = noDepth,
        .out = .bf16,
        .served = &.{
            .{ .rows = 8, .n = 1, .k = 1152 }, // one decode row's eight route pairs
            .{ .rows = 136, .n = 1, .k = 1152 }, // a 17-row prompt's pairs
            .{ .rows = 1024, .n = 1, .k = 1152 }, // a wide prompt's pairs
        },
    };
}

fn spread(rng: std.Random) f32 {
    const mag = std.math.pow(f32, 2, @floatFromInt(rng.intRangeAtMost(i32, -8, 9)));
    return (rng.float(f32) * 2 - 1) * mag;
}

/// i2x_swiglu in float64: the sigmoid reads the raw gate, both streams cap through tanh, BF16 out.
fn yOf(gate: f64, up: f64) f64 {
    const sigmoid = 1.0 / (1.0 + @exp(-gate));
    const gc = std.math.tanh(gate / gate_cap) * gate_cap;
    const uc = std.math.tanh(up / up_cap) * up_cap;
    return (gc * sigmoid) * uc;
}

pub const Proj = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    name: []const u8,
    lib: ?mtl.Library = null,
    pipe: ?mtl.Pipeline = null,
    shape: c.Shape = undefined,
    gbf: []u16 = &.{},
    ubf: []u16 = &.{},
    in_g: rig_.Ref = undefined,
    in_u: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, source: ?[]const u8, name: []const u8) !Proj {
        const text = source orelse sources.inkling2_experts;
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        errdefer lib.deinit();
        const pipe = try mtl.Pipeline.init(device, lib, "inkling2_exp_swiglu", false);
        errdefer pipe.deinit();
        return .{
            .a = a,
            .arena = std.heap.ArenaAllocator.init(a),
            .rig = .{ .a = a, .device = device, .queue = queue },
            .name = name,
            .lib = lib,
            .pipe = pipe,
        };
    }

    pub fn deinit(p: *Proj) void {
        if (p.pipe) |x| x.deinit();
        if (p.lib) |x| x.deinit();
        p.rig.deinit();
        p.arena.deinit();
    }

    pub fn kernel(p: *Proj, control: ?g.Gate) g.Kernel {
        return .{ .ctx = p, .vt = &vt, .control = control, .spec = spec(p.name) };
    }

    const vt = g.hooks(Proj);

    pub fn count(p: *Proj) usize {
        return p.shape.rows * p.shape.k;
    }

    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k == 0) return false;
        p.rig.reset();
        _ = p.arena.reset(.retain_capacity);
        const a = p.arena.allocator();
        p.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = shape.rows * shape.k;
        p.gbf = try a.alloc(u16, n);
        p.ubf = try a.alloc(u16, n);
        for (p.gbf) |*o| o.* = c.toBf16(spread(rng));
        for (p.ubf) |*o| o.* = c.toBf16(spread(rng));
        if (fill == .nonfinite) {
            const nan = c.toBf16(std.math.nan(f32));
            const inf = c.toBf16(std.math.inf(f32));
            const ninf = c.toBf16(-std.math.inf(f32));
            p.gbf[n / 7] = nan;
            p.gbf[n / 3 + 1] = inf;
            p.gbf[2 * n / 5] = ninf;
            p.ubf[n / 11] = nan;
            p.ubf[3 * n / 7] = inf;
            p.ubf[n - 1] = ninf;
        }
        p.in_g = try p.rig.input(std.mem.sliceAsBytes(p.gbf));
        p.in_u = try p.rig.input(std.mem.sliceAsBytes(p.ubf));
        p.out = try p.rig.output(n * 2);
        return true;
    }

    pub fn encode(p: *Proj, e: mtl.ComputeEncoder) !void {
        e.setPipeline(p.pipe.?);
        e.setBuffer(p.in_g.buf, p.in_g.off, 0);
        e.setBuffer(p.in_u.buf, p.in_u.off, 1);
        e.setBuffer(p.out.buf, p.out.off, 2);
        e.setValue(Args{ .pairs = @intCast(p.shape.rows), .inter = @intCast(p.shape.k), .gate_cap = @floatCast(gate_cap), .up_cap = @floatCast(up_cap) }, 3);
        e.dispatchThreads(mtl.Size.of(@intCast(p.count()), 1, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn run(p: *Proj) !void {
        const pass = p.rig.begin();
        try p.encode(pass.e);
        try p.rig.finish(pass);
    }

    pub fn reference(p: *Proj, want: []f64, mag: []f64) !void {
        const n = p.count();
        for (0..n) |i| {
            want[i] = yOf(c.fromBf16(p.gbf[i]), c.fromBf16(p.ubf[i]));
            mag[i] = @abs(want[i]);
        }
    }

    /// One BF16 ulp at the magnitude; below ~2^-118 fp32 intermediates can flush, so the full value is allowed.
    pub fn bounds(_: *Proj, want: []const f64, mag: []const f64, allow: []f64) !void {
        _ = want;
        for (allow, mag) |*l, m| {
            l.* = if (!std.math.isFinite(m)) std.math.inf(f64) else if (m < 0x1p-118) m + 0x1p-133 else std.math.pow(f64, 2, @floatFromInt(std.math.frexp(m).exponent - 8)) + 0x1p-133;
        }
    }

    pub fn simulate(p: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        _ = p;
        _ = acc;
        _ = order;
        _ = out;
        return error.NoSums;
    }
};

const Mutation = struct { gate: g.Gate, label: []const u8, from: []const u8, to: []const u8 };
const mutations = [_]Mutation{
    .{
        .gate = .shapes,
        .label = "control: inkling2_exp_swiglu swaps the gate and up streams",
        .from = "i2x_swiglu(g, u, a.gate_cap, a.up_cap)",
        .to = "i2x_swiglu(u, g, a.up_cap, a.gate_cap)",
    },
    .{
        .gate = .shapes,
        .label = "control: inkling2_exp_swiglu swaps the soft caps",
        .from = "i2x_swiglu(g, u, a.gate_cap, a.up_cap)",
        .to = "i2x_swiglu(g, u, a.up_cap, a.gate_cap)",
    },
    .{
        .gate = .shapes,
        .label = "control: inkling2_exp_swiglu writes one element past its output",
        .from = "out[idx] = bfloat(i2x_swiglu(g, u, a.gate_cap, a.up_cap));",
        .to = "out[idx + 1] = bfloat(i2x_swiglu(g, u, a.gate_cap, a.up_cap));",
    },
    .{
        .gate = .nonfinite,
        .label = "control: inkling2_exp_swiglu reads a non-finite gate as zero",
        .from = "const float g = float(gate[idx]);",
        .to = "float g = float(gate[idx]); if (g != g) g = 0.0f;",
    },
};

/// One probe per registration and control; each mutated probe compiles its own copy of the expert unit.
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    held: std.ArrayList(*Proj) = .empty,

    fn add(s: *Suite, only: ?[]const u8, label: []const u8, mutation: ?Mutation) !void {
        if (only) |text| if (std.mem.indexOf(u8, label, text) == null) return;
        const source = if (mutation) |m| try u.mutate(s.a, sources.inkling2_experts, m.from, m.to) else null;
        defer if (source) |t| s.a.free(t);
        const p = try s.a.create(Proj);
        errdefer s.a.destroy(p);
        p.* = try Proj.init(s.a, s.device, s.queue, source, label);
        errdefer p.deinit();
        try s.held.append(s.a, p);
        errdefer _ = s.held.pop();
        try s.kernels.append(s.a, p.kernel(if (mutation) |m| m.gate else null));
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        try s.add(only, "inkling2 experts EXL3 swiglu inkling2_exp_swiglu", null);
        for (mutations) |m| try s.add(only, m.label, m);
    }

    pub fn deinit(s: *Suite) void {
        for (s.held.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.held.deinit(s.a);
    }
};
