//! sam3_mask_squash in its served modes against float64: active slots squashed, inactive 0, other channels untouched.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const arithmetic = @import("metal_sam3_arithmetic.zig");
const c = contract.spec;
const g = contract.gates;
const vt = g.hooks(Probe);

const u: f64 = 0x1p-24;
const frame = 1008 * 1008;

const Mode = enum { grouped, sparse, accurate, step };

const Params = extern struct { pixels: u32, stride: u32, slots: u32, sources: u32, active: u32, threshold: u32, accurate: u32, gain: f32, offset: f32 };

fn label(mode: Mode, control: bool) []const u8 {
    if (control) return "control: sam3 arithmetic squash drops its offset";
    return switch (mode) {
        .grouped => "metal sam3 arithmetic squash grouped slots",
        .sparse => "metal sam3 arithmetic squash one slot",
        .accurate => "metal sam3 arithmetic squash one slot precise exp",
        .step => "metal sam3 arithmetic squash one slot step",
    };
}

fn noSums(_: c.Shape) usize {
    return 0;
}

/// memory_writer.zig's squashParams: SAM3.1 grouped (16 slots into 32 channels), sparse (fast or precise), SAM3's step.
const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    mode: Mode,
    control: bool,
    shape: c.Shape = undefined,
    logits: []f32 = &.{},
    input: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    fn params(p: *const Probe) Params {
        const grouped = p.mode == .grouped;
        return .{
            .pixels = @intCast(p.shape.rows),
            .stride = @intCast(p.shape.k),
            .slots = @intCast(p.shape.n),
            .sources = @intCast(p.shape.n),
            .active = if (grouped) 0b1010_0101_1100_0011 else 1,
            .threshold = @intFromBool(p.mode == .step),
            .accurate = @intFromBool(p.mode == .accurate),
            .gain = if (p.mode == .step) 20 else 2,
            .offset = if (p.mode == .step) -10 else -1,
        };
    }

    fn spec(p: *const Probe) c.Spec {
        const grouped = p.mode == .grouped;
        return .{
            .name = label(p.mode, p.control),
            .accum = .none,
            .depth = noSums,
            .out = .f32,
            .nan = .{ .stated = "a NaN logit squashes to NaN, or to the offset in step mode (NaN > 0 is false)" },
            .tiles = .{ .{ .size = if (grouped) 16 else 256 }, .{}, .{} },
            .served = if (grouped) &.{.{ .rows = frame, .n = 16, .k = 32 }} else &.{.{ .rows = frame, .n = 1, .k = 1 }},
        };
    }

    fn reset(p: *Probe) void {
        p.a.free(p.logits);
        p.logits = &.{};
        p.rig.reset();
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const grouped = p.mode == .grouped;
        if (shape.rows == 0 or shape.n != @as(usize, if (grouped) 16 else 1) or shape.k != @as(usize, if (grouped) 32 else 1)) return false;
        p.reset();
        p.shape = shape;
        p.logits = try p.a.alloc(f32, shape.rows * shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (p.logits) |*v| v.* = (rng.float(f32) - 0.5) * 24;
        if (fill == .nonfinite) {
            p.logits[0] = std.math.nan(f32);
            p.logits[1] = std.math.inf(f32);
            p.logits[2] = -std.math.inf(f32);
            p.logits[shape.n] = -std.math.inf(f32);
            p.logits[2 * shape.n] = std.math.inf(f32);
        } else if (p.logits.len > 3) {
            p.logits[0] = 100;
            p.logits[1] = -100;
            p.logits[2] = 0;
            p.logits[3] = -0.0;
        }
        p.input = try p.rig.input(std.mem.sliceAsBytes(p.logits));
        p.out = try p.rig.output(p.count() * 4);
        return true;
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        pass.e.setPipeline(p.pipe);
        pass.e.setBuffer(p.input.buf, p.input.off, 0);
        pass.e.setBuffer(p.out.buf, p.out.off, 1);
        pass.e.setValue(p.params(), 2);
        pass.e.dispatchThreads(mtl.Size.of(p.shape.rows * p.shape.n, 1, 1), mtl.Size.of(256, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn count(p: *const Probe) usize {
        return p.shape.rows * p.shape.k;
    }

    /// Output i's float64 value and allowance; null: the channel is not the kernel's to write.
    fn expect(p: *const Probe, i: usize) ?[2]f64 {
        const pr = p.params();
        const pixel = i / p.shape.k;
        const slot = i % p.shape.k;
        if (slot >= p.shape.n) return null;
        if ((pr.active >> @intCast(slot)) & 1 == 0) return .{ 0, 0 };
        const l: f64 = p.logits[pixel * p.shape.n + slot];
        const gain: f64 = pr.gain;
        const offset: f64 = pr.offset;
        if (p.mode == .step) {
            const w = (if (l > 0) gain else 0) + offset;
            return .{ w, u * (@abs(gain) + @abs(w)) + 0x1p-149 };
        }
        const e = @exp(-l);
        const s = 1 / (1 + e);
        const w = s * gain + offset;
        if (!std.math.isFinite(w)) return .{ w, 0 };
        // fast exp (40 + 2|l|) ulps as core/attention_range.zig, precise 4; the add; a 2-ulp divide; exact at |l| = Inf
        const delta = if (std.math.isInf(l)) 0 else if (p.mode == .accurate) 4 * 0x1p-23 else (40 + 2 * @abs(l)) * 0x1p-23;
        const share = if (std.math.isInf(e)) 1 else e / (1 + e);
        const rel = (delta * share + 5 * u) / (1 - delta - 5 * u);
        const ds = @abs(gain) * s * rel;
        return .{ w, (ds + u * @abs(gain * s) + u * (@abs(w) + ds)) * (1 + 0x1p-20) + 0x1p-149 };
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*w, *m, i| {
            w.* = if (p.expect(i)) |e| e[0] else 0;
            m.* = @abs(w.*);
        }
    }

    pub fn check(p: *Probe, at: g.Gate) !?[]const u8 {
        const out = p.rig.bytes(p.out, p.count() * 4);
        var planted: usize = 0;
        for (0..p.count()) |i| {
            const bits = std.mem.readInt(u32, out[4 * i ..][0..4], .little);
            const got: f64 = @as(f32, @bitCast(bits));
            const e = p.expect(i) orelse {
                if (bits != 0xffffffff) return "a channel past the slots was written";
                continue;
            };
            if (c.classOf(e[0]) != .finite) planted += 1;
            if (c.classOf(got) != c.classOf(e[0])) return "an output's class differs from float64's";
            if (std.math.isFinite(e[0]) and !(@abs(got - e[0]) <= e[1])) return "an output leaves its float64 bound";
        }
        if (at == .nonfinite and planted == 0 and p.mode != .step) return "the planted values give no non-finite output: no proof";
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    probes: [5]?*Probe = @splat(null),

    pub fn collect(s: *Suite, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        for (&s.probes, 0..) |*slot, i| {
            const control = i == 4;
            const mode: Mode = if (control) .grouped else @fromBackingInt(@intCast(i));
            if (only) |needle| if (std.mem.indexOf(u8, label(mode, control), needle) == null) continue;
            try kernels.ensureUnusedCapacity(s.a, 1);
            const p = try s.a.create(Probe);
            errdefer s.a.destroy(p);
            p.* = .{ .a = s.a, .rig = .{ .a = s.a, .device = device, .queue = queue }, .pipe = try arithmetic.pipeline(s.a, device, "sam3_mask_squash", control), .mode = mode, .control = control };
            slot.* = p;
            kernels.appendAssumeCapacity(.{ .spec = p.spec(), .ctx = p, .vt = &vt, .control = if (control) .shapes else null });
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes) |maybe| if (maybe) |p| {
            p.reset();
            p.rig.deinit();
            p.pipe.deinit();
            s.a.destroy(p);
        };
    }
};
