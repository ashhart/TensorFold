//! Nemotron's tf_sample_full as Ripple draws with it, held to its stated rule: each draw equals lanes' gpu_full twin.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const rc = @import("metal_ricursive.zig");
const c = contract.spec;
const g = contract.gates;
const full = tf.lanes.gpu_full;
const Sampling = tf.lanes.Sampling;

const rule = "each row's draw equals lanes' gpu_full twin: the plan (top_k, top_p, min_p) then the keyed race";

/// The settings rows cycle through: top_k off with a nucleus, a wide top_k, a narrow one at T 0.7, and min_p.
const settings = [_]Sampling{
    .{ .seed = 11, .temperature = 1.0, .top_k = 0, .top_p = 0.9 },
    .{ .seed = 12, .temperature = 1.0, .top_k = 3000, .top_p = 0.5 },
    .{ .seed = 13, .temperature = 0.7, .top_k = 20, .top_p = 0.95 },
    .{ .seed = 14, .temperature = 1.0, .top_k = 0, .top_p = 1.0, .min_p = 0.01 },
};
const position: u32 = 40000;

pub fn spec() c.Spec {
    return .{ .name = "ricursive sample tf_sample_full", .accum = .none, .depth = rc.none, .out = .raw, .nan = .{ .stated = rule }, .tiles = .{ .{}, .{}, .{ .size = 1024, .step = 1, .min = 1 } }, .served = &.{.{ .rows = 8, .n = 1, .k = rc.R.vocab }}, .sums = rc.no_sums };
}

pub const Sample = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    logits: []f32 = &.{},
    in: [5]rig_.Ref = undefined, // logits, seeds, positions, cfgs, caps
    out: rig_.Ref = undefined,
    msg: [160]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Sample {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, sources.nemotron_sample, "tf_sample_full") };
    }

    pub fn deinit(p: *Sample) void {
        p.a.free(p.logits);
        p.rig.deinit();
        p.pipe.deinit();
    }

    const vt = g.hooks(Sample);

    pub fn kernel(p: *Sample) g.Kernel {
        return .{ .spec = spec(), .ctx = p, .vt = &vt };
    }

    /// A row of logits in [-3, 0] in bf16 (every fourth row a peak over 2,000 tokens); planted NaN and infinities.
    pub fn prepare(p: *Sample, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.k == 0 or s.n != 1) return false;
        p.a.free(p.logits);
        p.logits = &.{};
        p.rig.reset();
        p.shape = s;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        p.logits = try p.a.alloc(f32, s.rows * s.k);
        const words = try p.a.alloc(u16, s.rows * s.k);
        defer p.a.free(words);
        for (words, 0..) |*w, i| {
            const peak = (i / s.k) % 4 == 3 and i % s.k < 2000;
            w.* = c.toBf16(if (peak) 0 else -3 * rng.float(f32));
        }
        if (fill == .nonfinite) rc.plant(words, s.k, s.rows);
        for (p.logits, words) |*l, w| l.* = c.fromBf16(w);
        var seeds: [64]u32 = undefined;
        var positions: [32]u32 = undefined;
        var cfgs: [128]f32 = undefined;
        var caps: [32]u32 = undefined;
        if (s.rows > positions.len) return false;
        for (0..s.rows) |r| {
            const ru = tf.lanes.gpu_rule.rule(settings[r % settings.len]);
            seeds[2 * r ..][0..2].* = ru.seedHalves();
            positions[r] = position + @as(u32, @intCast(r));
            cfgs[4 * r ..][0..4].* = ru.cfg();
            caps[r] = ru.top_k;
        }
        p.in[0] = try p.rig.input(std.mem.sliceAsBytes(words));
        p.in[1] = try p.rig.input(std.mem.sliceAsBytes(seeds[0 .. 2 * s.rows]));
        p.in[2] = try p.rig.input(std.mem.sliceAsBytes(positions[0..s.rows]));
        p.in[3] = try p.rig.input(std.mem.sliceAsBytes(cfgs[0 .. 4 * s.rows]));
        p.in[4] = try p.rig.input(std.mem.sliceAsBytes(caps[0..s.rows]));
        p.out = try p.rig.output(s.rows * 4);
        return true;
    }

    pub fn count(p: *Sample) usize {
        return p.shape.rows * 4;
    }

    pub fn run(p: *Sample) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        for (p.in, 0..) |r, slot| e.setBuffer(r.buf, r.off, slot);
        e.setBuffer(p.out.buf, p.out.off, 5);
        e.setBuffer(p.out.buf, p.out.off, 6); // the id map, unread without MAP (forward.zig binds the same)
        e.setValue(@as(u32, @intCast(p.shape.k)), 7);
        e.dispatchGroups(mtl.Size.of(p.shape.rows, 1, 1), mtl.Size.of(1024, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn reference(_: *Sample, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// The twin's draw for every row (gates 1 and 3).
    pub fn check(p: *Sample, _: g.Gate) !?[]const u8 {
        const k = p.shape.k;
        const ids = std.mem.bytesAsSlice(u32, p.rig.bytes(p.out, p.shape.rows * 4));
        for (0..p.shape.rows) |r| {
            const s = settings[r % settings.len];
            const row = p.logits[r * k ..][0..k];
            const plan = try full.plan(p.a, row, s);
            const want = full.race(row, plan, s.seed, position + @as(u32, @intCast(r)), null);
            if (ids[r] != want) return try std.fmt.bufPrint(&p.msg, "row {d}: drew {d} where the twin draws {d}", .{ r, ids[r], want });
        }
        return null;
    }
};
