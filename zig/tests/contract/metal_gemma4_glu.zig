//! Gemma 4's gate|up on the row path (no tensor units): gate rows then up's, against the 16-row interleave it replaces.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const cfg = tf.gemma4.config;
const sqb = tf.row_projection.simd_bits;

const Variant = struct { control: bool = false };
const variants = [_]Variant{ .{}, .{ .control = true } };
const names = [_][]const u8{ "metal gemma4 gate|up split rows +geglu", "control: metal gemma4 gate|up split rows, geglu reading them as interleaved" };

/// Decode and a prompt chunk on the 31B and the 12B: [rows, I] from x [rows, hidden].
const served = [_]c.Shape{ .{ .rows = 1, .n = cfg.dense31b.intermediate, .k = cfg.dense31b.hidden }, .{ .rows = 16, .n = cfg.dense31b.intermediate, .k = cfg.dense31b.hidden }, .{ .rows = 1, .n = cfg.unified12b.intermediate, .k = cfg.unified12b.hidden }, .{ .rows = 9, .n = 1024, .k = 512 } };

fn depth(s: c.Shape) usize {
    return 64 + 11 + s.k / 64 + 8;
}

const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    v: Variant,
    name: []const u8,
    sq: sqb.Pipelines,
    geglu: mtl.Pipeline,
    rows: usize = 0,
    n: usize = 0,
    k: usize = 0,
    /// x; split w, s, b (gate rows then up's); interleaved w, s, b (alternate 16-row blocks).
    in: [7]rig_.Ref = undefined,
    work: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant, name: []const u8) !Probe {
        const glue = try tf.gemma4.kernels.glueSource(a, cfg.dense31b, true, false, 4);
        defer a.free(glue);
        var sq = try sqb.Pipelines.load(device);
        errdefer sq.deinit();
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .name = name, .sq = sq, .geglu = try rig_.pipeline(device, glue, "gemma_geglu") };
    }

    fn deinit(p: *Probe) void {
        p.rig.deinit();
        p.sq.deinit();
        p.geglu.deinit();
    }

    fn spec(p: *const Probe) c.Spec {
        // bit-equal to the twin on the same projection kernel and width: its sums are simd_qmm's own (metal_affine)
        return .{ .name = p.name, .accum = .none, .depth = depth, .out = .bf16, .replaces = "gemma_geglu on gate|up interleaved in 16-row blocks", .tiles = .{ .{ .size = 1 }, .{ .size = 64, .step = 64, .min = 64 }, .{ .size = 512, .step = 64, .min = 64 } }, .served = &served };
    }

    pub fn count(p: *Probe) usize {
        return p.rows * p.n;
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 16 or shape.n % 64 != 0 or shape.k % 64 != 0) return false;
        p.rig.reset();
        p.rows = shape.rows;
        p.n = shape.n;
        p.k = shape.k;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const words = p.k / 8;
        const groups = p.k / 64;
        const x = try p.a.alloc(u16, p.rows * p.k);
        defer p.a.free(x);
        for (x) |*e| e.* = c.toBf16(@floatCast(rng.floatNorm(f64)));
        if (fill == .nonfinite) {
            x[3] = c.toBf16(std.math.nan(f32));
            x[p.k - 1] = c.toBf16(std.math.inf(f32));
        }
        // gate and up: [n, words] codes and [n, groups] scales and biases each, side by side as 2n rows
        const w = try p.a.alloc(u32, 2 * p.n * words);
        defer p.a.free(w);
        const s = try p.a.alloc(u16, 2 * p.n * groups);
        defer p.a.free(s);
        const b = try p.a.alloc(u16, 2 * p.n * groups);
        defer p.a.free(b);
        for (w) |*e| e.* = rng.int(u32);
        for (s) |*e| e.* = c.toBf16(@floatCast(0.002 + rng.float(f64) * 0.01));
        for (b) |*e| e.* = c.toBf16(@floatCast((rng.float(f64) - 0.5) * 0.1));
        if (fill == .stress) for (s, 0..) |*e, i| { // tiny and huge scales: products far apart in magnitude
            if (i % 7 == 0) e.* = c.toBf16(1e-6) else if (i % 11 == 0) e.* = c.toBf16(4);
        };
        const wi = try p.a.alloc(u32, w.len);
        defer p.a.free(wi);
        const si = try p.a.alloc(u16, s.len);
        defer p.a.free(si);
        const bi = try p.a.alloc(u16, b.len);
        defer p.a.free(bi);
        for (0..2) |m| for (0..p.n) |r| { // weights.pack's interleave: row r of member m to 32 (r / 16) + 16 m + r % 16
            const src = m * p.n + r;
            const dst = 32 * (r / 16) + 16 * m + r % 16;
            @memcpy(wi[dst * words ..][0..words], w[src * words ..][0..words]);
            @memcpy(si[dst * groups ..][0..groups], s[src * groups ..][0..groups]);
            @memcpy(bi[dst * groups ..][0..groups], b[src * groups ..][0..groups]);
        };
        p.in[0] = try p.rig.input(std.mem.sliceAsBytes(x));
        for ([_][]const u8{ std.mem.sliceAsBytes(w), std.mem.sliceAsBytes(s), std.mem.sliceAsBytes(b), std.mem.sliceAsBytes(wi), std.mem.sliceAsBytes(si), std.mem.sliceAsBytes(bi) }, 1..) |bytes, i|
            p.in[i] = try p.rig.input(bytes);
        p.work[0] = try p.rig.scratch(p.rows * 2 * p.n * 2); // gate|up rows, either layout
        p.work[1] = try p.rig.scratch(p.count() * 2); // the twin's activations
        p.work[2] = try p.rig.scratch(4);
        p.out = try p.rig.output(p.count() * 2);
        return true;
    }

    /// The row projection of x by the pack at in[at..at + 3] into work[0], then gemma_geglu (IL) into `act`.
    fn pass(p: *Probe, at: usize, il: bool, act: rig_.Ref) !void {
        const call = try sqb.call(&p.sq, 4, 2 * p.n, p.k, p.rows, 2 * p.n, false, 0);
        const ps = p.rig.begin();
        const e = ps.e;
        sqb.encode(e, call, .{ .w = p.in[at].buf, .w_off = p.in[at].off, .scales = p.in[at + 1].buf, .s_off = p.in[at + 1].off, .biases = p.in[at + 2].buf, .b_off = p.in[at + 2].off }, p.in[0].buf, p.in[0].off, p.work[0].buf, p.work[0].off);
        e.setPipeline(p.geglu);
        e.setBuffer(p.work[0].buf, p.work[0].off, 0);
        e.setBuffer(act.buf, act.off, 1);
        e.setValue(@as(u32, @intCast(p.n)), 2);
        e.setBuffer(p.work[2].buf, p.work[2].off, 3);
        e.setValue(@as(i32, 0), 4);
        e.setValue(@as(i32, @intFromBool(il)), 5);
        e.dispatchGroups(.{ .width = p.n / 64, .height = p.rows }, .{ .width = 64 });
        try p.rig.finish(ps);
    }

    pub fn run(p: *Probe) !void {
        try p.pass(1, p.v.control, p.out);
    }

    pub fn replaced(p: *Probe) ![]const u8 {
        try p.pass(4, true, p.work[1]);
        return p.rig.bytes(p.work[1], p.count() * 2);
    }

    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        const got = try p.a.dupe(u8, try p.rig.outputBytes());
        defer p.a.free(got);
        if (!std.mem.eql(u8, got, try p.replaced())) return "its bytes differ from the interleaved twin's";
        return null;
    }

    pub fn reference(_: *Probe, _: []f64, _: []f64) !void {
        return error.ReferenceIsTheTwin;
    }
};

const vt = g.hooks(Probe);

/// The row path's gate|up and its control, on every Mac (the tensor-unit path keeps the interleave).
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (variants, names) |v, name| {
            if (only) |o| if (std.mem.indexOf(u8, name, o) == null) continue;
            const p = try s.a.create(Probe);
            errdefer s.a.destroy(p);
            p.* = try Probe.init(s.a, s.device, s.queue, v, name);
            errdefer p.deinit();
            try s.probes.append(s.a, p);
            try s.kernels.append(s.a, .{ .spec = p.spec(), .ctx = p, .vt = &vt, .control = if (v.control) .shapes else null });
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
    }
};
