//! `draft <drafter dir> <target dir>`: the HIP DFlash2 drafter's config, its quantized projections and its load (GPU).

const std = @import("std");
const hip = @import("hip");
const qwen35 = @import("qwen3_5");

const dc = qwen35.draft_config;
const dw = qwen35.draft_weights;
const Projection = hip.quant.Projection;

/// A product's error may reach this share of the sum of its terms' magnitudes (fp32 accumulation over K).
const tolerance = 1e-5;

const Gpu = struct {
    gpa: std.mem.Allocator,
    d: *hip.Runtime,
    o: hip.ops.Ops,
    stream: *hip.Stream,
    kind: hip.ops.Kind,
};

fn bf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// Weight (r, c) of an affine-4 g64 projection: scale * code + bias, in fp64.
fn weight(q: *const dw.Quantized, r: usize, c: usize) f64 {
    const at = r * q.k + c;
    const code: f64 = @floatFromInt((q.words[at / 8] >> @intCast(4 * (at % 8))) & 15);
    const g = r * (q.k / 64) + c / 64;
    return code * bf16(q.scales[g]) + bf16(q.biases[g]);
}

/// `m` rows of x times `proj` on the device against the fp64 product of `q`: the worst error over its bound.
fn compare(g: Gpu, q: *const dw.Quantized, proj: Projection, x: []const f32, m: usize) !f64 {
    if (proj.n != q.n or proj.k != q.k or x.len != m * q.k) return error.ShapeMismatch;
    const xs = try g.gpa.alloc(u16, x.len);
    defer g.gpa.free(xs);
    for (xs, x) |*h, v| h.* = if (g.kind == .f16) @bitCast(@as(f16, @floatCast(v))) else hip.quant.convert.bf16(v);
    var xb = try hip.DeviceBuffer.fromHost(g.d, std.mem.sliceAsBytes(xs));
    defer xb.free();
    g.o.arena.reset();
    const y = try g.o.project(.{ .ptr = xb.base(), .kind = g.kind }, proj, m, true);
    try g.stream.synchronize();
    const got = try g.gpa.alloc(f32, m * q.n);
    defer g.gpa.free(got);
    try hip.runtime.check(g.d.api.hipMemcpyDtoH(@ptrCast(got.ptr), @ptrFromInt(y.ptr), got.len * 4));
    var worst: f64 = 0;
    for (0..m) |r| for (0..q.n) |c| {
        var sum: f64 = 0;
        var mag: f64 = 0;
        for (0..q.k) |i| {
            const xv: f64 = if (g.kind == .f16) @as(f16, @bitCast(xs[r * q.k + i])) else bf16(xs[r * q.k + i]);
            const t = weight(q, c, i) * xv;
            sum += t;
            mag += @abs(t);
        }
        const err = @abs(@as(f64, got[r * q.n + c]) - sum) / (tolerance * mag + 1e-6);
        if (!(err <= 1)) {
            const fmt = "row {d} output {d}: device {d}, fp64 {d} (|terms| {d})\n";
            std.debug.print(fmt, .{ r, c, got[r * q.n + c], sum, mag });
            return error.Mismatch;
        }
        worst = @max(worst, err);
    };
    return worst;
}

fn randomRows(gpa: std.mem.Allocator, rnd: std.Random, n: usize, scale: f32) ![]f32 {
    const out = try gpa.alloc(f32, n);
    for (out) |*v| v.* = (rnd.float(f32) - 0.5) * scale;
    return out;
}

/// A random bf16 (n, k) matrix quantized, uploaded through the HIP MLX format, projected over 1, 3 and 16 rows.
fn synthetic(g: Gpu, rnd: std.Random, n: usize, k: usize) !void {
    const w = try g.gpa.alloc(u16, n * k);
    defer g.gpa.free(w);
    for (w) |*h| h.* = hip.quant.convert.bf16((rnd.float(f32) - 0.5) * 0.1);
    var q = try dw.quantize(g.gpa, w, n, k);
    defer q.deinit(g.gpa);
    var buffers: std.ArrayList(hip.DeviceBuffer) = .empty;
    defer {
        for (buffers.items) |*b| b.free();
        buffers.deinit(g.gpa);
    }
    const up: hip.Upload = .{ .r = g.d, .buffers = &buffers, .gpa = g.gpa };
    const proj = try hip.quant.view(try hip.quant.upload(up.uploader(), .{ .mlx = q.host() }), false);
    for ([_]usize{ 1, 3, 16 }) |m| {
        const x = try randomRows(g.gpa, rnd, m * k, 2);
        defer g.gpa.free(x);
        const worst = try compare(g, &q, proj, x, m);
        const fmt = "PASS draft project synthetic ({d}, {d}) x {d} rows: worst {e:.2} of the bound\n";
        std.debug.print(fmt, .{ n, k, m, worst });
    }
}

/// A loaded projection against the checkpoint's tensor quantized again on the host, over 1 and 4 rows.
fn real(g: Gpu, ck: *dw.Checkpoint, rnd: std.Random, name: []const u8, proj: Projection) !void {
    const t = try ck.expect(name, .bf16, &.{ proj.n, proj.k });
    var q = try dw.quantize(g.gpa, std.mem.bytesAsSlice(u16, t.bytes), proj.n, proj.k);
    defer q.deinit(g.gpa);
    for ([_]usize{ 1, 4 }) |m| {
        const x = try randomRows(g.gpa, rnd, m * proj.k, 2);
        defer g.gpa.free(x);
        const worst = try compare(g, &q, proj, x, m);
        const fmt = "PASS draft project {s} ({d}, {d}) x {d} rows: worst {e:.2} of the bound\n";
        std.debug.print(fmt, .{ name, proj.n, proj.k, m, worst });
    }
}

fn seconds(io: std.Io, t0: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0.toNanoseconds())) / 1e9;
}

pub fn run(gpa: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    if (args.len != 2) {
        std.debug.print("usage: tf-qwen35-test draft <drafter dir> <target dir>\n", .{});
        return error.Usage;
    }
    var target = try qwen35.Config.read(gpa, io, args[1]);
    defer target.deinit();
    const c = try dc.Config.read(gpa, io, args[0], dc.Target.of(target.spec));
    const fmt = "PASS draft config: {d} layers, taps {any}, window {d} (ring {d}), block {d}, rank {d}, top-k {d}\n";
    std.debug.print(fmt, .{ c.layers, c.tapLayers(), c.window, dc.tap_window, c.block, c.rank, c.topk });
    var d = try hip.Runtime.open();
    defer d.close();
    var ctx = try hip.Context.init(&d, 0);
    defer ctx.deinit();
    const caps = try hip.Device.kernelCaps(&d, 0);
    var notes: hip.Policy.Notes = .{};
    var lib = try hip.Launcher.load(&d, (try hip.Policy.resolve("", .current, &notes)).choices(), hip.kernels.images);
    defer lib.unload();
    var stream = try hip.Stream.init(&d);
    defer stream.deinit();
    var arena = try hip.Arena.init(&d, 16 << 20);
    defer arena.deinit();
    const o: hip.ops.Ops = .{ .l = &lib, .bf16 = caps.act == .bf16, .stream = stream.handle, .arena = &arena };
    const g: Gpu = .{ .gpa = gpa, .d = &d, .o = o, .stream = &stream, .kind = if (caps.act == .f16) .f16 else .bf16 };
    var prng = std.Random.DefaultPrng.init(0xDF2);
    const rnd = prng.random();
    try synthetic(g, rnd, 200, 1024);
    try synthetic(g, rnd, 1280, 5120);
    const t0 = std.Io.Clock.awake.now(io);
    var w = try dw.load(gpa, io, &d, args[0], dw.Shape.of(c));
    defer w.deinit();
    const dt = seconds(io, t0);
    const bytes = w.deviceBytes();
    const gib = @as(f64, @floatFromInt(bytes)) / (1 << 30);
    const loaded = "PASS draft load: {d} device buffers, {d:.3} GiB uploaded in {d:.1} s ({d:.2} GiB/s)\n";
    std.debug.print(loaded, .{ w.buffers.items.len, gib, dt, gib / dt });
    var ck = try dw.Checkpoint.openModel(gpa, io, args[0]);
    defer ck.close();
    try real(g, &ck, rnd, "candidate_selector.hidden_projection.weight", w.selector);
    try real(g, &ck, rnd, "layers.0.self_attn.q_proj.weight", w.layers[0].q);
    const last = try std.fmt.allocPrint(gpa, "layers.{d}.mlp.down_proj.weight", .{w.layers.len - 1});
    defer gpa.free(last);
    try real(g, &ck, rnd, last, w.layers[w.layers.len - 1].down);
    try real(g, &ck, rnd, "fc.weight", w.fc);
}
