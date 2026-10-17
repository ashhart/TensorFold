//! Gemma 4 dense decode kernels against tools/zig/gemma4_dump.py's fixtures: each output's bits, by window width.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");

const lane = tf.lane_projection;
const g4 = tf.gemma4;
const npy = tf.npy;
const shared = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

const Ctx = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    device: mtl.Device,
    queue: mtl.Queue,
    ckpt: *tf.checkpoint_metal.Checkpoint,
    dump: []const u8,
    failed: usize = 0,
};

/// A dumped array's values as bf16 bits (the dump writes bf16 tensors as exact float32).
fn loadBf16(c: *Ctx, name: []const u8) ![]u16 {
    const path = try std.fmt.allocPrint(c.gpa, "{s}/{s}.npy", .{ c.dump, name });
    defer c.gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(c.io, path, c.gpa, .unlimited);
    defer c.gpa.free(bytes);
    const a = try npy.parse(bytes);
    if (!std.mem.eql(u8, a.descr, "<f4")) return error.UnexpectedDtype;
    const out = try c.gpa.alloc(u16, a.count());
    for (out, 0..) |*o, i| o.* = @truncate(std.mem.readInt(u32, a.data[i * 4 ..][0..4], .little) >> 16);
    return out;
}

/// The members' MLX affine tensors stacked along N, packed into one lane projection's layout.
const Stack = struct {
    proj: lane.Projection,
    w: mtl.Buffer,
    sb: mtl.Buffer,
    n: usize,
    k: usize,

    /// `decode`: the split the model decodes at (weights.splitK) rather than the reference's (referenceSplit).
    fn init(c: *Ctx, names: []const []const u8, precompute: bool, decode: bool) !Stack {
        const reg = reg_mode;
        var n: usize = 0;
        var k: usize = 0;
        for (names) |m| {
            const t = try c.ckpt.get(try key(c, m, "weight"));
            n += t.shape[0];
            k = t.shape[1] * 8;
        }
        const words = try c.gpa.alloc(u32, n * k / 8);
        defer c.gpa.free(words);
        const scales = try c.gpa.alloc(u16, n * k / 64);
        defer c.gpa.free(scales);
        const biases = try c.gpa.alloc(u16, n * k / 64);
        defer c.gpa.free(biases);
        var at: usize = 0;
        for (names) |m| {
            const w = (try c.ckpt.get(try key(c, m, "weight"))).host(u32);
            const s = (try c.ckpt.get(try key(c, m, "scales"))).host(u16);
            const b = (try c.ckpt.get(try key(c, m, "biases"))).host(u16);
            @memcpy(words[at * k / 8 ..][0..w.len], w);
            @memcpy(scales[at * k / 64 ..][0..s.len], s);
            @memcpy(biases[at * k / 64 ..][0..b.len], b);
            at += w.len / (k / 8);
        }
        const sk = if (decode) g4.weights.splitK(n, k) else g4.weights.referenceSplit(n, k);
        const l = lane.Layout{ .n = n, .k = k, .format = .{ .bits = 4, .group = 64 }, .sk = sk, .precompute_sums = precompute or reg, .reg = reg };
        const wb = try c.device.buffer(l.weightWords() * 4, shared);
        const sbb = try c.device.buffer(l.metadataElements() * 2, shared);
        try lane.pack(l, words, scales, biases, wb.slice(u32, l.weightWords()), sbb.slice(u16, l.metadataElements()));
        if (reg) try lane.regOrder(c.device, &.{.{ .w = .{ .buf = wb }, .n = n, .k = k }});
        return .{ .proj = try lane.Projection.init(c.gpa, c.device, l), .w = wb, .sb = sbb, .n = n, .k = k };
    }

    fn deinit(s: Stack) void {
        s.proj.deinit();
        s.w.deinit();
        s.sb.deinit();
    }
};

var key_buf: [256]u8 = undefined;
var reg_mode = false;
fn key(_: *Ctx, member: []const u8, part: []const u8) ![]const u8 {
    return std.fmt.bufPrint(&key_buf, "language_model.model.{s}.{s}", .{ member, part });
}

/// Runs `s` on the dumped input and counts outputs whose bits differ; a decode-split stack's are reported, not counted.
fn checkProjection(c: *Ctx, label: []const u8, s: Stack, rows: usize, in_name: []const u8, out_name: []const u8, decode: bool) !void {
    const x = try loadBf16(c, in_name);
    defer c.gpa.free(x);
    const want = try loadBf16(c, out_name);
    defer c.gpa.free(want);
    if (x.len != rows * s.k or want.len != rows * s.n) return error.FixtureShape;
    const xb = try c.device.buffer(x.len * 2, shared);
    defer xb.deinit();
    @memcpy(xb.slice(u16, x.len), x);
    const yb = try c.device.buffer(want.len * 2, shared);
    defer yb.deinit();
    const cb = c.queue.commandBuffer();
    const e = cb.compute(.serial);
    try s.proj.encode(e, .{ .buf = xb }, .{ .buf = s.w }, .{ .buf = s.sb }, .{ .buf = yb }, @intCast(rows));
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("{s}: GPU failure {s}\n", .{ label, msg });
        return error.GpuFailed;
    }
    const got = yb.slice(u16, want.len);
    var diff: usize = 0;
    var worst: f32 = 0;
    for (got, want) |g, w| if (g != w) {
        diff += 1;
        worst = @max(worst, @abs(@as(f32, @bitCast(@as(u32, g) << 16)) - @as(f32, @bitCast(@as(u32, w) << 16))));
    };
    if (decode) {
        std.debug.print("{s} rows {d} at the decode split: {d} of {d} outputs differ, worst |diff| {d}\n", .{ label, rows, diff, want.len, worst });
        return;
    }
    c.failed += diff;
    std.debug.print("{s} rows {d}: {d} of {d} outputs differ{s}\n", .{ label, rows, diff, want.len, if (diff == 0) "" else " (first at a column of row 0 or later)" });
}

/// One affine_mm product (layer 0's q_proj, from the checkpoint) on the dumped input against the dumped output.
fn checkAffine(c: *Ctx) !void {
    const text = try tf.gemma4.affine_mm.source(c.device, c.gpa, .{});
    defer c.gpa.free(text);
    const lib = try mtl.Library.fromSource(c.device, text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    var pipes: tf.gemma4.affine_mm.Pipes = undefined;
    for (&pipes, tf.gemma4.affine_mm.names) |*p, nm| p.* = try mtl.Pipeline.init(c.device, lib, nm, false);
    const x = try loadBf16(c, "r16.L0.x");
    defer c.gpa.free(x);
    const want = try loadBf16(c, "r16.L0.qkv");
    defer c.gpa.free(want);
    const xb = try c.device.buffer(x.len * 2, shared);
    defer xb.deinit();
    @memcpy(xb.slice(u16, x.len), x);
    const yb = try c.device.buffer(16 * 16384 * 2, shared);
    defer yb.deinit();
    const w = try c.ckpt.get("language_model.model.layers.0.self_attn.q_proj.weight");
    const sc = try c.ckpt.get("language_model.model.layers.0.self_attn.q_proj.scales");
    const bi = try c.ckpt.get("language_model.model.layers.0.self_attn.q_proj.biases");
    const Q = struct { w: lane.Ref, s: lane.Ref, b: lane.Ref, n: u32, k: u32 };
    const q = Q{ .w = .{ .buf = w.buffer, .off = w.offset }, .s = .{ .buf = sc.buffer, .off = sc.offset }, .b = .{ .buf = bi.buffer, .off = bi.offset }, .n = 8192, .k = 5376 };
    const cb = c.queue.commandBuffer();
    const e = cb.compute(.serial);
    tf.gemma4.affine_mm.denseBatch(e, pipes, lane.Ref{ .buf = xb }, q, lane.Ref{ .buf = yb }, 16, 1, .{ .x_row = 5376, .y_row = 16384 });
    e.end();
    cb.commit();
    cb.wait();
    const got = yb.slice(u16, 16 * 16384);
    var worst: f32 = 0;
    var exact: usize = 0;
    for (0..16) |r| for (0..8192) |j| {
        const a: f32 = @bitCast(@as(u32, got[r * 16384 + j]) << 16);
        const b: f32 = @bitCast(@as(u32, want[r * 16384 + j]) << 16);
        worst = @max(worst, @abs(a - b));
        exact += @intFromBool(got[r * 16384 + j] == want[r * 16384 + j]);
    };
    std.debug.print("affine_mm q_proj vs lane: {d} of {d} bit-equal, worst |diff| {d}; sample got {x} want {x}\n", .{ exact, 16 * 8192, worst, got[0], want[0] });
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer args.deinit();
    _ = args.next();
    const model = args.next() orelse return error.Usage;
    const dump = args.next() orelse return error.Usage;
    const flag = args.next();
    const precompute = if (flag) |a| std.mem.eql(u8, a, "--sums") else false;
    reg_mode = if (flag) |a| std.mem.eql(u8, a, "--reg") else false;
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var ckpt = tf.checkpoint_metal.Checkpoint.init(gpa);
    defer ckpt.deinit();
    const files = try tf.checkpoint_host.shardFiles(gpa, init.io, model);
    defer tf.checkpoint_host.freeShardFiles(gpa, files);
    for (files) |path| try ckpt.addFileSelected(device, path, "", "language_model.");
    var c = Ctx{ .gpa = gpa, .io = init.io, .device = device, .queue = queue, .ckpt = &ckpt, .dump = dump };
    std.debug.print("device {s}, precomputed sums {}, reg {}\n", .{ device.name(), precompute, reg_mode });
    if (flag) |f| if (std.mem.eql(u8, f, "--affine")) return checkAffine(&c);
    for ([_]usize{ 0, 5 }) |layer| {
        var names: [4][64]u8 = undefined;
        const q = try std.fmt.bufPrint(&names[0], "layers.{d}.self_attn.q_proj", .{layer});
        const k = try std.fmt.bufPrint(&names[1], "layers.{d}.self_attn.k_proj", .{layer});
        const v = try std.fmt.bufPrint(&names[2], "layers.{d}.self_attn.v_proj", .{layer});
        var mlp: [3][64]u8 = undefined;
        const gate = try std.fmt.bufPrint(&mlp[0], "layers.{d}.mlp.gate_proj", .{layer});
        const up = try std.fmt.bufPrint(&mlp[1], "layers.{d}.mlp.up_proj", .{layer});
        const o_name = try std.fmt.bufPrint(&names[3], "layers.{d}.self_attn.o_proj", .{layer});
        const down_name = try std.fmt.bufPrint(&mlp[2], "layers.{d}.mlp.down_proj", .{layer});
        for ([_]bool{ false, true }) |decode| {
            const qkv = try Stack.init(&c, if (layer % 6 == 5) &.{ q, k } else &.{ q, k, v }, precompute, decode);
            defer qkv.deinit();
            const o = try Stack.init(&c, &.{o_name}, precompute, decode);
            defer o.deinit();
            const gu = try Stack.init(&c, &.{ gate, up }, precompute, decode);
            defer gu.deinit();
            const down = try Stack.init(&c, &.{down_name}, precompute, decode);
            defer down.deinit();
            for ([_]usize{ 1, 2, 16 }) |rows| {
                if (decode and rows != 1) continue;
                var p: [6][48]u8 = undefined;
                const pre = try std.fmt.bufPrint(&p[0], "r{d}.L{d}", .{ rows, layer });
                try checkProjection(&c, "qkv", qkv, rows, try cat(&p[1], pre, ".x"), try cat(&p[2], pre, ".qkv"), decode);
                try checkProjection(&c, "o", o, rows, try cat(&p[1], pre, ".o_in"), try cat(&p[2], pre, ".o"), decode);
                try checkProjection(&c, "gate_up", gu, rows, try cat(&p[1], pre, ".n_mlp"), try cat(&p[2], pre, ".gate_up"), decode);
                try checkProjection(&c, "down", down, rows, try cat(&p[1], pre, ".act"), try cat(&p[2], pre, ".y"), decode);
            }
        }
        std.debug.print("layer {d} done\n", .{layer});
    }
    for ([_]bool{ false, true }) |decode| {
        const head = try Stack.init(&c, &.{"embed_tokens"}, precompute, decode);
        defer head.deinit();
        for ([_]usize{ 1, 2, 16 }) |rows| {
            if (decode and rows != 1) continue;
            var p: [3][48]u8 = undefined;
            const pre = try std.fmt.bufPrint(&p[0], "r{d}", .{rows});
            try checkProjection(&c, "head", head, rows, try cat(&p[1], pre, ".head_in"), try cat(&p[2], pre, ".head_raw"), decode);
        }
    }
    std.debug.print("{s}: {d} differing outputs\n", .{ if (c.failed == 0) "all bits equal" else "bits differ", c.failed });
    if (c.failed > 0) std.process.exit(1);
}

fn cat(buf: []u8, a: []const u8, b: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}{s}", .{ a, b });
}
