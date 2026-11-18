//! tf-glm-mla-norms: an MLA layer's cache writes in one launch against the kept kernels, and those kept; time.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const glm = tf.glm;
const R = glm.weights.Ref;
const Rows = glm.forward.Rows;

const c: glm.config.Config = .{};
const XP = 2240; // x_proj rows' widest stride: a prompt chunk's (2208 aligned to 64); decode's is 2208
const M = 300;
const layers = 11; // a round's MLA layers

const Side = struct { keys: mtl.Buffer, ik: mtl.Buffer, iw: mtl.Buffer };
const Rig = struct { xp: mtl.Buffer, w: glm.weights.Mla, out: [3]Side, refs: [3]mtl.Pipeline, k: *const glm.kernels.Kernels };

/// Side 0: the kept kernels; 1: glm_mla_norms; 2: rms, layer_norm and scale as they are now.
fn launch(e: mtl.ComputeEncoder, r: *const Rig, side: usize, rows: u32, stride: u32) void {
    const o = r.out[side];
    const xp = R{ .buf = r.xp };
    if (side == 1) return glm.forward.mlaNorms(r.k, &c, e, &r.w, xp, stride, .{ .buf = o.keys }, .{ .buf = o.ik }, .{ .buf = o.iw }, rows);
    const p = if (side == 0) r.refs else [3]mtl.Pipeline{ r.k.rms, r.k.layer_norm, r.k.scale };
    const K = c.kv_lora;
    e.setPipeline(p[0]);
    inline for (.{ xp.at(c.q_lora * 2), r.w.kv_norm, R{ .buf = o.keys } }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(Rows{ .rows = @intCast(rows), .width = K, .x_stride = @intCast(stride), .y_stride = K, .eps = c.eps }, 3);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(((K + 3) / 4 + 31) / 32 * 32, 1, 1));
    e.setPipeline(p[1]);
    inline for (.{ xp.at((c.q_lora + K) * 2), r.w.k_norm_w, r.w.k_norm_b, R{ .buf = o.ik } }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(Rows{ .rows = @intCast(rows), .width = c.i_dim, .x_stride = @intCast(stride), .y_stride = c.i_dim, .eps = 1e-6 }, 4);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(32, 1, 1));
    e.setPipeline(p[2]);
    inline for (.{ xp.at((c.q_lora + K + c.i_dim) * 2), R{ .buf = o.iw } }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(Rows{ .rows = @intCast(rows), .width = c.i_heads, .x_stride = @intCast(stride), .y_stride = c.i_heads, .eps = 1.0 / 64.0 }, 2);
    e.dispatchThreads(mtl.Size.of(c.i_heads, rows, 1), mtl.Size.of(c.i_heads, 1, 1));
}

fn run(q: mtl.Queue, r: *const Rig, side: usize, rows: u32, stride: u32, n: usize) !f64 {
    const cb = q.commandBuffer();
    const e = cb.compute(.serial);
    for (0..n) |_| launch(e, r, side, rows, stride);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("GPU failed: {s}\n", .{msg});
        return error.GpuFailed;
    }
    return cb.gpuSeconds() * 1e6 / @as(f64, @floatFromInt(n));
}

fn time(q: mtl.Queue, r: *const Rig, side: usize, rows: u32) !f64 {
    var t: [8]f64 = undefined;
    for (&t) |*v| v.* = try run(q, r, side, rows, 2208, 4 * layers);
    std.mem.sort(f64, t[1..], {}, std.sort.asc(f64));
    return t[4];
}

fn toBf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    const queue = try device.queue();
    const k = try glm.kernels.load(gpa, device);
    const lib = try mtl.Library.fromSource(device, try std.mem.concat(gpa, u8, &.{ sources.glm_glue, @embedFile("glm_mla_norms_ref.metal") }), mtl.CompileOptions.mlx());
    const opts = mtl.ResourceOptions.shared;
    var r: Rig = .{ .xp = try device.buffer(M * XP * 2, opts), .w = undefined, .out = undefined, .refs = undefined, .k = k };
    for (&r.refs, [_][:0]const u8{ "glm_rms_ref", "glm_layer_norm_ref", "glm_scale_ref" }) |*p, n| p.* = try mtl.Pipeline.init(device, lib, n, false);
    const wb = try device.buffer((c.kv_lora + 2 * c.i_dim) * 2, opts);
    r.w.kv_norm = .{ .buf = wb };
    r.w.k_norm_w = (R{ .buf = wb }).at(c.kv_lora * 2);
    r.w.k_norm_b = (R{ .buf = wb }).at((c.kv_lora + c.i_dim) * 2);
    for (&r.out) |*o| o.* = .{ .keys = try device.buffer(M * c.kv_lora * 2, opts), .ik = try device.buffer(M * c.i_dim * 2, opts), .iw = try device.buffer(M * c.i_heads * 2, opts) };
    var prng = std.Random.DefaultPrng.init(0x6d);
    const rnd = prng.random();
    std.debug.print("GPU {s}: an MLA layer's key norm, indexer key norm and indexer weights in one launch against three\n", .{device.name()});
    var checked: usize = 0;
    for (0..3) |d| { // draw 1: wider values with zeros, -0, large and tiny ones; draw 2: rows of one value, zero rows
        const span: f32 = if (d == 0) 2 else 64;
        for (r.xp.slice(u16, M * XP), 0..) |*v, i| v.* = if (d == 2 and (i / XP) % 3 == 0) toBf(@floatFromInt((i / XP) % 7)) else if (d == 1 and i % 61 == 0) 0x8000 else if (d == 1 and i % 67 == 0) 0 else if (d == 1 and i % 71 == 0) toBf(3e4) else if (d == 1 and i % 73 == 0) toBf(1e-30) else toBf((rnd.float(f32) - 0.5) * span);
        for (wb.slice(u16, c.kv_lora + 2 * c.i_dim), 0..) |*v, i| v.* = if (d == 1 and i % 29 == 0) 0x8000 else toBf((rnd.float(f32) - 0.3) * 2);
        for ([_]u32{ 2208, XP }) |stride| for ([_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 37, M }) |rows| {
            for (r.out) |o| for ([_]mtl.Buffer{ o.keys, o.ik, o.iw }, [_]usize{ c.kv_lora, c.i_dim, c.i_heads }) |b, w| @memset(b.slice(u16, M * w), 0x7fc1);
            for (0..3) |side| _ = try run(queue, &r, side, rows, stride, 1);
            for (1..3) |side| for ([_][2]mtl.Buffer{ .{ r.out[0].keys, r.out[side].keys }, .{ r.out[0].ik, r.out[side].ik }, .{ r.out[0].iw, r.out[side].iw } }, [_]usize{ c.kv_lora, c.i_dim, c.i_heads }) |pair, w| {
                if (!std.mem.eql(u16, pair[0].slice(u16, M * w), pair[1].slice(u16, M * w))) {
                    std.debug.print("draw {d}, {d} rows (stride {d}), side {d}: BITS DIFFER\n", .{ d, rows, stride, side });
                    return error.BitsDiffer;
                }
            };
            checked += rows * (c.kv_lora + c.i_dim + c.i_heads);
        };
    }
    for (0..64) |_| _ = try run(queue, &r, 0, 16, 2208, 4 * layers); // clocks up
    for ([_]u32{ 1, 2, 4, 8, 16 }) |rows| {
        const t0 = try time(queue, &r, 0, rows);
        const t1 = try time(queue, &r, 1, rows);
        std.debug.print("  {d:2} rows: three launches {d:5.2} us, one {d:5.2} us a layer\n", .{ rows, t0, t1 });
    }
    std.debug.print("{d} outputs equal (merged and the kernels as they are now) over three draws\n", .{checked});
}
