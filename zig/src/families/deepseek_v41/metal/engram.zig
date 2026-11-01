//! DeepSeek V4.1 Engram on Metal: dequantise gathered FP8 rows, act_quant, wkv by fp8_rows, gate, add into the streams.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const fp8 = @import("fp8_rows.zig");

pub const head = 256; // engram_head_dim: codes a row, one E8M0 scale per 32
pub const hc = 4; // hc_mult
pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };

/// rows tokens, cols hash columns (K = cols * 256), n wkv outputs ((hc + 1) * dim), dim hidden, eps the norm epsilon.
pub const Call = struct { rows: u32, cols: u32, n: u32, dim: u32, eps: f32 = 1e-6 };

const Args = extern struct { R: u32, cols: u32, dim: u32, eps: f32, dscale: f32 };

pub const Pipelines = struct {
    embed: mtl.Pipeline,
    aq: mtl.Pipeline,
    gate: mtl.Pipeline,
    add: mtl.Pipeline,
    fp8: fp8.Pipelines,

    pub fn load(device: mtl.Device) !Pipelines {
        return loadSource(device, sources.dsv41.engram);
    }

    /// The same kernels from other text: the contract's planted controls prefix a TF_CTL define.
    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var p: Pipelines = undefined;
        p.embed = try mtl.Pipeline.init(device, lib, "tf_dsv41_engram_embed", false);
        p.aq = try mtl.Pipeline.init(device, lib, "tf_dsv41_engram_aq", false);
        p.gate = try mtl.Pipeline.init(device, lib, "tf_dsv41_engram_gate", false);
        p.add = try mtl.Pipeline.init(device, lib, "tf_dsv41_engram_add", false);
        p.fp8 = try fp8.Pipelines.load(device);
        return p;
    }

    pub fn deinit(p: *Pipelines) void {
        p.embed.deinit();
        p.aq.deinit();
        p.gate.deinit();
        p.add.deinit();
        p.fp8.deinit();
    }
};

pub fn check(c: Call) !void {
    if (c.rows == 0 or c.cols == 0 or c.n == 0 or c.n % 32 != 0 or c.dim == 0 or c.dim % 32 != 0) return error.BadEngramShape;
}

/// float(dim ^ -0.5) as torch scales a float32 tensor by the Python float.
pub fn dscale(dim: u32) f32 {
    return @floatCast(std.math.pow(f64, @floatFromInt(dim), -0.5));
}

fn args(c: Call) Args {
    return .{ .R = c.rows, .cols = c.cols, .dim = c.dim, .eps = c.eps, .dscale = dscale(c.dim) };
}

/// codes [rows][cols][256] E4M3, scales [rows][cols][8] E8M0 -> emb [rows][cols * 256] bf16.
pub fn encodeEmbed(p: *const Pipelines, e: mtl.ComputeEncoder, codes: Ref, scales: Ref, emb: Ref, c: Call) !void {
    try check(c);
    e.setPipeline(p.embed);
    e.setBuffer(codes.buf, codes.off, 0);
    e.setBuffer(scales.buf, scales.off, 1);
    e.setBuffer(emb.buf, emb.off, 2);
    e.setValue(args(c), 3);
    e.dispatchThreads(mtl.Size.of(c.cols * head, c.rows, 1), mtl.Size.of(32, 1, 1));
}

/// emb [rows][K] bf16 -> y [rows][K] bf16 as act_quant (32-value blocks, E8M0 scale) leaves it; y may be emb.
pub fn encodeAq(p: *const Pipelines, e: mtl.ComputeEncoder, emb: Ref, y: Ref, c: Call) !void {
    try check(c);
    e.setPipeline(p.aq);
    e.setBuffer(emb.buf, emb.off, 0);
    e.setBuffer(y.buf, y.off, 1);
    e.setValue(args(c), 2);
    e.dispatchThreads(mtl.Size.of(c.cols * 8, c.rows, 1), mtl.Size.of(32, 1, 1));
}

/// xq [rows][K] (after `encodeAq`), W [n][K] E4M3, S [n/32][K/32] E8M0 -> kv [rows][n] bf16 by fp8_rows.
pub fn encodeWkv(p: *const Pipelines, e: mtl.ComputeEncoder, xq: Ref, w: Ref, s: Ref, scratch: Ref, kv: Ref, c: Call) !void {
    try check(c);
    const k: usize = @as(usize, c.cols) * head;
    var done: u32 = 0;
    while (done < c.rows) {
        const rows = @min(fp8.max_rows, c.rows - done);
        const call: fp8.Call = .{ .k = @intCast(k), .n = c.n, .rows = rows, .split = fp8.splitOf(@intCast(k)), .out = .bf16 };
        const at = struct {
            fn f(r: Ref, extra: usize) fp8.Ref {
                return .{ .buf = r.buf, .off = r.off + extra };
            }
        }.f;
        _ = scratch;
        try fp8.encodeWhole(&p.fp8, e, at(xq, 2 * done * k), at(w, 0), at(s, 0), at(kv, 2 * @as(usize, done) * c.n), call, .{});
        done += rows;
    }
}

/// Scratch bytes `encodeWkv` needs: the fp8 roots of a 16-row chunk.
pub fn fp8Scratch(c: Call) usize {
    return fp8.scratchBytes(.{ .k = c.cols * head, .n = c.n, .rows = @min(fp8.max_rows, c.rows) });
}

/// x [rows][hc][dim], kv [rows][hc + 1][dim], q and k [hc][dim] (all bf16; weight = q * k), mask [rows] u8 -> gate f32.
pub fn encodeGate(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, kv: Ref, q: Ref, k: Ref, mask: Ref, gate: Ref, c: Call) !void {
    try check(c);
    e.setPipeline(p.gate);
    e.setBuffer(x.buf, x.off, 0);
    e.setBuffer(kv.buf, kv.off, 1);
    e.setBuffer(q.buf, q.off, 2);
    e.setBuffer(k.buf, k.off, 3);
    e.setBuffer(mask.buf, mask.off, 4);
    e.setBuffer(gate.buf, gate.off, 5);
    e.setValue(args(c), 6);
    e.dispatchGroups(mtl.Size.of(c.rows * hc, 1, 1), mtl.Size.of(32, 1, 1));
}

/// y = bf16(x + gate * value); y may be x (each element is read, then written, by one thread).
pub fn encodeAdd(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, kv: Ref, gate: Ref, y: Ref, c: Call) !void {
    try check(c);
    e.setPipeline(p.add);
    e.setBuffer(x.buf, x.off, 0);
    e.setBuffer(kv.buf, kv.off, 1);
    e.setBuffer(gate.buf, gate.off, 2);
    e.setBuffer(y.buf, y.off, 3);
    e.setValue(args(c), 4);
    e.dispatchThreads(mtl.Size.of(hc * c.dim, c.rows, 1), mtl.Size.of(32, 1, 1));
}

fn align16(n: usize) usize {
    return (n + 15) & ~@as(usize, 15);
}

fn embBytes(c: Call) usize {
    return align16(2 * @as(usize, c.rows) * c.cols * head);
}

fn kvBytes(c: Call) usize {
    return align16(2 * @as(usize, c.rows) * c.n);
}

fn gateBytes(c: Call) usize {
    return align16(4 * @as(usize, c.rows) * hc);
}

/// Bytes of scratch a whole call needs: emb (act-quantised in place), kv, the gates, then the fp8 roots.
pub fn scratchBytes(c: Call) usize {
    return embBytes(c) + kvBytes(c) + gateBytes(c) + fp8Scratch(c);
}

/// The whole Engram step for `rows` tokens, five dispatches per 16 rows' wkv: y = x + the n-gram lookup.
pub fn encode(p: *const Pipelines, e: mtl.ComputeEncoder, codes: Ref, scales: Ref, w: Ref, ws: Ref, x: Ref, q: Ref, k: Ref, mask: Ref, scratch: Ref, y: Ref, c: Call) !void {
    const emb = Ref{ .buf = scratch.buf, .off = scratch.off };
    const kv = Ref{ .buf = scratch.buf, .off = scratch.off + embBytes(c) };
    const gate = Ref{ .buf = scratch.buf, .off = kv.off + kvBytes(c) };
    const roots = Ref{ .buf = scratch.buf, .off = gate.off + gateBytes(c) };
    try encodeEmbed(p, e, codes, scales, emb, c);
    try encodeAq(p, e, emb, emb, c);
    try encodeWkv(p, e, emb, w, ws, roots, kv, c);
    try encodeGate(p, e, x, kv, q, k, mask, gate, c);
    try encodeAdd(p, e, x, kv, gate, y, c);
}
