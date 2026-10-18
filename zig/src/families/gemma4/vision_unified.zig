//! Gemma 4 12B's image embedder on Metal: 48-px patches through LayerNorm, Linear and LayerNorm, positions, projection.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const frags = @import("../../core/frags.zig");
const ck = @import("../../core/checkpoint_metal.zig");
const lane = @import("../../core/lane_projection.zig");
const affine_mm = @import("affine_mm.zig");
const Mlx = @import("weights.zig").Mlx;
const Width = @import("widths.zig").Width;
const image = @import("image.zig");
const vision = @import("vision.zig");

pub const width = 3840; // mm_embed_dim and output_proj_dims
const table_rows = 1120; // mm_posemb_size: each axis's rows
const ln_eps: f32 = 1e-5; // nn.LayerNorm's default
const rms_eps: f32 = 1e-6;

pub const Ref = lane.Ref;
fn ref(t: ck.Tensor) Ref {
    return .{ .buf = t.buffer, .off = t.offset };
}

fn never(_: []const u8, _: usize) bool {
    return false;
}

/// The checkpoint's embedder tensors, packed 256-byte aligned beside the language model's (the checkpoint frees them).
pub fn addTensors(c: *ck.Checkpoint, device: mtl.Device, files: []const [:0]u8) !void {
    for (files) |path| for ([_][]const u8{ "vision_embedder.", "embed_vision." }) |p| try c.addFileSplit(device, path, "", p, never);
}

/// Whether the checkpoint has the embedder (tools/zig/gemma4_12b_convert.py keeps it).
pub fn present(c: *const ck.Checkpoint) bool {
    return c.has("vision_embedder.patch_dense.weight");
}

pub const Embedder = struct {
    gpa: std.mem.Allocator,
    lib: mtl.Library,
    gemm: [2][2]mtl.Pipeline, // bf16 a b^T: [M % 64 == 0][N % 128 == 0]
    norm: mtl.Pipeline,
    rms: mtl.Pipeline,
    rows: mtl.Pipeline,
    out_hidden: usize,
    ln1_w: Ref,
    ln1_b: Ref,
    dense_w: Ref,
    dense_b: Ref,
    ln2_w: Ref,
    ln2_b: Ref,
    table: Ref,
    pos_w: Ref,
    pos_b: Ref,
    proj: Ref,
    /// The projection quantized (an oQ pack's, at its config width) with the model's affine_mm pipelines for it.
    proj_q: ?struct { w: Mlx, mm: affine_mm.Pipes } = null,
    scratch: mtl.Buffer,
    a: Ref, // [T, merged_width]: patch_ln1's rows
    b: Ref, // [T, width]: patch_dense's, before its bias
    c: Ref, // [T, width]: patch_ln2's, then the RMSNorm's
    d: Ref, // [T, width]: pos_norm's
    /// Host-written per image: the merged patches as bf16.
    patches: mtl.Buffer,

    /// `mm` and `q`: the model's affine_mm pipelines and the config's width, for a quantized projection.
    pub fn load(gpa: std.mem.Allocator, device: mtl.Device, ckpt: *const ck.Checkpoint, out_hidden: usize, mm: affine_mm.Pipes, q: Width) !*Embedder {
        if (out_hidden != width) return error.UnexpectedGemmaVisionTensor;
        const v = try gpa.create(Embedder);
        errdefer gpa.destroy(v);
        v.* = undefined;
        v.proj_q = null;
        v.gpa = gpa;
        v.out_hidden = out_hidden;
        const body = try frags.source(device, gpa, ks.gemma4_vision);
        defer gpa.free(body);
        const src = try std.fmt.allocPrint(gpa, "#define G4V_OUT {d}\n{s}", .{ out_hidden, body });
        defer gpa.free(src);
        v.lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        errdefer v.lib.deinit();
        var made: usize = 0;
        errdefer for (0..made) |i| v.pipeline(i).deinit();
        for (0..2) |m| for (0..2) |n| {
            var name: [64]u8 = undefined;
            v.gemm[m][n] = try mtl.Pipeline.init(device, v.lib, try std.fmt.bufPrintSentinel(&name, "g4v_gemm_bf16_nt_{d}{d}", .{ m, n }, 0), false);
            made += 1;
        };
        for ([_]*mtl.Pipeline{ &v.norm, &v.rms, &v.rows }, [_][:0]const u8{ "g4u_norm", "g4v_rms", "g4v_rows" }) |p, n| {
            p.* = try mtl.Pipeline.init(device, v.lib, n, false);
            made += 1;
        }
        const t = struct {
            fn get(cp: *const ck.Checkpoint, name: []const u8, count: usize) !Ref {
                const x = try cp.get(name);
                if (x.dtype != .bf16 or x.count() != count) {
                    std.log.err("gemma4 vision: {s} is not bf16[{d}]", .{ name, count });
                    return error.UnexpectedGemmaVisionTensor;
                }
                return ref(x);
            }
        };
        const e = "vision_embedder.";
        const mw = image.merged_width;
        v.ln1_w = try t.get(ckpt, e ++ "patch_ln1.weight", mw);
        v.ln1_b = try t.get(ckpt, e ++ "patch_ln1.bias", mw);
        v.dense_w = try t.get(ckpt, e ++ "patch_dense.weight", width * mw);
        v.dense_b = try t.get(ckpt, e ++ "patch_dense.bias", width);
        v.ln2_w = try t.get(ckpt, e ++ "patch_ln2.weight", width);
        v.ln2_b = try t.get(ckpt, e ++ "patch_ln2.bias", width);
        v.table = try t.get(ckpt, e ++ "pos_embedding", table_rows * 2 * width);
        v.pos_w = try t.get(ckpt, e ++ "pos_norm.weight", width);
        v.pos_b = try t.get(ckpt, e ++ "pos_norm.bias", width);
        const pp = "embed_vision.embedding_projection.";
        if (ckpt.has(pp ++ "scales")) {
            const s = try ckpt.get(pp ++ "scales");
            const w = try ckpt.get(pp ++ "weight");
            const b = try ckpt.get(pp ++ "biases");
            const words = @as(usize, width) * q.bits / 32;
            const groups = width / q.group;
            const fits = w.dtype == .u32 and w.rank == 2 and w.shape[0] == out_hidden and w.shape[1] == words and
                w.count() == out_hidden * words and s.dtype == .bf16 and b.dtype == .bf16 and s.rank == 2 and b.rank == 2 and
                s.shape[0] == out_hidden and s.shape[1] == groups and b.shape[0] == out_hidden and b.shape[1] == groups and
                s.count() == out_hidden * groups and b.count() == s.count();
            if (!fits) {
                std.log.err("gemma4 vision: {s}* is not {d}-bit in groups of {d} [{d}, {d}]", .{ pp, q.bits, q.group, out_hidden, width });
                return error.UnexpectedGemmaVisionTensor;
            }
            v.proj_q = .{ .w = .{ .w = ref(w), .s = ref(s), .b = ref(b), .n = @intCast(out_hidden), .k = width, .bits = q.bits }, .mm = mm };
        } else v.proj = try t.get(ckpt, pp ++ "weight", out_hidden * width);
        const T: usize = image.max_budget;
        const sizes = [_]usize{ T * mw * 2, T * width * 2, T * width * 2, T * width * 2 };
        var total: usize = 0;
        for (sizes) |s| total += std.mem.alignForward(usize, s, 256);
        v.scratch = try device.buffer(total, mtl.ResourceOptions.private);
        errdefer v.scratch.deinit();
        var off: usize = 0;
        for ([_]*Ref{ &v.a, &v.b, &v.c, &v.d }, sizes) |f, s| {
            f.* = .{ .buf = v.scratch, .off = off };
            off += std.mem.alignForward(usize, s, 256);
        }
        v.patches = try device.buffer(T * mw * 2, mtl.ResourceOptions.shared);
        return v;
    }

    fn pipeline(v: *Embedder, i: usize) mtl.Pipeline {
        return switch (i) {
            0...3 => v.gemm[i / 2][i % 2],
            4 => v.norm,
            5 => v.rms,
            else => v.rows,
        };
    }

    pub fn deinit(v: *Embedder) void {
        for (0..7) |i| v.pipeline(i).deinit();
        v.lib.deinit();
        v.scratch.deinit();
        v.patches.deinit();
        v.gpa.destroy(v);
    }

    /// Merged patches (image.prepare's .merged) to `out` [gh * gw / 9, out_hidden]; `stop` 0: patch_ln2's in c, 1: d's.
    pub fn encode(v: *Embedder, e: mtl.ComputeEncoder, pixels: []const f32, gh: u32, gw: u32, out: Ref, stop: ?u32) !void {
        if (gh == 0 or gw == 0 or gh % image.pool != 0 or gw % image.pool != 0) return error.GemmaImageGrid;
        const T: u32 = gh * gw / (image.pool * image.pool);
        if (T > image.max_budget or pixels.len != @as(usize, T) * image.merged_width) return error.GemmaImageGrid;
        if (gw / image.pool > table_rows or gh / image.pool > table_rows) return error.GemmaImageGrid;
        for (pixels, v.patches.slice(u16, pixels.len)) |p, *d| d.* = vision.bf16(p);
        const mw: u32 = image.merged_width;
        v.layerNorm(e, .{ .buf = v.patches }, v.ln1_w, v.ln1_b, v.ln1_b, v.a, T, mw, 0, 0);
        v.linear(e, v.a, v.dense_w, v.b, T, width, mw);
        v.layerNorm(e, v.b, v.ln2_w, v.ln2_b, v.dense_b, v.c, T, width, 1, 0);
        if (stop) |s| if (s == 0) return;
        v.layerNorm(e, v.c, v.pos_w, v.pos_b, v.table, v.d, T, width, 2, gw / image.pool);
        if (stop) |s| if (s == 1) return;
        e.setPipeline(v.rms);
        for ([_]Ref{ v.d, v.d, v.c }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(extern struct { width: u32, weighted: u32, eps: f32 }{ .width = width, .weighted = 0, .eps = rms_eps }, 3);
        e.dispatchGroups(mtl.Size.of(T, 1, 1), mtl.Size.of(256, 1, 1));
        if (v.proj_q) |p| return affine_mm.dense(e, p.mm, v.c, p.w, out, T);
        v.linear(e, v.c, v.proj, out, T, @intCast(v.out_hidden), width);
    }

    /// Rows of an encoded image over `n` embedded prompt rows (`dst`, out_hidden bf16 a row), in the prompt's encoder.
    pub fn inject(v: *Embedder, e: mtl.ComputeEncoder, src: Ref, dst: Ref, n: u32) void {
        e.setPipeline(v.rows);
        e.setBuffer(src.buf, src.off, 0);
        e.setBuffer(dst.buf, dst.off, 1);
        e.dispatchThreads(mtl.Size.of(v.out_hidden / 8, n, 1), mtl.Size.of(64, 1, 1));
    }

    fn layerNorm(v: *Embedder, e: mtl.ComputeEncoder, x: Ref, w: Ref, b: Ref, extra: Ref, y: Ref, rows: u32, cols: u32, add: u32, gw: u32) void {
        e.setPipeline(v.norm);
        for ([_]Ref{ x, w, b, extra, y }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(extern struct { width: u32, add: u32, gw: u32, eps: f32 }{ .width = cols, .add = add, .gw = gw, .eps = ln_eps }, 5);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// y [M, N] = x [M, K] w^T in bf16 (w as stored, [N, K]): tfp::gemm_nax as the 31B's tower runs it.
    fn linear(v: *Embedder, e: mtl.ComputeEncoder, x: Ref, w: Ref, y: Ref, M: u32, N: u32, K: u32) void {
        const tn = (N + 127) / 128;
        const tm = (M + 63) / 64;
        e.setPipeline(v.gemm[@intFromBool(M % 64 == 0)][@intFromBool(N % 128 == 0)]);
        e.setBuffer(x.buf, x.off, 0);
        e.setBuffer(w.buf, w.off, 1);
        var p: [16]i32 = @splat(0);
        const vals = [_]usize{ M, N, K, K, K, N, tn, tm, 2, K / 256, 0, 0, 0, 0, 0 };
        for (vals, 0..) |x_, i| p[i] = @intCast(x_);
        e.setBytes(std.mem.asBytes(&p), 2);
        e.setBuffer(y.buf, y.off, 3);
        e.dispatchThreads(mtl.Size.of(@as(usize, tn << 2) * 32, (tm + 3) / 4 * 4, 2), mtl.Size.of(32, 4, 2));
    }
};

test "the embedder's sizes fit its kernels" {
    try std.testing.expectEqual(@as(usize, 6912), image.merged_width);
    try std.testing.expect(image.merged_width % 256 == 0 and width % 256 == 0 and width % 128 == 0);
}
