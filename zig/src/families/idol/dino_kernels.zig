//! Idol's DINOv3 pipelines (idol_dino.metal): bf16 projections, LayerNorms, RoPE heads, attention, the steps between.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("../../core/frags.zig");
const cfg = @import("dino_config.zig");
pub const Ref = @import("../../core/qwen_vl/weights.zig").Ref;

/// Each pipeline by its function.
pub const names = [_][]const u8{ "dn_linear_0", "dn_linear_1", "dn_norm", "dn_final", "dn_heads", "dn_scale_add", "dn_swiglu", "dn_attention" };

pub const Kernels = struct {
    own: mtl.Library,
    p: [names.len]mtl.Pipeline,

    pub fn load(gpa: std.mem.Allocator, device: mtl.Device) !Kernels {
        return loadSource(gpa, device, sources.idol_dino);
    }

    /// The pipelines from `text` (the contract's controls pass an edited source).
    pub fn loadSource(gpa: std.mem.Allocator, device: mtl.Device, text: []const u8) !Kernels {
        var k: Kernels = undefined;
        const own = try frags.source(device, gpa, text);
        defer gpa.free(own);
        k.own = try mtl.Library.fromSource(device, own, mtl.CompileOptions.mlx());
        errdefer k.own.deinit();
        var made: usize = 0;
        errdefer for (k.p[0..made]) |p| p.deinit();
        for (names, 0..) |name, i| {
            k.p[i] = try mtl.Pipeline.init(device, k.own, name, false);
            made += 1;
        }
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        for (k.p) |p| p.deinit();
        k.own.deinit();
    }

    pub fn get(k: *const Kernels, comptime name: []const u8) mtl.Pipeline {
        inline for (names, 0..) |n, i| if (comptime std.mem.eql(u8, n, name)) return k.p[i];
        @compileError("no pipeline " ++ name);
    }

    /// y [M, N] (rows `ldd` apart) = bf16(x [M, K] w^T + b), w as stored [N, K]; N % 128 = 0, K % 32 = 0.
    pub fn linear(k: *const Kernels, e: mtl.ComputeEncoder, x: Ref, w: Ref, b: Ref, y: Ref, M: usize, N: usize, K: usize, ldd: usize) void {
        std.debug.assert(N % 128 == 0 and K % 32 == 0);
        const tn = N / 128;
        const tm = (M + 63) / 64;
        e.setPipeline(if (M % 64 == 0) k.get("dn_linear_1") else k.get("dn_linear_0"));
        e.setBuffer(x.buf, x.off, 0);
        e.setBuffer(w.buf, w.off, 1);
        var p: [16]i32 = @splat(0);
        const vals = [_]usize{ M, N, K, K, K, ldd, tn, tm, 0, K / 256 };
        for (vals, 0..) |v, i| p[i] = @intCast(v);
        e.setBytes(std.mem.asBytes(&p), 2);
        e.setBuffer(y.buf, y.off, 3);
        e.setBuffer(b.buf, b.off, 4);
        e.dispatchGroups(mtl.Size.of(tn, tm, 1), mtl.Size.of(256, 1, 1));
    }

    /// y = nn.LayerNorm of x's bf16 rows (fp32 statistics), bf16(n w + b) with bf16 w, b.
    pub fn norm(k: *const Kernels, e: mtl.ComputeEncoder, x: Ref, w: Ref, b: Ref, y: Ref, rows: usize) void {
        const params = extern struct { width: u32, eps: f32 }{ .width = cfg.width, .eps = cfg.eps };
        bind(e, k.get("dn_norm"), &.{ x, w, b, y }, params);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// h = the last norm of x's rows (bf16), y = F.layer_norm of h in fp32, no affine: ImageEncoder's feature1.
    pub fn final(k: *const Kernels, e: mtl.ComputeEncoder, x: Ref, w: Ref, b: Ref, h: Ref, y: Ref, rows: usize) void {
        const params = extern struct { width: u32, eps: f32 }{ .width = cfg.width, .eps = cfg.eps };
        bind(e, k.get("dn_final"), &.{ x, w, b, h, y }, params);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// q, k, v heads [20, rows, 64] from qkv rows [rows, 3840]; past the prefix, q and k turned by cos, sin.
    pub fn heads(k: *const Kernels, e: mtl.ComputeEncoder, qkv: Ref, cos: Ref, sin: Ref, q: Ref, kk: Ref, v: Ref, rows: usize) void {
        const params = extern struct { rows: u32, prefix: u32 }{ .rows = @intCast(rows), .prefix = cfg.prefix };
        bind(e, k.get("dn_heads"), &.{ qkv, cos, sin, q, kk, v }, params);
        e.dispatchThreads(mtl.Size.of(cfg.heads * 32, rows, 1), mtl.Size.of(128, 1, 1));
    }

    /// x [rows, 1280] = bf16(x + bf16(y g)), in place: LayerScale then the residual.
    pub fn scaleAdd(k: *const Kernels, e: mtl.ComputeEncoder, x: Ref, y: Ref, g: Ref, rows: usize) void {
        bind(e, k.get("dn_scale_add"), &.{ x, y, g }, @as(u32, cfg.width));
        e.dispatchThreads(mtl.Size.of(cfg.width, rows, 1), mtl.Size.of(256, 1, 1));
    }

    /// h [rows, 5120] = bf16(bf16(silu(gate)) up) from gu rows [gate | up].
    pub fn swiglu(k: *const Kernels, e: mtl.ComputeEncoder, gu: Ref, h: Ref, rows: usize) void {
        bind(e, k.get("dn_swiglu"), &.{ gu, h }, @as(u32, cfg.ffn));
        e.dispatchThreads(mtl.Size.of(cfg.ffn, rows, 1), mtl.Size.of(256, 1, 1));
    }

    /// o rows [lq, heads * 64] = softmax(q k^T / 8) v for q [heads, lq, 64] and k, v [heads, lk, 64], bf16.
    pub fn attention(k: *const Kernels, e: mtl.ComputeEncoder, q: Ref, kb: Ref, vb: Ref, o: Ref, heads_: usize, lq: usize, lk: usize) void {
        const nq = (lq + 63) / 64;
        const nk = (lk + 31) / 32;
        e.setPipeline(k.get("dn_attention"));
        for ([_]Ref{ q, kb, vb }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        // heads, q rows, k rows, k blocks, last q block, last k block, rows in the last q block and in the last k block
        var params: [8]i32 = .{ @intCast(heads_), @intCast(lq), @intCast(lk), @intCast(nk), @intCast(nq - 1), @intCast(nk - 1), @intCast(lq - (nq - 1) * 64), @intCast(lk - (nk - 1) * 32) };
        e.setBytes(std.mem.asBytes(&params), 3);
        var f: [4]f32 = .{ 1.0 / @sqrt(@as(f32, cfg.head_dim)), 0, 0, 0 };
        e.setBytes(std.mem.asBytes(&f), 4);
        e.setBuffer(o.buf, o.off, 5);
        e.dispatchGroups(mtl.Size.of(nq, heads_, 1), mtl.Size.of(32, 4, 1));
    }
};

/// One kernel's buffers in order, then its parameters at the next index.
pub fn bind(e: mtl.ComputeEncoder, p: mtl.Pipeline, refs: []const Ref, params: anytype) void {
    e.setPipeline(p);
    for (refs, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(params, refs.len);
}

test "every Idol DINOv3 pipeline compiles and loads on this GPU" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    if (!device.tensorUnits()) return error.SkipZigTest;
    var k = try Kernels.load(std.testing.allocator, device);
    k.deinit();
}
