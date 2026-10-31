//! The decoders' own Metal pipelines (idol_decoder.metal); their projections, norms and attention are the denoiser's.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("../../core/frags.zig");
const ks = @import("kernels.zig");
const dc = @import("decoder_config.zig");
const Ref = ks.Ref;
const bind = ks.bind;

pub const names = [_][]const u8{ "dec_embed", "dec_half", "dec_kv", "dec_heads", "dec_attention_few", "dec_attention_00", "dec_attention_01", "dec_attention_10", "dec_attention_11" };
const attention_first = 5;

/// Query rows the fp32 attention takes: torch on MPS runs its fp32 vector kernel up to 8, its fp16 graph past them.
pub const few_rows = 8;

comptime {
    std.debug.assert(std.mem.eql(u8, names[attention_first], "dec_attention_00"));
}

pub const Kernels = struct {
    own: mtl.Library,
    p: [names.len]mtl.Pipeline,

    pub fn load(gpa: std.mem.Allocator, device: mtl.Device) !Kernels {
        var k: Kernels = undefined;
        const own = try frags.source(device, gpa, sources.idol_decoder);
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

    fn get(k: *const Kernels, comptime name: []const u8) mtl.Pipeline {
        inline for (names, 0..) |n, i| if (comptime std.mem.eql(u8, n, name)) return k.p[i];
        @compileError("no pipeline " ++ name);
    }

    /// h [rows, 1024] fp32 = in_proj(p) + the points' sinusoidal embedding, p [rows, 3] fp32.
    pub fn embed(k: *const Kernels, e: mtl.ComputeEncoder, p: Ref, w: Ref, b: Ref, freqs: Ref, h: Ref, rows: usize) void {
        buffers(e, k.get("dec_embed"), &.{ p, w, b, freqs, h });
        e.dispatchThreads(mtl.Size.of(dc.hidden, rows, 1), mtl.Size.of(256, 1, 1));
    }

    /// o [rows, width] fp16 = y + b by column.
    pub fn toHalf(k: *const Kernels, e: mtl.ComputeEncoder, y: Ref, b: Ref, o: Ref, rows: usize, width: usize) void {
        bind(e, k.get("dec_half"), &.{ y, b, o }, @as(u32, @intCast(width)));
        e.dispatchThreads(mtl.Size.of(width, rows, 1), mtl.Size.of(256, 1, 1));
    }

    /// kv [8192, 2048] fp16 = to_kv of the fp16 latent rows.
    pub fn kv(k: *const Kernels, e: mtl.ComputeEncoder, ctx: Ref, w: Ref, b: Ref, out: Ref) void {
        buffers(e, k.get("dec_kv"), &.{ ctx, w, b, out });
        e.dispatchThreads(mtl.Size.of(2 * dc.hidden, dc.context_rows, 1), mtl.Size.of(256, 1, 1));
    }

    /// dst [16, rows, 64] from columns `offset` on of `src` rows `stride` apart, RMS-normed by `gamma` when given.
    pub fn heads(k: *const Kernels, e: mtl.ComputeEncoder, src: Ref, gamma: ?Ref, dst: Ref, rows: usize, stride: usize, offset: usize) void {
        const P = extern struct { rows: u32, stride: u32, offset: u32, norm: u32 };
        const params: P = .{ .rows = @intCast(rows), .stride = @intCast(stride), .offset = @intCast(offset), .norm = @intFromBool(gamma != null) };
        bind(e, k.get("dec_heads"), &.{ src, gamma orelse src, dst }, params);
        e.dispatchGroups(mtl.Size.of(rows, 2, 1), mtl.Size.of(256, 1, 1));
    }

    /// o rows [lq, 1024] = softmax(q k^T / 8) v for q [16, lq, 64], k, v [16, lk, 64]; P V in two fp16 halves of P.
    pub fn attention(k: *const Kernels, e: mtl.ComputeEncoder, q: Ref, kb: Ref, vb: Ref, o: Ref, lq: usize, lk: usize) void {
        const form = @as(usize, @intFromBool(lq % 64 == 0)) * 2 + @intFromBool(lk % 32 == 0);
        const nq = (lq + 63) / 64;
        const nk = (lk + 31) / 32;
        e.setPipeline(k.p[attention_first + form]);
        for ([_]Ref{ q, kb, vb }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        // heads, q rows, k rows, k blocks, last q block, last k block, rows in the last q block and in the last k block
        var params: [8]i32 = .{ dc.heads, @intCast(lq), @intCast(lk), @intCast(nk), @intCast(nq - 1), @intCast(nk - 1), @intCast(lq - (nq - 1) * 64), @intCast(lk - (nk - 1) * 32) };
        e.setBytes(std.mem.asBytes(&params), 3);
        var f: [4]f32 = .{ 0.125, 0, 0, 0 };
        e.setBytes(std.mem.asBytes(&f), 4);
        e.setBuffer(o.buf, o.off, 5);
        e.dispatchGroups(mtl.Size.of(nq, dc.heads, 1), mtl.Size.of(32, 4, 1));
    }

    /// o rows [lq, 1024] = softmax(q k^T / 8) v in fp32 for lq <= 8 query rows and up to 8,192 keys.
    pub fn attentionFew(k: *const Kernels, e: mtl.ComputeEncoder, q: Ref, kb: Ref, vb: Ref, o: Ref, lq: usize, lk: usize) void {
        std.debug.assert(lq <= few_rows and lk <= 32 * 256);
        bind(e, k.get("dec_attention_few"), &.{ q, kb, vb, o }, [2]u32{ @intCast(lq), @intCast(lk) });
        e.dispatchGroups(mtl.Size.of(lq, dc.heads, 1), mtl.Size.of(256, 1, 1));
    }
};

/// One kernel's buffers in order, no parameters.
fn buffers(e: mtl.ComputeEncoder, p: mtl.Pipeline, refs: []const Ref) void {
    e.setPipeline(p);
    for (refs, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
}

/// y [M, N] (rows N apart) = x [M, K] w^T in fp32 FMA by the denoiser's id_gemm32; any N % 8 = 0 (its last tile).
pub fn gemm32(k: *const ks.Kernels, e: mtl.ComputeEncoder, x: Ref, w: Ref, y: Ref, M: usize, N: usize, K: usize) void {
    std.debug.assert(K % 32 == 0 and N % 8 == 0);
    const params = [4]u32{ @intCast(M), @intCast(N), @intCast(K), @intCast(N) };
    bind(e, k.get("id_gemm32"), &.{ x, w, y }, params);
    e.dispatchGroups(mtl.Size.of((N + 63) / 64, (M + 63) / 64, 1), mtl.Size.of(128, 1, 1));
}

test "every decoder pipeline compiles and loads on this GPU" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    if (!device.tensorUnits()) return error.SkipZigTest;
    var k = try Kernels.load(std.testing.allocator, device);
    k.deinit();
}
