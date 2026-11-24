//! The prompt path's Metal pipelines (h3_prompt.metal, compiled at load, any Apple GPU) and their dispatches.
const std = @import("std");
const mtl = @import("metal");
const src = @import("h3_prompt_kernels");
const Ref = @import("tensors.zig").Ref;

/// How a norm rounds: as the Qwen text model (w times the bf16 normed value) or as H3's blocks (x r w at once).
pub const Mode = enum(u32) { qwen = 0, h3 = 1 };

/// Where a projection's heads sit: rows `row` values apart, heads `head` apart (128 alone, 384 in a fused q k v row).
pub const Heads = struct { count: usize, row: usize, head: usize };

pub const Kernels = struct {
    lib: mtl.Library,
    p: [src.names.len]mtl.Pipeline,

    pub fn load(device: mtl.Device) !Kernels {
        var k: Kernels = undefined;
        k.lib = try mtl.Library.fromSource(device, src.source, mtl.CompileOptions.mlx());
        errdefer k.lib.deinit();
        var made: usize = 0;
        errdefer for (k.p[0..made]) |p| p.deinit();
        for (src.names, 0..) |name, i| {
            k.p[i] = try mtl.Pipeline.init(device, k.lib, name, false);
            made += 1;
        }
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        for (k.p) |p| p.deinit();
        k.lib.deinit();
    }

    fn get(k: *const Kernels, comptime name: []const u8) mtl.Pipeline {
        inline for (src.names, 0..) |n, i| if (comptime std.mem.eql(u8, n, name)) return k.p[i];
        @compileError("no pipeline " ++ name);
    }

    fn bind(k: *const Kernels, e: mtl.ComputeEncoder, comptime name: []const u8, refs: []const Ref, params: anytype) void {
        e.setPipeline(k.get(name));
        for (refs, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
        e.setValue(params, refs.len);
    }

    /// y [M, N] = bf16(x [M, K] w [N, K]^T (+ b)) with fp32 sums; N a multiple of 64 and K of 32.
    pub fn gemm(k: *const Kernels, e: mtl.ComputeEncoder, x: Ref, w: Ref, b: ?Ref, y: Ref, M: usize, N: usize, K: usize) void {
        std.debug.assert(N % 64 == 0 and K % 32 == 0 and M > 0);
        const params = extern struct { M: u32, N: u32, K: u32, bias: u32 }{ .M = @intCast(M), .N = @intCast(N), .K = @intCast(K), .bias = @intFromBool(b != null) };
        k.bind(e, "hp_gemm", &.{ x, w, b orelse w, y }, params);
        e.dispatchGroups(mtl.Size.of(N / 64, (M + 63) / 64, 1), mtl.Size.of(128, 1, 1));
    }

    /// y [rows, width] = RMSNorm(x) by w, rounded as `mode`.
    pub fn norm(k: *const Kernels, e: mtl.ComputeEncoder, x: Ref, w: Ref, y: Ref, rows: usize, width: usize, eps: f32, mode: Mode) void {
        const params = extern struct { width: u32, mode: u32, eps: f32 }{ .width = @intCast(width), .mode = @backingInt(mode), .eps = eps };
        k.bind(e, "hp_norm", &.{ x, w, y }, params);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// q and k heads normed in place by wq and wk, then turned by cos, sin [rows, 64] (fp32 of bf16 values) if given.
    pub fn heads(k: *const Kernels, e: mtl.ComputeEncoder, q: Ref, qh: Heads, kr: Ref, kh: Heads, wq: Ref, wk: Ref, rot: ?[2]Ref, rows: usize, eps: f32, mode: Mode) void {
        std.debug.assert(qh.head == kh.head);
        const params = extern struct { heads: u32, kv_heads: u32, q_row: u32, k_row: u32, head: u32, mode: u32, rotary: u32, eps: f32 }{
            .heads = @intCast(qh.count),
            .kv_heads = @intCast(kh.count),
            .q_row = @intCast(qh.row),
            .k_row = @intCast(kh.row),
            .head = @intCast(qh.head),
            .mode = @backingInt(mode),
            .rotary = @intFromBool(rot != null),
            .eps = eps,
        };
        const t = rot orelse .{ wq, wq };
        k.bind(e, "hp_heads", &.{ q, kr, wq, wk, t[0], t[1] }, params);
        e.dispatchGroups(mtl.Size.of(rows, (qh.count + kh.count + 7) / 8, 1), mtl.Size.of(256, 1, 1));
    }

    /// o [rows, q heads x 128] = softmax(q k^T / sqrt(128)) v, each query over earlier rows when `causal`, else all.
    pub fn attention(k: *const Kernels, e: mtl.ComputeEncoder, q: Ref, qh: Heads, kr: Ref, vr: Ref, kh: Heads, o: Ref, rows: usize, causal: bool) void {
        std.debug.assert(qh.count % kh.count == 0 and qh.head == kh.head);
        const params = extern struct { rows: u32, heads: u32, kv_heads: u32, q_row: u32, k_row: u32, v_row: u32, head: u32, causal: u32, scale: f32 }{
            .rows = @intCast(rows),
            .heads = @intCast(qh.count),
            .kv_heads = @intCast(kh.count),
            .q_row = @intCast(qh.row),
            .k_row = @intCast(kh.row),
            .v_row = @intCast(kh.row),
            .head = @intCast(qh.head),
            .causal = @intFromBool(causal),
            .scale = 1.0 / @sqrt(128.0),
        };
        k.bind(e, "hp_attention", &.{ q, kr, vr, o }, params);
        e.dispatchGroups(mtl.Size.of(rows, (qh.count + 7) / 8, 1), mtl.Size.of(256, 1, 1));
    }

    /// o [rows, width] = silu(g) v, g and v rows `g_row` and `v_row` apart, rounded as `mode`.
    pub fn glu(k: *const Kernels, e: mtl.ComputeEncoder, g: Ref, v: Ref, o: Ref, rows: usize, width: usize, g_row: usize, v_row: usize, mode: Mode) void {
        const params = extern struct { width: u32, g_row: u32, v_row: u32, mode: u32 }{ .width = @intCast(width), .g_row = @intCast(g_row), .v_row = @intCast(v_row), .mode = @backingInt(mode) };
        k.bind(e, "hp_glu", &.{ g, v, o }, params);
        e.dispatchThreads(mtl.Size.of(width, rows, 1), mtl.Size.of(256, 1, 1));
    }

    /// x = bf16(x + y) over n values.
    pub fn add(k: *const Kernels, e: mtl.ComputeEncoder, x: Ref, y: Ref, n: usize) void {
        k.bind(e, "hp_add", &.{ x, y }, @as(u32, @intCast(n)));
        e.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// x rows `idx` (u32 [n]) = bf16(x + y [n, width]).
    pub fn scatterAdd(k: *const Kernels, e: mtl.ComputeEncoder, x: Ref, y: Ref, idx: Ref, n: usize, width: usize) void {
        k.bind(e, "hp_scatter_add", &.{ x, y, idx }, @as(u32, @intCast(width)));
        e.dispatchThreads(mtl.Size.of(width, n, 1), mtl.Size.of(256, 1, 1));
    }
};
