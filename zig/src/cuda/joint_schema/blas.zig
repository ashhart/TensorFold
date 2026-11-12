//! The head's products on cuBLASLt, FP32 compute only, as hip/blas.zig runs them on rocBLAS: a plan per shape, kept.
const std = @import("std");
const lt = @import("cuda").cublaslt;
const Driver = @import("cuda").Driver;
const DeviceBuffer = @import("cuda").DeviceBuffer;
const Stream = @import("cuda").Stream;

/// cuBLASLt's workspace for every plan: its heuristics choose within it.
pub const workspace_bytes = 32 << 20;

/// Attention's heads as batches: rows of `heads * head_dim` values, head h at column h * head_dim.
pub const Heads = struct { keys: usize, queries: usize, heads: usize, head_dim: usize };

pub const Blas = struct {
    gpa: std.mem.Allocator,
    lib: lt.Library,
    handle: lt.Handle,
    plans: std.AutoHashMapUnmanaged(lt.Spec, lt.Gemm) = .empty,
    ws: DeviceBuffer,

    pub fn init(gpa: std.mem.Allocator, d: *const Driver) !Blas {
        var lib = try lt.Library.open();
        errdefer lib.close();
        const handle = try lt.createHandle(&lib);
        errdefer _ = lib.api.cublasLtDestroy(handle);
        return .{ .gpa = gpa, .lib = lib, .handle = handle, .ws = try DeviceBuffer.alloc(d, workspace_bytes) };
    }

    /// Plans point at `lib`: `b` stays in place once a product ran.
    pub fn deinit(b: *Blas) void {
        var it = b.plans.valueIterator();
        while (it.next()) |g| g.deinit();
        b.plans.deinit(b.gpa);
        b.ws.free();
        _ = b.lib.api.cublasLtDestroy(b.handle);
        b.lib.close();
        b.* = undefined;
    }

    /// out[rows, n] = x[rows, k] w[n, k]ᵀ, BF16 in and out, FP32 sums, one rounding (a torch Linear without bias).
    pub fn project(b: *Blas, s: Stream, x: u64, w: u64, out: u64, rows: usize, n: usize, k: usize) !void {
        try b.linear(s, x, w, out, rows, n, k, 0);
    }

    /// out = x wᵀ + out, `out` already holding the bias rows: the bias adds inside the one rounding.
    pub fn projectAdd(b: *Blas, s: Stream, x: u64, w: u64, out: u64, rows: usize, n: usize, k: usize) !void {
        try b.linear(s, x, w, out, rows, n, k, 1);
    }

    fn linear(b: *Blas, s: Stream, x: u64, w: u64, out: u64, rows: usize, n: usize, k: usize, beta: f32) !void {
        if (rows == 0) return;
        // column-major outᵀ [n, rows] = (w stored k x n)ᵀ xᵀ
        try b.run(s, .{ .ta = .t, .m = n, .n = rows, .k = k, .a = .bf16, .b = .bf16, .c = .bf16, .lda = k, .ldb = k, .ldc = n }, 1, w, x, beta, out);
    }

    /// Per head h, FP32 `S_h = (Q_h K_hᵀ) alpha` as rows [heads][queries][keys] from BF16 q[queries, W] and k[keys, W].
    pub fn scores(b: *Blas, s: Stream, q: u64, k: u64, out: u64, h: Heads, alpha: f32) !void {
        const w = h.heads * h.head_dim;
        // column-major S_hᵀ [keys, queries] = K_hᵀ Q_h: the keys of one query row are contiguous for the softmax
        try b.run(s, .{ .ta = .t, .m = h.keys, .n = h.queries, .k = h.head_dim, .a = .bf16, .b = .bf16, .c = .f32, .lda = w, .ldb = w, .ldc = h.keys, .batch = @intCast(h.heads), .sa = h.head_dim, .sb = h.head_dim, .sc = h.keys * h.queries }, alpha, k, q, 0, out);
    }

    /// Per head h, FP32 `ctx_h = P_h V_h` into out[queries, W] from p [heads][queries][keys] and FP32 v[keys, W].
    pub fn context(b: *Blas, s: Stream, p: u64, v: u64, out: u64, h: Heads) !void {
        const w = h.heads * h.head_dim;
        // column-major ctx_h [head_dim, queries] = V_h [head_dim, keys] P_h [keys, queries], at column h * head_dim
        try b.run(s, .{ .m = h.head_dim, .n = h.queries, .k = h.keys, .a = .f32, .b = .f32, .c = .f32, .lda = w, .ldb = h.keys, .ldc = w, .batch = @intCast(h.heads), .sa = h.head_dim, .sb = h.keys * h.queries, .sc = h.head_dim }, 1, v, p, 0, out);
    }

    fn run(b: *Blas, s: Stream, spec: lt.Spec, alpha: f32, a: u64, x: u64, beta: f32, c: u64) !void {
        if (a == 0 or x == 0 or c == 0) return error.Invalid;
        const gop = try b.plans.getOrPut(b.gpa, spec);
        if (!gop.found_existing) {
            gop.value_ptr.* = lt.Gemm.init(&b.lib, b.handle, spec, workspace_bytes) catch |e| {
                _ = b.plans.remove(spec);
                return e;
            };
        }
        try gop.value_ptr.run(alpha, a, x, beta, c, b.ws.ptr, b.ws.len, s.handle);
    }
};
