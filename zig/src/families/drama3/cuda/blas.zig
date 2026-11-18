//! cuBLAS (the classic API) opened at run time: the strided batched GEMM Drama 3's prompt rows and codec take.
const std = @import("std");
const platform = @import("sys");
const cuda = @import("cuda");

pub const Status = c_int;
pub const Handle = ?*opaque {};
pub const Op = enum(c_int) { n = 0, t = 1 };
/// cudaDataType_t
pub const DataType = enum(c_int) { f32 = 0, f16 = 2, bf16 = 14 };
/// cublasComputeType_t CUBLAS_COMPUTE_32F (fp32 compute, no TF32)
pub const compute_32f: c_int = 68;
/// cublasGemmAlgo_t CUBLAS_GEMM_DEFAULT_TENSOR_OP
pub const gemm_default_tensor_op: c_int = 99;
/// cublasMath_t CUBLAS_DEFAULT_MATH
pub const default_math: c_int = 0;

pub const Error = @import("sys").optional_libraries.Failure || error{ LibraryUnavailable, MissingSymbol, CublasFailed };

const S = Status;
const CPtr = ?*const anyopaque;
const Ptr = cuda.abi.DevicePtr;
const StreamHandle = cuda.abi.Stream;

pub const Api = struct {
    cublasCreate_v2: *const fn (*Handle) callconv(.c) S,
    cublasDestroy_v2: *const fn (Handle) callconv(.c) S,
    cublasSetStream_v2: *const fn (Handle, StreamHandle) callconv(.c) S,
    cublasSetWorkspace_v2: *const fn (Handle, Ptr, usize) callconv(.c) S,
    cublasSetMathMode: *const fn (Handle, c_int) callconv(.c) S,
    cublasGetStatusName: *const fn (S) callconv(.c) ?[*:0]const u8,
    cublasGemmStridedBatchedEx: *const fn (Handle, Op, Op, c_int, c_int, c_int, CPtr, Ptr, DataType, c_int, c_longlong, Ptr, DataType, c_int, c_longlong, CPtr, Ptr, DataType, c_int, c_longlong, c_int, c_int, c_int) callconv(.c) S,
};

/// The library and one handle with its workspace, bound to a stream per call.
pub const Blas = struct {
    lib: platform.DynLib,
    api: Api,
    h: Handle,

    /// The CUDA 13 soname first, then the unversioned link name, as cuda/cublaslt.zig opens cuBLASLt.
    pub fn open(workspace: Ptr, workspace_bytes: usize) Error!Blas {
        for ([_][]const u8{ "libcublas.so.13", "libcublas.so" }) |path| {
            return openPath(path, workspace, workspace_bytes) catch |e| switch (e) {
                error.LibraryUnavailable => continue,
                error.MissingSymbol => return @import("sys").optional_libraries.unavailable(.cublas),
                else => return e,
            };
        }
        return @import("sys").optional_libraries.unavailable(.cublas);
    }

    fn openPath(path: []const u8, workspace: Ptr, workspace_bytes: usize) Error!Blas {
        var lib = platform.DynLib.open(path) catch return error.LibraryUnavailable;
        errdefer lib.close();
        var b: Blas = .{ .lib = lib, .api = undefined, .h = null };
        const info = @typeInfo(Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(b.api, name) = b.lib.lookup(T, name) orelse {
                std.log.err("{s} has no {s}", .{ path, name });
                return error.MissingSymbol;
            };
        }
        try b.check(b.api.cublasCreate_v2(&b.h), "cublasCreate");
        errdefer _ = b.api.cublasDestroy_v2(b.h);
        try b.check(b.api.cublasSetWorkspace_v2(b.h, workspace, workspace_bytes), "cublasSetWorkspace");
        try b.check(b.api.cublasSetMathMode(b.h, default_math), "cublasSetMathMode");
        return b;
    }

    pub fn close(b: *Blas) void {
        _ = b.api.cublasDestroy_v2(b.h);
        b.lib.close();
    }

    fn check(b: *const Blas, s: Status, what: []const u8) Error!void {
        if (s == 0) return;
        const name = b.api.cublasGetStatusName(s);
        std.log.err("{s}: {s} ({d})", .{ what, if (name) |p| std.mem.span(p) else "?", s });
        return error.CublasFailed;
    }

    /// Column-major C[i] = alpha op(A[i]) op(B[i]) + beta C[i] for i < batch, fp32 compute, on stream `s`.
    pub fn gemm(b: *const Blas, s: cuda.Stream, ta: Op, tb: Op, m: usize, n: usize, k: usize, alpha: f32, a: Ptr, lda: usize, sa: usize, bm: Ptr, ldb: usize, sb: usize, beta: f32, c: Ptr, ldc: usize, sc: usize, batch: usize, in_t: DataType, out_t: DataType) Error!void {
        try b.check(b.api.cublasSetStream_v2(b.h, s.handle), "cublasSetStream");
        try b.check(b.api.cublasGemmStridedBatchedEx(b.h, ta, tb, @intCast(m), @intCast(n), @intCast(k), &alpha, a, in_t, @intCast(lda), @intCast(sa), bm, in_t, @intCast(ldb), @intCast(sb), &beta, c, out_t, @intCast(ldc), @intCast(sc), @intCast(batch), compute_32f, gemm_default_tensor_op), "cublasGemmStridedBatchedEx");
    }
};
