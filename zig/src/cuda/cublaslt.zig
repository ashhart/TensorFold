//! cuBLASLt opened at runtime: descriptors, layouts, heuristics, matmul with fp32 compute only (no TF32 or fast modes).

const std = @import("std");
const platform = @import("sys");
const abi = @import("abi.zig");

pub const Status = c_int;
pub const Handle = ?*opaque {};
pub const MatmulDesc = ?*opaque {};
pub const Layout = ?*opaque {};
pub const Preference = ?*opaque {};

pub const Algo = extern struct { data: [8]u64 };

pub const Heuristic = extern struct {
    algo: Algo,
    workspace_size: usize,
    state: Status,
    waves_count: f32,
    reserved: [4]c_int,
};

pub const DataType = enum(c_int) { f32 = 0, f16 = 2, bf16 = 14 };
pub const compute_32f: c_int = 68;
pub const Op = enum(c_int) { n = 0, t = 1 };
pub const Order = enum(c_int) { col = 0, row = 1 };

pub const desc_transa: c_int = 3;
pub const desc_transb: c_int = 4;
pub const desc_epilogue: c_int = 7;
pub const desc_bias_pointer: c_int = 8;
pub const layout_order: c_int = 1;
pub const layout_batch_count: c_int = 5;
pub const layout_batch_stride: c_int = 6;
pub const pref_max_workspace_bytes: c_int = 1;

/// cublasLtMatrixLayoutAttribute_t's CUBLASLT_MATRIX_LAYOUT_COLS (cublasLt.h), a uint64_t.
pub const layout_cols: c_int = 3;

/// cublasLtMatmulAlgoConfigAttributes_t's split-K factor and reduction scheme, as ints.
pub const algo_config_splitk: c_int = 2;
pub const algo_config_reduction: c_int = 3;

/// What D gets after the product: nothing, or a bias over its columns added in f32 before D's one rounding.
pub const Epilogue = enum(u32) { none = 1, bias = 4 };

pub const Error = platform.optional_libraries.Failure || error{ LibraryUnavailable, MissingSymbol, CublasFailed, NoAlgorithm, Invalid };

const S = Status;
const CPtr = ?*const anyopaque;

/// Device pointers are declared as u64: the same register class as `void*` on every LP64 target.
pub const Api = struct {
    cublasLtCreate: *const fn (*Handle) callconv(.c) S,
    cublasLtDestroy: *const fn (Handle) callconv(.c) S,
    cublasLtGetVersion: *const fn () callconv(.c) usize,
    cublasLtGetStatusName: *const fn (S) callconv(.c) ?[*:0]const u8,
    cublasLtMatmulDescCreate: *const fn (*MatmulDesc, c_int, DataType) callconv(.c) S,
    cublasLtMatmulDescDestroy: *const fn (MatmulDesc) callconv(.c) S,
    cublasLtMatmulDescSetAttribute: *const fn (MatmulDesc, c_int, CPtr, usize) callconv(.c) S,
    cublasLtMatrixLayoutCreate: *const fn (*Layout, DataType, u64, u64, i64) callconv(.c) S,
    cublasLtMatrixLayoutDestroy: *const fn (Layout) callconv(.c) S,
    cublasLtMatrixLayoutSetAttribute: *const fn (Layout, c_int, CPtr, usize) callconv(.c) S,
    cublasLtMatmulPreferenceCreate: *const fn (*Preference) callconv(.c) S,
    cublasLtMatmulPreferenceDestroy: *const fn (Preference) callconv(.c) S,
    cublasLtMatmulPreferenceSetAttribute: *const fn (Preference, c_int, CPtr, usize) callconv(.c) S,
    cublasLtMatmulAlgoGetHeuristic: *const fn (Handle, MatmulDesc, Layout, Layout, Layout, Layout, Preference, c_int, [*]Heuristic, *c_int) callconv(.c) S,
    cublasLtMatmulAlgoCheck: *const fn (Handle, MatmulDesc, Layout, Layout, Layout, Layout, *const Algo, *Heuristic) callconv(.c) S,
    cublasLtMatmulAlgoConfigGetAttribute: *const fn (*const Algo, c_int, ?*anyopaque, usize, ?*usize) callconv(.c) S,
    cublasLtMatmul: *const fn (Handle, MatmulDesc, CPtr, abi.DevicePtr, Layout, abi.DevicePtr, Layout, CPtr, abi.DevicePtr, Layout, abi.DevicePtr, Layout, ?*const Algo, abi.DevicePtr, usize, abi.Stream) callconv(.c) S,
};

pub const Library = struct {
    lib: platform.DynLib,
    api: Api,

    /// The CUDA 13 soname first, then the unversioned link name; on Windows the toolkit's DLL.
    pub fn open() Error!Library {
        const names: []const []const u8 = if (@import("builtin").os.tag == .windows) &.{"cublasLt64_13.dll"} else &.{ "libcublasLt.so.13", "libcublasLt.so" };
        for (names) |path| {
            return openCandidate(path) catch |e| switch (e) {
                error.LibraryUnavailable => continue,
                error.MissingSymbol => return platform.optional_libraries.unavailable(.cublaslt),
                else => return e,
            };
        }
        return platform.optional_libraries.unavailable(.cublaslt);
    }

    pub fn openPath(path: []const u8) Error!Library {
        return openCandidate(path) catch |e| switch (e) {
            error.LibraryUnavailable, error.MissingSymbol => platform.optional_libraries.unavailable(.cublaslt),
            else => e,
        };
    }

    fn openCandidate(path: []const u8) Error!Library {
        var lib = platform.DynLib.open(path) catch return error.LibraryUnavailable;
        errdefer lib.close();
        var api: Api = undefined;
        const info = @typeInfo(Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(api, name) = lib.lookup(T, name) orelse {
                std.log.err("{s} has no {s}", .{ path, name });
                return error.MissingSymbol;
            };
        }
        return .{ .lib = lib, .api = api };
    }

    pub fn close(self: *Library) void {
        self.lib.close();
    }

    pub fn check(self: *const Library, s: Status, what: []const u8) Error!void {
        if (s == 0) return;
        const n = self.api.cublasLtGetStatusName(s);
        std.log.err("{s}: {s} ({d})", .{ what, if (n) |p| std.mem.span(p) else "?", s });
        return error.CublasFailed;
    }
};

/// One row-major product D[m,n] = X[m,k] . W[n,k]^T (a torch Linear), fp32 accumulate; bf16 inputs unless typed.
pub const Linear = struct {
    lt: *const Library,
    handle: Handle,
    desc: MatmulDesc,
    a: Layout,
    b: Layout,
    c: Layout,
    pref: Preference,
    algo: Algo,
    workspace_need: usize,
    epilogue: Epilogue, // fused into `algo`: `initEpilogue` asks, the algorithm decides

    /// Picks the heuristic's first algorithm for this shape once; `workspace_limit` bounds what it may ask for.
    pub fn init(lt: *const Library, m: u64, n: u64, k: u64, out: DataType, workspace_limit: usize) Error!Linear {
        return initEpilogue(lt, m, n, k, out, workspace_limit, .none);
    }

    /// `init`'s plan, its algorithm also running `epilogue` when it supports it (`epilogue` says if); see `runBias`.
    pub fn initEpilogue(lt: *const Library, m: u64, n: u64, k: u64, out: DataType, workspace_limit: usize, epilogue: Epilogue) Error!Linear {
        return initTyped(lt, m, n, k, .bf16, out, workspace_limit, epilogue);
    }

    /// `initEpilogue` with X and W of type `in` (bf16, fp16, or fp32 for true fp32 products), fp32 compute.
    pub fn initTyped(lt: *const Library, m: u64, n: u64, k: u64, in: DataType, out: DataType, workspace_limit: usize, epilogue: Epilogue) Error!Linear {
        var self: Linear = undefined;
        self.lt = lt;
        try lt.check(lt.api.cublasLtCreate(&self.handle), "cublasLtCreate");
        errdefer _ = lt.api.cublasLtDestroy(self.handle);
        try lt.check(lt.api.cublasLtMatmulDescCreate(&self.desc, compute_32f, .f32), "cublasLtMatmulDescCreate");
        errdefer _ = lt.api.cublasLtMatmulDescDestroy(self.desc);
        const ta: c_int = @backingInt(Op.t);
        const tb: c_int = @backingInt(Op.n);
        try lt.check(lt.api.cublasLtMatmulDescSetAttribute(self.desc, desc_transa, &ta, @sizeOf(c_int)), "set transa");
        try lt.check(lt.api.cublasLtMatmulDescSetAttribute(self.desc, desc_transb, &tb, @sizeOf(c_int)), "set transb");
        // column-major view: D^T[n,m] = W[n,k] (stored k x n, transposed) . X^T[k,m]
        try lt.check(lt.api.cublasLtMatrixLayoutCreate(&self.a, in, k, n, @intCast(k)), "layout W");
        errdefer _ = lt.api.cublasLtMatrixLayoutDestroy(self.a);
        try lt.check(lt.api.cublasLtMatrixLayoutCreate(&self.b, in, k, m, @intCast(k)), "layout X");
        errdefer _ = lt.api.cublasLtMatrixLayoutDestroy(self.b);
        try lt.check(lt.api.cublasLtMatrixLayoutCreate(&self.c, out, n, m, @intCast(n)), "layout D");
        errdefer _ = lt.api.cublasLtMatrixLayoutDestroy(self.c);
        try lt.check(lt.api.cublasLtMatmulPreferenceCreate(&self.pref), "cublasLtMatmulPreferenceCreate");
        errdefer _ = lt.api.cublasLtMatmulPreferenceDestroy(self.pref);
        const limit: u64 = workspace_limit;
        try lt.check(lt.api.cublasLtMatmulPreferenceSetAttribute(self.pref, pref_max_workspace_bytes, &limit, @sizeOf(u64)), "set workspace");
        var found: [1]Heuristic = undefined;
        var count: c_int = 0;
        try lt.check(lt.api.cublasLtMatmulAlgoGetHeuristic(self.handle, self.desc, self.a, self.b, self.c, self.c, self.pref, 1, &found, &count), "cublasLtMatmulAlgoGetHeuristic");
        if (count < 1 or found[0].state != 0) return error.NoAlgorithm;
        self.algo = found[0].algo;
        self.workspace_need = found[0].workspace_size;
        self.epilogue = .none;
        if (epilogue != .none) try self.fuse(epilogue, workspace_limit);
        return self;
    }

    pub fn deinit(self: *Linear) void {
        const api = self.lt.api;
        _ = api.cublasLtMatmulPreferenceDestroy(self.pref);
        _ = api.cublasLtMatrixLayoutDestroy(self.c);
        _ = api.cublasLtMatrixLayoutDestroy(self.b);
        _ = api.cublasLtMatrixLayoutDestroy(self.a);
        _ = api.cublasLtMatmulDescDestroy(self.desc);
        _ = api.cublasLtDestroy(self.handle);
        self.* = undefined;
    }

    /// D = X . W^T on `stream`; the workspace must hold `workspace_need` bytes.
    pub fn run(self: *const Linear, x: abi.DevicePtr, w: abi.DevicePtr, d: abi.DevicePtr, workspace: abi.DevicePtr, workspace_len: usize, stream: abi.Stream) Error!void {
        if (workspace_len < self.workspace_need) return error.Invalid;
        const one: f32 = 1;
        const zero: f32 = 0;
        try self.lt.check(self.lt.api.cublasLtMatmul(self.handle, self.desc, &one, w, self.a, x, self.b, &zero, d, self.c, d, self.c, &self.algo, workspace, workspace_len, stream), "cublasLtMatmul");
    }

    /// D = X . W^T + bias (bias [n], D's type) on a plan that fused `.bias`, one rounding to D.
    pub fn runBias(self: *const Linear, x: abi.DevicePtr, w: abi.DevicePtr, bias: abi.DevicePtr, d: abi.DevicePtr, workspace: abi.DevicePtr, workspace_len: usize, stream: abi.Stream) Error!void {
        if (self.epilogue != .bias) return error.Invalid;
        try self.lt.check(self.lt.api.cublasLtMatmulDescSetAttribute(self.desc, desc_bias_pointer, &bias, @sizeOf(abi.DevicePtr)), "set bias");
        try self.run(x, w, d, workspace, workspace_len, stream);
    }

    /// Fuses `epilogue` into the chosen algorithm when it supports it within `workspace_limit`; else leaves it unfused.
    fn fuse(self: *Linear, epilogue: Epilogue, workspace_limit: usize) Error!void {
        const api = self.lt.api;
        const e: u32 = @backingInt(epilogue);
        try self.lt.check(api.cublasLtMatmulDescSetAttribute(self.desc, desc_epilogue, &e, @sizeOf(u32)), "set epilogue");
        var checked: Heuristic = undefined;
        if (api.cublasLtMatmulAlgoCheck(self.handle, self.desc, self.a, self.b, self.c, self.c, &self.algo, &checked) == 0 and checked.state == 0 and checked.workspace_size <= workspace_limit) {
            self.epilogue = epilogue;
            self.workspace_need = @max(self.workspace_need, checked.workspace_size);
            return;
        }
        const none: u32 = @backingInt(Epilogue.none);
        try self.lt.check(api.cublasLtMatmulDescSetAttribute(self.desc, desc_epilogue, &none, @sizeOf(u32)), "unset epilogue");
    }
};

/// One column-major product `C = alpha op(A) op(B) + beta C`, C [m, n], batched with element strides; FP32 compute.
pub const Spec = struct {
    ta: Op = .n,
    tb: Op = .n,
    m: u64,
    n: u64,
    k: u64,
    a: DataType,
    b: DataType,
    c: DataType,
    lda: u64,
    ldb: u64,
    ldc: u64,
    batch: u32 = 1,
    sa: u64 = 0,
    sb: u64 = 0,
    sc: u64 = 0,
};

/// A planned `Spec` on a shared handle: the heuristic's first algorithm within the workspace limit.
pub const Gemm = struct {
    lt: *const Library,
    handle: Handle,
    desc: MatmulDesc,
    a: Layout,
    b: Layout,
    c: Layout,
    algo: Algo,
    workspace_need: usize,

    pub fn init(lt: *const Library, handle: Handle, s: Spec, workspace_limit: usize) Error!Gemm {
        if (s.m == 0 or s.n == 0 or s.k == 0 or s.batch == 0) return error.Invalid;
        var g: Gemm = undefined;
        g.lt = lt;
        g.handle = handle;
        const api = lt.api;
        try lt.check(api.cublasLtMatmulDescCreate(&g.desc, compute_32f, .f32), "cublasLtMatmulDescCreate");
        errdefer _ = api.cublasLtMatmulDescDestroy(g.desc);
        const ta: c_int = @backingInt(s.ta);
        const tb: c_int = @backingInt(s.tb);
        try lt.check(api.cublasLtMatmulDescSetAttribute(g.desc, desc_transa, &ta, @sizeOf(c_int)), "set transa");
        try lt.check(api.cublasLtMatmulDescSetAttribute(g.desc, desc_transb, &tb, @sizeOf(c_int)), "set transb");
        // op(A) is m x k and op(B) k x n: a transposed operand is stored the other way round
        g.a = try layout(lt, s.a, if (s.ta == .n) s.m else s.k, if (s.ta == .n) s.k else s.m, s.lda, s.batch, s.sa);
        errdefer _ = api.cublasLtMatrixLayoutDestroy(g.a);
        g.b = try layout(lt, s.b, if (s.tb == .n) s.k else s.n, if (s.tb == .n) s.n else s.k, s.ldb, s.batch, s.sb);
        errdefer _ = api.cublasLtMatrixLayoutDestroy(g.b);
        g.c = try layout(lt, s.c, s.m, s.n, s.ldc, s.batch, s.sc);
        errdefer _ = api.cublasLtMatrixLayoutDestroy(g.c);
        var pref: Preference = null;
        try lt.check(api.cublasLtMatmulPreferenceCreate(&pref), "cublasLtMatmulPreferenceCreate");
        defer _ = api.cublasLtMatmulPreferenceDestroy(pref);
        const limit: u64 = workspace_limit;
        try lt.check(api.cublasLtMatmulPreferenceSetAttribute(pref, pref_max_workspace_bytes, &limit, @sizeOf(u64)), "set workspace");
        var found: [1]Heuristic = undefined;
        var count: c_int = 0;
        try lt.check(api.cublasLtMatmulAlgoGetHeuristic(handle, g.desc, g.a, g.b, g.c, g.c, pref, 1, &found, &count), "cublasLtMatmulAlgoGetHeuristic");
        if (count < 1 or found[0].state != 0) return error.NoAlgorithm;
        g.algo = found[0].algo;
        g.workspace_need = found[0].workspace_size;
        return g;
    }

    pub fn deinit(g: *Gemm) void {
        const api = g.lt.api;
        _ = api.cublasLtMatrixLayoutDestroy(g.c);
        _ = api.cublasLtMatrixLayoutDestroy(g.b);
        _ = api.cublasLtMatrixLayoutDestroy(g.a);
        _ = api.cublasLtMatmulDescDestroy(g.desc);
        g.* = undefined;
    }

    /// C = alpha op(A) op(B) + beta C on `stream`, C read and written in place.
    pub fn run(g: *const Gemm, alpha: f32, a: abi.DevicePtr, b: abi.DevicePtr, beta: f32, c: abi.DevicePtr, workspace: abi.DevicePtr, workspace_len: usize, stream: abi.Stream) Error!void {
        if (workspace_len < g.workspace_need) return error.Invalid;
        try g.lt.check(g.lt.api.cublasLtMatmul(g.handle, g.desc, &alpha, a, g.a, b, g.b, &beta, c, g.c, c, g.c, &g.algo, workspace, workspace_len, stream), "cublasLtMatmul");
    }

    fn layout(lt: *const Library, t: DataType, rows: u64, cols: u64, ld: u64, batch: u32, stride: u64) Error!Layout {
        if (ld < rows) return error.Invalid;
        var l: Layout = null;
        try lt.check(lt.api.cublasLtMatrixLayoutCreate(&l, t, rows, cols, @intCast(ld)), "cublasLtMatrixLayoutCreate");
        errdefer _ = lt.api.cublasLtMatrixLayoutDestroy(l);
        if (batch > 1) {
            const n: i32 = @intCast(batch);
            const st: i64 = @intCast(stride);
            try lt.check(lt.api.cublasLtMatrixLayoutSetAttribute(l, layout_batch_count, &n, @sizeOf(i32)), "set batch count");
            try lt.check(lt.api.cublasLtMatrixLayoutSetAttribute(l, layout_batch_stride, &st, @sizeOf(i64)), "set batch stride");
        }
        return l;
    }
};

/// A handle for `Gemm` plans, which share it.
pub fn createHandle(lt: *const Library) Error!Handle {
    var h: Handle = null;
    try lt.check(lt.api.cublasLtCreate(&h), "cublasLtCreate");
    return h;
}

comptime {
    std.debug.assert(@sizeOf(Algo) == 64 and @sizeOf(Heuristic) == 96);
}
