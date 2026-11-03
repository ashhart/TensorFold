//! rocBLAS BF16 matrix products with FP32 accumulation on the caller's HIP stream.
const std = @import("std");
const platform = @import("sys");
const abi = @import("abi.zig");
const Stream = @import("stream.zig").Stream;
const Handle = ?*anyopaque;
const Const = ?*const anyopaque;
const Ptr = ?*anyopaque;
pub const Error = platform.optional_libraries.Failure || error{ MissingSymbol, RocblasFailed, Invalid };

const Api = struct {
    rocblas_create_handle: *const fn (*Handle) callconv(.c) c_int,
    rocblas_destroy_handle: *const fn (Handle) callconv(.c) c_int,
    rocblas_set_stream: *const fn (Handle, abi.Stream) callconv(.c) c_int,
    rocblas_gemm_ex: *const fn (Handle, c_int, c_int, c_int, c_int, c_int, Const, Const, c_int, c_int, Const, c_int, c_int, Const, Const, c_int, c_int, Ptr, c_int, c_int, c_int, c_int, i32, u32) callconv(.c) c_int,
    rocblas_gemm_strided_batched_ex: *const fn (Handle, c_int, c_int, c_int, c_int, c_int, Const, Const, c_int, c_int, i64, Const, c_int, c_int, i64, Const, Const, c_int, c_int, i64, Ptr, c_int, c_int, i64, c_int, c_int, c_int, i32, u32) callconv(.c) c_int,
};

/// The workspace calls, which only the BF16 path makes: looked up when it first sets a workspace, never at `open`.
pub const WorkspaceApi = struct {
    rocblas_set_workspace: *const fn (Handle, Ptr, usize) callconv(.c) c_int,
    rocblas_get_device_memory_size: *const fn (Handle, *usize) callconv(.c) c_int,
    rocblas_is_managing_device_memory: *const fn (Handle) callconv(.c) bool,
};

/// Every call of `WorkspaceApi` from `lib` (anything with DynLib's `lookup`), or the name of the first one it lacks.
pub fn resolveWorkspace(lib: anytype, out: *WorkspaceApi) ?[]const u8 {
    const info = @typeInfo(WorkspaceApi).@"struct";
    inline for (info.field_names, info.field_types) |field, T| @field(out, field) = lib.lookup(T, field) orelse return field;
    return null;
}

/// A handle's workspace: rocBLAS's own default size, which every product it serves runs within.
pub const workspace_bytes: usize = 32 << 20;

pub const Blas = struct {
    lib: platform.DynLib,
    api: Api,
    handle: Handle,
    stream: Stream,
    /// The workspace calls once a workspace was set (null: never asked for).
    ws: ?WorkspaceApi = null,

    /// The handle and all its calls belong to this device context and one nonblocking stream.
    pub fn open(stream: Stream) Error!Blas {
        const spec = platform.optional_libraries.spec(.rocblas);
        for (spec.names) |name| {
            var lib = platform.DynLib.open(name) catch continue;
            errdefer lib.close();
            var api: Api = undefined;
            const info = @typeInfo(Api).@"struct";
            inline for (info.field_names, info.field_types) |field, T|
                @field(api, field) = lib.lookup(T, field) orelse return platform.optional_libraries.unavailable(.rocblas);
            var handle: Handle = null;
            try check(api.rocblas_create_handle(&handle));
            if (handle == null) return error.RocblasFailed;
            errdefer _ = api.rocblas_destroy_handle(handle);
            try check(api.rocblas_set_stream(handle, stream.handle));
            return .{ .lib = lib, .api = api, .handle = handle, .stream = stream };
        }
        return platform.optional_libraries.unavailable(.rocblas);
    }

    /// The workspace calls, looked up on first use; a rocBLAS without one is refused by the call's name.
    fn workspaceApi(b: *Blas) Error!WorkspaceApi {
        if (b.ws) |w| return w;
        var w: WorkspaceApi = undefined;
        if (resolveWorkspace(&b.lib, &w)) |name| {
            std.log.err("this rocBLAS has no {s}, which the BF16 projections' planned workspace needs", .{name});
            return error.MissingSymbol;
        }
        b.ws = w;
        return w;
    }

    /// rocBLAS works in `bytes` at `ptr` (the caller's planned buffer) and frees its default: it allocates no more.
    pub fn useWorkspace(b: *Blas, ptr: u64, bytes: usize) Error!void {
        if (ptr == 0 or bytes == 0) return error.Invalid;
        const w = try b.workspaceApi();
        try check(w.rocblas_set_workspace(b.handle, @ptrFromInt(ptr), bytes));
    }

    /// The device bytes the handle works in, and whether rocBLAS allocated them (false: the caller's workspace).
    pub fn workspace(b: *Blas) Error!struct { bytes: usize, managed: bool } {
        const w = try b.workspaceApi();
        var bytes: usize = 0;
        try check(w.rocblas_get_device_memory_size(b.handle, &bytes));
        return .{ .bytes = bytes, .managed = w.rocblas_is_managing_device_memory(b.handle) };
    }

    pub fn close(b: *Blas) void {
        _ = b.stream.r.api.hipStreamSynchronize(b.stream.handle);
        _ = b.api.rocblas_destroy_handle(b.handle);
        b.lib.close();
        b.* = undefined;
    }

    /// BF16 x[M,K] times BF16 w[N,K] transposed, BF16 out[M,N]; FP32 scalar/accumulator, beta zero.
    pub fn project(b: *const Blas, x: u64, w: u64, out: u64, m: usize, n: usize, k: usize) Error!void {
        return b.product(x, w, out, m, n, k, false);
    }

    /// Add existing BF16 out[M,N] to the FP32 dot before its one BF16 rounding; bias rows use this epilogue.
    pub fn projectAdd(b: *const Blas, x: u64, w: u64, out: u64, m: usize, n: usize, k: usize) Error!void {
        return b.product(x, w, out, m, n, k, true);
    }

    fn product(b: *const Blas, x: u64, w: u64, out: u64, m: usize, n: usize, k: usize, addend: bool) Error!void {
        const shape = try Shape.of(m, n, k);
        if (x == 0 or w == 0 or out == 0 or x % 2 != 0 or w % 2 != 0 or out % 2 != 0) return error.Invalid;
        const xb = try extent(x, m, k, 2);
        const wb = try extent(w, n, k, 2);
        const yb = try extent(out, m, n, 2);
        if (overlap(x, xb, out, yb) or overlap(w, wb, out, yb)) return error.Invalid;
        const alpha: f32 = 1;
        const beta: f32 = if (addend) 1 else 0;
        try check(b.api.rocblas_gemm_ex(
            b.handle,
            112,
            111,
            shape.n,
            shape.m,
            shape.k,
            &alpha,
            @ptrFromInt(w),
            168,
            shape.k,
            @ptrFromInt(x),
            168,
            shape.k,
            &beta,
            @ptrFromInt(out),
            168,
            shape.n,
            @ptrFromInt(out),
            168,
            shape.n,
            151,
            0,
            0,
            0,
        ));
    }

    /// Per head h, FP32 `S_h = (Q_h K_hᵀ) alpha` as rows [heads][queries][keys] from BF16 q[queries, W] and k[keys, W].
    pub fn scores(b: *const Blas, q: u64, k: u64, out: u64, h: Heads, alpha: f32) Error!void {
        const z = try h.sizes();
        const w = z.w;
        if (q % 2 != 0 or k % 2 != 0 or out % 4 != 0) return error.Invalid;
        const qb = try extent(q, h.queries, w, 2);
        const kb = try extent(k, h.keys, w, 2);
        const sb = try extent(out, z.rows, h.keys, 4);
        if (overlap(q, qb, out, sb) or overlap(k, kb, out, sb)) return error.Invalid;
        // column-major S_hᵀ [keys, queries] = K_hᵀ Q_h: the keys of one query row are contiguous for the softmax
        try b.batched(.{ .ta = transpose, .m = h.keys, .n = h.queries, .k = h.head_dim, .a = k, .a_type = bf16_r, .lda = w, .sa = h.head_dim, .b = q, .b_type = bf16_r, .ldb = w, .sb = h.head_dim, .c = out, .ldc = h.keys, .sc = z.block, .batch = h.heads, .alpha = alpha });
    }

    /// Per head h, FP32 `ctx_h = P_h V_h` into out[queries, W] from p [heads][queries][keys] and FP32 v[keys, W].
    pub fn context(b: *const Blas, p: u64, v: u64, out: u64, h: Heads) Error!void {
        const z = try h.sizes();
        const w = z.w;
        if (p % 4 != 0 or v % 4 != 0 or out % 4 != 0) return error.Invalid;
        const pb = try extent(p, z.rows, h.keys, 4);
        const vb = try extent(v, h.keys, w, 4);
        const ob = try extent(out, h.queries, w, 4);
        if (overlap(p, pb, out, ob) or overlap(v, vb, out, ob)) return error.Invalid;
        // column-major ctx_h [head_dim, queries] = V_h [head_dim, keys] P_h [keys, queries], at column h * head_dim
        try b.batched(.{ .ta = none, .m = h.head_dim, .n = h.queries, .k = h.keys, .a = v, .a_type = f32_r, .lda = w, .sa = h.head_dim, .b = p, .b_type = f32_r, .ldb = h.keys, .sb = z.block, .c = out, .ldc = w, .sc = h.head_dim, .batch = h.heads, .alpha = 1 });
    }

    fn batched(b: *const Blas, g: Batched) Error!void {
        const shape = try Shape.of(g.m, g.n, g.k);
        if (g.a == 0 or g.b == 0 or g.c == 0 or g.batch == 0) return error.Invalid;
        const ld = [_]usize{ g.lda, g.ldb, g.ldc, g.batch };
        for (ld) |x| if (x == 0 or x > std.math.maxInt(c_int)) return error.Invalid;
        const alpha = g.alpha;
        const beta: f32 = 0;
        try check(b.api.rocblas_gemm_strided_batched_ex(b.handle, g.ta, none, shape.m, shape.n, shape.k, &alpha, @ptrFromInt(g.a), g.a_type, @intCast(g.lda), @intCast(g.sa), @ptrFromInt(g.b), g.b_type, @intCast(g.ldb), @intCast(g.sb), &beta, @ptrFromInt(g.c), f32_r, @intCast(g.ldc), @intCast(g.sc), @ptrFromInt(g.c), f32_r, @intCast(g.ldc), @intCast(g.sc), @intCast(g.batch), f32_r, 0, 0, 0));
    }
};

const bf16_r = 168;
const f32_r = 151;
const none = 111;
const transpose = 112;

/// One batched rocBLAS product: column-major `C_b = op(A_b) op(B_b)` for b < batch, FP32 accumulation, beta zero.
const Batched = struct { ta: c_int, m: usize, n: usize, k: usize, a: u64, a_type: c_int, lda: usize, sa: usize, b: u64, b_type: c_int, ldb: usize, sb: usize, c: u64, ldc: usize, sc: usize, batch: usize, alpha: f32 };

/// Attention's heads as rocBLAS batches: rows of `heads * head_dim` values, head h at column h * head_dim.
pub const Heads = struct {
    keys: usize,
    queries: usize,
    heads: usize,
    head_dim: usize,

    /// The products the batches take, each checked: W, heads x queries and keys x queries (Invalid: zero or overflow).
    fn sizes(h: Heads) error{Invalid}!struct { w: usize, rows: usize, block: usize } {
        if (h.keys == 0 or h.queries == 0 or h.heads == 0 or h.head_dim == 0) return error.Invalid;
        return .{
            .w = std.math.mul(usize, h.heads, h.head_dim) catch return error.Invalid,
            .rows = std.math.mul(usize, h.heads, h.queries) catch return error.Invalid,
            .block = std.math.mul(usize, h.keys, h.queries) catch return error.Invalid,
        };
    }
};

pub const Shape = struct {
    m: c_int,
    n: c_int,
    k: c_int,

    pub fn of(m: usize, n: usize, k: usize) error{Invalid}!Shape {
        if (m == 0 or n == 0 or k == 0 or @max(m, n, k) > std.math.maxInt(c_int)) return error.Invalid;
        return .{ .m = @intCast(m), .n = @intCast(n), .k = @intCast(k) };
    }
};

fn extent(start: u64, rows: usize, width: usize, size: usize) error{Invalid}!u64 {
    const count = std.math.mul(u64, rows, width) catch return error.Invalid;
    const bytes = std.math.mul(u64, count, size) catch return error.Invalid;
    _ = std.math.add(u64, start, bytes) catch return error.Invalid;
    return bytes;
}

fn overlap(a: u64, an: u64, b: u64, bn: u64) bool {
    return a < b + bn and b < a + an;
}

fn check(status: c_int) Error!void {
    if (status != 0) {
        std.log.err("rocBLAS status {d}", .{status});
        return error.RocblasFailed;
    }
}

test "HIP BLAS: BLAS refuses impossible geometry and output aliases before library calls" {
    var b: Blas = undefined;
    try std.testing.expectError(error.Invalid, b.project(2, 4, 6, 0, 1, 1));
    try std.testing.expectError(error.Invalid, b.project(2, 4, 6, 1, 1, @as(usize, std.math.maxInt(c_int)) + 1));
    try std.testing.expectError(error.Invalid, b.project(0, 4, 6, 1, 1, 1));
    try std.testing.expectError(error.Invalid, b.project(2, 4, 3, 1, 1, 1));
    try std.testing.expectError(error.Invalid, b.project(100, 200, 102, 2, 2, 2));
    try std.testing.expectError(error.Invalid, b.project(100, 200, 202, 2, 2, 2));
    try std.testing.expectError(error.Invalid, b.project(std.math.maxInt(u64) - 1, 100, 200, 1, 1, 2));
    try std.testing.expectError(error.Invalid, b.useWorkspace(0, workspace_bytes));
    try std.testing.expectError(error.Invalid, b.useWorkspace(256, 0));
    const h: Heads = .{ .keys = 4, .queries = 2, .heads = 2, .head_dim = 8 };
    try std.testing.expectError(error.Invalid, b.scores(2, 1000, 1002, h, 1)); // FP32 out not on 4 bytes
    try std.testing.expectError(error.Invalid, b.scores(2, 1000, 4000, .{ .keys = 0, .queries = 2, .heads = 2, .head_dim = 8 }, 1));
    try std.testing.expectError(error.Invalid, b.scores(4000, 1000, 4000, h, 1)); // out over q
    try std.testing.expectError(error.Invalid, b.context(1000, 2002, 8000, h)); // FP32 V on 2 bytes
    try std.testing.expectError(error.Invalid, b.context(8000, 4000, 8000 + 60, h)); // out over p
    try std.testing.expectError(error.Invalid, b.context(80000, 8000, 0, h)); // a null output
    // products of the head fields that overflow, and zero dimensions, refuse before any extent
    const wide: Heads = .{ .keys = 4, .queries = 2, .heads = 1 << 63, .head_dim = 2 };
    try std.testing.expectError(error.Invalid, b.scores(2, 1000, 4000, wide, 1));
    try std.testing.expectError(error.Invalid, b.context(4000, 8000, 16000, wide));
    const tall: Heads = .{ .keys = 1 << 62, .queries = 8, .heads = 1, .head_dim = 8 };
    try std.testing.expectError(error.Invalid, b.context(4000, 8000, 16000, tall));
    try std.testing.expectError(error.Invalid, b.scores(2, 1000, 4000, .{ .keys = 4, .queries = 2, .heads = 2, .head_dim = 0 }, 1));
}

test "HIP BLAS: the workspace calls resolve together, and the first one missing is named" {
    const Fake = struct {
        missing: ?[]const u8,
        fn dummy() callconv(.c) void {}
        pub fn lookup(f: *const @This(), comptime T: type, name: [:0]const u8) ?T {
            if (f.missing) |m| if (std.mem.eql(u8, m, name)) return null;
            return @ptrCast(&dummy);
        }
    };
    var w: WorkspaceApi = undefined;
    try std.testing.expect(resolveWorkspace(&Fake{ .missing = null }, &w) == null);
    inline for (@typeInfo(WorkspaceApi).@"struct".field_names) |name| {
        try std.testing.expectEqualStrings(name, resolveWorkspace(&Fake{ .missing = name }, &w).?);
        // `open` resolves the product calls only, so a loader that sets no workspace never needs these
        try std.testing.expect(!@hasField(Api, name));
    }
}
