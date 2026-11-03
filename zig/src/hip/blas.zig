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
        const xb = try extent(x, m, k);
        const wb = try extent(w, n, k);
        const yb = try extent(out, m, n);
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

fn extent(start: u64, rows: usize, width: usize) error{Invalid}!u64 {
    const count = std.math.mul(u64, rows, width) catch return error.Invalid;
    const bytes = std.math.mul(u64, count, 2) catch return error.Invalid;
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
