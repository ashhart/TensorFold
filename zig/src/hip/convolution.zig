//! MIOpen BF16 convolutions on one owned HIP stream, with admitted workspace and FP32 accumulation.
const std = @import("std");
const platform = @import("sys");
const hip = @import("runtime.zig");
const Stream = @import("stream.zig").Stream;
const Buffer = @import("memory.zig").DeviceBuffer;
const Ptr = ?*anyopaque;
const Const = ?*const anyopaque;
const Perf = extern struct { algorithm: c_int, time: f32, memory: usize };
pub const Error = hip.Error || platform.optional_libraries.Failure || error{ MiopenFailed, BadConvolutionShape, ConvolutionWorkspaceDoesNotFit };

const Api = struct {
    miopenCreateWithStream: *const fn (*Ptr, @import("abi.zig").Stream) callconv(.c) c_int,
    miopenDestroy: *const fn (Ptr) callconv(.c) c_int,
    miopenCreateTensorDescriptor: *const fn (*Ptr) callconv(.c) c_int,
    miopenDestroyTensorDescriptor: *const fn (Ptr) callconv(.c) c_int,
    miopenSetTensorDescriptor: *const fn (Ptr, c_int, c_int, [*]const c_int, [*]const c_int) callconv(.c) c_int,
    miopenCreateConvolutionDescriptor: *const fn (*Ptr) callconv(.c) c_int,
    miopenDestroyConvolutionDescriptor: *const fn (Ptr) callconv(.c) c_int,
    miopenInitConvolutionNdDescriptor: *const fn (Ptr, c_int, [*]const c_int, [*]const c_int, [*]const c_int, c_int) callconv(.c) c_int,
    miopenConvolutionForwardGetWorkSpaceSize: *const fn (Ptr, Ptr, Ptr, Ptr, Ptr, *usize) callconv(.c) c_int,
    miopenFindConvolutionForwardAlgorithm: *const fn (Ptr, Ptr, Const, Ptr, Const, Ptr, Ptr, Ptr, c_int, *c_int, *Perf, Ptr, usize, bool) callconv(.c) c_int,
    miopenConvolutionForward: *const fn (Ptr, Const, Ptr, Const, Ptr, Const, Ptr, c_int, Const, Ptr, Ptr, Ptr, usize) callconv(.c) c_int,
    miopenGetVersion: *const fn (*usize, *usize, *usize) callconv(.c) c_int,
};

pub const TensorShape = struct {
    rank: c_int,
    dims: [5]c_int = @splat(1),
    strides: [5]c_int = @splat(1),
    values: usize,

    pub fn of(dims: []const usize) Error!TensorShape {
        if (dims.len != 4 and dims.len != 5) return error.BadConvolutionShape;
        var s: TensorShape = .{ .rank = @intCast(dims.len), .values = 1 };
        var i = dims.len;
        while (i > 0) {
            i -= 1;
            if (dims[i] == 0 or dims[i] > std.math.maxInt(c_int)) return error.BadConvolutionShape;
            s.dims[i] = @intCast(dims[i]);
            s.strides[i] = @intCast(s.values);
            s.values = std.math.mul(usize, s.values, dims[i]) catch return error.BadConvolutionShape;
            if (s.values > std.math.maxInt(c_int)) return error.BadConvolutionShape;
        }
        return s;
    }
};

pub const Shape = struct {
    x: TensorShape,
    w: TensorShape,
    y: TensorShape,
    stride: [3]c_int = @splat(1),

    pub fn of(input: []const usize, weight: []const usize, stride: []const usize) Error!Shape {
        const x = try TensorShape.of(input);
        const w = try TensorShape.of(weight);
        if (x.rank != w.rank or input[1] != weight[1] or stride.len != input.len - 2) return error.BadConvolutionShape;
        var dims: [5]usize = @splat(1);
        dims[0] = input[0];
        dims[1] = weight[0];
        var s: Shape = .{ .x = x, .w = w, .y = undefined };
        for (stride, 0..) |step, i| {
            if (step == 0 or step > std.math.maxInt(c_int) or weight[i + 2] > input[i + 2]) return error.BadConvolutionShape;
            s.stride[i] = @intCast(step);
            dims[i + 2] = (input[i + 2] - weight[i + 2]) / step + 1;
        }
        s.y = try TensorShape.of(dims[0..input.len]);
        return s;
    }

    /// Logical dimensions remain N,C,H,W; these strides select MIOpen's physical channels-last layout.
    pub fn channelsLast(s: Shape) Error!Shape {
        if (s.x.rank != 4) return error.BadConvolutionShape;
        var out = s;
        inline for (.{ "x", "w", "y" }) |name| {
            const t = &@field(out, name);
            const channels = t.dims[1];
            t.strides = .{ t.dims[2] * t.dims[3] * channels, 1, t.dims[3] * channels, channels, 1 };
        }
        return out;
    }

    pub fn pointers(s: Shape, x: u64, w: u64, y: u64) Error!void {
        const inputs = [_]u64{ x, w, y };
        const sizes = [_]usize{ s.x.values * 2, s.w.values * 2, s.y.values * 2 };
        for (inputs, sizes) |p, size| {
            if (p == 0 or p % 2 != 0) return error.Invalid;
            _ = std.math.add(u64, p, size) catch return error.Invalid;
        }
        for (inputs[0..2], sizes[0..2]) |p, size|
            if (p < y + sizes[2] and y < p + size) return error.Invalid;
    }
};

pub const Convolution = struct {
    lib: platform.DynLib,
    api: Api,
    handle: Ptr,
    descs: [3]Ptr, // input, weight, output
    conv: Ptr,
    shape: Shape,
    stream: Stream,
    workspace: Buffer,
    perf: Perf,
    version: [3]usize,

    /// Non-exhaustive search loads the fixed GEMM reduction; its workspace must fit the caller's budget.
    pub fn init(stream: Stream, shape: Shape, x: u64, w: u64, y: u64, workspace_budget: usize) Error!Convolution {
        try shape.pointers(x, w, y);
        var lib = try openLibrary();
        errdefer lib.close();
        var api: Api = undefined;
        const info = @typeInfo(Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T|
            @field(api, name) = lib.lookup(T, name) orelse return platform.optional_libraries.unavailable(.miopen);
        var version: [3]usize = undefined;
        try check(api.miopenGetVersion(&version[0], &version[1], &version[2]));
        var handle: Ptr = null;
        try check(api.miopenCreateWithStream(&handle, stream.handle));
        errdefer _ = api.miopenDestroy(handle);
        var descs: [3]Ptr = @splat(null);
        errdefer for (descs) |d| if (d != null) {
            _ = api.miopenDestroyTensorDescriptor(d);
        };
        for (&descs, [_]TensorShape{ shape.x, shape.w, shape.y }) |*d, s| {
            try check(api.miopenCreateTensorDescriptor(d));
            try check(api.miopenSetTensorDescriptor(d.*, 5, s.rank, &s.dims, &s.strides));
        }
        var conv: Ptr = null;
        try check(api.miopenCreateConvolutionDescriptor(&conv));
        errdefer _ = api.miopenDestroyConvolutionDescriptor(conv);
        const zero = [_]c_int{ 0, 0, 0 };
        const one = [_]c_int{ 1, 1, 1 };
        try check(api.miopenInitConvolutionNdDescriptor(conv, shape.x.rank - 2, &zero, &shape.stride, &one, 0));
        var needed: usize = 0;
        try check(api.miopenConvolutionForwardGetWorkSpaceSize(handle, descs[1], descs[0], conv, descs[2], &needed));
        if (needed > workspace_budget) return error.ConvolutionWorkspaceDoesNotFit;
        var workspace = try Buffer.alloc(stream.r, needed);
        errdefer workspace.free();
        errdefer stream.synchronize() catch {};
        var count: c_int = 0;
        var found: [8]Perf = undefined;
        try check(api.miopenFindConvolutionForwardAlgorithm(handle, descs[0], @ptrFromInt(x), descs[1], @ptrFromInt(w), conv, descs[2], @ptrFromInt(y), found.len, &count, &found[0], workspace.ptr, needed, false));
        try stream.synchronize();
        const perf = try gemm(&found, count, needed);
        return .{ .lib = lib, .api = api, .handle = handle, .descs = descs, .conv = conv, .shape = shape, .stream = stream, .workspace = workspace, .perf = perf, .version = version };
    }

    pub fn run(c: *const Convolution, x: u64, w: u64, y: u64) Error!void {
        try c.shape.pointers(x, w, y);
        const one: f32 = 1;
        const zero: f32 = 0;
        try check(c.api.miopenConvolutionForward(c.handle, &one, c.descs[0], @ptrFromInt(x), c.descs[1], @ptrFromInt(w), c.conv, c.perf.algorithm, &zero, c.descs[2], @ptrFromInt(y), c.workspace.ptr, c.workspace.len));
    }

    pub fn deinit(c: *Convolution) void {
        c.stream.synchronize() catch {};
        c.workspace.free();
        for (c.descs) |d| _ = c.api.miopenDestroyTensorDescriptor(d);
        _ = c.api.miopenDestroyConvolutionDescriptor(c.conv);
        _ = c.api.miopenDestroy(c.handle);
        c.lib.close();
        c.* = undefined;
    }
};

fn openLibrary() Error!platform.DynLib {
    const spec = platform.optional_libraries.spec(.miopen);
    for (spec.names) |name| return platform.DynLib.open(name) catch continue;
    return platform.optional_libraries.unavailable(.miopen);
}

fn check(status: c_int) Error!void {
    if (status != 0) {
        std.log.err("MIOpen status {d}", .{status});
        return error.MiopenFailed;
    }
}

fn gemm(found: []const Perf, count: c_int, workspace: usize) Error!Perf {
    if (count <= 0 or count > found.len) return error.MiopenFailed;
    for (found[0..@intCast(count)]) |item| {
        if (item.algorithm != 0) continue;
        if (item.memory > workspace) return error.ConvolutionWorkspaceDoesNotFit;
        return item;
    }
    return error.MiopenFailed;
}

test "HIP convolution: GEMM is selected even when a cold search ranks another family first" {
    const fastest: Perf = .{ .algorithm = 5, .time = 1, .memory = 0 };
    const stable: Perf = .{ .algorithm = 0, .time = 2, .memory = 32 };
    try std.testing.expectEqualDeep(stable, try gemm(&.{ fastest, stable }, 2, 32));
    try std.testing.expectEqualDeep(stable, try gemm(&.{ stable, fastest }, 2, 32));
    try std.testing.expectError(error.MiopenFailed, gemm(&.{fastest}, 1, 32));
    try std.testing.expectError(error.MiopenFailed, gemm(&.{stable}, 2, 32));
    try std.testing.expectError(error.ConvolutionWorkspaceDoesNotFit, gemm(&.{stable}, 1, 31));
}

test "HIP convolution: shapes and output aliases are refused before library calls" {
    const s = try Shape.of(&.{ 1564, 3, 2, 14, 14 }, &.{ 1024, 3, 2, 14, 14 }, &.{ 2, 14, 14 });
    try std.testing.expectEqualSlices(c_int, &.{ 1564, 1024, 1, 1, 1 }, &s.y.dims);
    try std.testing.expectError(error.Invalid, s.pointers(1024, 1024, 1024));
    try std.testing.expectError(error.BadConvolutionShape, Shape.of(&.{ 1, 3, 2, 14, 14 }, &.{ 1024, 4, 2, 14, 14 }, &.{ 2, 14, 14 }));
    try std.testing.expectError(error.BadConvolutionShape, Shape.of(&.{ 1, 3, 2, 14, 14 }, &.{ 1024, 3, 2, 14, 14 }, &.{ 0, 14, 14 }));
    try std.testing.expectError(error.BadConvolutionShape, TensorShape.of(&.{ 1, 1024, std.math.maxInt(usize), 1 }));
    const down = try (try Shape.of(&.{ 391, 1024, 2, 2 }, &.{ 1536, 1024, 2, 2 }, &.{ 2, 2 })).channelsLast();
    try std.testing.expectEqualSlices(c_int, &.{ 4096, 1, 2048, 1024, 1 }, &down.x.strides);
    try std.testing.expectEqualSlices(c_int, &.{ 1536, 1, 1536, 1536, 1 }, &down.y.strides);
    try std.testing.expectError(error.BadConvolutionShape, s.channelsLast());
    std.testing.refAllDecls(Convolution);
}
