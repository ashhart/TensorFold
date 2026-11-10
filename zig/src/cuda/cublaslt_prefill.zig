//! Optional prompt-only cuBLASLt bridge; all GPU scratch belongs to the engine before cache admission.
const std = @import("std");
const platform = @import("sys");
const Driver = @import("driver.zig").Driver;
const Buffer = @import("memory.zig").DeviceBuffer;
const Stream = @import("stream.zig").Stream;
const qmmf = @import("qmmf.zig");

pub fn selected(value: ?[]const u8) !bool {
    const raw = value orelse return false;
    if (std.mem.eql(u8, raw, "0")) return false;
    if (std.mem.eql(u8, raw, "1")) return true;
    return error.BadCublasPrefill;
}

pub fn requested() !bool {
    return selected(try @import("sys").environment.given("TF_CUBLAS_PREFILL"));
}

pub const Mode = enum(u32) { affine4 = 1, nvfp4 = 2, fp8 = 3, fp8_row = 4 };
const DeviceSpan = extern struct { address: u64 = 0, bytes: u64 = 0 };
const Create = extern struct {
    abi_version: u32 = 1,
    struct_size: u32 = 128,
    max_plans: u32 = 64,
    flags: u32 = 0,
    weight_raw: DeviceSpan = .{},
    activation_raw: DeviceSpan = .{},
    weight_scales: DeviceSpan = .{},
    activation_scales: DeviceSpan = .{},
    workspace: DeviceSpan = .{},
    reserved: [4]u64 = @splat(0),
};
pub const Run = extern struct {
    abi_version: u32 = 1,
    struct_size: u32 = 160,
    mode: Mode,
    out_kind: u32 = 1,
    m: u32,
    n: u32,
    k: u32,
    npad: u32,
    ldx: u32,
    flags: u32 = 0,
    x: u64 = 0,
    weight: u64 = 0,
    scale: u64 = 0,
    bias: u64 = 0,
    out: u64 = 0,
    act_scale: f32 = 0,
    weight_scale: f32 = 1,
    x_bytes: u64 = 0,
    weight_bytes: u64 = 0,
    scale_bytes: u64 = 0,
    bias_bytes: u64 = 0,
    out_bytes: u64 = 0,
    stream: u64 = 0,
    reserved: [3]u64 = @splat(0),
};
const Requirements = extern struct {
    abi_version: u32 = 1,
    struct_size: u32 = 64,
    weight_raw_bytes: u64 = 0,
    activation_raw_bytes: u64 = 0,
    weight_scales_bytes: u64 = 0,
    activation_scales_bytes: u64 = 0,
    minimum_workspace_bytes: u64 = 0,
    preferred_workspace_bytes: u64 = 0,
    reserved: u64 = 0,
};
const Handle = ?*anyopaque;
const Tune = *const fn (Handle) callconv(.c) u32;
const Api = struct {
    version: *const fn () callconv(.c) u32,
    requirements: *const fn (*const Run, *Requirements) callconv(.c) u32,
    create: *const fn (*const Create, *Handle) callconv(.c) u32,
    destroy: *const fn (Handle) callconv(.c) u32,
    run: *const fn (Handle, *const Run) callconv(.c) u32,
    last_error: *const fn (Handle, [*]u8, u64) callconv(.c) u64,
    tune: ?Tune,
};

pub const Kit = struct {
    d: *const Driver,
    lib: platform.DynLib,
    api: Api,
    handle: Handle = null,
    sizes: [5]usize = @splat(0),
    buffers: [5]?Buffer = @splat(null),
    arena: ?Buffer = null,
    last_stream: ?Stream = null,
    admitted: [32]Admitted = undefined,
    admitted_count: usize = 0,

    const Admitted = struct { mode: Mode, n: usize, k: usize, npad: usize };

    pub fn open(d: *const Driver) !Kit {
        const path = (try platform.environment.given("TF_CUBLAS_BRIDGE")) orelse return error.MissingCublasBridge;
        var lib = try platform.DynLib.open(path);
        errdefer lib.close();
        const api: Api = .{
            .version = lib.lookup(@FieldType(Api, "version"), "tf_lt_abi_version") orelse return error.BadCublasBridge,
            .requirements = lib.lookup(@FieldType(Api, "requirements"), "tf_lt_get_requirements") orelse return error.BadCublasBridge,
            .create = lib.lookup(@FieldType(Api, "create"), "tf_lt_create") orelse return error.BadCublasBridge,
            .destroy = lib.lookup(@FieldType(Api, "destroy"), "tf_lt_destroy") orelse return error.BadCublasBridge,
            .run = lib.lookup(@FieldType(Api, "run"), "tf_lt_run") orelse return error.BadCublasBridge,
            .last_error = lib.lookup(@FieldType(Api, "last_error"), "tf_lt_last_error") orelse return error.BadCublasBridge,
            .tune = lib.lookup(Tune, "tf_lt_tune_last"),
        };
        if (api.version() != 1) return error.BadCublasBridge;
        return .{ .d = d, .lib = lib, .api = api };
    }

    pub fn deinit(v: *Kit) void {
        if (v.last_stream) |stream| @import("completion.zig").drain(v.d, stream, "cuBLAS prompt work did not drain");
        if (v.handle != null) _ = v.api.destroy(v.handle);
        for (&v.buffers) |*buffer| if (buffer.*) |*b| b.free();
        if (v.arena) |*arena| arena.free();
        v.lib.close();
    }

    pub fn include(v: *Kit, mode: Mode, rows: usize, n: usize, k: usize, npad: usize) !void {
        var op = try shape(mode, rows, n, k, npad, k, .bf16);
        var req: Requirements = .{};
        const status = v.api.requirements(&op, &req);
        if (status == 1) return;
        try v.check(status);
        if (!v.allows(mode, n, k, npad)) {
            if (v.admitted_count == v.admitted.len) return error.TooManyCublasShapes;
            v.admitted[v.admitted_count] = .{ .mode = mode, .n = n, .k = k, .npad = npad };
            v.admitted_count += 1;
        }
        const sizes = [_]u64{ req.weight_raw_bytes, req.activation_raw_bytes, req.weight_scales_bytes, req.activation_scales_bytes, @max(req.minimum_workspace_bytes, req.preferred_workspace_bytes) };
        for (&v.sizes, sizes) |*capacity, byte_count| capacity.* = @max(capacity.*, std.math.cast(usize, byte_count) orelse
            return error.BadCublasShape);
        // Startup tuning writes a private BF16 result after this call's live activation scales.
        v.sizes[3] = @max(v.sizes[3], try tuningSpan(mode, rows, n, req.activation_scales_bytes));
    }

    fn allows(v: *const Kit, mode: Mode, n: usize, k: usize, npad: usize) bool {
        for (v.admitted[0..v.admitted_count]) |s| if (s.mode == mode and s.n == n and s.k == k and s.npad == npad)
            return true;
        return false;
    }

    pub fn allocate(v: *Kit) !void {
        if (v.handle != null or v.arena != null) return error.CublasAlreadyAllocated;
        var offsets: [5]usize = @splat(0);
        var total: usize = 0;
        for (v.sizes, &offsets) |byte_count, *offset| {
            total = (try std.math.add(usize, total, 255)) & ~@as(usize, 255);
            offset.* = total;
            total = try std.math.add(usize, total, byte_count);
        }
        v.arena = try Buffer.alloc(v.d, total);
        for (v.sizes, offsets, &v.buffers) |byte_count, offset, *buffer| if (byte_count > 0) {
            buffer.* = .{ .d = v.d, .ptr = v.arena.?.ptr + offset, .len = byte_count, .borrowed = true };
        };
        const views = v.spans();
        const args: Create = .{ .weight_raw = views[0], .activation_raw = views[1], .weight_scales = views[2], .activation_scales = views[3], .workspace = views[4] };
        try v.check(v.api.create(&args, &v.handle));
    }

    fn spans(v: *const Kit) [5]DeviceSpan {
        var out: [5]DeviceSpan = @splat(.{});
        for (v.buffers, &out) |buffer, *span| if (buffer) |b| {
            span.* = .{ .address = b.ptr, .bytes = b.len };
        };
        return out;
    }

    pub fn bytes(v: *const Kit) usize {
        return if (v.arena) |arena| arena.len else 0;
    }

    pub fn affine4(v: *Kit, stream: Stream, x: u64, rows: usize, w: anytype, out: u64, kind: qmmf.Out) !bool {
        if (!v.usesAffine4(rows, w)) return false;
        var op = try shape(.affine4, rows, w.n, w.k, w.npad, w.k, kind);
        op.x = x;
        op.weight = w.w;
        op.scale = w.s;
        op.bias = w.b;
        op.out = out;
        return v.execute(stream, &op);
    }

    pub fn usesAffine4(v: *const Kit, rows: usize, w: anytype) bool {
        return supports(rows, w.n, w.k) and v.allows(.affine4, w.n, w.k, w.npad) and
            (w.n == 48 or (rows == 1024 and w.n == 6144 and w.k == 5120));
    }

    pub fn nvfp4(v: *Kit, stream: Stream, x: u64, rows: usize, w: qmmf.Weight, out: u64, kind: qmmf.Out) !bool {
        if (w.mode != .fp4 or !supports(rows, w.n, w.k) or !v.allows(.nvfp4, w.n, w.k, w.npad) or w.act <= 0) return false;
        var op = try shape(.nvfp4, rows, w.n, w.k, w.npad, w.k, kind);
        op.x = x;
        op.weight = w.codes;
        op.scale = w.scales;
        op.act_scale = w.act;
        op.weight_scale = w.scale;
        op.out = out;
        return v.execute(stream, &op);
    }

    pub fn acceptsFp8(v: *const Kit, w: qmmf.Weight) bool {
        const mode: Mode = switch (w.mode) {
            .fp8 => .fp8,
            .fp8g => .fp8_row,
            else => return false,
        };
        return v.allows(mode, w.n, w.k, w.npad);
    }

    pub fn fp8(v: *Kit, stream: Stream, x: u64, rows: usize, w: qmmf.Weight, out: u64, kind: qmmf.Out) !bool {
        if (!v.usesFp8(rows, w)) return false;
        const mode: Mode = if (w.mode == .fp8) .fp8 else .fp8_row;
        var op = try shape(mode, rows, w.n, w.k, w.npad, w.k, kind);
        op.x = x;
        op.weight = w.codes;
        op.scale = if (w.mode == .fp8g) w.scales else 0;
        op.act_scale = w.act;
        op.weight_scale = w.scale;
        op.out = out;
        return v.execute(stream, &op);
    }

    pub fn usesFp8(v: *const Kit, rows: usize, w: qmmf.Weight) bool {
        if (!supports(rows, w.n, w.k) or !v.acceptsFp8(w) or rows < 1024) return false;
        if (rows < 2048) return w.n == 6144 or w.n == 5120 or (w.mode == .fp8 and w.n == 48);
        if (w.mode == .fp8) return true;
        return w.n == 10240 or w.n == 48 or (w.n == 5120 and w.k == 6144);
    }

    pub fn tuneLast(v: *Kit) !void {
        _ = try v.tryTuneLast();
    }

    pub fn tryTuneLast(v: *Kit) !bool {
        const tune = v.api.tune orelse return false;
        const status = tune(v.handle);
        if (status == 1) return false;
        try v.check(status);
        return true;
    }

    fn execute(v: *Kit, stream: Stream, op: *Run) !bool {
        if (v.handle == null) return error.CublasNotAllocated;
        op.stream = if (stream.handle) |pointer| @intFromPtr(pointer) else 0;
        const status = v.api.run(v.handle, op);
        // CUDA failures can follow queued work; drain that stream before freeing caller-owned scratch.
        if (status == 0 or status == 4 or status == 5 or status == 7 or status == 9) v.last_stream = stream;
        if (status == 1) return false;
        try v.check(status);
        return true;
    }

    fn check(v: *const Kit, status: u32) !void {
        if (status == 0) return;
        var message: [512]u8 = @splat(0);
        _ = v.api.last_error(v.handle, &message, message.len);
        const end = std.mem.indexOfScalar(u8, &message, 0) orelse message.len;
        std.log.err("cuBLAS prompt bridge status {d}: {s}", .{ status, message[0..end] });
        return switch (status) {
            1 => error.UnsupportedCublasShape,
            2, 3 => error.BadCublasShape,
            6 => error.CublasContextMismatch,
            7 => error.OutOfMemory,
            8 => error.CublasStreamBusy,
            else => error.CublasPrefillFailed,
        };
    }
};

fn tuningSpan(mode: Mode, rows: usize, columns: usize, used: u64) !usize {
    const live = std.math.cast(usize, used) orelse return error.BadCublasShape;
    if (mode == .fp8 or mode == .fp8_row) return live;
    const start = (try std.math.add(usize, live, 255)) & ~@as(usize, 255);
    const output = try std.math.mul(usize, try std.math.mul(usize, rows, columns), 2);
    return std.math.add(usize, start, output);
}

test "cuBLAS prefill startup tuning has a private result beyond live scale storage" {
    const live: usize = 8192 * (5120 / 64) * 4;
    const span = try tuningSpan(.nvfp4, 8192, 17408, live);
    try std.testing.expect(span > 64 << 20);
    try std.testing.expect(span - live >= 8192 * 17408 * 2);
    try std.testing.expectEqual(@as(usize, 256 + 128 * 48 * 2), try tuningSpan(.affine4, 128, 48, 17));
    try std.testing.expectEqual(@as(usize, 12345), try tuningSpan(.fp8_row, 8192, 17408, 12345));
    try std.testing.expectError(error.Overflow, tuningSpan(.nvfp4, std.math.maxInt(usize), 2, 0));
}

pub fn supports(rows: usize, n: usize, k: usize) bool {
    return rows >= 128 and rows <= @import("qwen_prefill_config").max_rows and rows % 16 == 0 and n > 0 and n % 16 == 0 and k > 0 and k % 64 == 0;
}

fn shape(mode: Mode, rows: usize, n: usize, k: usize, npad: usize, ldx: usize, kind: qmmf.Out) !Run {
    if (rows == 0 or n == 0 or k == 0 or npad < n or ldx < k) return error.BadCublasShape;
    const input = try std.math.add(usize, try std.math.mul(usize, rows - 1, ldx), k);
    const codes = try std.math.mul(usize, npad, k);
    const scales = switch (mode) {
        .nvfp4 => codes / 16,
        .affine4 => codes / 64 * 2,
        .fp8 => 0,
        .fp8_row => codes / 64 * 4,
    };
    const weight_bytes = if (mode == .fp8 or mode == .fp8_row) codes else codes / 2;
    return .{ .mode = mode, .out_kind = if (kind == .bf16) 1 else 2, .m = std.math.cast(u32, rows) orelse return error.BadCublasShape, .n = std.math.cast(u32, n) orelse return error.BadCublasShape, .k = std.math.cast(u32, k) orelse return error.BadCublasShape, .npad = std.math.cast(u32, npad) orelse return error.BadCublasShape, .ldx = std.math.cast(u32, ldx) orelse return error.BadCublasShape, .x_bytes = try std.math.mul(usize, input, 2), .weight_bytes = weight_bytes, .scale_bytes = scales, .bias_bytes = if (mode == .affine4) scales else 0, .out_bytes = try std.math.mul(usize, try std.math.mul(usize, rows, n), if (kind == .bf16) 2 else 4) };
}

test "cuBLAS prefill bridge ABI and source extents match the versioned C interface" {
    try std.testing.expectEqual(@as(usize, 128), @sizeOf(Create));
    try std.testing.expectEqual(@as(usize, 160), @sizeOf(Run));
    try std.testing.expectEqual(@as(usize, 64), @sizeOf(Requirements));
    try std.testing.expectEqual(@as(usize, 40), @offsetOf(Run, "x"));
    try std.testing.expectEqual(@as(usize, 88), @offsetOf(Run, "x_bytes"));
    try std.testing.expectEqual(@as(usize, 128), @offsetOf(Run, "stream"));
    const op = try shape(.affine4, 128, 48, 5120, 64, 5184, .bf16);
    try std.testing.expectEqual(@as(u64, (127 * 5184 + 5120) * 2), op.x_bytes);
    try std.testing.expectEqual(@as(u64, 64 * 5120 / 2), op.weight_bytes);
    try std.testing.expectEqual(@as(u64, 64 * 5120 / 64 * 2), op.scale_bytes);
    try std.testing.expectEqual(op.scale_bytes, op.bias_bytes);
}

test "cuBLAS prefill selection rejects ambiguous values and leaves short prompt tails on the native path" {
    try std.testing.expect(!try selected(null));
    try std.testing.expect(!try selected("0"));
    try std.testing.expect(try selected("1"));
    for ([_][]const u8{ "", "true", "2", "01" }) |value| try std.testing.expectError(error.BadCublasPrefill, selected(value));
    try std.testing.expect(supports(1024, 17408, 5120));
    try std.testing.expect(!supports(1, 17408, 5120));
    try std.testing.expect(!supports(129, 17408, 5120));
}

test "cuBLAS prefill excludes unregistered proposal-head spans before touching scratch" {
    var kit: Kit = undefined;
    kit.admitted_count = 1;
    kit.admitted[0] = .{ .mode = .affine4, .n = 10240, .k = 5120, .npad = 10240 };
    try std.testing.expect(kit.allows(.affine4, 10240, 5120, 10240));
    try std.testing.expect(!kit.allows(.affine4, 98304, 5120, 248320));
    try std.testing.expect(!kit.allows(.nvfp4, 10240, 5120, 10240));
}

test "FP8 bridge extents retain complete codes and row-scale tiles" {
    const scalar = try shape(.fp8, 128, 48, 5120, 64, 5120, .bf16);
    try std.testing.expectEqual(@as(u64, 64 * 5120), scalar.weight_bytes);
    try std.testing.expectEqual(@as(u64, 0), scalar.scale_bytes);
    try std.testing.expectEqual(@as(u64, 0), scalar.bias_bytes);
    const rows = try shape(.fp8_row, 128, 48, 5120, 64, 5120, .f32);
    try std.testing.expectEqual(@as(u64, 64 * (5120 / 64) * 4), rows.scale_bytes);
    try std.testing.expectEqual(@as(u64, 128 * 48 * 4), rows.out_bytes);
}
