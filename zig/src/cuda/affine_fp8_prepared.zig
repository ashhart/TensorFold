//! Optional per-projection integer-code expansion; one owned buffer and no cross-call weight cache.
const std = @import("std");
const Driver = @import("driver.zig").Driver;
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const Buffer = @import("memory.zig").DeviceBuffer;
const Stream = @import("stream.zig").Stream;
const launch = @import("launch.zig");
const kernels = @import("kernels.zig");
const Affine4 = @import("qlinear.zig").Affine4;
const Out = @import("qmmf.zig").Out;

pub fn selected(raw: ?[]const u8) !bool {
    const value = raw orelse return false;
    if (std.mem.eql(u8, value, "0")) return false;
    if (std.mem.eql(u8, value, "1")) return true;
    return error.InvalidAffinePrepared;
}
pub fn requested() !bool {
    return selected(try @import("sys").environment.given("TF_QWEN_AFFINE_PREPARED"));
}
pub fn selectedRows(rows: usize) bool {
    return rows >= 1024 and rows <= @import("qwen_prefill_config").max_rows;
}
pub fn required(w: Affine4) ?usize {
    const max = std.math.maxInt(i32);
    if (w.n == 0 or w.n > max or w.k == 0 or w.k > max or w.k % 64 != 0 or
        w.npad > max or w.npad % 64 != 0 or w.npad < (w.n + 63) / 64 * 64) return null;
    return std.math.mul(usize, w.npad, w.k) catch null;
}

pub const Prepared = struct {
    module: Module,
    prepare: Function,
    products: [2]Function,
    buffer: ?Buffer = null,
    last_stream: ?Stream = null,

    pub fn load(d: *const Driver) !Prepared {
        const image = if (@hasDecl(kernels, "affine_fp8_prepared")) kernels.affine_fp8_prepared else return error.BuiltWithoutAffinePrepared;
        var module = try Module.load(d, image);
        errdefer module.unload();
        return .{ .module = module, .prepare = try module.function("tf_affine_fp8_prepare"), .products = .{ try module.function("tf_affine_fp8_prepared"), try module.function("tf_affine_fp8_prepared_f32") } };
    }
    pub fn reserve(p: *Prepared, bytes: usize) !void {
        if (bytes == 0) return;
        if (p.buffer) |buffer| if (buffer.len >= bytes) return;
        if (p.last_stream != null) return error.AffinePreparedAlreadyUsed;
        const next = try Buffer.alloc(p.module.d, bytes);
        if (p.buffer) |*buffer| buffer.free();
        p.buffer = next;
    }
    pub fn unload(p: *Prepared) void {
        if (p.last_stream) |s| @import("completion.zig").drain(p.module.d, s, "affine FP8 prepared work did not drain");
        if (p.buffer) |*buffer| buffer.free();
        p.module.unload();
        p.* = undefined;
    }
    pub fn fits(p: *const Prepared, w: Affine4) bool {
        const bytes = required(w) orelse return false;
        return if (p.buffer) |buffer| buffer.len >= bytes else false;
    }
    pub fn capacity(p: *const Prepared) usize {
        return if (p.buffer) |buffer| buffer.len else 0;
    }
    pub fn admit(p: *Prepared, s: Stream) !void {
        if (p.last_stream) |last| if (last.handle != s.handle or last.d != s.d) {
            if (last.d != s.d or !try last.done()) return error.AffinePreparedBusy;
        };
        p.last_stream = s;
    }

    pub fn matmul(p: *Prepared, s: Stream, x: u64, scales: u64, sums: u64, rows: usize, w: Affine4, out: u64, kind: Out) !bool {
        if (!p.fits(w)) return false;
        if (rows == 0 or rows > std.math.maxInt(i32)) return error.InvalidFp8PromptShape;
        const pool = try Span.init(p.buffer.?.ptr, p.buffer.?.len);
        const output = try Span.init(out, try std.math.mul(usize, rows, try std.math.mul(usize, w.n, if (kind == .f32) 4 else 2)));
        if (pool.overlaps(output)) return error.InvalidFp8PromptAlias;
        const groups = w.k / 64;
        const reads = [_]Span{
            try Span.init(x, try std.math.mul(usize, rows, w.k)),
            try Span.init(scales, try std.math.mul(usize, rows, groups * 4)),
            try Span.init(sums, try std.math.mul(usize, rows, groups * 4)),
            try Span.init(w.w, required(w).? / 2),
            try Span.init(w.s, try std.math.mul(usize, w.npad, groups * 2)),
            try Span.init(w.b, try std.math.mul(usize, w.npad, groups * 2)),
        };
        for (reads) |read| if (pool.overlaps(read) or output.overlaps(read)) return error.InvalidFp8PromptAlias;
        try p.admit(s);
        var pack: launch.Args = .{};
        pack.add(w.w);
        pack.add(pool.begin);
        pack.add(@as(i32, @intCast(w.k)));
        try launch.launch(p.prepare, .{ .grid = .{ .x = @intCast(groups), .y = @intCast(w.npad / 64) }, .block = .{ .x = 256 } }, s, &pack);
        var args: launch.Args = .{};
        for ([_]u64{ x, scales, sums, pool.begin, w.s, w.b, out }) |address| args.add(address);
        for ([_]usize{ rows, w.n, w.k, w.npad }) |value| args.add(@as(i32, @intCast(value)));
        try launch.launch(p.products[@intFromBool(kind == .f32)], .{
            .grid = .{ .x = @intCast((w.n + 127) / 128), .y = @intCast((rows + 63) / 64) },
            .block = .{ .x = 256 },
        }, s, &args);
        return true;
    }
};

const Span = struct {
    begin: u64,
    end: u64,
    fn init(address: u64, count: usize) !Span {
        if (address == 0) return error.InvalidFp8PromptAlias;
        return .{ .begin = address, .end = try std.math.add(u64, address, count) };
    }
    fn overlaps(a: Span, b: Span) bool {
        return a.begin < b.end and b.begin < a.end;
    }
};

test "FP8 prompt affine prepared flag is strict and row selection preserves short prompts" {
    try std.testing.expect(!try selected(null));
    try std.testing.expect(!try selected("0"));
    try std.testing.expect(try selected("1"));
    for ([_][]const u8{ "", "true", "2", "01", " 1", "1 " }) |bad|
        try std.testing.expectError(error.InvalidAffinePrepared, selected(bad));
    const max_rows = @import("qwen_prefill_config").max_rows;
    for ([_]usize{ 1, 17, 1023, max_rows + 1 }) |rows| try std.testing.expect(!selectedRows(rows));
    try std.testing.expect(selectedRows(1024) and selectedRows(4096));
    try std.testing.expectEqual(max_rows == 8192, selectedRows(8192));
}

test "FP8 prompt affine prepared owns one buffer and drains before freeing it" {
    const abi = @import("abi.zig");
    const Fake = struct {
        var events: [6]u8 = undefined;
        var count: usize = 0;
        fn event(value: u8) void {
            events[count] = value;
            count += 1;
        }
        fn alloc(out: *u64, _: usize) callconv(.c) abi.Result {
            event(1);
            out.* = 0x900000 + count * 0x10000;
            return abi.success;
        }
        fn free(_: u64) callconv(.c) abi.Result {
            event(2);
            return abi.success;
        }
        fn sync(_: abi.Stream) callconv(.c) abi.Result {
            event(3);
            return abi.success;
        }
        fn unload(_: abi.Module) callconv(.c) abi.Result {
            event(4);
            return abi.success;
        }
    };
    var d: Driver = undefined;
    d.api.cuMemAlloc_v2 = Fake.alloc;
    d.api.cuMemFree_v2 = Fake.free;
    d.api.cuStreamSynchronize = Fake.sync;
    d.api.cuModuleUnload = Fake.unload;
    var p: Prepared = .{ .module = .{ .d = &d, .handle = @ptrFromInt(1) }, .prepare = undefined, .products = undefined };
    Fake.count = 0;
    try p.reserve(0);
    try std.testing.expectEqual(@as(usize, 0), p.capacity());
    try p.reserve(4096);
    try p.reserve(2048);
    try std.testing.expectEqual(@as(usize, 1), Fake.count);
    try p.reserve(8192);
    try std.testing.expectEqual(@as(usize, 8192), p.capacity());
    try p.admit(.{ .d = &d, .handle = null });
    try std.testing.expectError(error.AffinePreparedAlreadyUsed, p.reserve(16384));
    try std.testing.expectEqual(@as(usize, 3), Fake.count);
    p.unload();
    try std.testing.expectEqualSlices(u8, &.{ 1, 1, 2, 3, 2, 4 }, &Fake.events);
}
