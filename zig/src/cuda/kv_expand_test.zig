//! Real-wrapper launch and ownership checks without a CUDA device.
const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;
const kv = @import("kv_expand.zig");
const T = std.testing;

const Fake = struct {
    const Call = struct { symbol: usize, args: [11]u64 = @splat(0), x: u32, y: u32, shared: u32 };
    var calls: [8]Call = undefined;
    var count: usize = 0;
    var allocated: usize = 0;
    var freed: usize = 0;
    var synced: usize = 0;
    var resolved: usize = 0;
    fn notReady(_: abi.Stream) callconv(.c) abi.Result {
        return abi.error_not_ready;
    }
    fn arg(comptime V: type, args: ?[*]?*anyopaque, i: usize) V {
        return @as(*const V, @ptrCast(@alignCast(args.?[i].?))).*;
    }
    fn launch(f: abi.Function, x: c_uint, y: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, shared: c_uint, _: abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
        var call: Call = .{ .symbol = @intFromPtr(f.?), .x = x, .y = y, .shared = shared };
        const pointers: usize = if (call.symbol == 1) 5 else 4;
        for (0..pointers) |i| call.args[i] = arg(u64, args, i);
        if (call.symbol == 1) {
            for (5..8) |i| call.args[i] = @intCast(arg(i32, args, i));
            call.args[8] = arg(u64, args, 8);
            call.args[9] = @intCast(arg(i64, args, 9));
        } else {
            for (4..9) |i| call.args[i] = @intCast(arg(i32, args, i));
            call.args[9] = @as(u32, @bitCast(arg(f32, args, 9)));
        }
        call.args[10] = @intCast(arg(i32, args, 10));
        calls[count] = call;
        count += 1;
        return abi.success;
    }
    fn alloc(out: *u64, bytes: usize) callconv(.c) abi.Result {
        allocated = bytes;
        out.* = 0x400000000;
        return abi.success;
    }
    fn free(_: u64) callconv(.c) abi.Result {
        freed += 1;
        return abi.success;
    }
    fn sync(_: abi.Stream) callconv(.c) abi.Result {
        synced += 1;
        return abi.success;
    }
    fn function(out: *abi.Function, _: abi.Module, name: [*:0]const u8) callconv(.c) abi.Result {
        if (!std.mem.eql(u8, std.mem.span(name), kv.symbol)) return abi.error_not_found;
        resolved += 1;
        out.* = @ptrFromInt(1);
        return abi.success;
    }
};
fn function(d: *const Driver, id: usize) Function {
    return .{ .d = d, .handle = @ptrFromInt(id) };
}
fn operator(d: *const Driver) kv.Expanded {
    return .{ .d = d, .expand = function(d, 1), .attention = function(d, 2), .scratch = .{ .d = d, .ptr = 0x400000000, .len = kv.scratch_bytes, .borrowed = true } };
}
fn input(paged: bool) kv.Input {
    const rows: u64 = 4 * (if (paged) @as(u64, 48) else 8192);
    const k: u64 = 0x200000000;
    return .{ .q = 0x100000000, .k = k, .v = k + rows * 256, .k_scales = k + rows * 512, .v_scales = k + rows * 516, .output = 0x110000000, .pages = if (paged) 0x300000000 else 0, .shape = .{ .rows = 17, .prefix = 65, .heads = 24, .kv_heads = 4, .capacity = 8192, .scale = 1.0 / 16.0, .page_tokens = if (paged) 48 else 0, .page_bytes = if (paged) 4 * 48 * 520 else 0 } };
}

test "KV expansion flag is strict and the scratch is bounded" {
    try T.expect(!try kv.selected(null));
    try T.expect(!try kv.selected("0"));
    try T.expect(try kv.selected("1"));
    for ([_][]const u8{ "", "true", "2", "01", " 1" }) |bad| try T.expectError(error.InvalidExpandKVFlag, kv.selected(bad));
    try T.expectEqual(@as(usize, 64 << 20), kv.scratch_bytes);
}

test "KV expansion preserves source strides and uses filled length for BF16 attention" {
    var d: Driver = undefined;
    d.api.cuLaunchKernel = Fake.launch;
    const stream: Stream = .{ .d = &d, .handle = null };
    var op = operator(&d);
    for ([_]bool{ false, true }) |paged| {
        const p = input(paged);
        Fake.count = 0;
        try T.expect(try op.run(stream, p));
        try T.expectEqual(@as(usize, 2), Fake.count);
        const expand = Fake.calls[0];
        const attention = Fake.calls[1];
        try T.expectEqual(@as(usize, 1), expand.symbol);
        try T.expectEqualSlices(u64, &.{ p.k, p.v, p.k_scales, p.v_scales, op.scratch.ptr }, expand.args[0..5]);
        try T.expectEqualSlices(u64, &.{ 82, 4, 8192, p.pages, p.shape.page_bytes, p.shape.page_tokens }, expand.args[5..11]);
        try T.expectEqual(@as(u32, 82), expand.x);
        try T.expectEqual(@as(usize, 2), attention.symbol);
        try T.expectEqualSlices(u64, &.{ p.q, op.scratch.ptr, op.scratch.ptr + 4 * 82 * 256 * 2, p.output }, attention.args[0..4]);
        try T.expectEqualSlices(u64, &.{ 65, 17, 24, 4, 6 }, attention.args[4..9]);
        try T.expectEqual(@as(u64, 82), attention.args[10]);
        try T.expectEqual(@as(u32, 1), attention.x);
        try T.expectEqual(@as(u32, 12), attention.y);
        try T.expectEqual(kv.attention_shared, attention.shared);
        try T.expect(try op.run(stream, p));
        try T.expectEqual(@as(usize, 4), Fake.count);
        try T.expectEqual(@as(usize, 1), Fake.calls[2].symbol);
    }
}

test "KV expansion refuses aliases and short scratch before launching and falls back above cap" {
    var d: Driver = undefined;
    d.api.cuLaunchKernel = Fake.launch;
    const stream: Stream = .{ .d = &d, .handle = null };
    var op = operator(&d);
    var p = input(false);
    Fake.count = 0;
    p.shape.prefix = kv.max_filled;
    p.shape.rows = 1;
    p.shape.capacity = 32768;
    try T.expect(!try op.run(stream, p));
    p = input(false);
    p.output = p.q + 2;
    try T.expectError(error.ExpandKVAlias, op.run(stream, p));
    p.output = op.scratch.ptr + 16;
    try T.expectError(error.ExpandKVAlias, op.run(stream, p));
    p = input(false);
    p.k = op.scratch.ptr;
    try T.expectError(error.ExpandKVAlias, op.run(stream, p));
    p = input(true);
    p.pages = op.scratch.ptr;
    try T.expectError(error.ExpandKVAlias, op.run(stream, p));
    p = input(false);
    op.scratch.len = 1024;
    try T.expectError(error.InsufficientExpandKVScratch, op.run(stream, p));
    try T.expectEqual(@as(usize, 0), Fake.count);
}

test "KV expansion owns one allocation and borrows both module and BF16 kernel" {
    var d: Driver = undefined;
    d.api.cuModuleGetFunction = Fake.function;
    d.api.cuMemAlloc_v2 = Fake.alloc;
    d.api.cuMemFree_v2 = Fake.free;
    d.api.cuStreamSynchronize = Fake.sync;
    Fake.allocated = 0;
    Fake.freed = 0;
    Fake.synced = 0;
    Fake.resolved = 0;
    const module: Module = .{ .d = &d, .handle = @ptrFromInt(7) };
    var op = try kv.Expanded.borrow(module, function(&d, 2));
    try T.expectEqual(@as(usize, 1), Fake.resolved);
    try T.expectEqual(kv.scratch_bytes, Fake.allocated);
    op.last_stream = .{ .d = &d, .handle = null };
    op.deinit();
    try T.expectEqual(@as(usize, 1), Fake.synced);
    try T.expectEqual(@as(usize, 1), Fake.freed);
}

test "KV expansion refuses an active cross-stream scratch rebind before either launch" {
    var d: Driver = undefined;
    d.api.cuLaunchKernel = Fake.launch;
    d.api.cuStreamQuery = Fake.notReady;
    var op = operator(&d);
    op.last_stream = .{ .d = &d, .handle = @ptrFromInt(1) };
    Fake.count = 0;
    try T.expectError(error.ExpandKVBusy, op.run(.{ .d = &d, .handle = @ptrFromInt(2) }, input(false)));
    try T.expectEqual(@as(usize, 0), Fake.count);
}

test "KV expansion falls back for padded physical capacities without poisoning later calls" {
    var d: Driver = undefined;
    d.api.cuLaunchKernel = Fake.launch;
    const stream: Stream = .{ .d = &d, .handle = null };
    var op = operator(&d);
    for ([_]u32{ 131072, 131073, 139264 }) |capacity| {
        for ([_]u32{ 16384, 16385 }) |filled| {
            var p = input(true);
            p.shape.capacity = capacity;
            p.shape.prefix = filled - 1;
            p.shape.rows = 1;
            Fake.count = 0;
            const use_expansion = capacity == 131072 and filled == 16384;
            try T.expectEqual(use_expansion, try op.run(stream, p));
            try T.expectEqual(@as(usize, if (use_expansion) 2 else 0), Fake.count);
            if (capacity > 131072)
                try T.expectError(error.UnsupportedFp8AttentionShape, p.validate());
        }
    }
    var long = input(true);
    long.shape.capacity = 139264;
    long.shape.prefix = 90112;
    long.shape.rows = 705;
    Fake.count = 0;
    try T.expect(!try op.run(stream, long));
    try T.expectEqual(@as(usize, 0), Fake.count);
    Fake.count = 0;
    try T.expect(try op.run(stream, input(true)));
    try T.expectEqual(@as(usize, 2), Fake.count);
    try T.expectEqual(@as(u64, 82), Fake.calls[0].args[5]);
    try T.expectEqual(@as(u64, 82), Fake.calls[1].args[10]);
}

test "KV expansion validates malformed fallback shapes and addresses before launching" {
    var d: Driver = undefined;
    d.api.cuLaunchKernel = Fake.launch;
    const stream: Stream = .{ .d = &d, .handle = null };
    var op = operator(&d);
    var p = input(true);
    p.shape.capacity = 139264;
    p.shape.prefix = 16384;
    p.shape.rows = 1;
    Fake.count = 0;
    p.shape.page_tokens = 0;
    try T.expectError(error.UnsupportedFp8AttentionPages, op.run(stream, p));
    p.shape.page_tokens = 48;
    const query = p.q;
    p.q = 0;
    try T.expectError(error.InvalidFp8AttentionPointer, op.run(stream, p));
    p.q = query;
    p.pages += 2;
    try T.expectError(error.InvalidFp8AttentionPointer, op.run(stream, p));
    p.pages -= 2;
    p.k = std.math.maxInt(u64) - 15;
    try T.expectError(error.Overflow, op.run(stream, p));
    p.k = input(true).k;
    p.output = p.q + 2;
    try T.expectError(error.ExpandKVAlias, op.run(stream, p));
    p.output = input(true).output;
    p.shape.prefix = p.shape.capacity;
    try T.expectError(error.UnsupportedFp8AttentionShape, op.run(stream, p));
    p.shape.prefix = 0;
    p.shape.capacity = std.math.maxInt(u32);
    try T.expectError(error.UnsupportedFp8AttentionShape, op.run(stream, p));
    try T.expectEqual(@as(usize, 0), Fake.count);
}
