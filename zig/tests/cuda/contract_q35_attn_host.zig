//! Record the real contract adapter's ABI and geometry, without loading a GPU library or image.
const std = @import("std");
const cuda = @import("cuda");
const adapter = @import("contract_q35_attn.zig");
const T = std.testing;
const Geo = extern struct { hid: c_int, nk: c_int, nv: c_int, qh: c_int, kvh: c_int, nl: c_int };
const Call = struct {
    id: usize,
    grid: [3]u32,
    block: [3]u32,
    shared: u32,
    geo: Geo,
    args: [12]u64 = @splat(0),
};
const Trace = struct {
    var calls: std.ArrayList(Call) = .empty;
    var attrs: std.ArrayList(usize) = .empty;
    var bad_attribute = false;
    fn reset() void {
        calls.clearRetainingCapacity();
        attrs.clearRetainingCapacity();
        bad_attribute = false;
    }
    fn deinit() void {
        calls.deinit(T.allocator);
        attrs.deinit(T.allocator);
        calls = .empty;
        attrs = .empty;
    }
    fn function(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) cuda.abi.Result {
        const names = [_][]const u8{ "tf_q35_attn_partial", "tf_q35_attn_rows", "tf_q35_attn_merge", "tf_q35_attn_partial8", "tf_q35_attn_rows8" };
        for (names, 1..) |n, id| if (std.mem.eql(u8, n, std.mem.span(name))) {
            out.* = @ptrFromInt(id);
            return 0;
        };
        return cuda.abi.error_not_found;
    }
    fn attribute(f: cuda.abi.Function, a: cuda.abi.FunctionAttribute, bytes: c_int) callconv(.c) cuda.abi.Result {
        if (a != .max_dynamic_shared_size_bytes or bytes != 65536) bad_attribute = true;
        attrs.append(T.allocator, @intFromPtr(f)) catch return 2;
        return 0;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn download(dst: ?*anyopaque, _: u64, bytes: usize) callconv(.c) cuda.abi.Result {
        @memset(@as([*]u8, @ptrCast(dst.?))[0..bytes], 0);
        return 0;
    }
    fn launch(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, _: cuda.abi.Stream, a: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const id = @intFromPtr(f);
        const long = id == 2 or id == 5;
        var call: Call = .{ .id = id, .grid = .{ gx, gy, gz }, .block = .{ bx, by, bz }, .shared = shared, .geo = read(Geo, a, 0) };
        if (id == 3) {
            for (1..9) |i| call.args[i] = read(u64, a, i);
        } else {
            for (1..5) |i| call.args[i] = read(u64, a, i);
            call.args[5] = @intCast(read(c_int, a, 5));
            call.args[6] = read(u64, a, 6);
            call.args[7] = @as(u32, @bitCast(read(f32, a, 7)));
            if (long) call.args[8] = @intCast(read(c_int, a, 8));
            for (8 + @as(usize, @intFromBool(long))..11 + @as(usize, @intFromBool(long))) |i| call.args[i] = read(u64, a, i);
        }
        calls.append(T.allocator, call) catch return 2;
        return 0;
    }
    fn read(comptime V: type, a: ?[*]?*anyopaque, i: usize) V {
        return @as(*const V, @ptrCast(@alignCast(a.?[i].?))).*;
    }
};
fn driver() cuda.Driver {
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuModuleGetFunction = Trace.function;
    d.api.cuFuncSetAttribute = Trace.attribute;
    d.api.cuLaunchKernel = Trace.launch;
    d.api.cuStreamSynchronize = Trace.sync;
    d.api.cuMemcpyDtoH_v2 = Trace.download;
    return d;
}
fn make(d: *cuda.Driver, form: adapter.Form, rows: usize, keys: usize) !adapter.Q35 {
    var q = try adapter.Q35.init(.{ .gpu = .{ .d = d, .ctx = undefined, .gpa = T.allocator, .io = undefined }, .s = .borrowed(d, null) }, form, .{ .d = d, .handle = @ptrFromInt(100) }, .{ .d = d, .handle = @ptrFromInt(200) });
    q.shape = .{ .rows = rows, .n = 24, .k = keys };
    q.dev = .{ 0x100000, 0x200000, 0x300000 };
    q.segs = 0x400000;
    q.part = .{ 0x500000, 0x600000, 0x700000 };
    q.out = 0x800000;
    q.base = .{ 0x900000, 0xa00000, 0xb00000, 0xc00000 };
    q.chunks = 10;
    return q;
}
fn partial(call: Call, cache: adapter.Cache, long: bool, rows: usize, first: usize, chunks: u32, p: [3]u64) !void {
    try T.expectEqual(if (cache == .bf16) (if (long) @as(usize, 2) else 1) else if (long) @as(usize, 5) else 4, call.id);
    // Engine: short rows each get a block, long rows pack 64/G rows, chunk is y and KV head is z.
    const groups = (rows + 64 / 6 - 1) / (64 / 6);
    try T.expectEqual([3]u32{ @intCast(if (long) groups else rows), chunks, 4 }, call.grid);
    try T.expectEqual([3]u32{ if (long) 256 else 128, 1, 1 }, call.block);
    try T.expectEqual(@as(u32, if (cache == .bf16) 65536 else 0), call.shared);
    try T.expectEqualDeep(Geo{ .hid = 5120, .nk = 16, .nv = 48, .qh = 24, .kvh = 4, .nl = 1 }, call.geo);
    try T.expectEqualSlices(u64, &.{ 0x100000 + first * 24 * 256 * 2, 0x400000, 0x200000 + first * 4, 0x300000 + first * 16, 0, 0, @as(u32, @bitCast(@as(f32, 1.0 / 16.0))) }, call.args[1..8]);
    if (long) try T.expectEqual(rows, call.args[8]);
    const at: usize = if (long) 9 else 8;
    try T.expectEqualSlices(u64, &.{ p[0] + first * 24 * 64 * 4, p[1] + first * 24 * 64 * 4, p[2] + first * 24 * 64 * 256 * 4 }, call.args[at..][0..3]);
}
fn merge(call: Call, rows: usize, p: [3]u64, out: u64) !void {
    try T.expectEqual(@as(usize, 3), call.id);
    try T.expectEqual([3]u32{ 24, @intCast(rows), 1 }, call.grid);
    try T.expectEqual([3]u32{ 32, 1, 1 }, call.block);
    try T.expectEqual(@as(u32, 0), call.shared);
    try T.expectEqualSlices(u64, &.{ p[0], p[1], p[2], 0x400000, 0x200000, 0x300000, 0, out }, call.args[1..9]);
}

test "q35 contract host: actual adapter launches short and long TC paths then merge for both caches" {
    defer Trace.deinit();
    var d = driver();
    for (adapter.forms) |form| for ([_]usize{ 1, 9, 10, 11, 16, 64, 128 }) |rows| {
        // Both entries are needed even for mixed streams: length-based selection happens on the device.
        for ([_]usize{ 700, 8191, 8192, 9000 }) |keys| {
            Trace.reset();
            var q = try make(&d, form, rows, keys);
            defer q.deinit();
            try q.run();
            try T.expectEqual(@as(usize, 3), Trace.calls.items.len);
            const z: u32 = if (form.control) 9 else 64;
            try partial(Trace.calls.items[0], form.cache, false, rows, 0, z, q.part);
            try partial(Trace.calls.items[1], form.cache, true, rows, 0, z, q.part);
            try merge(Trace.calls.items[2], rows, q.part, q.out);
            try T.expect(!Trace.bad_attribute);
            try T.expectEqualSlices(usize, if (form.cache == .bf16) &.{ 1, 2 } else &.{}, Trace.attrs.items);
        }
    };
}

test "q35 contract host: each-row replacement retains TC paths, offsets and long-row total" {
    defer Trace.deinit();
    var d = driver();
    for (adapter.forms[0..2]) |form| {
        Trace.reset();
        var q = try make(&d, form, 17, 9000);
        defer q.deinit();
        _ = try q.replaced();
        try T.expectEqual(@as(usize, 35), Trace.calls.items.len);
        for (0..17) |row| {
            try partial(Trace.calls.items[2 * row], form.cache, false, 1, row, 64, q.base[0..3].*);
            try partial(Trace.calls.items[2 * row + 1], form.cache, true, 1, row, 64, q.base[0..3].*);
        }
        try merge(Trace.calls.items[34], 17, q.base[0..3].*, q.base[3]);
    }
}
