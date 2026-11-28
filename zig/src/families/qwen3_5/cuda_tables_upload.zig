//! Compact HtoD spans for the tables consumed by a text forward.
const std = @import("std");
const cuda = @import("cuda");
const Round = @import("cuda_round.zig");
const Seg = @import("cuda_cache.zig").Seg;
const Step = @import("core").tree_slots.Step;
const Shape = @import("cuda_shapes.zig").Shape;

pub const Span = struct { offset: usize, len: usize };

fn add(out: *[9]Span, used: *usize, offset: usize, len: usize) void {
    out[used.*] = .{ .offset = offset, .len = len };
    used.* += 1;
}

/// Device row counts come from params; captured padding reads only the shape's ID bucket.
pub fn plan(out: *[9]Span, bucket: usize, total: usize, segments: usize, record: bool, draws: usize) []const Span {
    var used: usize = 0;
    add(out, &used, Round.ids_at, bucket * 4);
    add(out, &used, Round.segs_at, segments * @sizeOf(Seg));
    add(out, &used, Round.rowseg_at, total * 4);
    add(out, &used, Round.rowpar_at, total * 4);
    add(out, &used, Round.rowanc_at, total * @sizeOf(Round.Ancestor));
    if (record) add(out, &used, Round.steps_at, total * @sizeOf(Step));
    if (draws > 0) {
        add(out, &used, Round.rules_at, draws * @sizeOf(cuda.sample.Rule));
        add(out, &used, Round.meta_at, draws * 4);
    }
    add(out, &used, Round.params_at, Round.src_at - Round.params_at);
    return out[0..used];
}

pub fn upload(e: anytype, st: []const u8, total: usize, segments: usize, record: bool, draws: usize, images: bool) !void {
    if (e.lever7.active_tables and record) {
        if (e.active_staged == null) e.active_staged = try cuda.copy_source.Buffer.alloc(e.d, Round.src_at);
        const staging = &e.active_staged.?;
        try staging.wait();
        const bucket: usize = Shape.init(total, segments, record, false, 1).total;
        var spans: [9]Span = undefined;
        const live = plan(&spans, bucket, total, segments, record, draws);
        for (live) |span| @memcpy(staging.bytes[span.offset .. span.offset + span.len], st[span.offset .. span.offset + span.len]);
        for (live) |span| try staging.copyTo(e.tables, span.offset, staging.bytes[span.offset .. span.offset + span.len], e.stream);
    } else try e.tables.uploadAsync(0, st[0..if (images) Round.bytes else Round.src_at], e.stream.handle);
}

/// Independent consumer geometry: bucket IDs and live rows; padding elsewhere remains unread.
fn consume(device: []const u8, out: []u8, bucket: usize, total: usize, segments: usize, record: bool, draws: usize) void {
    var at: usize = 0;
    const take = struct {
        fn f(src: []const u8, dst: []u8, offset: usize, len: usize, cursor: *usize) void {
            @memcpy(dst[cursor.* .. cursor.* + len], src[offset .. offset + len]);
            cursor.* += len;
        }
    }.f;
    take(device, out, Round.ids_at, bucket * 4, &at);
    take(device, out, Round.segs_at, segments * @sizeOf(Seg), &at);
    take(device, out, Round.rowseg_at, total * 4, &at);
    take(device, out, Round.rowpar_at, total * 4, &at);
    take(device, out, Round.rowanc_at, total * @sizeOf(Round.Ancestor), &at);
    if (record) take(device, out, Round.steps_at, total * @sizeOf(Step), &at);
    if (draws > 0) {
        take(device, out, Round.rules_at, draws * @sizeOf(cuda.sample.Rule), &at);
        take(device, out, Round.meta_at, draws * 4, &at);
    }
    take(device, out, Round.params_at, Round.src_at - Round.params_at, &at);
}

fn consumedLen(bucket: usize, total: usize, segments: usize, record: bool, draws: usize) usize {
    return bucket * 4 + segments * @sizeOf(Seg) + total * (8 + @sizeOf(Round.Ancestor)) +
        (if (record) total * @sizeOf(Step) else 0) + (if (draws > 0) draws * (@sizeOf(cuda.sample.Rule) + 4) else 0) + Round.src_at - Round.params_at;
}

test "REV295 active tables preserve consumed bytes while inactive planes stay poisoned" {
    var staged: [Round.bytes]u8 = undefined;
    for (&staged, 0..) |*byte, i| byte.* = @truncate(i *% 37 +% 11);
    var full: [Round.bytes]u8 = @splat(0xa5);
    var active: [Round.bytes]u8 = @splat(0xa5);
    var d = Upload.driver();
    const previous = cuda.copy_source.current();
    cuda.copy_source.setCurrent(@ptrFromInt(0x351a));
    defer cuda.copy_source.setCurrent(previous);
    var e: UploadEngine = .{ .d = &d, .stream = .{ .d = &d, .handle = null }, .tables = .{ .d = &d, .ptr = @intFromPtr(&active), .len = active.len, .borrowed = true } };
    defer if (e.active_staged) |*b| b.free();
    @memcpy(&full, &staged);
    try upload(&e, &staged, 37, 9, true, 0, false);
    try std.testing.expectEqual(@as(usize, 7), Upload.copies);
    var full_read: [Round.bytes]u8 = undefined;
    var active_read: [Round.bytes]u8 = undefined;
    consume(&full, &full_read, 48, 37, 9, true, 0);
    consume(&active, &active_read, 48, 37, 9, true, 0);
    const len = consumedLen(48, 37, 9, true, 0);
    try std.testing.expectEqualSlices(u8, full_read[0..len], active_read[0..len]);
    try std.testing.expectEqual(@as(u8, 0xa5), active[Round.ids_at + 48 * 4]);
    try std.testing.expectEqual(@as(u8, 0xa5), active[Round.rowseg_at + 37 * 4]);
}

test "REV295 draw metadata and non-record rounds retain exact active spans" {
    var spans: [9]Span = undefined;
    const draw_spans = plan(&spans, 8, 7, 4, false, 7);
    try std.testing.expectEqual(@as(usize, 8), draw_spans.len);
    try std.testing.expectEqual(Round.rules_at, draw_spans[5].offset);
    const no_draws = plan(&spans, 8, 7, 4, true, 0);
    try std.testing.expectEqual(@as(usize, 7), no_draws.len);
    for (no_draws) |span| try std.testing.expect(span.offset != Round.rules_at and span.offset != Round.meta_at);
}

const Upload = struct {
    var host: ?[]align(16) u8 = null;
    var copies: usize = 0;
    fn copy(dst: u64, src: ?*const anyopaque, len: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        @memcpy(@as([*]u8, @ptrFromInt(dst))[0..len], @as([*]const u8, @ptrCast(src.?))[0..len]);
        copies += 1;
        return 0;
    }
    fn hostAlloc(out: *?*anyopaque, len: usize, _: c_uint) callconv(.c) cuda.abi.Result {
        host = std.testing.allocator.alignedAlloc(u8, .@"16", len) catch return cuda.abi.error_not_ready;
        out.* = host.?.ptr;
        return 0;
    }
    fn hostFree(_: ?*anyopaque) callconv(.c) cuda.abi.Result {
        std.testing.allocator.free(host.?);
        host = null;
        return 0;
    }
    fn event(out: *cuda.abi.Event, _: c_uint) callconv(.c) cuda.abi.Result {
        out.* = @ptrFromInt(1);
        return 0;
    }
    fn eventSync(_: cuda.abi.Event) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn driver() cuda.Driver {
        copies = 0;
        var d: cuda.Driver = undefined;
        d.api.cuMemcpyHtoDAsync_v2 = copy;
        d.api.cuMemHostAlloc = hostAlloc;
        d.api.cuMemFreeHost = hostFree;
        d.api.cuEventCreate = event;
        d.api.cuEventRecord = record;
        d.api.cuEventSynchronize = eventSync;
        d.api.cuEventDestroy_v2 = eventSync;
        return d;
    }
    fn record(_: cuda.abi.Event, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
};

const UploadEngine = struct {
    d: *const cuda.Driver,
    stream: cuda.Stream,
    tables: cuda.DeviceBuffer,
    lever7: @import("cuda_lever7.zig").Options = .{ .active_tables = true },
    active_staged: ?cuda.copy_source.Buffer = null,
};
