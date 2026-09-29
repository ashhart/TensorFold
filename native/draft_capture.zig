const std = @import("std");
const Sampling = @import("sampling.zig").Sampling;
const max_record_bytes = 16 * 1024 * 1024;

fn put(comptime T: type, bytes: []u8, at: *usize, value: T) void {
    const U = @Int(.unsigned, @bitSizeOf(T));
    std.mem.writeInt(U, bytes[at.*..][0..@sizeOf(T)], @bitCast(value), .little);
    at.* += @sizeOf(T);
}

pub fn contextRecord(a: std.mem.Allocator, start: i64, width: usize, tokens: []const i32, bits: []const u16) ![]u8 {
    if (start < 0 or width == 0 or width > std.math.maxInt(i32) or tokens.len == 0 or tokens.len > std.math.maxInt(i32)) return error.InvalidCaptureShape;
    if (bits.len != try std.math.mul(usize, tokens.len, width)) return error.InvalidCaptureShape;
    const bytes = try std.math.add(usize, 16, try std.math.add(usize, try std.math.mul(usize, bits.len, 2), try std.math.mul(usize, tokens.len, 4)));
    if (bytes > max_record_bytes) return error.CaptureRecordTooLarge;
    const out = try a.alloc(u8, bytes);
    var at: usize = 0;
    put(i64, out, &at, start);
    put(i32, out, &at, @intCast(tokens.len));
    put(i32, out, &at, @intCast(width));
    for (bits) |value| put(u16, out, &at, value);
    for (tokens) |value| put(i32, out, &at, value);
    return out;
}

pub fn targetRecord(a: std.mem.Allocator, positions: []const i64, k: usize, ids: []const i32, logits: []const f32) ![]u8 {
    if (positions.len == 0 or positions.len > std.math.maxInt(i32) or k == 0 or k > std.math.maxInt(i32)) return error.InvalidCaptureShape;
    if (ids.len != logits.len or ids.len != try std.math.mul(usize, positions.len, k)) return error.InvalidCaptureShape;
    const bytes = try std.math.add(usize, 8, try std.math.add(usize, try std.math.mul(usize, positions.len, 8), try std.math.mul(usize, ids.len, 8)));
    if (bytes > max_record_bytes) return error.CaptureRecordTooLarge;
    const out = try a.alloc(u8, bytes);
    var at: usize = 0;
    put(i32, out, &at, @intCast(positions.len));
    put(i32, out, &at, @intCast(k));
    for (positions) |value| put(i64, out, &at, value);
    for (ids) |value| put(i32, out, &at, value);
    for (logits) |value| put(f32, out, &at, value);
    return out;
}

pub const Writer = struct {
    const Record = struct { target: bool, bytes: []u8 };
    a: std.mem.Allocator,
    io: std.Io,
    base: []const u8,
    context: std.Io.File,
    target: std.Io.File,
    storage: [4]Record = undefined,
    queue: std.Io.Queue(Record),
    thread: ?std.Thread = null,
    failed: std.atomic.Value(bool) = .init(false),

    pub fn init(a: std.mem.Allocator, io: std.Io, folder: []const u8, settings: Sampling, first: i64, width: usize) !*Writer {
        try std.Io.Dir.cwd().createDirPath(io, folder);
        var nonce: [16]u8 = undefined;
        io.random(&nonce);
        const base = try std.fmt.allocPrint(a, "{s}/capture-{s}", .{ folder, std.fmt.bytesToHex(nonce, .lower) });
        errdefer a.free(base);
        const bin_path = try std.fmt.allocPrint(a, "{s}.bin", .{base});
        defer a.free(bin_path);
        const target_path = try std.fmt.allocPrint(a, "{s}.logits", .{base});
        defer a.free(target_path);
        const meta_path = try std.fmt.allocPrint(a, "{s}.json", .{base});
        defer a.free(meta_path);
        const paths = .{ bin_path, target_path, meta_path };
        const context = try std.Io.Dir.cwd().createFile(io, paths[0], .{ .exclusive = true });
        errdefer {
            context.close(io);
            std.Io.Dir.cwd().deleteFile(io, paths[0]) catch {};
        }
        const target = try std.Io.Dir.cwd().createFile(io, paths[1], .{ .exclusive = true });
        errdefer {
            target.close(io);
            std.Io.Dir.cwd().deleteFile(io, paths[1]) catch {};
        }
        const meta = try std.Io.Dir.cwd().createFile(io, paths[2], .{ .exclusive = true });
        defer meta.close(io);
        errdefer std.Io.Dir.cwd().deleteFile(io, paths[2]) catch {};
        const greedy = settings.temperature == 0;
        const bytes = try std.json.Stringify.valueAlloc(a, .{
            .seed = if (greedy) @as(?u64, null) else settings.seed,
            .temperature = if (greedy) @as(?f64, null) else settings.temperature,
            .top_k = if (greedy) @as(?usize, null) else settings.top_k,
            .top_p = if (greedy) @as(?f64, null) else settings.top_p,
            .first_position = first,
            .width = width,
        }, .{});
        defer a.free(bytes);
        try meta.writeStreamingAll(io, bytes);
        const writer = try a.create(Writer);
        errdefer a.destroy(writer);
        writer.* = .{ .a = a, .io = io, .base = base, .context = context, .target = target, .queue = undefined };
        writer.queue = .init(&writer.storage);
        writer.thread = try std.Thread.spawn(.{}, drain, .{writer});
        return writer;
    }

    pub fn deinit(w: *Writer) void {
        w.finish();
        w.context.close(w.io);
        w.target.close(w.io);
        w.a.free(w.base);
        const a = w.a;
        a.destroy(w);
    }

    pub fn finish(w: *Writer) void {
        if (w.thread) |thread| {
            w.queue.close(w.io);
            thread.join();
            w.thread = null;
        }
    }

    pub fn disable(w: *Writer, err: anyerror) void {
        if (!w.failed.swap(true, .acq_rel)) std.debug.print("DFlash capture disabled: {s}\n", .{@errorName(err)});
    }

    fn drain(w: *Writer) void {
        while (w.queue.getOneUncancelable(w.io)) |record| {
            defer w.a.free(record.bytes);
            if (w.failed.load(.acquire)) continue;
            const file = if (record.target) w.target else w.context;
            file.writeStreamingAll(w.io, record.bytes) catch |err| w.disable(err);
        } else |_| {}
    }

    fn enqueue(w: *Writer, target: bool, bytes: []u8) !void {
        errdefer w.a.free(bytes);
        if (w.failed.load(.acquire)) return error.CaptureDisabled;
        try w.queue.putOneUncancelable(w.io, .{ .target = target, .bytes = bytes });
    }

    pub fn recordContext(w: *Writer, start: i64, width: usize, tokens: []const i32, bits: []const u16) !void {
        try w.enqueue(false, try contextRecord(w.a, start, width, tokens, bits));
    }

    pub fn recordTarget(w: *Writer, positions: []const i64, k: usize, ids: []const i32, logits: []const f32) !void {
        try w.enqueue(true, try targetRecord(w.a, positions, k, ids, logits));
    }
};

pub fn check(io: std.Io, folder: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Context = struct { start: i64, width: usize, tokens: []const i32, bits: []const u16 };
    const Target = struct { positions: []const i64, k: usize, ids: []const i32, logits: []const f32 };
    const Meta = struct { seed: ?u64, temperature: ?f64, top_k: ?usize, top_p: ?f64, first_position: i64, width: usize };
    const Fixture = struct { settings: Sampling, contexts: []const Context, targets: []const Target, metadata: Meta };
    const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(a, "{s}/fixture.json", .{folder}), a, .limited(32 * 1024 * 1024));
    const fixture = (try std.json.parseFromSlice(Fixture, a, text, .{})).value;
    const output = try std.fmt.allocPrint(a, "{s}/native", .{folder});
    const writer = try Writer.init(std.heap.page_allocator, io, output, fixture.settings, fixture.metadata.first_position, fixture.metadata.width);
    defer writer.deinit();
    for (fixture.contexts, fixture.targets) |context, target| {
        const bits = try a.dupe(u16, context.bits);
        try writer.recordContext(context.start, context.width, context.tokens, bits);
        @memset(bits, 0);
        try writer.recordTarget(target.positions, target.k, target.ids, target.logits);
    }
    writer.finish();
    try std.testing.expect(!writer.failed.load(.acquire));
    for ([_][]const u8{ "bin", "logits" }) |suffix| {
        const actual = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(a, "{s}.{s}", .{ writer.base, suffix }), a, .limited(32 * 1024 * 1024));
        const expected = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(a, "{s}/expected.{s}", .{ folder, suffix }), a, .limited(32 * 1024 * 1024));
        try std.testing.expectEqualSlices(u8, expected, actual);
    }
    const meta = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(a, "{s}.json", .{writer.base}), a, .limited(4096));
    try std.testing.expectEqualDeep(fixture.metadata, (try std.json.parseFromSlice(Meta, a, meta, .{})).value);
    const fault = try Writer.init(std.heap.page_allocator, io, output, fixture.settings, 0, 1);
    defer fault.deinit();
    // The worker is waiting on an empty queue; a read-only descriptor makes its next write fail.
    const read_only = try std.Io.Dir.cwd().openFile(io, try std.fmt.allocPrint(a, "{s}.bin", .{fault.base}), .{});
    fault.context.close(io);
    fault.context = read_only;
    try fault.recordContext(0, 1, &.{42}, &.{0x3f80});
    fault.finish();
    try std.testing.expect(fault.failed.load(.acquire));
    try std.testing.expectError(error.CaptureDisabled, fault.recordContext(1, 1, &.{43}, &.{0x4000}));
    std.debug.print("PASS: {d} upstream capture and target records byte-exact, metadata, queue drain, copied buffers and background write failure\n", .{fixture.contexts.len});
}

test "capture records reject truncated shapes and preserve every BF16 bit" {
    const a = std.testing.allocator;
    const bits = [_]u16{ 0, 0x8000, 0x7f80, 0xff80, 0x7fc1, 0xffff };
    const bytes = try contextRecord(a, 0x100000007, 3, &.{ 42, 248319 }, &bits);
    defer a.free(bytes);
    try std.testing.expectEqual(@as(i64, 0x100000007), std.mem.readInt(i64, bytes[0..8], .little));
    for (bits, 0..) |value, i| try std.testing.expectEqual(value, std.mem.readInt(u16, bytes[16 + i * 2 ..][0..2], .little));
    try std.testing.expectError(error.InvalidCaptureShape, contextRecord(a, 0, 3, &.{42}, &bits));
    try std.testing.expectError(error.InvalidCaptureShape, targetRecord(a, &.{0}, 3, &.{ 1, 2 }, &.{ 1, 2 }));
}
