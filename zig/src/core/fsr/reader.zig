//! Exact bounded positional reads; cache-bypass policy is enforced by the platform file owner.
const std = @import("std");
pub const Span = struct { file: usize, offset: u64, length: usize };
pub const Request = struct { span: Span, destination: []u8 };
pub const max_read_bytes = 4 * 1024 * 1024;
pub const width = 9;
pub const unchecked = std.math.maxInt(u64);
pub const Stats = struct { calls: u64 = 0, bytes: u64 = 0, logical_bytes: u64 = 0, size_checks: u64 = 0 };

pub fn validate(span: Span, size: u64, destination: usize) !void {
    if (span.length == 0 or span.length != destination) return error.InvalidTensorSpan;
    const end = std.math.add(u64, span.offset, span.length) catch return error.InvalidTensorSpan;
    if (end > size or end > std.math.maxInt(i64)) return error.InvalidTensorSpan;
}

/// `reader` supplies fileSize and readAt; a short successful read is retried, zero progress is a refusal.
pub fn read(reader: anytype, span: Span, destination: []u8) !void {
    const size = try reader.fileSize(span.file);
    try validate(span, size, destination.len);
    var done: usize = 0;
    while (done < destination.len) {
        const n = try reader.readAt(span.file, destination[done..], span.offset + done);
        if (n == 0) return error.ShortRead;
        if (n > destination.len - done) return error.InvalidReadCount;
        done += n;
    }
    if (try reader.fileSize(span.file) != size) return error.CheckpointChanged;
}

/// All components are filled before the caller may publish the expert's address-map entry.
pub fn expert(reader: anytype, spans: []const Span, destinations: []const []u8) !void {
    if (spans.len == 0 or spans.len != destinations.len) return error.InvalidExpertComponents;
    for (spans, destinations) |span, dst| try validate(span, try reader.fileSize(span.file), dst.len);
    for (spans, destinations) |span, dst| try read(reader, span, dst);
}

/// One bounded batch validates every file before reading, then checks all lengths again before publication.
pub fn batch(source: anytype, jobs: anytype, experts: anytype, strides: [9]usize, slabs: [9][]u8, sizes: []u64, stats: *Stats) !void {
    if (jobs.len == 0) return;
    @memset(sizes, unchecked);
    for (jobs) |job| for (experts[job.ticket.key]) |span| {
        if (span.file >= sizes.len) return error.NoFile;
        if (sizes[span.file] == unchecked) {
            sizes[span.file] = try source.fileSize(span.file);
            stats.size_checks += 1;
        }
        try validate(span, sizes[span.file], span.length);
        stats.logical_bytes += span.length;
    };
    var requests: [width]Request = undefined;
    var count: usize = 0;
    for (0..9) |part| {
        var current: ?Request = null;
        for (jobs) |job| {
            const span = experts[job.ticket.key][part];
            const dst = slabs[part][job.ticket.slot * strides[part] ..][0..span.length];
            if (current) |*run| {
                if (run.span.file == span.file and run.span.offset + run.span.length == span.offset and
                    @intFromPtr(run.destination.ptr) + run.destination.len == @intFromPtr(dst.ptr) and
                    span.length <= max_read_bytes -| run.span.length)
                {
                    run.span.length += span.length;
                    run.destination = run.destination.ptr[0..run.span.length];
                    continue;
                }
                try enqueue(source, &requests, &count, run.*, stats);
            }
            current = .{ .span = span, .destination = dst };
        }
        if (current) |run| try enqueue(source, &requests, &count, run, stats);
    }
    try flush(source, requests[0..count], stats);
    for (sizes, 0..) |size, file| if (size != unchecked) {
        const actual = try source.fileSize(file);
        stats.size_checks += 1;
        if (actual != size) return error.CheckpointChanged;
    };
}

fn enqueue(source: anytype, requests: *[width]Request, count: *usize, run: Request, stats: *Stats) !void {
    var at: usize = 0;
    while (at < run.span.length) {
        const n = @min(max_read_bytes, run.span.length - at);
        requests[count.*] = .{ .span = .{ .file = run.span.file, .offset = run.span.offset + at, .length = n }, .destination = run.destination[at..][0..n] };
        count.* += 1;
        if (count.* == width) {
            try flush(source, requests, stats);
            count.* = 0;
        }
        at += n;
    }
}

fn flush(source: anytype, requests: []const Request, stats: *Stats) !void {
    if (requests.len == 0) return;
    const T = switch (@typeInfo(@TypeOf(source))) {
        .pointer => |p| p.child,
        else => @TypeOf(source),
    };
    if (@hasDecl(T, "readMany")) return source.readMany(requests, stats);
    for (requests) |request| {
        var at: usize = 0;
        while (at < request.destination.len) {
            const n = try source.readAt(request.span.file, request.destination[at..], request.span.offset + at);
            stats.calls += 1;
            if (n == 0) return error.ShortRead;
            if (n > request.destination.len - at) return error.InvalidReadCount;
            stats.bytes += n;
            at += n;
        }
    }
}
