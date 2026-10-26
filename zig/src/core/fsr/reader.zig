//! Exact bounded positional reads; cache-bypass policy is enforced by the platform file owner.
const std = @import("std");
pub const Span = struct { file: usize, offset: u64, length: usize };

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
