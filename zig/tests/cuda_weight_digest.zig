//! Final device-weight digests use pinned asynchronous readback, separate from timed loading.
const std = @import("std");
const cuda = @import("cuda");
pub const Span = struct { name: []const u8, ptr: u64, len: usize };
pub const Digest = struct { name: []const u8, bytes: usize, sha256: [64]u8 };
pub const piece_bytes = 4 << 20;

pub fn collect(a: std.mem.Allocator, d: *const cuda.Driver, stream: cuda.Stream, spans: []const Span, control: bool) ![]Digest {
    if (spans.len == 0) return error.NoWeightSpans;
    var host = try cuda.copy_source.Buffer.alloc(d, piece_bytes);
    defer host.free();
    const out = try a.alloc(Digest, spans.len);
    errdefer a.free(out);
    if (control) {
        if (spans[0].len == 0) return error.EmptyControlSpan;
        try read(d, stream, &host, host.bytes[0..1], spans[0].ptr);
        try d.check(d.api.cuMemsetD8Async(spans[0].ptr, host.bytes[0] ^ 1, 1, stream.handle), "cuMemsetD8Async");
        try stream.synchronize();
    }
    for (spans, out) |s, *o| {
        if (s.len == 0 or s.ptr == 0) return error.EmptyWeightSpan;
        var hash: std.crypto.hash.sha2.Sha256 = .init(.{});
        var at: usize = 0;
        while (at < s.len) {
            const n = @min(s.len - at, host.bytes.len);
            try read(d, stream, &host, host.bytes[0..n], try std.math.add(u64, s.ptr, at));
            hash.update(host.bytes[0..n]);
            at += n;
        }
        o.* = .{ .name = s.name, .bytes = s.len, .sha256 = std.fmt.bytesToHex(hash.finalResult(), .lower) };
    }
    return out;
}
fn read(d: *const cuda.Driver, stream: cuda.Stream, host: *cuda.copy_source.Buffer, into: []u8, src: u64) !void {
    try host.mark(stream);
    try d.check(d.api.cuMemcpyDtoHAsync_v2(into.ptr, src, into.len, stream.handle), "cuMemcpyDtoHAsync");
    try host.wait();
}
