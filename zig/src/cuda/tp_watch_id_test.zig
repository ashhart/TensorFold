//! Independent NCCL-ID wire oracle and bounded socket reads for the maintained TP fanout tests.
const std = @import("std");
const posix = @import("sys").posix;

pub fn receive(fd: posix.socket_t, records: anytype) !void {
    for (records) |*record| {
        const data = std.mem.asBytes(record);
        var at: usize = 0;
        while (at < data.len) {
            var fds = [_]posix.pollfd{.{ .fd = fd, .events = posix.POLL.IN, .revents = 0 }};
            if (try posix.poll(&fds, 2000) == 0) return error.IdTimeout;
            const rc = posix.system.read(fd, data[at..].ptr, data.len - at);
            if (posix.errno(rc) == .INTR) continue;
            if (posix.errno(rc) != .SUCCESS or rc <= 0) return error.IdReadFailed;
            at += @intCast(rc);
        }
    }
}

/// This checks the declared wire format independently of idRecord, takePart and the production completion mask.
pub fn verify(records: anytype, expected: []const u8) !void {
    if (records.len != 3) return error.IdPartsMissing;
    if (expected.len != 128) return error.BadIdLength;
    var got: [128]u8 = undefined;
    for (records, 0..) |record, part| {
        if (@sizeOf(@TypeOf(record)) != 64 or record.kind != 4 or record.rank != 0) return error.BadIdHeader;
        if (record.text[0] != part) return error.BadIdPart;
        for (record.text[1..8]) |byte| if (byte != 0) return error.BadIdPadding;
        const at = part * 48;
        const n = @min(48, got.len - at);
        @memcpy(got[at..][0..n], record.text[8..][0..n]);
        for (record.text[8 + n ..]) |byte| if (byte != 0) return error.BadIdPadding;
    }
    if (!std.mem.eql(u8, &got, expected)) return error.BadIdBytes;
}
