//! The 27B's MTP head loads beside a deferred model: TF_QWEN27_DIR the checkpoint, TF_QWEN27_MTP_DIR its head's folder.
const std = @import("std");
const platform = @import("sys");
const Model = @import("model.zig").Model;

test "qwen27 MTP head: loads from the file beside the checkpoint, once, and refuses after the files close" {
    const dir = std.mem.span(platform.c.getenv("TF_QWEN27_DIR") orelse return error.SkipZigTest);
    const mtp_dir = std.mem.span(platform.c.getenv("TF_QWEN27_MTP_DIR") orelse return error.SkipZigTest);
    const a = std.testing.allocator;
    const io = std.testing.io;
    const m = try Model.loadDeferred(a, io, dir, 128);
    defer m.deinit();
    try std.testing.expectError(error.MissingMtpFile, m.loadMtp(io, "/nonexistent"));
    try m.loadMtp(io, mtp_dir);
    const h = m.weights.mtp orelse return error.MtpNotLoaded;
    const hidden: u32 = @intCast(m.config.hidden);
    try std.testing.expectEqual(hidden, h.fc.n);
    try std.testing.expectEqual(2 * hidden, h.fc.k);
    try std.testing.expect(h.layer.mixer == .attention);
    try m.loadMtp(io, mtp_dir); // a second call keeps the first head
    try m.loadShare(0);
    try std.testing.expectError(error.MtpAfterFilesClosed, m.loadMtp(io, mtp_dir));
}
