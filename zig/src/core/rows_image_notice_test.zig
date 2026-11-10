//! A disk store adds a RAM retention notice along image prefixes it still holds.
const std = @import("std");
const platform = @import("sys");
const pc = @import("prompt_cache.zig");
const rows = @import("rows_store.zig");
const fake = @import("rows_fake.zig");
const notes = @import("prompt_cache_note.zig");
const a = std.testing.allocator;

test "image disk notice: image conversations keep RAM state and suppress notices along retained prefixes" {
    for ([_]bool{ false, true }) |sync| for (0..3) |kind| {
        const tmp = if (platform.c.getenv("TMPDIR")) |p| std.mem.span(p) else ".";
        const root = try std.fmt.allocPrint(a, "{s}/tf-image-note-{any}-{d}-{d}", .{ tmp, sync, kind, platform.c.getpid() });
        defer a.free(root);
        defer @import("prompt_imprint_test.zig").rmTree(root);
        var f: fake.Fake = .{ .gpa = a };
        var s = pc.Store.init(a, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0 }, 1 << 30);
        const r = try rows.attach(a, &s, .{ .root = root, .identity = 7, .cap = 1 << 30, .daily = 1 << 30, .floor = 0, .lookahead = 1, .sync_writes = sync });
        defer r.deinit();
        defer s.deinit();
        var tokens: [16]u32 = undefined;
        for (&tokens, 0..) |*t, i| t.* = @intCast(i + 1);
        if (kind == 2) tokens[0] = 0x8000_0001;
        f.prompt = &tokens;
        f.at = 8;
        const Image = @import("lanes").stream.Image;
        const picture: Image = .{ .pixels = &.{}, .gh = 1, .gw = 1, .at = 1, .tokens = 1, .hash = 17 };
        var heard: notes.Heard = .{};
        notes.sink = &heard;
        defer notes.sink = null;
        try std.testing.expect(s.keepBornFrame(&tokens, 8, 8, null, &.{}, &.{}, if (kind == 0) &.{picture} else &.{}, if (kind == 1) 7 else 0));
        try std.testing.expectEqual(@as(usize, 1), s.entries.items.len);
        try std.testing.expectEqual(@as(u64, 1), s.counts.kept);
        try std.testing.expectEqual(@as(usize, 0), r.queue.items.len);
        try std.testing.expect(!rows.write.pending(r));
        try std.testing.expectEqual(@as(usize, 0), f.gathered);
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, heard.text(), "ImageConversation"));
        // A later mark and its repeated warm keep extend the same conversation.
        s.most = 1; // carrying the notice must survive eviction of its only earlier entry
        f.at = 12;
        for (0..2) |_| try std.testing.expect(s.keepBornFrame(&tokens, 12, 12, null, &.{}, &.{}, if (kind == 0) &.{picture} else &.{}, if (kind == 1) 7 else 0));
        try std.testing.expectEqual(@as(u64, 2), s.counts.kept);
        try std.testing.expectEqual(@as(usize, 1), s.entries.items.len);
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, heard.text(), "ImageConversation"));
        // A different token prefix is a different conversation, even with the same image.
        tokens[1] = 99;
        try std.testing.expect(s.keepBornFrame(&tokens, 12, 12, null, &.{}, &.{}, if (kind == 0) &.{picture} else &.{}, if (kind == 1) 7 else 0));
        try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, heard.text(), "ImageConversation"));
        // Once every related entry is gone, the same prefix can add a new notice.
        s.remove(0);
        try std.testing.expect(s.keepBornFrame(&tokens, 12, 12, null, &.{}, &.{}, if (kind == 0) &.{picture} else &.{}, if (kind == 1) 7 else 0));
        try std.testing.expectEqual(@as(usize, 3), std.mem.count(u8, heard.text(), "ImageConversation"));
        try std.testing.expectEqual(@as(usize, 0), r.queue.items.len);
        try std.testing.expect(!rows.write.pending(r));
        try std.testing.expectEqual(@as(usize, 0), f.gathered);
    };
}
