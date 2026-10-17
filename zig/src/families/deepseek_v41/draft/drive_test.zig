//! drive.zig on the CPU twin: lanes with oracle-corrupted drafts == the serial decode; drafts off == serial.
const std = @import("std");
const twin = @import("twin.zig");
const drive = @import("drive.zig");

const gpa = std.testing.allocator;

test "oracle drafts and no drafts both give the serial decode" {
    const dims: twin.Dims = .{};
    const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3 };
    const max_new = 90;
    const want = blk: {
        const t = try twin.Twin.init(gpa, dims, 1);
        defer t.deinit();
        break :blk try t.serial(0, &prompt, max_new, null, null);
    };
    defer gpa.free(want);
    const shape: @import("dspark.zig").Shape = .{ .block = dims.block, .window = dims.window, .hidden = dims.dim, .candidates = 8 };
    for ([_]bool{ false, true }) |drafts| {
        const t = try twin.Twin.init(gpa, dims, 2);
        defer t.deinit();
        var nd: drive.NoDraft = .{};
        var o: drive.Oracle = .{ .serial = want, .prompt_len = prompt.len, .vocab = dims.vocab };
        const r = try drive.generate(gpa, t.target(), if (drafts) o.pass() else nd.pass(), shape, &prompt, max_new, drafts);
        defer gpa.free(r.tokens);
        try std.testing.expectEqualSlices(u32, want, r.tokens);
        if (drafts) {
            try std.testing.expectEqual(@as(u64, 0), o.off_anchor);
            try std.testing.expect(r.accepted > 0 and r.accepted < r.drafted); // some kept, some rejected
            try std.testing.expect(o.round >= 8); // every corruption kind came round twice
        } else try std.testing.expectEqual(@as(u64, 0), r.drafted);
    }
}
