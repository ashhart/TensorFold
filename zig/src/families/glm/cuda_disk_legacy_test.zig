//! A deterministic whole-form file producer shared unchanged by the main and candidate byte comparison.
const std = @import("std");
const cuda = @import("cuda");
const sys = @import("sys");
const image = @import("cuda_disk_image.zig");
const fixture = @import("cuda_disk_image_test.zig");
const State = @import("cuda_state.zig").State;
const Plan = @import("cuda_snapshot_layout.zig").Plan;
const pages = @import("cuda_snapshot_pages.zig");
const shard = @import("cuda_shard.zig");
const file = @import("engine_api").prompt_cache.rows.file;
const t = std.testing;

test "GLM legacy disk: write a fixed two-rank whole image with the production image and file writers" {
    const dir = sys.getenv("TENSORFOLD_GLM_DISK_FIXTURE_DIR") orelse return error.SkipZigTest;
    const saved = sys.getenv("TF_GLM_KV_LIVE_HALF");
    const copy = if (saved) |s| try std.fmt.allocPrintSentinel(t.allocator, "{s}", .{s}, 0) else null;
    defer {
        if (copy) |s| {
            _ = sys.env.setenv("TF_GLM_KV_LIVE_HALF", s, 1);
            t.allocator.free(s);
        } else _ = sys.env.unsetenv("TF_GLM_KV_LIVE_HALF");
    }
    _ = sys.env.unsetenv("TF_GLM_KV_LIVE_HALF");
    var d = fixture.H.driver();
    var writer = try file.Writer.open(dir, "image", ".st");
    errdefer writer.abort();
    for (0..2) |rank| {
        var m: fixture.Model = undefined;
        try m.init();
        m.w.config.hidden = 16;
        m.w.config.kda_heads = 3;
        m.w.config.kv_lora = 8;
        m.w.config.i_dim = 8;
        m.w.share = try shard.Share.of(&m.w.config, rank, 2);
        var st = try State.initFormat(t.allocator, &d, &m.w, 64, false);
        defer st.deinit();
        const sh = try image.Shape.of(&st, &m.w.config, 64, true);
        const plan = try Plan.init(&st, &m.w.config, 32, 31, 0);
        for (plan.parts[0..plan.count], 0..) |part, j| {
            const bytes = @as([*]u8, @ptrFromInt(part.live))[0..part.len];
            for (bytes, 0..) |*b, i| b.* = @truncate(j *% 17 +% i *% 31 +% 7);
        }
        var pool = try pages.Pool.init(t.allocator, &d, 8 << 20, false);
        defer pool.deinit();
        var lease = try pool.take(plan.len);
        defer pool.give(&lease);
        try pool.move(.{ .d = &d, .handle = null }, &lease, &plan, true);
        var segs: std.ArrayList(image.Seg) = .empty;
        defer segs.deinit(t.allocator);
        try image.segments(sh, 32, 31, &segs, t.allocator);
        const bytes = try t.allocator.alloc(u8, @intCast(image.imageBytes(sh, 32)));
        defer t.allocator.free(bytes);
        var words = (image.Words{ .at = 32, .mtp_len = 31, .base = 0, .mtp_base = 0, .cur = 2, .world = 2 }).encode();
        try (image.Copy{ .d = &d, .pool = &pool, .lease = &lease, .segs = segs.items, .words = &words }).run(0, bytes, true);
        try writer.append(if (rank == 0) bytes else bytes[image.Words.bytes..][0..@intCast(sh.prefix)]);
    }
    _ = try writer.finish(.{ .kind = @backingInt(file.Kind.state), .payload = 0, .payload_hash = 0, .name = @splat(7), .at = 32, .block = 256 }, &.{}, false);
}
