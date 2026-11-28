const reply = @import("qwen27_reply.zig");
const replies = @import("qwen27_reply_cache_test.zig");

fn noOpening(_: anytype, _: anytype, _: anytype) !void {
    @panic("fullCache has no rewind/opening path");
}

test "qwen27 reply cache: served full-cache warm prefixes and pinned history" {
    try replies.fullCache(reply.keepEnd, noOpening);
}
