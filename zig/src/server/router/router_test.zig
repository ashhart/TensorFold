//! The router's tests: every one is named ``serve --router:`` for ``zig build test-router-host``.
const std = @import("std");

test {
    _ = @import("config_test.zig");
    _ = @import("pool_test.zig");
    _ = @import("front_test.zig");
}

test "serve --router: every module compiles, its public declarations included" {
    inline for (.{ @import("args.zig"), @import("catalog.zig"), @import("config.zig"), @import("room.zig"), @import("pool.zig"), @import("relay.zig"), @import("replies.zig"), @import("front.zig"), @import("model_route.zig"), @import("listing.zig"), @import("router.zig") }) |m|
        std.testing.refAllDecls(m);
}
