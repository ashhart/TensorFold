//! Own the image tower and feature buffers across every fallible post-image startup phase.
pub fn finish(e: anytype, args: anytype, comptime after: anytype) @typeInfo(@TypeOf(after)).@"fn".return_type.? {
    errdefer if (e.images) |v| {
        e.images = null;
        v.deinit();
    };
    return after(args);
}
