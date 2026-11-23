//! Load-time settings: a chunk, solo chunk or load-thread count the table refuses is named before any driver call.
const std = @import("std");
const cuda = @import("cuda");

pub fn Cases(comptime Cuda: type, comptime openFn: anytype) type {
    return struct {
        test "a bad TF_FLASHNEXT_CHUNK, _CHUNK_SOLO or TF_FLASHNEXT_LOAD_THREADS is refused at load before the loader takes a stream, a module or a buffer" {
            _ = Cuda;
            const env = @import("sys").env;
            // every driver entry point undefined: the refusal must come before the first call
            var driver: cuda.Driver = undefined;
            const ctx: cuda.Context = .{ .d = &driver, .device = 0, .handle = null };
            // a chunk the table refuses is named at startup, so it never reaches the reader's own BadChunk
            for ([_]struct { [*:0]const u8, [*:0]const u8, anyerror }{ .{ "TF_FLASHNEXT_CHUNK", "0", error.InvalidEnvironment }, .{ "TF_FLASHNEXT_CHUNK", "16385", error.InvalidEnvironment }, .{ "TF_FLASHNEXT_CHUNK", "4k", error.InvalidEnvironment }, .{ "TF_FLASHNEXT_CHUNK_SOLO", "1024", error.BadChunk }, .{ "TF_FLASHNEXT_LOAD_THREADS", "0", error.BadLoadThreads }, .{ "TF_FLASHNEXT_LOAD_THREADS", "33", error.BadLoadThreads }, .{ "TF_FLASHNEXT_LOAD_THREADS", "8x", error.BadLoadThreads } }) |bad| {
                _ = env.setenv(bad[0], bad[1], 1);
                defer _ = env.unsetenv(bad[0]);
                try std.testing.expectError(bad[2], openFn(std.testing.allocator, std.testing.io, &ctx, "no-such-checkpoint", null, .{ .context = 4096, .drafts = true }));
            }
        }
    };
}
