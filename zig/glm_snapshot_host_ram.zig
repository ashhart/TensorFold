//! GLM kept states in pinned host memory (TF_GLM_HOST_CACHE) through the actual rank owner with a recording driver.
const std = @import("std");
const glm = @import("glm");
const cuda = @import("cuda");
const t = std.testing;
const rec = @import("test_support/glm_snap_driver.zig").Driver;
const Fixture = @import("test_support/glm_snap_fixture.zig").Fixture;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "GLM snapshots: TF_GLM_HOST_CACHE keeps states in pinned host memory and restores them byte for byte" {
    var d = rec.init();
    _ = setenv("TF_GLM_HOST_CACHE", "1", 1);
    defer _ = unsetenv("TF_GLM_HOST_CACHE");
    const before = cuda.usage(false).device;
    for (0..2) |rank| {
        var f: Fixture = undefined;
        try f.init(&d, rank, 2, true);
        defer f.deinit();
        const device_allocs = rec.allocs;
        const host_allocs = rec.host_allocs;
        try f.e.prepareSnapshots(2 * f.e.stateBytes(8192));
        try t.expectEqual(device_allocs, rec.allocs); // the arena is host memory: no device allocation
        try t.expectEqual(host_allocs + 1, rec.host_allocs);
        try t.expect(f.e.snap_pages.?.arena.? == .host);
        for ([_]usize{ 2048, 4096, 8192 }) |at| {
            try f.mark(at, at, @intCast(at % 251));
            const saved = try t.allocator.dupe(u8, rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len));
            defer t.allocator.free(saved);
            const id = try f.e.keepState();
            @memset(rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len), 0x5A);
            try f.seq.st.restoreHost(f.e.stream, 17, 18, &.{0}, .{ 0, 0 });
            try f.e.restoreState(id);
            try f.expectPrefix(saved, at, at);
            try t.expectEqual(at, f.seq.st.pos);
            f.e.dropState(id);
            f.e.dropped.clearRetainingCapacity();
        }
    }
    try t.expectEqual(before, cuda.usage(false).device);
}
