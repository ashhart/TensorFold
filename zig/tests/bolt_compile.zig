//! Generate the native Bolt load and lane vtable bodies without opening a CUDA device.
const std = @import("std");
const cuda = @import("cuda");
const n = @import("nemotron");
export fn boltCompile(ctx: *const cuda.Context, io: *const std.Io, path: [*:0]const u8, images: bool) void {
    const loaded = n.native.open(std.heap.page_allocator, io.*, ctx, std.mem.span(path), null, .{ .context = 256, .drafts = true, .vision = if (images) .{ .image_tokens = 3328, .max_images = 1 } else null }) catch return;
    defer loaded.deinit(loaded.ctx);
    loaded.backend.vtable.verify(loaded.backend.ptr, &.{}, &.{}) catch {};
}
