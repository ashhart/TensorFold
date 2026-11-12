//! Refuse unqualified SM110 modes before Qwen's generic paging checks or weight loading.
const std = @import("std");
const cuda = @import("cuda");
const native = @import("cuda_native.zig");
const modes = @import("cuda_sm110_modes.zig");

pub fn check(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, o: native.Options) !void {
    var now = modes.current(o.kv8, o.prompt_int8, o.rope_yarn != null);
    now.slide = o.slide;
    try modes.checkDevice(ctx, now);
    const dir = o.drafter orelse return;
    if (try ctx.capability() != 110) return;
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(bytes);
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer doc.deinit();
    try modes.check(110, .{ .dspine = @import("core").dspine.declared(doc.value) });
}
