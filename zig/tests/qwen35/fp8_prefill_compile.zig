//! Fake images compile the GPU-enabled FP8 prompt path without linking or executing this object as a service.
const std = @import("std");
const cuda = @import("cuda");
const qwen = @import("qwen35");

export fn fp8OpenCompile(a: *const std.mem.Allocator, io: *const std.Io, ctx: *const cuda.Context) ?*qwen.Engine {
    const e = qwen.Engine.init(a.*, io.*, ctx, "/not-a-real-checkpoint", true) catch return null;
    e.prompt8.enable(e, 121) catch {
        e.deinit();
        return null;
    };
    return e;
}

export fn fp8ForwardCompile(e: *qwen.Engine, cache: *qwen.engine.Cache) bool {
    e.forward(&.{.{ .cache = cache, .rows = 1 }}, &.{1}, .last) catch return false;
    return true;
}

export fn fp8CloseCompile(e: *qwen.Engine) void {
    e.deinit();
}
