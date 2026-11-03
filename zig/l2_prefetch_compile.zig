//! Emit the actual family table construction and decode hooks without GPU images or execution.
const q = @import("qwen35");
const n = @import("nemotron");
export fn qwenPrefetch(e: *q.engine.Engine, enabled: bool) void {
    var p = q.prefetch.init(e, .{ .on = enabled }, &.{}) catch return;
    if (p) |*value| value.deinit();
}
export fn qwenDecode() usize {
    return @intFromPtr(&q.engine.Engine.forward);
}
export fn nemotronPrefetch(e: *n.Engine) void {
    var p = n.prefetch.init(e) catch return;
    if (p) |*value| value.deinit();
    const f = e.forward(null);
    f.window(1) catch {};
}
