//! Code generation of the complete native adapter without loading a driver, model or codec.
const std = @import("std");
const api = @import("engine_api");
const host = @import("drama_host.zig");
export fn compileNativeDrama(io: *const std.Io, dir: [*:0]const u8, r: *const api.SpeechRequest) void {
    const opened = host.open(std.heap.page_allocator, io.*, std.mem.span(dir), "", .{}) catch return;
    defer opened.close(opened.ctx);
    opened.engine.submitSpeech(1, r, .{ .ctx = opened.ctx, .event = event }) catch {};
    opened.engine.cancel(1);
    opened.engine.drain();
}
fn event(_: *anyopaque, _: api.Id, _: *const api.Event) void {}
