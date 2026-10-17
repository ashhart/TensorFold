//! Compiles real GPU entry points into an object; this object is never executed by host tests.
const std = @import("std");
const d3 = @import("drama3");

export fn dramaCompile(io: *const std.Io, model: [*:0]const u8, codec: [*:0]const u8) void {
    const a = std.heap.page_allocator;
    const e = d3.Engine.create(a, io.*, std.mem.span(model), .{ .warmup = false }) catch return;
    defer e.destroy();
    const t = d3.tts.Tts.create(a, io.*, e, std.mem.span(codec), 4) catch return;
    defer t.destroy();
    const codes = t.encodeWav(&.{}) catch return;
    defer a.free(codes);
    var rows: [2][11]i32 = @splat(@splat(0));
    const result = t.speak(io.*, &rows, .{}, 4, .{}, .{ .ctx = e, .audio = audio }) catch return;
    defer a.free(result.frames);
    d3.replay.verify(a, io.*, e, t, &rows, .{}) catch {};
    if (@hasDecl(d3, "speed_probe")) d3.speed_probe.verify(e) catch {};
    const frames = e.generate(io.*, &rows, .{}, 4, false, 1) catch return;
    defer a.free(frames.frames);
}

fn audio(_: *anyopaque, _: []const f32) !void {}
