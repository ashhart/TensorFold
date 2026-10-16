//! `prefill <model dir> <length>...`: cold synthetic prompts of each length through prefill, as prompt tokens a second.

const std = @import("std");
const qwen35 = @import("qwen3_5");
const ids_file = @import("ids_file.zig");

/// The median seconds of three cold prefills of `prompt` after one warm-up.
pub fn seconds(gpa: std.mem.Allocator, io: std.Io, e: *qwen35.engine.Engine, prompt: []const u32) !f64 {
    var times: [3]f64 = undefined;
    for (0..4) |rep| {
        var caches = try e.newCaches(prompt.len + 8);
        defer {
            e.drain();
            caches.deinit(gpa);
        }
        const t0 = std.Io.Clock.awake.now(io);
        _ = try e.prefill(&caches, prompt, 0, null, .{ .sampling = null, .position = prompt.len }, null);
        const dt = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0.toNanoseconds())) / 1e9;
        if (rep > 0) times[rep - 1] = dt;
    }
    std.mem.sort(f64, &times, {}, std.sort.asc(f64));
    return times[1];
}

pub fn run(gpa: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    if (args.len < 2) return error.MissingArgument;
    var longest: usize = 0;
    for (args[1..]) |a| longest = @max(longest, try std.fmt.parseInt(usize, a, 10));
    const e = try qwen35.engine.Engine.open(gpa, io, args[0], .{ .capacity = longest + 64, .prompt_rows = longest + 64, .streams = 1, .batch_rows = 32, .policy = (try @import("group.zig").resolve("", 0)).policy });
    defer e.deinit();
    const prompt = try ids_file.synthetic(gpa, longest);
    defer gpa.free(prompt);
    for (args[1..]) |a| {
        const len = try std.fmt.parseInt(usize, a, 10);
        const t = try seconds(gpa, io, e, prompt[0..len]);
        std.debug.print("RESULT prefill {d} tokens: {d:.3} s, {d:.0} tok/s\n", .{ len, t, @as(f64, @floatFromInt(len)) / t });
    }
}
