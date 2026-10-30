const std = @import("std");
const register = @import("speech_register.zig");
const log = @import("log.zig");

pub fn jump(a: std.mem.Allocator, id: []const u8, pcm: []const f32, sample_rate: u32, rule: register.Rule) ?f64 {
    const measured = register.jump(a, pcm, sample_rate, rule) catch |e| {
        log.line("speech {s} register unmeasured: {s}", .{ id, @errorName(e) });
        return null;
    };
    if (measured == null) log.line("speech {s} register unmeasured: {s}", .{ id, "no voiced phrase comparison" });
    return measured;
}

test "register errors and absent measurements leave the take unmeasured and log the request" {
    try @import("speech_unmeasured_test.zig").check(@import("speech.zig").Test.redrawTake);
}
