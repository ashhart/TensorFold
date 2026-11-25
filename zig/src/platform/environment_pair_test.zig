//! The pair's own environment rules, tested beside `environment_test.zig` to keep that file under the lean line cap.
const std = @import("std");
const e = @import("environment.zig");
const t = std.testing;
const a = t.allocator;
const all: e.Flags = .{ .cuda = true, .metal = true, .xpu = true, .slide = true, .hooks = true, .dsv41 = true, .gemma4 = true, .drama3 = true, .glm = true, .flashnext = true, .nemotron = true, .minimax_m3 = true, .chat = true, .hub = true };
const Env = struct {
    name: []const u8,
    raw: ?[]const u8,
    pub fn get(v: Env, name: []const u8) ?[]const u8 {
        if (std.mem.eql(u8, name, v.name)) return v.raw;
        return null;
    }
};

test "environment: the pair trace interval refuses 0, which would divide by zero, and names its variable" {
    for ([_][]const u8{ "0", "4294967296" }) |raw| {
        try t.expectError(error.InvalidEnvironment, e.parse("TF_Q27_PAIR_TRACE_EVERY", raw));
        const why = (try e.refusal(a, Env{ .name = "TF_Q27_PAIR_TRACE_EVERY", .raw = raw }, all)).?;
        defer a.free(why);
        try t.expect(std.mem.indexOf(u8, why, "TF_Q27_PAIR_TRACE_EVERY") != null);
    }
}
