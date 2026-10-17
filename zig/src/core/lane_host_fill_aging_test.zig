//! Sustained shorter arrivals must not starve any admitted prompt, including background work.
const std = @import("std");
const api = @import("engine_api.zig");
const Host = @import("lane_host.zig").LaneHost;
const fill = @import("lane_host_fill.zig");
const a = std.testing.allocator;

fn event(_: *anyopaque, _: api.Id, _: *const api.Event) void {}
fn job(h: *Host, r: *const api.Request) Host.Job {
    var j: Host.Job = .{ .host = h, .id = 1, .request = r, .sink = .{ .ctx = h, .event = event } };
    j.stream.prompt_len = r.prompt.len;
    j.stream.cached = 0;
    return j;
}

test "fill aging: every admitted fill gets chunks within a fixed round bound under steady short arrivals" {
    const tokens: [1024]u32 = @splat(7);
    for ([_]usize{ 2, 3, 8 }) |n| for ([_]usize{ 1, n - 1 }) |slow| for ([_]bool{ false, true }) |background| {
        var h = Host.init(a, std.testing.io, undefined, .{ .lanes = @intCast(n) });
        defer h.filling.deinit(a);
        const long: api.Request = .{ .prompt = &tokens, .max_tokens = 1, .background = background };
        const short: api.Request = .{ .prompt = &.{1}, .max_tokens = 1 };
        var jobs: [8]Host.Job = undefined;
        for (jobs[0..n], 0..) |*j, i| {
            j.* = job(&h, if (i < slow) &long else &short);
            try h.filling.append(a, j);
        }
        const bound = 8 + n;
        var last: [8]usize = @splat(0);
        var chunks: [8]usize = @splat(0);
        for (1..bound * 16 + 1) |round| {
            const at = fill.next(&h);
            if (at < slow) {
                try std.testing.expect(round - last[at] <= bound);
                last[at] = round;
                chunks[at] += 1;
            } else jobs[at] = job(&h, &short); // the completed short fill is replaced by a new arrival
            for (last[0..slow]) |previous| try std.testing.expect(round - previous < bound);
        }
        for (chunks[0..slow]) |count| try std.testing.expect(count >= 16);
    };
}
