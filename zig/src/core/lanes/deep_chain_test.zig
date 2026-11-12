//! A lone stream chains up to a model's most drafts while shared rounds keep the model's shared cap.
const std = @import("std");
const depth = @import("depth.zig");

const empty: @import("table.zig").Table = .{};
const high: [7]f64 = @splat(0.95);
const low: [7]f64 = @splat(0.5);

const NoCopy = struct {
    fn copy(_: *anyopaque, _: *const depth.Who) bool {
        return false;
    }
};

fn rule(prior: []const f64, shared_cap: u32) depth.Rule {
    return .{
        .gpa = std.testing.allocator,
        .prior = prior,
        .most_drafts = 7,
        .shared_cap = shared_cap,
        .family_costs = &empty,
        .shared_costs = &empty,
        .mtp_step_ms = 0,
        .plain_guard = false,
        .node_probabilities = false,
        .batch_rows = 16,
    };
}

fn who(state: *?depth.State) depth.Who {
    return .{ .state = state, .draft_room = 100, .finished = false, .forced = false };
}

test "a lone stream asks for most_drafts on a high first-draft acceptance and stays low on a low one" {
    var unused: u8 = 0;
    const probe: depth.Probe = .{ .ptr = &unused, .copy = NoCopy.copy };
    var r = rule(&high, 3);
    defer r.deinit();
    var s: ?depth.State = null;
    defer if (s) |*x| x.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(i64, 7), try r.headDepth(&who(&s), null, probe));
    r.prior = &low;
    var t: ?depth.State = null;
    defer if (t) |*x| x.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(i64, 1), try r.headDepth(&who(&t), null, probe));
}

test "streams that share a round chain at most the shared cap each" {
    var unused: u8 = 0;
    const probe: depth.Probe = .{ .ptr = &unused, .copy = NoCopy.copy };
    var r = rule(&high, 3);
    defer r.deinit();
    var a: ?depth.State = null;
    var b: ?depth.State = null;
    defer if (a) |*x| x.deinit(std.testing.allocator);
    defer if (b) |*x| x.deinit(std.testing.allocator);
    var whos = [_]depth.Who{ who(&a), who(&b) };
    var out: [2]i64 = undefined;
    try r.budgets(&whos, probe, false, &out);
    try std.testing.expect(out[0] > 0 and out[1] > 0);
    try std.testing.expect(out[0] <= 3 and out[1] <= 3);
}

test "a zero shared cap keeps the ladder's top of three for a lone stream and most_drafts for shared budgets" {
    var unused: u8 = 0;
    const probe: depth.Probe = .{ .ptr = &unused, .copy = NoCopy.copy };
    var r = rule(&high, 0);
    defer r.deinit();
    var a: ?depth.State = null;
    var b: ?depth.State = null;
    defer if (a) |*x| x.deinit(std.testing.allocator);
    defer if (b) |*x| x.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(i64, 3), try r.headDepth(&who(&a), null, probe));
    var whos = [_]depth.Who{ who(&a), who(&b) };
    var out: [2]i64 = undefined;
    try r.budgets(&whos, probe, false, &out);
    try std.testing.expect(out[0] > 3 or out[1] > 3);
}
