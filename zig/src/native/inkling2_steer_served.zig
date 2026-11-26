//! The control vector a served Inkling 2 Mini takes: its fit to the visit plan, its device directions and its line.
const std = @import("std");
const api = @import("engine_api");
const Model = @import("../families/inkling2/metal/model.zig").Model;
const steer = @import("../families/inkling2/metal/steer.zig");
const Allocator = std.mem.Allocator;

/// Check the vector against the model that loaded, then put its directions on the device; strength 0 leaves main's.
pub fn apply(gpa: Allocator, model: *Model, dir: []const u8, g: api.glp.Glp, a: Allocator, why: *[]const u8) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var reason: api.glp.Why = .{};
    const source = try api.glp.checkpoint.source(arena.allocator(), dir);
    const repo = if (source) |s| s.repo else "";
    api.glp.checkModel(g.vector, steer.hidden, repo, &reason) catch |err| {
        why.* = try std.fmt.allocPrint(a, "the control vector does not fit {s}: {s}", .{ dir, reason.text() });
        return err;
    };
    if (source != null and g.vector.base_model.len > 0) api.glp.checkPairedRevision(g.vector, source.?.revision, &reason) catch |err| {
        why.* = try std.fmt.allocPrint(a, "the control vector does not fit {s}: {s}", .{ dir, reason.text() });
        return err;
    };
    // alpha 0 adds nothing: no pipeline is compiled and every visit launches main's kernels
    if (g.off()) return;
    const visits = try gpa.alloc(steer.Visit, g.vector.layers.len);
    defer gpa.free(visits);
    for (g.vector.layers, 0..) |l, i| visits[i] = .{ .id = l.id, .dir = l.dir };
    model.steer = steer.init(gpa, model.device, .{ .alpha = g.alpha, .visits_total = model.weight_plan.n_visits, .visits = visits }) catch |err| {
        why.* = try std.fmt.allocPrint(a, "the control vector does not fit {s}: {s}", .{ dir, problem(err) });
        return err;
    };
}

/// The line that reports the visits the vector steers, or that strength 0 leaves every visit unsteered.
pub fn report(model: *const Model, g: api.glp.Glp) void {
    const file_sha = std.fmt.bytesToHex(g.vector.file_sha256, .lower);
    const plan = model.steer orelse {
        std.log.info("Inkling 2 Mini: control vector loaded at alpha 0, replies are unsteered: file sha256 {s}", .{file_sha[0..]});
        return;
    };
    const ids = plan.ids;
    std.log.info("Inkling 2 Mini: control vector in force, replies are steered: alpha {d:.3}, {d} decoder visits ({d}-{d}) of {d}, file sha256 {s}", .{
        plan.alpha,
        ids.len,
        ids[0],
        ids[ids.len - 1],
        model.weight_plan.n_visits,
        file_sha[0..],
    });
}

/// Why a vector's directions cannot be prepared, as the operator reads it.
pub fn problem(err: anyerror) []const u8 {
    return switch (err) {
        error.BadSteerLayer => "it names visit 0 or a decoder visit past the 48 a token makes",
        error.BadSteerWidth => "its directions are not 2304 wide",
        error.BadSteerShape => "it holds no direction",
        else => "its directions cannot be prepared",
    };
}
