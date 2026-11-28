//! The control vector a served Qwen3.8-27B takes: its fit, its plan on the device and the line that reports it.
const std = @import("std");
const api = @import("engine_api");
const q = @import("tensorfold").qwen27;
const Allocator = std.mem.Allocator;

/// Check whether a control vector matches the Qwen3.8-27B model's geometry, repo, and snapshot revision.
pub fn checkFit(vector: api.glp.Vector, hidden: usize, dir: []const u8, a: Allocator, reason: *api.glp.Why) !void {
    const src = cachedSource(a, dir);
    try api.glp.checkModel(vector, hidden, src.repo, reason);
    try api.glp.checkRevision(vector, src.repo, src.revision, reason);
}

/// Check the vector against the model that loaded, then put its directions on the device.
pub fn apply(gpa: Allocator, model: *q.model.Model, dir: []const u8, g: api.glp.Glp, a: Allocator, why: *[]const u8) !void {
    // The steered norm has neither the looped model's mHC blocks nor its shifted gains.
    if (model.config.looped or model.config.shift_norms) {
        why.* = try std.fmt.allocPrint(a, "the control vector cannot steer {s}: SenseNova models take no control vector", .{dir});
        return error.SteeringUnsupported;
    }
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var reason: api.glp.Why = .{};
    checkFit(g.vector, model.config.hidden, dir, arena.allocator(), &reason) catch |err| {
        why.* = try std.fmt.allocPrint(a, "the control vector does not fit {s}: {s}", .{ dir, reason.text() });
        return err;
    };
    const layers = try gpa.alloc(q.steer.Layer, g.vector.layers.len);
    defer gpa.free(layers);
    for (g.vector.layers, 0..) |l, i| layers[i] = .{ .id = l.id, .dir = l.dir };
    model.steer = q.steer.init(gpa, model.device, .{
        .alpha = g.alpha,
        .hidden = @intCast(model.config.hidden),
        .layers_total = @intCast(model.config.layers),
        .layers = layers,
    }) catch |err| {
        why.* = try std.fmt.allocPrint(a, "the control vector does not fit {s}: {s}", .{ dir, steerProblem(err) });
        return err;
    };
}

/// The line that reports the control vector this model steers with.
pub fn report(g: api.glp.Glp) void {
    const v = g.vector;
    const file_sha = std.fmt.bytesToHex(v.file_sha256, .lower);
    var first: u32 = 0;
    var last: u32 = 0;
    for (v.layers, 0..) |l, i| {
        if (i == 0) first = l.id;
        last = l.id;
    }
    var base: [320]u8 = undefined;
    std.log.info("Qwen3.8-27B: control vector in force, replies are steered: alpha {d:.3}, {d} layers ({d}-{d}), base {s}, file sha256 {s}", .{
        g.alpha,
        v.layers.len,
        first,
        last,
        api.glp.baseLine(v, base[0..]),
        file_sha[0..],
    });
    if (v.transferred) std.log.warn("Qwen3.8-27B: the control vector was derived at {f}, not at {f}: it is applied at the residual stream", .{ std.zig.fmtString(v.derived_at), std.zig.fmtString(v.hook_point) });
}

/// The repo id and revision a cached checkpoint came from; empty strings when not a cache snapshot.
fn cachedSource(a: Allocator, dir: []const u8) struct { repo: []const u8, revision: []const u8 } {
    const at = std.mem.lastIndexOf(u8, dir, "models--") orelse return .{ .repo = "", .revision = "" };
    const rest = dir[at + "models--".len ..];
    const sep = std.mem.indexOf(u8, rest, "/snapshots/") orelse return .{ .repo = "", .revision = "" };
    const owner = rest[0..sep];
    const dash = std.mem.indexOf(u8, owner, "--") orelse return .{ .repo = "", .revision = "" };
    const tail = rest[sep + "/snapshots/".len ..];
    const rev = tail[0 .. std.mem.indexOfScalar(u8, tail, '/') orelse tail.len];
    const repo = std.fmt.allocPrint(a, "{s}/{s}", .{ owner[0..dash], owner[dash + 2 ..] }) catch "";
    return .{ .repo = repo, .revision = if (repo.len > 0) rev else "" };
}

/// Why a vector's directions cannot be prepared, as the operator reads it.
fn steerProblem(err: anyerror) []const u8 {
    return switch (err) {
        error.BadSteerLayer => "it names a layer this model does not have",
        error.BadSteerWidth => "its directions are not as wide as this model",
        error.BadSteerShape => "it holds no direction",
        else => "its directions cannot be prepared",
    };
}

test "checkFit rejects a mismatched snapshot revision on the same Qwen repo while keeping conversions" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = [_]f32{ 1, 0, 0, 0, 0, 0, 0, 0 };
    const vector: api.glp.Vector = .{
        .alpha = 1,
        .hook_point = api.glp.residual_site,
        .derived_at = api.glp.residual_site,
        .base_model = "Qwen/Qwen3.8-27B",
        .base_revision = "aaaa1111bbbb",
        .method = "",
        .contrast = "",
        .width = 8,
        .layers = &.{.{ .id = 1, .dir = &dir }},
        .sha256 = @splat(0),
        .claims_digest = false,
        .file_sha256 = @splat(0),
        .transferred = false,
    };
    var why: api.glp.Why = .{};
    try checkFit(vector, 8, "/cache/models--Qwen--Qwen3.8-27B/snapshots/aaaa1111bbbb", a, &why);
    try checkFit(vector, 8, "/cache/models--mlx-community--Qwen3.8-27B-4bit/snapshots/cccc2222dddd", a, &why);
    try std.testing.expectError(error.BadGlpVector, checkFit(vector, 8, "/cache/models--Qwen--Qwen3.8-27B/snapshots/cccc2222dddd", a, &why));
    try std.testing.expect(std.mem.indexOf(u8, why.text(), "revision") != null);
}
