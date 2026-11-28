//! The control vector a served Qwen3.8-27B takes: its fit, its plan on the device and the line that reports it.
const std = @import("std");
const api = @import("engine_api");
const q = @import("tensorfold").qwen27;
const Allocator = std.mem.Allocator;

/// Check whether a control vector matches the Qwen3.8-27B model's geometry, repo, and snapshot revision.
pub fn checkFit(vector: api.glp.Vector, hidden: usize, dir: []const u8, a: Allocator, reason: *api.glp.Why) !void {
    const source = try api.glp.checkpoint.source(a, dir);
    const repo = if (source) |s| s.repo else "";
    try api.glp.checkModel(vector, hidden, repo, reason);
    if (source) |s| if (vector.base_model.len > 0 and std.ascii.eqlIgnoreCase(api.glp.repoId(vector.base_model), api.glp.repoId(s.repo))) {
        try api.glp.checkPairedRevision(vector, s.revision, reason);
    };
}

/// Check the vector against the model that loaded, then put its directions on the device.
pub fn apply(gpa: Allocator, model: *q.model.Model, dir: []const u8, g: api.glp.Glp, a: Allocator, why: *[]const u8) !void {
    if (shiftRefusal(model.config.shift_norms)) |r| {
        why.* = r;
        return error.GlpRefused;
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
    var base: [192]u8 = undefined;
    std.log.info("Qwen3.8-27B: control vector in force, replies are steered: alpha {d:.3}, {d} layers ({d}-{d}), base {s}, file sha256 {s}", .{
        g.alpha,
        v.layers.len,
        first,
        last,
        baseOf(v, base[0..]),
        file_sha[0..],
    });
    if (v.transferred) std.log.warn("Qwen3.8-27B: the control vector was derived at {f}, not at {f}: it is applied at the residual stream", .{ std.zig.fmtString(v.derived_at), std.zig.fmtString(v.hook_point) });
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

/// The checkpoint the vector came from, with its revision when it names one.
fn baseOf(v: api.glp.Vector, buf: []u8) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    if (v.base_model.len == 0) {
        w.writeAll("an unnamed checkpoint") catch {};
    } else {
        w.writeAll(v.base_model) catch {};
        if (v.base_revision.len > 0) {
            w.writeAll(" @ ") catch {};
            w.writeAll(v.base_revision) catch {};
        }
    }
    return buf[0..w.end];
}

/// The Metal engines other than the 27B-class one hold no steering hooks, so they refuse a control vector.
pub fn refusal(o: api.Open, dense: bool) ?[]const u8 {
    if (o.glp != null and !dense) return "--glp: only the Qwen3.8-27B-class engine applies a control vector: serve this checkpoint without --glp";
    return null;
}

/// SenseNova's shifted norms (weight + 1) are not what the steered norm applies, so such a checkpoint refuses a vector.
pub fn shiftRefusal(shift_norms: bool) ?[]const u8 {
    if (shift_norms) return "--glp: SenseNova checkpoints shift their norms, which the steered norm does not: serve it without --glp";
    return null;
}

test "--glp is refused where the Qwen3.8-27B-class engine does not open" {
    const vector: api.glp.Vector = .{ .alpha = 1, .hook_point = "", .derived_at = "", .base_model = "", .base_revision = "", .method = "", .contrast = "", .width = 0, .layers = &.{}, .sha256 = @splat(0), .file_sha256 = @splat(0), .transferred = false };
    const steered: api.Open = .{ .dir = "d", .model_type = "qwen3_5", .glp = .{ .vector = vector, .alpha = 1 } };
    try std.testing.expect(refusal(steered, true) == null);
    try std.testing.expectStringStartsWith(refusal(steered, false) orelse "", "--glp: only the Qwen3.8-27B-class engine");
}

test "--glp is refused by name on a checkpoint whose norms are shifted" {
    try std.testing.expect(shiftRefusal(false) == null);
    try std.testing.expectStringStartsWith(shiftRefusal(true) orelse "", "--glp: SenseNova checkpoints shift their norms");
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
        .file_sha256 = @splat(0),
        .transferred = false,
    };
    var why: api.glp.Why = .{};
    try checkFit(vector, 8, "/cache/models--Qwen--Qwen3.8-27B/snapshots/aaaa1111bbbb", a, &why);
    try checkFit(vector, 8, "/cache/models--mlx-community--Qwen3.8-27B-4bit/snapshots/cccc2222dddd", a, &why);
    try std.testing.expectError(error.BadGlpVector, checkFit(vector, 8, "/cache/models--Qwen--Qwen3.8-27B/snapshots/cccc2222dddd", a, &why));
    try std.testing.expect(std.mem.indexOf(u8, why.text(), "revision") != null);
}
