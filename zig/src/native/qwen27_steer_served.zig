//! The control vector a served Qwen3.8-27B takes: its fit, its plan on the device and the line that reports it.
const std = @import("std");
const api = @import("engine_api");
const q = @import("tensorfold").qwen27;
const Allocator = std.mem.Allocator;

/// Check the vector against the model that loaded, then put its directions on the device.
pub fn apply(gpa: Allocator, model: *q.model.Model, dir: []const u8, g: api.glp.Glp, a: Allocator, why: *[]const u8) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var reason: api.glp.Why = .{};
    api.glp.checkModel(g.vector, model.config.hidden, repoId(arena.allocator(), dir), &reason) catch |err| {
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
    if (v.transferred) std.log.warn("Qwen3.8-27B: the control vector was derived at {s}, not at {s}: it is applied at the residual stream", .{ v.derived_at, v.hook_point });
}

/// The repo id a cached checkpoint came from ("Org/Name"); empty when the folder is not a cache snapshot.
fn repoId(a: Allocator, dir: []const u8) []const u8 {
    const at = std.mem.lastIndexOf(u8, dir, "models--") orelse return "";
    const rest = dir[at + "models--".len ..];
    const sep = std.mem.indexOf(u8, rest, "/snapshots/") orelse return "";
    const owner = rest[0..sep];
    const dash = std.mem.indexOf(u8, owner, "--") orelse return "";
    return std.fmt.allocPrint(a, "{s}/{s}", .{ owner[0..dash], owner[dash + 2 ..] }) catch "";
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
