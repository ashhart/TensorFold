//! DeepSeek-V4.1's Metal entry point for the native server: a release checkpoint served through the lane core.
const std = @import("std");
const api = @import("engine_api");
const pkg = @import("dsv41_metal");
const host = @import("dsv41");

const serve_lanes = host.serve_lanes;
const Allocator = std.mem.Allocator;

pub const backends: []const []const u8 = &.{"metal"};
pub const families: []const api.Family = &.{.{ .model_type = "deepseek_v41", .formats = &.{"fp8"}, .raw_text = false }};

pub fn chip(_: Allocator) ?[]const u8 {
    return null;
}

/// The engine for `o.dir`, or null with `problem` set (the native_engines contract).
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (@import("sys").experimental.refusal(o.model_type, @import("sys").getenv("TF_EXPERIMENTAL")) != null) {
        problem.* = "DeepSeek-V4.1 on Metal is experimental: its served lanes have no whole-model gate yet; set TF_EXPERIMENTAL=deepseek to opt in";
        return null;
    }
    if (!std.mem.eql(u8, o.model_type, "deepseek_v41")) {
        problem.* = try std.fmt.allocPrint(a, "this build serves deepseek_v41 checkpoints only, not {s}", .{o.model_type});
        return null;
    }
    const opts: pkg.boot.Options = .{
        .slots = @max(1, @min(o.lanes, 16)),
        .context = if (o.context) |c| @intCast(@max(c, 1)) else null,
        .drafts = o.drafts,
        .cache_bytes = if (o.prompt_cache_gib) |g| @intFromFloat(@max(g, 0) * (1 << 30)) else 4 << 30,
        .vision = o.vision != null,
        .image_tokens = if (o.vision) |v| v.image_tokens else 4096,
        .max_images = if (o.vision) |v| v.max_images else 4,
        .small = if (std.c.getenv("TF_DSV41_SMALL")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false,
    };
    const m = pkg.boot.Boot.open(gpa, io, o.dir, opts) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "the DeepSeek-V4.1 Metal engine cannot load {s} ({s})", .{ o.dir, @errorName(e) });
        return null;
    };
    const s = serve_lanes.Served.open(gpa, io, m.model(), o.dir, o.context) catch |e| {
        m.model().close(m);
        problem.* = try std.fmt.allocPrint(a, "the DeepSeek-V4.1 Metal engine cannot start its lanes ({s})", .{@errorName(e)});
        return null;
    };
    return .{ .engine = s.engine(), .close = serve_lanes.Served.close, .ctx = s, .halt = serve_lanes.Served.halt };
}
