//! The drafters made for known targets, and the one serve pairs by itself: found in local files only, never downloaded.
const std = @import("std");
const json = @import("json");
const cli = @import("cli.zig");
const hub = @import("hub.zig");
const log = @import("log.zig");

/// ``model_type``, else ``text_config.model_type``, when it is a non-empty string.
pub fn modelType(config: json.Value) ?[]const u8 {
    if (config.get("model_type")) |t| if (t.truthy()) return t.str();
    const text = config.get("text_config") orelse return null;
    const t = text.get("model_type") orelse return null;
    return t.str();
}

/// A drafter made for one family and shape: the target's hidden size and layer count.
const Known = struct { model_type: []const u8, hidden: i64, layers: i64, repo: []const u8 };
const known_drafters = [_]Known{.{ .model_type = "qwen3_5", .hidden = 5120, .layers = 64, .repo = "z-lab/Qwen3.8-27B-DFlash2" }};

/// The repo ID of the drafter made for this checkpoint's family and shape (Qwen3.8-27B's DFlash2), else null.
pub fn knownDrafter(config: json.Value) ?[]const u8 {
    const kind = modelType(config) orelse return null;
    for (known_drafters) |d| {
        if (!std.mem.eql(u8, kind, d.model_type)) continue;
        for ([_]?json.Value{ config, config.get("text_config") }) |source| {
            const src = source orelse continue;
            const hidden = (src.get("hidden_size") orelse continue).int64() orelse continue;
            const layers = (src.get("num_hidden_layers") orelse continue).int64() orelse continue;
            if (hidden == d.hidden and layers == d.layers) return d.repo;
        }
    }
    return null;
}

/// The drafter's directory and the name it was given; `automatic` when serve took it from the table by itself.
pub const Selection = struct { dir: ?[]const u8 = null, name: ?[]const u8 = null, automatic: bool = false };

/// --drafter as given (none: no drafter), else the target's known drafter when local files hold it; null: `problem`.
pub fn select(a: std.mem.Allocator, io: std.Io, env: ?*const std.process.Environ.Map, target: []const u8, explicit: ?[]const u8, no_drafts: bool, backend: []const u8, problem: *[]const u8) !?Selection {
    if (explicit) |name| {
        if (std.mem.eql(u8, name, "none")) return .{};
        const dir = try hub.resolve(a, io, env, name, problem) orelse return null;
        return .{ .dir = dir, .name = name };
    }
    // --no-drafts serves the plain reference, and the HIP engine runs no DFlash2 yet
    if (no_drafts or std.mem.eql(u8, backend, "rocm")) return .{};
    const bytes = @import("checkpoint_config").optional(a, io, target) orelse return .{};
    const parsed = try json.parse(a, bytes);
    if (parsed != .ok) return .{};
    const name = knownDrafter(parsed.ok) orelse return .{};
    var uncached: []const u8 = "";
    const dir = try hub.resolve(a, io, env, name, &uncached) orelse return .{};
    return .{ .dir = dir, .name = name, .automatic = true };
}

/// `open`'s engine; a drafter serve chose by itself that does not open is dropped, and the target opens alone.
pub fn openOrPlain(open: anytype, head: anytype, args: *cli.Args, automatic: bool, problem: *[]const u8) @typeInfo(@TypeOf(open)).@"fn".return_type.? {
    if (!automatic) return @call(.auto, open, head ++ .{ args.*, problem });
    if (@call(.auto, open, head ++ .{ args.*, problem })) |opened| {
        if (opened != null) return opened;
    } else |e| {
        if (problem.*.len == 0) problem.* = @errorName(e);
    }
    const why = problem.*;
    const name = args.drafter_name orelse "";
    args.drafter = null;
    args.drafter_name = null;
    problem.* = "";
    const plain = (try @call(.auto, open, head ++ .{ args.*, problem })) orelse return null;
    log.line("drafter {s} did not open ({s}): serving without it", .{ name, why });
    return plain;
}
