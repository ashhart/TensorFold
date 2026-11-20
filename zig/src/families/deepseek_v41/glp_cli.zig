//! Worker GLP loading through the native reader; original format and paired-site concept: Matt Suiche/weightless.
const std = @import("std");
const api = @import("engine_api");
const quiet = @import("builtin").is_test;
pub const Split = struct { positional: []const []const u8, glp: ?api.glp.Glp };
pub const Error = error{ BadGlpFlag, BadGlpVector } || std.mem.Allocator.Error;

fn value(x: []const u8, args: []const []const u8, i: *usize, comptime flag: []const u8) Error!?[]const u8 {
    if (std.mem.eql(u8, x, flag)) {
        i.* += 1;
        if (i.* >= args.len) {
            if (!quiet) std.debug.print("FAIL {s} needs a value\n", .{flag});
            return error.BadGlpFlag;
        }
        return args[i.*];
    }
    if (std.mem.startsWith(u8, x, flag ++ "=")) return x[flag.len + 1 ..];
    return null;
}

/// Separate the worker's positional arguments; the caller's arena keeps the accepted immutable file alive.
pub fn split(a: std.mem.Allocator, io: std.Io, args: []const []const u8) Error!Split {
    const flagged = for (args) |x| {
        if (std.mem.eql(u8, x, "--glp") or std.mem.startsWith(u8, x, "--glp=") or
            std.mem.eql(u8, x, "--glp-alpha") or std.mem.startsWith(u8, x, "--glp-alpha=")) break true;
    } else false;
    if (!flagged) return .{ .positional = args, .glp = null };
    var pos: std.ArrayList([]const u8) = .empty;
    var path: ?[]const u8 = null;
    var alpha: ?f32 = null;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const x = args[i];
        if (try value(x, args, &i, "--glp")) |v| {
            path = v;
        } else if (try value(x, args, &i, "--glp-alpha")) |v| {
            const parsed = std.fmt.parseFloat(f64, v) catch return error.BadGlpFlag;
            alpha = api.glp.strength(parsed) orelse return error.BadGlpFlag;
        } else try pos.append(a, x);
    }
    const file = path orelse {
        if (alpha != null) return error.BadGlpFlag;
        return .{ .positional = pos.items, .glp = null };
    };
    var why: api.glp.Why = .{};
    const g = api.glp.load(a, io, file, alpha, &why) catch |e| {
        if (e == error.OutOfMemory) return e;
        if (!quiet) std.debug.print("FAIL --glp {f}: {f}\n", .{ std.zig.fmtString(file), std.zig.fmtString(why.text()) });
        return error.BadGlpVector;
    };
    if (!quiet) {
        const sha = std.fmt.bytesToHex(g.vector.file_sha256, .lower);
        std.debug.print("glp: file {f}, SHA256 {s}, add strength {d}, {s}\n", .{ std.zig.fmtString(file), sha, g.alpha, if (g.off()) "inactive load" else "responses are steered" });
    }
    return .{ .positional = pos.items, .glp = g };
}

/// Standalone workers and trace generation check the same geometry/provenance as HTTP startup.
pub fn admit(a: std.mem.Allocator, io: std.Io, pack: []const u8, given: ?api.glp.Glp) !void {
    const g = given orelse return;
    const cfg = try @import("config.zig").Config.read(a, io, pack);
    const source = try api.glp.checkpoint.source(a, pack);
    var why: api.glp.Why = .{};
    api.glp.checkPairedModel(g.vector, cfg.hidden, cfg.layers, if (source) |s| s.repo else "", &why) catch
        return refusal(&why);
    if (source) |s| {
        if (g.vector.base_model.len > 0) api.glp.checkPairedRevision(g.vector, s.revision, &why) catch
            return refusal(&why);
    }
}

fn refusal(why: *const api.glp.Why) Error {
    if (!quiet) std.debug.print("FAIL --glp: {f}\n", .{std.zig.fmtString(why.text())});
    return error.BadGlpVector;
}
