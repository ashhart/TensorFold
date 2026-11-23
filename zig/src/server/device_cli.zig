//! `tensorfold device [--json] [--rescan]`, the summary bare `tensorfold` and `--help` print, and serve's early check.
const std = @import("std");
const engines = @import("engines.zig");
const support = @import("device/support.zig");
const profile = @import("device/profile.zig");
const scan = @import("device/scan.zig");
const read = @import("device/read.zig");
const report = @import("device/report.zig");
const Allocator = std.mem.Allocator;

pub const usage = "usage: tensorfold device [--json] [--rescan]\n";
const version = @import("build_options").version;

fn binary(a: Allocator) !scan.Binary {
    return .{ .version = version, .offer = try engines.offer(a), .driver = engines.driverGpus };
}

fn now(io: std.Io, buf: *[20]u8) []const u8 {
    return profile.stamp(buf, @intCast(@divFloor(std.Io.Clock.real.now(io).toNanoseconds(), std.time.ns_per_s)));
}

/// Scans when the profile is missing, stale or quick (asking the GPU's driver here), then reports; the exit code.
pub fn main(init: std.process.Init, args: []const [:0]const u8) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    var json = false;
    var rescan = false;
    for (args) |arg| {
        if (std.mem.eql(u8, arg, "--json")) json = true else if (std.mem.eql(u8, arg, "--rescan")) rescan = true else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            try std.Io.File.stdout().writeStreamingAll(io, usage);
            return 0;
        } else {
            try std.Io.File.stderr().writeStreamingAll(io, usage);
            return 2;
        }
    }
    const file = try profile.path(a, init.environ_map);
    const bin = try binary(a);
    var stamp: [20]u8 = undefined;
    const got = try scan.current(a, io, file, try read.quick(a, io), bin, now(io, &stamp), rescan, true);
    var fits: std.ArrayList(report.Fit) = .empty;
    for (@import("checkpoint_cli").onDisk(a, io, init.environ_map, null) catch &.{}) |d| try fits.append(a, .{
        .repo = d.repo,
        .title = d.family,
        .model_type = d.model_type,
        .bytes = d.bytes,
        .verdict = support.fit(got.support, bin.offer.families, d.model_type, d.bytes),
    });
    var out: std.Io.Writer.Allocating = .init(a);
    if (json) {
        const d = got.doc;
        try std.json.Stringify.value(.{
            .schema = d.schema,
            .scanned_at = d.scanned_at,
            .binary_version = d.binary_version,
            .fingerprint = d.fingerprint,
            .scan = d.scan,
            .machine = d.machine,
            .support = got.support,
            .models = fits.items,
        }, .{ .whitespace = .indent_2 }, &out.writer);
        try out.writer.writeByte('\n');
    } else try report.full(&out.writer, got.doc, got.support, version, fits.items);
    try std.Io.File.stdout().writeStreamingAll(io, out.written());
    if (file == null) try std.Io.File.stderr().writeStreamingAll(io, "tensorfold device: no HOME or USERPROFILE, so the profile was not saved\n");
    if (file != null and got.scanned != null and !got.saved) try std.Io.File.stderr().writeStreamingAll(io, "tensorfold device: could not write ~/.tensorfold/device.json\n");
    return 0;
}

/// The two summary lines, from the saved profile or a quick scan saved for next time; empty when they cannot be made.
pub fn summary(init: std.process.Init) []const u8 {
    const a = init.arena.allocator();
    const file = profile.path(a, init.environ_map) catch return "";
    const quick = read.quick(a, init.io) catch return "";
    var stamp: [20]u8 = undefined;
    const got = scan.current(a, init.io, file, quick, binary(a) catch return "", now(init.io, &stamp), false, false) catch return "";
    var out: std.Io.Writer.Allocating = .init(a);
    report.summary(&out.writer, got.doc.machine, got.support) catch return "";
    return out.written();
}

/// Why serve should stop before loading `model_type`: a current profile's certain facts only; null to go on as before.
pub fn serveRefusal(a: Allocator, io: std.Io, env: ?*const std.process.Environ.Map, model_type: []const u8) ?[]const u8 {
    // TENSORFOLD_DEVICE_CHECK=0 skips this check, for a profile that names the machine wrongly
    if (env) |e| if (e.get("TENSORFOLD_DEVICE_CHECK")) |v| if (std.mem.eql(u8, v, "0")) return null;
    const file = (profile.path(a, env) catch return null) orelse return null;
    const doc = scan.fresh(a, io, file, read.quick(a, io) catch return null, version) orelse return null;
    const s = support.derive(a, doc.machine, engines.offer(a) catch return null) catch return null;
    const why = (support.refusal(a, s, model_type) catch return null) orelse return null;
    return std.fmt.allocPrint(a, "{s} (from ~/.tensorfold/device.json; `tensorfold device --rescan` checks again)", .{why}) catch why;
}
