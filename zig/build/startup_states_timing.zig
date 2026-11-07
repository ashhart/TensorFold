//! test-startup-states-timing, test-imprint-ops-timing: kept states' open, Imprint's calls; ReleaseFast, host only.
const std = @import("std");

const stores = [_][]const u8{ "rows", "imprint" };
const runs = 5;

pub fn step(b: *std.Build) void {
    const host = b.graph.host;
    if (host.result.os.tag != .macos and host.result.os.tag != .linux) return;
    const counts = b.option([]const u8, "startup-states-counts", "Kept-state counts test-startup-states-timing seeds") orelse "1000,10000,100000";
    const ops_counts = b.option([]const u8, "imprint-ops-counts", "Learned-state counts test-imprint-ops-timing seeds") orelse "1000,10000,100000";
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = host, .optimize = .fast, .link_libc = true });
    const root = b.createModule(.{
        .root_source_file = b.path("zig/src/core/rows_startup_timing.zig"),
        .target = host,
        .optimize = .fast,
        .link_libc = true,
        .imports = &.{.{ .name = "lanes", .module = lanes }},
    });
    const exe = b.addExecutable(.{ .name = "tf-startup-states", .root_module = root });
    const corpus = b.addRunArtifact(b.addTest(.{ .root_module = root, .filters = &.{"conversations keep canonical spans"} }));
    const about = "Time startup's read of 1K, 10K and 100K kept states on disk, each store's open in 5 processes (no GPU)";
    b.step("test-startup-states-timing", about).dependOn(chain(b, exe, &corpus.step, counts, &stores, true).?);
    const ops_about = "Time Imprint's per-request calls at 1K, 10K and 100K learned states, in 5 processes each (no GPU)";
    const ops = b.step("test-imprint-ops-timing", ops_about);
    if (chain(b, exe, null, ops_counts, &.{"ops"}, false)) |last| ops.dependOn(last);
}

/// For each count and store: seed, `runs` opens in fresh processes, the phases if asked, the summary, then clean.
fn chain(b: *std.Build, exe: *std.Build.Step.Compile, first: ?*std.Build.Step, counts: []const u8, names: []const []const u8, phases: bool) ?*std.Build.Step {
    var last = first;
    var it = std.mem.tokenizeScalar(u8, counts, ',');
    while (it.next()) |n| for (names) |s| {
        var modes: [runs + 4][]const u8 = undefined;
        modes[0] = "seed";
        for (modes[1 .. runs + 1]) |*m| m.* = "open";
        modes[runs + 1 ..].* = .{ "phases", "summary", "clean" };
        for (modes) |mode| {
            if (!phases and std.mem.eql(u8, mode, "phases")) continue;
            const run = b.addRunArtifact(exe);
            run.addArgs(&.{ s, mode, n });
            run.has_side_effects = true; // timings: run every time, one process after another
            if (last) |l| run.step.dependOn(l);
            last = &run.step;
        }
    };
    return last;
}
