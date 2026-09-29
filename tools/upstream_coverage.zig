const std = @import("std");
const Snapshot = struct { path: []const u8, sha256: []const u8 };
const manifest = "native/upstream_sources.json";

fn watched(path: []const u8) bool {
    // This CUDA-directory data file is embedded by the native Flash MTP head.
    if (std.mem.eql(u8, path, "src/tensorfold/families/qwen4_exp/cuda/draft_vocab.txt")) return true;
    if (!std.mem.startsWith(u8, path, "src/tensorfold/") or std.mem.indexOf(u8, path, "/cuda/") != null or std.mem.endsWith(u8, path, "_cuda.py")) return false;
    for ([_][]const u8{ ".py", ".cpp", ".c", ".cc", ".h", ".hpp", ".mm", ".metal", ".json", ".txt", ".jinja" }) |extension| if (std.mem.endsWith(u8, path, extension)) return true;
    return false;
}

fn compare(before: []const Snapshot, after: []const Snapshot, report: bool) usize {
    var changes: usize = 0;
    var i: usize = 0;
    var j: usize = 0;
    while (i < before.len or j < after.len) {
        const order: std.math.Order = if (i == before.len) .gt else if (j == after.len) .lt else std.mem.order(u8, before[i].path, after[j].path);
        switch (order) {
            .lt => {
                if (report) std.debug.print("REMOVED upstream source: {s}\n", .{before[i].path});
                changes += 1;
                i += 1;
            },
            .gt => {
                if (report) std.debug.print("NEW upstream source: {s}\n", .{after[j].path});
                changes += 1;
                j += 1;
            },
            .eq => {
                if (!std.mem.eql(u8, before[i].sha256, after[j].sha256)) {
                    if (report) std.debug.print("CHANGED upstream source: {s}\n", .{after[j].path});
                    changes += 1;
                }
                i += 1;
                j += 1;
            },
        }
    }
    return changes;
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    var record = false;
    var complete = false;
    var steps: std.ArrayList([]const u8) = .empty;
    var arg: usize = 1;
    while (arg < args.len) : (arg += 1) {
        if (std.mem.eql(u8, args[arg], "--record-reviewed")) {
            record = true;
        } else if (std.mem.eql(u8, args[arg], "--require-complete")) {
            complete = true;
        } else if (std.mem.eql(u8, args[arg], "--test-step") and arg + 1 < args.len) {
            arg += 1;
            try steps.append(a, args[arg]);
        } else return error.InvalidArguments;
    }
    if (steps.items.len == 0) return error.UseRegisteredCoverageBuildStep;
    if (record and complete) return error.InvalidArguments;
    const source_dir = try std.Io.Dir.cwd().openDir(init.io, "src/tensorfold", .{ .iterate = true });
    defer source_dir.close(init.io);
    var walker = try source_dir.walk(a);
    defer walker.deinit();
    var snapshots: std.ArrayList(Snapshot) = .empty;
    while (try walker.next(init.io)) |entry| {
        if (entry.kind != .file) continue;
        const path = try std.fmt.allocPrint(a, "src/tensorfold/{s}", .{entry.path});
        if (!watched(path)) continue;
        const bytes = std.Io.Dir.cwd().readFileAlloc(init.io, path, a, .limited(16 * 1024 * 1024)) catch |err| switch (err) {
            error.FileNotFound => continue,
            else => return err,
        };
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
        try snapshots.append(a, .{ .path = path, .sha256 = try a.dupe(u8, &std.fmt.bytesToHex(digest, .lower)) });
    }
    std.mem.sort(Snapshot, snapshots.items, {}, struct {
        fn less(_: void, x: Snapshot, y: Snapshot) bool {
            return std.mem.lessThan(u8, x.path, y.path);
        }
    }.less);
    if (snapshots.items.len == 0) return error.NoUpstreamSources;
    const source_paths = try a.alloc([]const u8, snapshots.items.len);
    for (source_paths, snapshots.items) |*path, snapshot| path.* = snapshot.path;
    try @import("feature_coverage.zig").check(a, init.io, source_paths, steps.items, complete);
    if (record) {
        const bytes = try std.json.Stringify.valueAlloc(a, snapshots.items, .{ .whitespace = .indent_2 });
        const file = try std.Io.Dir.cwd().createFile(init.io, manifest, .{});
        defer file.close(init.io);
        try file.writeStreamingAll(init.io, bytes);
        try file.writeStreamingAll(init.io, "\n");
        std.debug.print("Recorded {d} reviewed upstream source identities; this does not change implementation or verification status.\n", .{snapshots.items.len});
        return;
    }
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, manifest, a, .limited(4 * 1024 * 1024));
    const previous = try std.json.parseFromSlice([]const Snapshot, a, bytes, .{});
    if (compare(previous.value, snapshots.items, true) != 0) {
        std.debug.print("Review native implementation and oracle coverage for these changes before recording new source identities.\n", .{});
        return error.UnreviewedUpstreamChanges;
    }
    std.debug.print("PASS: {d} upstream Mac source identities reviewed (not a claim of complete native feature coverage).\n", .{snapshots.items.len});
}

test {
    _ = @import("feature_coverage.zig");
}

test "new families, changed kernels and deleted execution paths all require review" {
    const before = [_]Snapshot{ .{ .path = "a", .sha256 = "1" }, .{ .path = "b", .sha256 = "2" } };
    try std.testing.expectEqual(@as(usize, 0), compare(&before, &before, false));
    const after = [_]Snapshot{ .{ .path = "a", .sha256 = "3" }, .{ .path = "new-family", .sha256 = "4" } };
    try std.testing.expectEqual(@as(usize, 3), compare(&before, &after, false));
    try std.testing.expect(watched("src/tensorfold/families/new_model/runtime.py"));
    try std.testing.expect(watched("src/tensorfold/kernels/new_model/attention.py"));
    try std.testing.expect(watched("src/tensorfold/server/prompt_fill.py"));
    try std.testing.expect(!watched("src/tensorfold/families/qwen3_5/cuda/decode.py"));
    try std.testing.expect(watched("src/tensorfold/streaming/hostsync/hostsync.cpp"));
    try std.testing.expect(watched("src/tensorfold/families/qwen3_5/dflash2_calibration.json"));
    try std.testing.expect(watched("src/tensorfold/families/qwen4_exp/cuda/draft_vocab.txt"));
    try std.testing.expect(!watched("src/tensorfold/cuda/bridge.cpp"));
}
