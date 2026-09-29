const std = @import("std");
const Snapshot = struct { path: []const u8, sha256: []const u8 };
const manifest = "native/upstream_sources.json";

fn watched(path: []const u8) bool {
    return std.mem.startsWith(u8, path, "src/tensorfold/") and
        std.mem.endsWith(u8, path, ".py") and
        std.mem.indexOf(u8, path, "/cuda/") == null and
        !std.mem.endsWith(u8, path, "_cuda.py");
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
    const record = args.len == 2 and std.mem.eql(u8, args[1], "--record-reviewed");
    if (args.len > 1 and !record) return error.InvalidArguments;
    const listed = try std.process.run(a, init.io, .{ .argv = &.{ "git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "src/tensorfold" } });
    if (!listed.term.success()) return error.GitCommandFailed;
    var paths = std.mem.splitScalar(u8, listed.stdout, 0);
    var snapshots: std.ArrayList(Snapshot) = .empty;
    while (paths.next()) |path| {
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

test "new families, changed kernels and deleted execution paths all require review" {
    const before = [_]Snapshot{ .{ .path = "a", .sha256 = "1" }, .{ .path = "b", .sha256 = "2" } };
    try std.testing.expectEqual(@as(usize, 0), compare(&before, &before, false));
    const after = [_]Snapshot{ .{ .path = "a", .sha256 = "3" }, .{ .path = "new-family", .sha256 = "4" } };
    try std.testing.expectEqual(@as(usize, 3), compare(&before, &after, false));
    try std.testing.expect(watched("src/tensorfold/families/new_model/runtime.py"));
    try std.testing.expect(watched("src/tensorfold/kernels/new_model/attention.py"));
    try std.testing.expect(watched("src/tensorfold/server/prompt_fill.py"));
    try std.testing.expect(!watched("src/tensorfold/families/qwen3_5/cuda/decode.py"));
}
