//! Persistent hashing must overlap distinct files without losing records or weakening identity checks.
const std = @import("std");
const cache = @import("digest_cache.zig");
const t = std.testing;
const Gate = struct {
    active: std.atomic.Value(usize) = .init(0),
    peak: std.atomic.Value(usize) = .init(0),
    pub fn read(g: *Gate, file: std.Io.File, size: u64) !cache.Value {
        const active = g.active.fetchAdd(1, .acq_rel) + 1;
        defer _ = g.active.fetchSub(1, .acq_rel);
        _ = g.peak.fetchMax(active, .monotonic);
        const start = std.Io.Clock.awake.now(t.io).toNanoseconds();
        while (g.peak.load(.acquire) < 2 and std.Io.Clock.awake.now(t.io).toNanoseconds() - start < 200_000_000)
            std.Thread.yield() catch {};
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        var buf: [16384]u8 = undefined;
        var at: u64 = 0;
        while (at < size) {
            const n = try file.readPositionalAll(t.io, buf[0..@intCast(@min(buf.len, size - at))], at);
            if (n == 0) return error.FileShrank;
            hash.update(buf[0..n]);
            at += n;
        }
        return .{ .sha256 = hash.finalResult() };
    }
};
const Job = struct {
    root: []const u8,
    path: []const u8,
    gate: *Gate,
    result: ?cache.Result = null,
    failure: ?anyerror = null,
    fn run(j: *Job) void {
        j.result = (cache.Cache{ .root = j.root }).get(t.allocator, t.io, j.path, false, j.gate) catch |err| {
            j.failure = err;
            return;
        };
    }
};

test "digest parallel: distinct persistent loaders overlap and publication retains every record" {
    if (!@import("digest_key.zig").persistent) return error.SkipZigTest;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    const index = try std.fs.path.join(a, &.{ root, "cache" });
    var gate: Gate = .{};
    var jobs: [8]Job = undefined;
    var threads: [8]?std.Thread = @splat(null);
    defer for (threads) |thread| if (thread) |x| x.join();
    for (&jobs, &threads, 0..) |*job, *thread, i| {
        const name = try std.fmt.allocPrint(a, "file-{d}", .{i});
        try tmp.dir.writeFile(t.io, .{ .sub_path = name, .data = name });
        job.* = .{ .root = index, .path = try std.fs.path.join(a, &.{ root, name }), .gate = &gate };
        thread.* = try std.Thread.spawn(.{}, Job.run, .{job});
    }
    for (&threads) |*thread| {
        thread.*.?.join();
        thread.* = null;
    }
    for (jobs) |job| {
        try t.expect(job.failure == null);
        try t.expect(job.result != null and !job.result.?.hit);
    }
    try t.expect(gate.peak.load(.acquire) >= 2);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(t.io, try std.fs.path.join(a, &.{ index, "index.bin" }), a, .limited(cache.max_bytes));
    var records = try cache.decode(a, bytes);
    defer records.deinit(a);
    try t.expectEqual(jobs.len, records.items.len);
    for (jobs) |job| {
        var found = false;
        for (records.items) |record| if (record.key.equal(job.result.?.key)) {
            try t.expectEqual(job.result.?.value.sha256, record.value.sha256);
            found = true;
        };
        try t.expect(found);
    }
}

test "digest parallel: duplicate real-file work computes once while the index lock is released" {
    if (!@import("digest_key.zig").persistent) return error.SkipZigTest;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const path = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "file", .data = "first" });
    const root = try std.fs.path.join(a, &.{ path, "cache" });
    const file = try std.fs.path.join(a, &.{ path, "file" });
    var gate: Gate = .{};
    var jobs = [_]Job{ .{ .root = root, .path = file, .gate = &gate }, .{ .root = root, .path = file, .gate = &gate } };
    var threads: [2]?std.Thread = @splat(null);
    defer for (threads) |thread| if (thread) |x| x.join();
    for (&jobs, &threads) |*job, *thread| thread.* = try std.Thread.spawn(.{}, Job.run, .{job});
    for (&threads) |*thread| {
        thread.*.?.join();
        thread.* = null;
    }
    for (jobs) |job| try t.expect(job.failure == null and job.result != null);
    try t.expectEqual(@as(usize, 1), gate.peak.load(.acquire));
    try t.expect(jobs[0].result.?.hit != jobs[1].result.?.hit);
    try t.expectEqual(jobs[0].result.?.value.sha256, jobs[1].result.?.value.sha256);
}

test "digest parallel: pathname replacement after hashing refuses the unchanged original descriptor" {
    if (!@import("digest_key.zig").persistent) return error.SkipZigTest;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const path = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "file", .data = "first" });
    try tmp.dir.writeFile(t.io, .{ .sub_path = "replacement", .data = "other" });
    const Reader = struct {
        dir: std.Io.Dir,
        pub fn read(r: @This(), file: std.Io.File, _: u64) !cache.Value {
            var bytes: [5]u8 = undefined;
            try t.expectEqual(@as(usize, 5), try file.readPositionalAll(t.io, &bytes, 0));
            try t.expectEqualStrings("first", &bytes);
            try r.dir.rename("replacement", r.dir, "file", t.io);
            var digest: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(&bytes, &digest, .{});
            return .{ .sha256 = digest };
        }
    };
    const root = try std.fs.path.join(a, &.{ path, "cache" });
    const file = try std.fs.path.join(a, &.{ path, "file" });
    try t.expectError(error.DigestFileChanged, (cache.Cache{ .root = root }).get(a, t.io, file, false, Reader{ .dir = tmp.dir }));
    try t.expectError(error.FileNotFound, std.Io.Dir.cwd().openFile(t.io, try std.fs.path.join(a, &.{ root, "index.bin" }), .{}));
}
