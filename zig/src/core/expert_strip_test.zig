//! Deterministic host tests for production planner, reader, fallback, queue and accounting code.
const std = @import("std");
const stripe = @import("expert_strip.zig");
const snapshot = @import("expert_strip_accounting.zig");
const Sha256 = std.crypto.hash.sha2.Sha256;

const tensor_len = 23;
const expected = [_]u8{ 7, 1, 9, 2, 8, 3, 6, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 };

fn makeIdentity(byte: u8) stripe.Identity {
    return .{ .tensor = @splat(byte), .format = @splat(byte + 1), .checkpoint = @splat(byte + 2) };
}

fn makePlan(a: std.mem.Allocator) !stripe.Plan {
    const checksums = try a.alloc(stripe.Digest, 5);
    defer a.free(checksums);
    for (checksums, 0..) |*checksum, i| {
        const at = i * 5;
        const length = @min(5, expected.len - at);
        Sha256.hash(expected[at..][0..length], checksum, .{});
    }
    const replicas = [_]stripe.Replica{
        .{ .disk = 100, .base = 0, .bytes = tensor_len },
        .{ .disk = 200, .base = 0, .bytes = tensor_len },
        .{ .disk = 300, .base = 0, .bytes = tensor_len },
    };
    const devices = [_]stripe.DeviceSpec{ .{ .id = 10 }, .{ .id = 20 } };
    return stripe.Plan.init(a, .{
        .identity = makeIdentity(20),
        .tensor_bytes = tensor_len,
        .stripe_bytes = 5,
        .replicas = &replicas,
        .devices = &devices,
        .checksums = checksums,
    });
}

const Source = struct {
    data: [3][]const u8,
    identities: [3]stripe.Identity,
    max_chunk: usize = std.math.maxInt(usize),
    zero_reads: bool = false,
    identity_error: ?usize = null,
    cancel_after_read: ?*stripe.Cancellation = null,
    reads: usize = 0,

    fn host(s: *Source) stripe.Host {
        return .{ .ctx = s, .artifact = artifact, .read = read };
    }

    fn of(ctx: ?*anyopaque) *Source {
        return @ptrCast(@alignCast(ctx.?));
    }

    fn artifact(ctx: ?*anyopaque, candidate: stripe.Candidate) !stripe.Identity {
        const s = of(ctx);
        if (s.identity_error) |replica| if (replica == candidate.replica) return error.ReadFailed;
        return s.identities[candidate.replica];
    }

    fn read(ctx: ?*anyopaque, candidate: stripe.Candidate, offset: u64, destination: []u8, _: ?*const stripe.Cancellation) !usize {
        const s = of(ctx);
        s.reads += 1;
        if (s.cancel_after_read) |cancel| cancel.cancel();
        if (s.zero_reads) return 0;
        const data = s.data[candidate.replica];
        if (offset >= data.len) return 0;
        const at: usize = @intCast(offset);
        const n = @min(destination.len, s.max_chunk, data.len - at);
        @memcpy(destination[0..n], data[at..][0..n]);
        return n;
    }
};

const DeviceRecorder = struct {
    staged: usize = 0,
    staged_bytes: usize = 0,
    commits: usize = 0,
    aborts: usize = 0,

    fn sink(s: *DeviceRecorder) stripe.Sink {
        return .{ .ctx = s, .stage = stage, .commit = commit, .abort = abort };
    }

    fn of(ctx: ?*anyopaque) *DeviceRecorder {
        return @ptrCast(@alignCast(ctx.?));
    }

    fn stage(ctx: ?*anyopaque, device: u32, _: usize, bytes: []const u8) !void {
        const s = of(ctx);
        try std.testing.expect(device == 10 or device == 20);
        s.staged += 1;
        s.staged_bytes += bytes.len;
    }

    fn commit(ctx: ?*anyopaque) !void {
        of(ctx).commits += 1;
    }

    fn abort(ctx: ?*anyopaque) void {
        of(ctx).aborts += 1;
    }
};

fn disks() [3][tensor_len]u8 {
    return .{ expected, expected, expected };
}

fn source(data: *[3][tensor_len]u8) Source {
    return .{
        .data = .{ &data[0], &data[1], &data[2] },
        .identities = @splat(makeIdentity(20)),
    };
}

test "expert stripes map devices and disks and assemble exact bytes through short host reads" {
    const a = std.testing.allocator;
    var plan = try makePlan(a);
    defer plan.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 4 }, plan.devices[0].stripes);
    try std.testing.expectEqualSlices(usize, &.{ 1, 3 }, plan.devices[1].stripes);
    const primary_disks = [_]u32{ 100, 200, 200, 300, 300 };
    for (plan.stripes, primary_disks, 0..) |part, disk, i| {
        try std.testing.expectEqual(disk, part.candidates[0].disk);
        try std.testing.expectEqual(i, part.tensor_offset / 5);
        try std.testing.expectEqual(@as(usize, 3), part.candidates.len);
    }
    var data = disks();
    var src = source(&data);
    src.max_chunk = 2;
    var reader = try stripe.Reader.init(a, &plan);
    defer reader.deinit(a);
    var recorder: DeviceRecorder = .{};
    var stats: stripe.Accounting = .{};
    var output: [tensor_len]u8 = @splat(0);
    try reader.read(src.host(), &output, recorder.sink(), null, &stats);
    try std.testing.expectEqualSlices(u8, &expected, &output);
    try std.testing.expectEqual(@as(usize, 14), src.reads);
    try std.testing.expectEqual(@as(usize, 5), recorder.staged);
    try std.testing.expectEqual(@as(usize, tensor_len), recorder.staged_bytes);
    try std.testing.expectEqual(@as(usize, 1), recorder.commits);
    try std.testing.expectEqual(@as(usize, 0), recorder.aborts);
    const totals = stats.snapshot();
    try std.testing.expectEqual(@as(u64, 14), totals.read_calls);
    try std.testing.expectEqual(@as(u64, tensor_len), totals.read_bytes);
    try std.testing.expectEqual(@as(u64, 5), totals.verified_stripes);
    try std.testing.expectEqual(@as(u64, tensor_len), totals.verified_bytes);
    try std.testing.expectEqual(@as(u64, 1), totals.committed_tensors);
    try std.testing.expectEqual(@as(u64, tensor_len), totals.committed_bytes);
}

test "expert stripe checksum failure selects a fallback atomically and never publishes corrupt data" {
    const a = std.testing.allocator;
    var plan = try makePlan(a);
    defer plan.deinit();
    var data = disks();
    data[0][0] ^= 0xff;
    var src = source(&data);
    var reader = try stripe.Reader.init(a, &plan);
    defer reader.deinit(a);
    var stats: stripe.Accounting = .{};
    var recorder: DeviceRecorder = .{};
    var output: [tensor_len]u8 = undefined;
    try reader.read(src.host(), &output, recorder.sink(), null, &stats);
    try std.testing.expectEqualSlices(u8, &expected, &output);
    try std.testing.expectEqual(@as(usize, 1), try reader.selectedCandidate(0));
    try std.testing.expectEqual(@as(u64, 1), stats.snapshot().fallback_selections);
    try std.testing.expectEqual(@as(u64, tensor_len + 5), stats.snapshot().read_bytes);
    const calls = src.reads;
    try reader.read(src.host(), &output, recorder.sink(), null, &stats);
    try std.testing.expectEqual(calls + 5, src.reads);
    for (0..3) |disk| data[disk][10] ^= 0x80;
    var failed_reader = try stripe.Reader.init(a, &plan);
    defer failed_reader.deinit(a);
    var failed_recorder: DeviceRecorder = .{};
    try std.testing.expectError(error.ChecksumMismatch, failed_reader.read(src.host(), &output, failed_recorder.sink(), null, &stats));
    try std.testing.expectEqual(@as(u64, 2), stats.snapshot().committed_tensors);
    try std.testing.expectEqual(@as(u64, 1), stats.snapshot().failed_tensors);
    try std.testing.expectEqual(@as(usize, 1), failed_recorder.aborts);
    try std.testing.expectEqual(@as(usize, 0), failed_recorder.commits);
}

test "expert stripes refuse wrong artifacts incomplete replicas and invalid outputs before publication" {
    const a = std.testing.allocator;
    var plan = try makePlan(a);
    defer plan.deinit();
    var data = disks();
    var src = source(&data);
    var reader = try stripe.Reader.init(a, &plan);
    defer reader.deinit(a);
    var stats: stripe.Accounting = .{};
    var output: [tensor_len]u8 = undefined;
    const wrong = makeIdentity(90);
    src.identities[0].format = wrong.format;
    try reader.read(src.host(), &output, null, null, &stats);
    try std.testing.expectEqual(@as(usize, 1), try reader.selectedCandidate(0));
    for (&src.identities) |*identity| identity.checkpoint = wrong.checkpoint;
    const reads = src.reads;
    try std.testing.expectError(error.WrongArtifact, reader.read(src.host(), &output, null, null, &stats));
    try std.testing.expectEqual(reads, src.reads);
    try std.testing.expectEqual(@as(u64, 1), stats.snapshot().committed_tensors);
    src.identities = @splat(makeIdentity(20));
    src.zero_reads = true;
    var fresh = try stripe.Reader.init(a, &plan);
    defer fresh.deinit(a);
    try std.testing.expectError(error.IncompleteRead, fresh.read(src.host(), &output, null, null, &stats));
    try std.testing.expectError(error.InvalidOutputSize, fresh.read(src.host(), output[0..4], null, null, &stats));
    try std.testing.expectEqual(@as(u64, 1), stats.snapshot().committed_tensors);
}

test "expert stripe queue caps queued reads cancels admission and stops an active host read" {
    const a = std.testing.allocator;
    var plan = try makePlan(a);
    defer plan.deinit();
    var data = disks();
    var src = source(&data);
    var queue = try stripe.Queue.init(a, &plan, 2);
    defer queue.deinit();
    var outputs: [4][tensor_len]u8 = @splat(@splat(0));
    const first = try queue.submit(.{ .user = 1, .host = src.host(), .output = &outputs[0] });
    const second = try queue.submit(.{ .user = 2, .host = src.host(), .output = &outputs[1] });
    try std.testing.expectError(error.QueueFull, queue.submit(.{ .user = 3, .host = src.host(), .output = &outputs[2] }));
    try std.testing.expectEqual(stripe.CancelResult.queued, try queue.cancel(first));
    try std.testing.expectEqual(@as(usize, 1), queue.pending());
    _ = try queue.submit(.{ .user = 3, .host = src.host(), .output = &outputs[2] });
    const one = queue.runNext().?;
    const two = queue.runNext().?;
    try std.testing.expectEqual(@as(u64, 2), one.user);
    try std.testing.expectEqual(@as(u64, 3), two.user);
    try std.testing.expect(one.result == null and two.result == null);
    try std.testing.expect(queue.runNext() == null);
    try std.testing.expectEqualSlices(u8, &expected, &outputs[1]);
    try std.testing.expectEqualSlices(u8, &expected, &outputs[2]);
    try std.testing.expectEqual(second.slot, one.ticket.slot);
    var cancel: stripe.Cancellation = .{};
    _ = try queue.submit(.{ .user = 4, .host = src.host(), .output = &outputs[3], .cancel = &cancel });
    src.cancel_after_read = &cancel;
    const stopped = queue.runNext().?;
    try std.testing.expectEqual(@as(u64, 4), stopped.user);
    try std.testing.expectEqual(error.Cancelled, stopped.result.?);
    const totals = queue.snapshot();
    try std.testing.expectEqual(@as(u64, 2), totals.committed_tensors);
    try std.testing.expectEqual(@as(u64, 2 * tensor_len), totals.committed_bytes);
    try std.testing.expectEqual(@as(u64, 2), totals.cancelled_tensors);
    try queue.restore(totals);
}

test "expert stripe accounting snapshots checksum restart totals without speculative commits" {
    const a = std.testing.allocator;
    var plan = try makePlan(a);
    defer plan.deinit();
    var data = disks();
    for (0..3) |disk| {
        for (5..10) |i| data[disk][i] ^= 0x40;
    }
    var src = source(&data);
    var reader = try stripe.Reader.init(a, &plan);
    var stats: stripe.Accounting = .{};
    var recorder: DeviceRecorder = .{};
    var output: [tensor_len]u8 = undefined;
    try std.testing.expectError(error.ChecksumMismatch, reader.read(src.host(), &output, recorder.sink(), null, &stats));
    reader.deinit(a);
    try std.testing.expectEqual(@as(u64, 0), stats.snapshot().committed_tensors);
    try std.testing.expectEqual(@as(u64, 0), stats.snapshot().committed_bytes);
    try std.testing.expectEqual(@as(u64, 1), recorder.aborts);
    const encoded = snapshot.encode(stats.snapshot());
    const restored = try snapshot.decode(&encoded);
    try std.testing.expectEqual(stats.snapshot(), restored);
    var damaged = encoded;
    damaged[9] ^= 1;
    try std.testing.expectError(error.CorruptAccountingSnapshot, snapshot.decode(&damaged));
    try std.testing.expectError(error.InvalidAccountingSnapshot, snapshot.decode(encoded[0 .. encoded.len - 1]));
    for (0..3) |disk| @memcpy(data[disk][5..10], expected[5..10]);
    var restarted = try stripe.Reader.init(a, &plan);
    defer restarted.deinit(a);
    var continued: stripe.Accounting = .{};
    continued.restore(restored);
    var completion: DeviceRecorder = .{};
    try restarted.read(src.host(), &output, completion.sink(), null, &continued);
    const totals = continued.snapshot();
    try std.testing.expectEqual(@as(u64, 1), totals.failed_tensors);
    try std.testing.expectEqual(@as(u64, 1), totals.committed_tensors);
    try std.testing.expectEqual(@as(u64, tensor_len), totals.committed_bytes);
    try std.testing.expectEqual(@as(usize, 1), completion.commits);
    try std.testing.expectEqualSlices(u8, &expected, &output);
}
