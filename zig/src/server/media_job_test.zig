//! Host fixtures for admission, ownership, cancellation, recovery, corruption and reclaim.
const std = @import("std");
const media = @import("media_job.zig");
const t = std.testing;
const Allocator = std.mem.Allocator;

const Fake = struct {
    serials: [16]u64 = @splat(0),
    starts: usize = 0,
    cancels: usize = 0,

    fn runner(f: *Fake) media.Runner {
        return .{ .ctx = f, .start = start, .cancel = cancel };
    }

    fn start(ctx: *anyopaque, invocation: media.Invocation) media.StartError!void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.serials[f.starts] = invocation.serial;
        f.starts += 1;
    }

    fn cancel(ctx: *anyopaque, serial: u64) void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        var found = false;
        for (f.serials[0..f.starts]) |started| found = found or started == serial;
        std.debug.assert(found);
        f.cancels += 1;
    }
};

fn directory(a: Allocator, tmp: std.testing.TmpDir) ![]u8 {
    return std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/media", .{tmp.sub_path});
}

fn owner(comptime byte: u8) [32]u8 {
    return @splat(byte);
}

fn add(r: *media.Registry, a: Allocator, key: [32]u8, bytes: []const u8) !media.Submission {
    return r.submit(a, key, .image, bytes);
}

fn succeed(r: *media.Registry, serial: u64, bytes: []const u8) !void {
    const outputs = [_]media.Output{.{ .media_type = "image/png", .bytes = bytes }};
    try r.finish(serial, .{ .succeeded = &outputs });
}

fn blob(a: Allocator, id: []const u8) ![]u8 {
    return std.fmt.allocPrint(a, "media/{s}.00000000.blob", .{id});
}

fn job(a: Allocator, id: []const u8) ![]u8 {
    return std.fmt.allocPrint(a, "media/{s}.job", .{id});
}

fn expectState(r: *media.Registry, a: Allocator, id: []const u8, key: [32]u8, state: media.State) !void {
    var snapshot = (try r.inspect(a, id, key)).?;
    defer snapshot.deinit(a);
    try t.expectEqual(state, snapshot.state);
}

test "media jobs: tenant and global admission refuse without durable side effects" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var registry: media.Registry = undefined;
    try registry.init(t.allocator, t.io, try directory(a, tmp), .{ .jobs = 2, .running = 1, .per_owner = 1, .max_request_bytes = 4, .request_bytes = 8, .retained_results = 4, .result_bytes = 64 });
    defer registry.deinit();
    var fake: Fake = .{};
    try registry.attach(fake.runner());
    const first = try add(&registry, a, owner(1), "1234");
    try t.expectEqual(@as(usize, 1), fake.starts);
    try t.expectError(error.OwnerLimit, add(&registry, a, owner(1), "x"));
    try t.expectError(error.RequestTooLarge, add(&registry, a, owner(2), "12345"));
    const second = try add(&registry, a, owner(2), "abcd");
    try t.expectEqual(@as(usize, 1), fake.starts);
    try t.expectError(error.JobLimit, add(&registry, a, owner(3), "x"));
    try t.expectEqual(@as(usize, 8), registry.stats().request_bytes);
    try t.expect((try registry.inspect(a, first.id, owner(2))) == null);
    try t.expectEqual(@as(?media.CancelStatus, null), try registry.cancel(second.id, owner(1)));
    try expectState(&registry, a, second.id, owner(2), .queued);
}

test "media jobs: request byte caps count only queued and running payloads" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var registry: media.Registry = undefined;
    try registry.init(t.allocator, t.io, try directory(a, tmp), .{ .jobs = 3, .running = 1, .per_owner = 2, .max_request_bytes = 4, .request_bytes = 5, .retained_results = 2, .result_bytes = 16 });
    defer registry.deinit();
    const first = try add(&registry, a, owner(1), "1234");
    _ = try add(&registry, a, owner(1), "5");
    try t.expectError(error.RequestByteLimit, add(&registry, a, owner(2), "6"));
    try t.expectError(error.RequestTooLarge, add(&registry, a, owner(2), "12345"));
    var fake: Fake = .{};
    try registry.attach(fake.runner());
    try succeed(&registry, first.serial, "ok");
    try t.expectEqual(@as(usize, 1), registry.stats().request_bytes);
    _ = try add(&registry, a, owner(2), "6789");
    try t.expectEqual(@as(usize, 5), registry.stats().request_bytes);
}

test "media jobs: cancel wins late completion and queued cancel never starts" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var registry: media.Registry = undefined;
    try registry.init(t.allocator, t.io, try directory(a, tmp), .{ .jobs = 4, .running = 1, .per_owner = 4, .max_request_bytes = 16, .request_bytes = 64, .retained_results = 8, .result_bytes = 16 });
    defer registry.deinit();
    var fake: Fake = .{};
    try registry.attach(fake.runner());
    const first = try add(&registry, a, owner(1), "one");
    const second = try add(&registry, a, owner(1), "two");
    const third = try add(&registry, a, owner(1), "three");
    try t.expectEqual(@as(?media.CancelStatus, .requested), try registry.cancel(first.id, owner(1)));
    try t.expectEqual(@as(usize, 1), fake.cancels);
    try t.expectEqual(@as(?media.CancelStatus, .already_requested), try registry.cancel(first.id, owner(1)));
    try succeed(&registry, first.serial, "must-not-be-published");
    try expectState(&registry, a, first.id, owner(1), .cancelled);
    try t.expect((try registry.result(a, first.id, owner(1), 0)) == null);
    try t.expectEqual(@as(usize, 2), fake.starts);
    try t.expectEqual(@as(usize, 0), registry.stats().result_bytes);
    try t.expectError(error.StaleCompletion, registry.finish(first.serial, .cancelled));
    try t.expectEqual(@as(?media.CancelStatus, .cancelled), try registry.cancel(third.id, owner(1)));
    try t.expectEqual(@as(usize, 2), fake.starts);
    try expectState(&registry, a, third.id, owner(1), .cancelled);
    try registry.finish(second.serial, .cancelled);
    try t.expectEqual(@as(usize, 0), registry.stats().active);
    try t.expectEqual(@as(usize, 0), registry.stats().request_bytes);
    try t.expectEqual(@as(usize, 3), registry.stats().retained);
    try t.expectEqual(@as(?media.CancelStatus, .already_cancelled), try registry.cancel(first.id, owner(1)));
}

test "media jobs: corrupt checksummed records refuse recovery without erasure" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try directory(a, tmp);
    var registry: media.Registry = undefined;
    try registry.init(t.allocator, t.io, root, .{ .jobs = 2, .running = 1, .per_owner = 2, .retained_results = 2, .result_bytes = 16 });
    const created = try add(&registry, a, owner(1), "request");
    registry.deinit();
    const name = try job(a, created.id);
    const damaged = try tmp.dir.readFileAlloc(t.io, name, a, .limited(1 << 20));
    damaged[12] ^= 1;
    try tmp.dir.writeFile(t.io, .{ .sub_path = name, .data = damaged });
    var recovered: media.Registry = undefined;
    try t.expectError(error.CorruptRecord, recovered.init(t.allocator, t.io, root, .{}));
    try t.expectEqualSlices(u8, damaged, try tmp.dir.readFileAlloc(t.io, name, a, .limited(1 << 20)));
}

test "media jobs: restart reconciles actors, replays queued work and keeps verified bytes" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try directory(a, tmp);
    const limits: media.Limits = .{ .jobs = 4, .running = 2, .per_owner = 4, .max_request_bytes = 32, .request_bytes = 128, .retained_results = 8, .result_bytes = 64 };
    var registry: media.Registry = undefined;
    try registry.init(t.allocator, t.io, root, limits);
    var fake: Fake = .{};
    try registry.attach(fake.runner());
    const cancelling = try add(&registry, a, owner(1), "cancel");
    const interrupted = try add(&registry, a, owner(1), "interrupt");
    const queued = try registry.submit(a, owner(1), .video, "queued");
    _ = try registry.cancel(cancelling.id, owner(1));
    registry.deinit();
    var recovered: media.Registry = undefined;
    try recovered.init(t.allocator, t.io, root, limits);
    try expectState(&recovered, a, cancelling.id, owner(1), .cancelled);
    try expectState(&recovered, a, interrupted.id, owner(1), .failed);
    try expectState(&recovered, a, queued.id, owner(1), .queued);
    var queued_snapshot = (try recovered.inspect(a, queued.id, owner(1))).?;
    defer queued_snapshot.deinit(a);
    try t.expectEqual(media.Kind.video, queued_snapshot.kind);
    var second_fake: Fake = .{};
    try recovered.attach(second_fake.runner());
    try t.expectEqual(@as(usize, 1), second_fake.starts);
    try t.expectEqual(queued.serial, second_fake.serials[0]);
    const fresh = try add(&recovered, a, owner(1), "fresh");
    try t.expect(fresh.serial > queued.serial);
    const video = [_]media.Output{.{ .media_type = "video/mp4", .bytes = "mp4" }};
    try recovered.finish(queued.serial, .{ .succeeded = &video });
    try recovered.finish(fresh.serial, .{ .failed = "engine failed" });
    var result = (try recovered.result(a, queued.id, owner(1), 0)).?;
    defer result.deinit(a);
    try t.expectEqualStrings("video/mp4", result.media_type);
    try t.expectEqualStrings("mp4", result.bytes);
    recovered.deinit();
    var durable: media.Registry = undefined;
    try durable.init(t.allocator, t.io, root, limits);
    defer durable.deinit();
    var reread = (try durable.result(a, queued.id, owner(1), 0)).?;
    defer reread.deinit(a);
    try t.expectEqualStrings("mp4", reread.bytes);
    const damaged_name = try blob(a, queued.id);
    try tmp.dir.writeFile(t.io, .{ .sub_path = damaged_name, .data = "bad" });
    try t.expectError(error.CorruptResult, durable.result(a, queued.id, owner(1), 0));
}

test "media jobs: result count and byte retention remove payloads and records" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var registry: media.Registry = undefined;
    try registry.init(t.allocator, t.io, try directory(a, tmp), .{ .jobs = 2, .running = 1, .per_owner = 2, .retained_results = 2, .result_bytes = 5 });
    defer registry.deinit();
    var fake: Fake = .{};
    try registry.attach(fake.runner());
    const first = try add(&registry, a, owner(1), "one");
    try succeed(&registry, first.serial, "123");
    const second = try add(&registry, a, owner(1), "two");
    try succeed(&registry, second.serial, "45");
    try t.expectEqual(@as(usize, 5), registry.stats().result_bytes);
    const third = try add(&registry, a, owner(1), "three");
    try succeed(&registry, third.serial, "6");
    try t.expect((try registry.inspect(a, first.id, owner(1))) == null);
    try t.expectEqual(@as(usize, 2), registry.stats().retained);
    try t.expectEqual(@as(usize, 3), registry.stats().result_bytes);
    try t.expectError(error.FileNotFound, tmp.dir.access(t.io, try job(a, first.id), .{}));
    try t.expectError(error.FileNotFound, tmp.dir.access(t.io, try blob(a, first.id), .{}));
    var kept = (try registry.result(a, third.id, owner(1), 0)).?;
    defer kept.deinit(a);
    try t.expectEqualStrings("6", kept.bytes);
    const oversized = try add(&registry, a, owner(1), "large");
    try succeed(&registry, oversized.serial, "123456");
    try expectState(&registry, a, oversized.id, owner(1), .failed);
    try t.expectEqual(@as(usize, 0), registry.stats().active);
    try t.expect((try registry.result(a, oversized.id, owner(1), 0)) == null);
    try t.expect(!(try registry.remove(second.id, owner(1))));
    try t.expect(try registry.remove(third.id, owner(1)));
    try t.expect(try registry.remove(oversized.id, owner(1)));
    try t.expectEqual(@as(usize, 0), registry.stats().retained);
    try t.expectEqual(@as(usize, 0), registry.stats().result_bytes);
    try t.expectError(error.FileNotFound, tmp.dir.access(t.io, try blob(a, second.id), .{}));
    try t.expectError(error.FileNotFound, tmp.dir.access(t.io, try blob(a, third.id), .{}));
    try t.expect(!(try registry.remove(second.id, owner(1))));
}
