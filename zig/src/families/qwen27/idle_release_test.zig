const std = @import("std");
const release = @import("idle_release.zig");

const Marks = struct {
    volatile_calls: usize = 0,
    nonvolatile_calls: usize = 0,
    discard: bool = true, // whether the OS discards the contents while volatile (the fake zeroes them)
    discarded: bool = false,
};

const Fake = struct {
    pub const Buffer = struct {
        bytes: []u8,
        marks: *Marks,
        reported_length: ?usize = null,

        pub fn contents(self: Buffer) [*]u8 {
            return self.bytes.ptr;
        }
        pub fn length(self: Buffer) usize {
            return self.reported_length orelse self.bytes.len;
        }
        pub fn makeVolatile(self: Buffer) void {
            if (self.marks.discard) @memset(self.bytes, 0);
            self.marks.discarded = self.marks.discard;
            self.marks.volatile_calls += 1;
        }
        // as Metal's setPurgeableState: NonVolatile, whose previous state is Empty after a discard
        pub fn makeNonVolatile(self: Buffer) bool {
            self.marks.nonvolatile_calls += 1;
            defer self.marks.discarded = false;
            return self.marks.discarded;
        }
    };
};

const Host = release.Owner(Fake);

fn fill(bytes: []u8, seed: u64) void {
    var x = seed;
    for (bytes) |*byte| {
        x = x *% 6_364_136_223_846_793_005 +% 1;
        byte.* = @truncate(x >> 24);
    }
}

fn requestWork(bytes: []const u8) [256]u32 {
    var state: u64 = 0x9e3779b97f4a7c15;
    for (bytes) |byte| state = (state ^ byte) *% 1_099_511_628_211;
    var tokens: [256]u32 = undefined;
    for (&tokens) |*token| {
        state = state *% 6_364_136_223_846_793_005 +% 1_442_695_040_888_963_407;
        token.* = @truncate((state >> 16) % 4096);
    }
    return tokens;
}

fn tempRoot(tmp: *std.testing.TmpDir, a: std.mem.Allocator) ![:0]u8 {
    return tmp.dir.realPathFileAlloc(std.testing.io, ".", a);
}

test "idle release honors quiet time and skips empty and oversized extents" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tempRoot(&tmp, a);
    defer a.free(root);
    var source: [65]u8 = undefined;
    fill(&source, 1);
    var original = source;
    var one: Marks = .{};
    var empty: Marks = .{};
    var huge: Marks = .{};
    const buffers = [_]Fake.Buffer{
        .{ .bytes = &source, .marks = &one },
        .{ .bytes = source[0..0], .marks = &empty },
        .{ .bytes = source[0..1], .marks = &huge, .reported_length = (4 << 30) + 1 },
    };
    const owner = (try Host.init(a, std.testing.io, root, &buffers, 100)).?;
    defer owner.deinit();
    owner.idle(0);
    owner.idle(99);
    try std.testing.expect(!owner.isReleased());
    try std.testing.expectEqualSlices(u8, &original, &source);
    owner.idle(100);
    try std.testing.expect(owner.isReleased());
    try std.testing.expectEqual(@as(usize, source.len), owner.releasedBytes());
    try std.testing.expectEqual(@as(usize, 1), one.volatile_calls);
    try std.testing.expectEqual(@as(usize, 0), empty.volatile_calls);
    try std.testing.expectEqual(@as(usize, 0), huge.volatile_calls);
    try owner.restore(101);
    try std.testing.expectEqualSlices(u8, &original, &source);
}

test "twenty release cycles restore every byte and deterministic request tokens" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tempRoot(&tmp, a);
    defer a.free(root);
    const lens = [_]usize{ 31, 4097, 70_001 };
    var sources: [3][]u8 = undefined;
    var controls: [3][]u8 = undefined;
    var marks: [3]Marks = .{ .{}, .{}, .{} };
    var buffers: [3]Fake.Buffer = undefined;
    for (lens, 0..) |len, i| {
        sources[i] = try a.alloc(u8, len);
        controls[i] = try a.alloc(u8, len);
        fill(sources[i], 100 + i);
        @memcpy(controls[i], sources[i]);
        buffers[i] = .{ .bytes = sources[i], .marks = &marks[i] };
    }
    defer for (sources, controls) |source, control| {
        a.free(source);
        a.free(control);
    };
    const owner = (try Host.init(a, std.testing.io, root, &buffers, 10)).?;
    defer owner.deinit();

    for (0..20) |cycle| {
        const now: i64 = @intCast(10 + cycle * 30);
        owner.idle(now);
        owner.idle(now + 10);
        try std.testing.expect(owner.isReleased());
        for (sources, controls) |source, control| {
            var zeros: [256]u8 = @splat(0);
            var at: usize = 0;
            while (at < source.len) {
                const n = @min(zeros.len, source.len - at);
                try std.testing.expectEqualSlices(u8, zeros[0..n], source[at..][0..n]);
                at += n;
            }
            try std.testing.expect(control.len == source.len);
        }
        // Request-like work is deliberately admitted only after restoration completes.
        try owner.restore(now + 15);
        try std.testing.expect(!owner.isReleased());
        for (sources, controls) |source, control| {
            try std.testing.expectEqualSlices(u8, control, source);
            const got = requestWork(source);
            const want = requestWork(control);
            try std.testing.expectEqualSlices(u32, &want, &got);
        }
        try std.testing.expectEqual(@as(usize, 0), owner.releasedBytes());
    }
    for (marks) |mark| {
        try std.testing.expectEqual(@as(usize, 20), mark.volatile_calls);
        try std.testing.expectEqual(@as(usize, 20), mark.nonvolatile_calls);
    }
}

test "corrupt and short snapshots refuse admission and permit a later retry" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tempRoot(&tmp, a);
    defer a.free(root);
    var source: [1027]u8 = undefined;
    fill(&source, 7);
    var original = source;
    var marks: Marks = .{};
    const buffers = [_]Fake.Buffer{.{ .bytes = &source, .marks = &marks }};
    const owner = (try Host.init(a, std.testing.io, root, &buffers, 10)).?;
    defer owner.deinit();
    owner.idle(0);
    owner.idle(10);
    const saved = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, owner.weights_path, a, .limited(4096));
    defer a.free(saved);

    const file = try std.Io.Dir.cwd().openFile(std.testing.io, owner.weights_path, .{ .mode = .read_write });
    defer file.close(std.testing.io);
    try file.writePositionalAll(std.testing.io, "X", 11);
    try std.testing.expectError(error.SnapshotDigestMismatch, owner.restore(20));
    try std.testing.expect(owner.isReleased());
    try std.testing.expectEqual(@as(usize, 1), marks.nonvolatile_calls);
    try file.writePositionalAll(std.testing.io, saved, 0);
    try owner.restore(21);
    try std.testing.expectEqualSlices(u8, &original, &source);
    try std.testing.expectEqual(@as(usize, 2), marks.nonvolatile_calls);

    owner.idle(31);
    owner.idle(41);
    try std.testing.expect(owner.isReleased());
    try file.setLength(std.testing.io, saved.len - 1);
    try std.testing.expectError(error.ShortSnapshot, owner.restore(42));
    try std.testing.expect(owner.isReleased());
    try std.testing.expectEqual(@as(usize, 3), marks.nonvolatile_calls);
    try file.setLength(std.testing.io, saved.len);
    try file.writePositionalAll(std.testing.io, saved, 0);
    try owner.restore(43);
    try std.testing.expectEqualSlices(u8, &original, &source);
    try std.testing.expectEqual(@as(usize, 4), marks.nonvolatile_calls);
    try std.testing.expect(!owner.isReleased());
}

test "serialization failure keeps originals intact and waits a fresh quiet interval" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const base = try tempRoot(&tmp, a);
    defer a.free(base);
    const root = try std.fs.path.join(a, &.{ base, "blocked" });
    defer a.free(root);
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = root, .data = "not a directory" });
    var source: [333]u8 = undefined;
    fill(&source, 9);
    var original = source;
    var marks: Marks = .{};
    const buffers = [_]Fake.Buffer{.{ .bytes = &source, .marks = &marks }};
    const owner = (try Host.init(a, std.testing.io, root, &buffers, 10)).?;
    defer owner.deinit();

    owner.idle(0);
    owner.idle(10);
    try std.testing.expect(!owner.isReleased());
    try std.testing.expectEqualSlices(u8, &original, &source);
    try std.testing.expectEqual(@as(usize, 0), marks.volatile_calls);
    owner.idle(19);
    try std.testing.expect(!owner.isReleased());
    try std.Io.Dir.cwd().deleteFile(std.testing.io, root);
    try std.Io.Dir.cwd().createDirPath(std.testing.io, root);
    owner.idle(20);
    try std.testing.expect(owner.isReleased());
    try std.testing.expectEqual(@as(usize, 1), marks.volatile_calls);
    try owner.restore(21);
    try std.testing.expectEqualSlices(u8, &original, &source);
}

const EventGate = struct {
    marks: *Marks,
    original: []const u8,
    refuse: bool = true,
    before_calls: usize = 0,
    after_calls: usize = 0,
    after_wired_and_restored: bool = false,

    fn before(ctx: ?*anyopaque, buffers: []const Fake.Buffer) !void {
        const self: *EventGate = @ptrCast(@alignCast(ctx.?));
        self.before_calls += 1;
        if (buffers.len != 1 or buffers[0].bytes.ptr != self.original.ptr) return error.BufferIdentity;
        if (self.refuse) return error.StopRelease;
    }
    fn after(ctx: ?*anyopaque, buffers: []const Fake.Buffer) !void {
        const self: *EventGate = @ptrCast(@alignCast(ctx.?));
        self.after_calls += 1;
        self.after_wired_and_restored = self.marks.nonvolatile_calls == 1 and buffers.len == 1 and std.mem.eql(u8, buffers[0].bytes, self.original);
    }
};

test "events bracket release transitions and a refused event backs off" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tempRoot(&tmp, a);
    defer a.free(root);
    var source: [91]u8 = undefined;
    fill(&source, 11);
    var original = source;
    var marks: Marks = .{};
    const buffers = [_]Fake.Buffer{.{ .bytes = &source, .marks = &marks }};
    const owner = (try Host.init(a, std.testing.io, root, &buffers, 10)).?;
    defer owner.deinit();
    var gate: EventGate = .{ .marks = &marks, .original = &source };
    owner.setEvents(.{ .ctx = &gate, .beforeRelease = EventGate.before, .afterRestore = EventGate.after });
    owner.idle(0);
    owner.idle(10);
    try std.testing.expect(!owner.isReleased());
    try std.testing.expectEqualSlices(u8, &original, &source);
    owner.idle(19);
    try std.testing.expectEqual(@as(usize, 1), gate.before_calls);
    gate.refuse = false;
    owner.idle(20);
    try std.testing.expect(owner.isReleased());
    try std.testing.expectEqual(@as(usize, 2), gate.before_calls);
    try owner.restore(30);
    try std.testing.expectEqual(@as(usize, 1), gate.after_calls);
    try std.testing.expect(gate.after_wired_and_restored);
    try std.testing.expectEqualSlices(u8, &original, &source);
}

const RestoreRace = struct {
    owner: *Host,
    original: []const u8,
    entered: std.atomic.Value(bool) = .init(false),
    proceed: std.atomic.Value(bool) = .init(false),
    second_done: std.atomic.Value(bool) = .init(false),
    first_error: ?anyerror = null,
    second_error: ?anyerror = null,
    tokens: [256]u32 = @splat(0),

    fn after(ctx: ?*anyopaque, buffers: []const Fake.Buffer) !void {
        const r: *RestoreRace = @ptrCast(@alignCast(ctx.?));
        std.debug.assert(buffers.len == 1 and std.mem.eql(u8, buffers[0].bytes, r.original));
        r.entered.store(true, .release);
        while (!r.proceed.load(.acquire)) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }

    fn first(r: *RestoreRace) void {
        r.owner.restore(20) catch |err| {
            r.first_error = err;
        };
    }

    fn second(r: *RestoreRace) void {
        r.owner.restore(21) catch |err| {
            r.second_error = err;
            r.second_done.store(true, .release);
            return;
        };
        r.tokens = requestWork(r.original);
        r.second_done.store(true, .release);
    }
};

test "a concurrent request waits for complete restoration before producing tokens" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tempRoot(&tmp, a);
    defer a.free(root);
    var source: [4097]u8 = undefined;
    fill(&source, 13);
    var original = source;
    var marks: Marks = .{};
    const buffers = [_]Fake.Buffer{.{ .bytes = &source, .marks = &marks }};
    const owner = (try Host.init(a, std.testing.io, root, &buffers, 10)).?;
    defer owner.deinit();
    var race: RestoreRace = .{ .owner = owner, .original = &original };
    owner.setEvents(.{ .ctx = &race, .afterRestore = RestoreRace.after });
    owner.idle(0);
    owner.idle(10);

    var second_started = false;
    var first_joined = false;
    var second_joined = false;
    const first = try std.Thread.spawn(.{}, RestoreRace.first, .{&race});
    defer if (!first_joined) {
        race.proceed.store(true, .release);
        first.join();
    };
    var second: std.Thread = undefined;
    defer if (second_started and !second_joined) {
        race.proceed.store(true, .release);
        second.join();
    };
    const deadline = std.Io.Clock.awake.now(std.testing.io).toNanoseconds() + 5 * std.time.ns_per_s;
    while (!race.entered.load(.acquire)) {
        if (std.Io.Clock.awake.now(std.testing.io).toNanoseconds() > deadline) return error.RestoreFixtureTimeout;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    second = try std.Thread.spawn(.{}, RestoreRace.second, .{&race});
    second_started = true;
    std.Io.sleep(std.testing.io, .fromMilliseconds(10), .awake) catch {};
    try std.testing.expect(!race.second_done.load(.acquire));
    race.proceed.store(true, .release);
    first.join();
    first_joined = true;
    second.join();
    second_joined = true;
    try std.testing.expect(race.second_done.load(.acquire));
    try std.testing.expect(race.first_error == null);
    try std.testing.expect(race.second_error == null);
    const want = requestWork(&original);
    try std.testing.expectEqualSlices(u32, &want, &race.tokens);
    try std.testing.expectEqualSlices(u8, &original, &source);
}

test "every failing initialization allocation cleans up and off allocates nothing" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tempRoot(&tmp, a);
    defer a.free(root);
    var source: [17]u8 = undefined;
    var marks: Marks = .{};
    const buffers = [_]Fake.Buffer{.{ .bytes = &source, .marks = &marks }};
    try std.testing.checkAllAllocationFailures(a, struct {
        fn run(allocator: std.mem.Allocator, path: []const u8, items: []const Fake.Buffer) !void {
            const owner = (try Host.init(allocator, std.testing.io, path, items, 10)).?;
            defer owner.deinit();
        }
    }.run, .{ root, buffers[0..] });

    for ([_]i64{ 0, -1 }) |quiet| {
        var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = 0 });
        const owner = try Host.init(failing.allocator(), std.testing.io, root, &buffers, quiet);
        try std.testing.expect(owner == null);
        try std.testing.expect(!failing.has_induced_failure);
        try std.testing.expectEqual(@as(usize, 0), failing.allocated_bytes);
    }
}

test "model allocation initializes idle release with the switch unset or zero" {
    const Model = struct { idle_release: ?*Host };
    for (0..2) |_| {
        const model = try release.createModel(Model, std.testing.allocator);
        defer std.testing.allocator.destroy(model);
        try std.testing.expect(model.idle_release == null);
    }
}

const RejoinGate = struct {
    fail: bool = true,

    fn after(ctx: ?*anyopaque, _: []const Fake.Buffer) !void {
        const gate: *RejoinGate = @ptrCast(@alignCast(ctx.?));
        if (gate.fail) return error.ResidencyRejoinFailed;
    }
};

test "failed residency rejoin admits no work and retries every restored byte" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tempRoot(&tmp, a);
    defer a.free(root);
    var source: [4097]u8 = undefined;
    fill(&source, 29);
    const original = try a.dupe(u8, &source);
    defer a.free(original);
    var marks: Marks = .{};
    const buffers = [_]Fake.Buffer{.{ .bytes = &source, .marks = &marks }};
    const owner = (try Host.init(a, std.testing.io, root, &buffers, 10)).?;
    defer owner.deinit();
    var gate: RejoinGate = .{};
    owner.setEvents(.{ .ctx = &gate, .afterRestore = RejoinGate.after });
    owner.idle(0);
    owner.idle(10);
    owner.restore(20) catch |err| try std.testing.expectEqual(error.ResidencyRejoinFailed, err);
    var backend_calls: usize = 0;
    if (!owner.isReleased()) backend_calls += 1;
    try std.testing.expectEqual(@as(usize, 0), backend_calls);
    try std.testing.expect(owner.isReleased());
    try std.testing.expectEqual(@as(usize, source.len), owner.releasedBytes());
    try std.testing.expectEqualSlices(u8, original, &source);
    gate.fail = false;
    try owner.restore(30);
    if (!owner.isReleased()) backend_calls += 1;
    try std.testing.expectEqual(@as(usize, 1), backend_calls);
    try std.testing.expect(!owner.isReleased());
    try std.testing.expectEqual(@as(usize, 0), owner.releasedBytes());
    const want = requestWork(original);
    const got = requestWork(&source);
    try std.testing.expectEqualSlices(u32, &want, &got);
}

test "restore makes every buffer non-volatile and reloads only the contents the OS discarded" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tempRoot(&tmp, a);
    defer a.free(root);
    var kept_bytes: [300]u8 = undefined;
    var lost_bytes: [500]u8 = undefined;
    fill(&kept_bytes, 41);
    fill(&lost_bytes, 43);
    const kept_original = kept_bytes;
    const lost_original = lost_bytes;
    var kept: Marks = .{ .discard = false };
    var lost: Marks = .{};
    const buffers = [_]Fake.Buffer{ .{ .bytes = &kept_bytes, .marks = &kept }, .{ .bytes = &lost_bytes, .marks = &lost } };
    const owner = (try Host.init(a, std.testing.io, root, &buffers, 10)).?;
    defer owner.deinit();
    owner.idle(0);
    owner.idle(10);
    try std.testing.expect(owner.isReleased());
    try std.testing.expectEqualSlices(u8, &kept_original, &kept_bytes);
    // the kept buffer's snapshot region is damaged: rereading it would refuse, and it must not be read
    const file = try std.Io.Dir.cwd().openFile(std.testing.io, owner.weights_path, .{ .mode = .read_write });
    defer file.close(std.testing.io);
    try file.writePositionalAll(std.testing.io, "XXXX", 0);
    try owner.restore(20);
    try std.testing.expect(!owner.isReleased());
    try std.testing.expectEqual(@as(usize, 1), kept.nonvolatile_calls);
    try std.testing.expectEqual(@as(usize, 1), lost.nonvolatile_calls);
    try std.testing.expectEqualSlices(u8, &kept_original, &kept_bytes);
    try std.testing.expectEqualSlices(u8, &lost_original, &lost_bytes);
}
