//! Lifetime placement priors: routed/metadata fingerprint; startup warmth may be discarded by prefill.
const std = @import("std");
const codec = @import("cuda_exl3_frequency.zig");
const Sha256 = std.crypto.hash.sha2.Sha256;
const namespace = "flashnext-exl3-placement-v1";

pub const State = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    path: []u8,
    identity: codec.Identity,
    lock: std.Io.File,

    pub fn deinit(state: *State) void {
        state.lock.close(state.io);
        state.gpa.free(state.path);
        state.* = undefined;
    }

    pub fn save(state: *State, own: anytype) void {
        const counts = snapshot(own) catch |err| {
            std.log.err("EXL3 placement profile rank {d}: unsafe snapshot skipped ({s})", .{ state.identity.rank, @errorName(err) });
            return;
        };
        codec.write(state.gpa, state.io, state.path, state.identity, &counts) catch |err| {
            std.log.warn("EXL3 placement profile rank {d}: save failed ({s})", .{ state.identity.rank, @errorName(err) });
            return;
        };
        std.log.info("EXL3 placement profile rank {d}: saved decoder lifetime placement snapshot to {s}", .{ state.identity.rank, state.path });
    }
};

pub const Prepared = struct {
    state: State,
    counts: ?codec.Counts,
};

pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, own: anytype, world: u32, rank: u32) !?State {
    const prepared = try prepare(gpa, io, dir, own.engine.layers[0..own.engine.n_layers], world, rank) orelse return null;
    var state = prepared.state;
    errdefer state.deinit();
    if (prepared.counts) |*counts| try finish(own, counts);
    return state;
}

pub fn prepare(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, layers: anytype, world: u32, rank: u32) !?Prepared {
    const root = std.c.getenv("TF_FLASHNEXT_EXL3_PROFILE_DIR") orelse return null;
    if (std.mem.span(root).len == 0) return null;
    if (comptime @import("builtin").os.tag == .windows) {
        std.log.warn("EXL3 placement profiles require POSIX storage; persistence disabled", .{});
        return null;
    }
    if (!available(layers)) {
        std.log.warn("EXL3 placement profile rank {d}: unavailable decoder pools; persistence disabled", .{rank});
        return null;
    }
    const before = std.Io.Clock.awake.now(io).toNanoseconds();
    const model = fingerprint(gpa, io, dir, layers) catch |err| {
        std.log.warn("EXL3 placement profile rank {d}: fingerprint unavailable ({s}); persistence disabled", .{ rank, @errorName(err) });
        return null;
    };
    std.log.info("EXL3 placement profile rank {d}: routed-bank/metadata fingerprint took {d:.3} s (excludes router/dense weights)", .{ rank, @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - before)) / 1e9 });
    var state = open(gpa, io, std.mem.span(root), .{ .model = model, .world = world, .rank = rank }) catch |err| {
        std.log.warn("EXL3 placement profile rank {d}: lock/storage unavailable ({s}); persistence disabled", .{ rank, @errorName(err) });
        return null;
    };
    errdefer state.deinit();
    var counts: ?codec.Counts = codec.read(gpa, io, state.path, state.identity) catch |err| blk: {
        std.log.warn("EXL3 placement profile rank {d}: cold start ({s})", .{ rank, @errorName(err) });
        break :blk null;
    };
    if (counts) |*value| {
        decay(value);
        std.log.info("EXL3 placement profile rank {d}: loaded and decayed rank-local placement counts before pool allocation", .{rank});
    }
    return .{ .state = state, .counts = counts };
}

pub fn finish(own: anytype, counts: *const codec.Counts) !void {
    const layers = own.engine.layers[0..own.engine.n_layers];
    if (!available(layers)) return error.ProfileUnavailable;
    for (layers, 0..) |layer, index| {
        const pool = layer.exl3.?.pool orelse return error.ProfileUnavailable;
        pool.counts[pool.opt.layer] = counts.decoder[index];
    }
    var warmed: usize = 0;
    for (layers) |layer| warmed += try layer.exl3.?.pool.?.warmProfile();
    std.log.info("EXL3 placement profile: imported and warmed {d} expert lines after pool allocation", .{warmed});
}

fn open(gpa: std.mem.Allocator, io: std.Io, root: []const u8, identity: codec.Identity) !State {
    const private = try std.fs.path.join(gpa, &.{ root, namespace });
    defer gpa.free(private);
    var directory = try std.Io.Dir.cwd().createDirPathOpen(io, private, .{
        .permissions = .fromMode(0o700),
        .open_options = .{ .iterate = true },
    });
    defer directory.close(io);
    try directory.setPermissions(io, .fromMode(0o700));
    const name = try std.fmt.allocPrint(gpa, "{s}-w{d}-r{d}.bin", .{ std.fmt.bytesToHex(&identity.model, .lower), identity.world, identity.rank });
    defer gpa.free(name);
    const path = try std.fs.path.join(gpa, &.{ private, name });
    errdefer gpa.free(path);
    const lock_path = try std.fmt.allocPrint(gpa, "{s}.lock", .{path});
    defer gpa.free(lock_path);
    const lock = try openLock(io, lock_path);
    errdefer lock.close(io);
    try lock.setPermissions(io, .fromMode(0o600));
    return .{ .gpa = gpa, .io = io, .path = path, .identity = identity, .lock = lock };
}

fn openLock(io: std.Io, path: []const u8) !std.Io.File {
    const file: std.Io.File = if (comptime @import("builtin").os.tag == .windows) return error.UnsupportedPlatform else .{
        .handle = try std.posix.openat(std.Io.Dir.cwd().handle, path, .{
            .ACCMODE = .RDWR,
            .CREAT = true,
            .NONBLOCK = true,
            .CLOEXEC = true,
            .NOCTTY = true,
            .NOFOLLOW = true,
        }, 0o600),
        .flags = .{ .nonblocking = true },
    };
    errdefer file.close(io);
    if ((try file.stat(io)).kind != .file) return error.ProfileLockNotRegular;
    if (!try file.tryLock(io, .exclusive)) return error.WouldBlock;
    return file;
}

fn restore(state: *State, own: anytype) !bool {
    var counts = codec.read(state.gpa, state.io, state.path, state.identity) catch |err| {
        std.log.warn("EXL3 placement profile rank {d}: cold start ({s})", .{ state.identity.rank, @errorName(err) });
        return false;
    };
    decay(&counts);
    const layers = own.engine.layers[0..own.engine.n_layers];
    for (layers, 0..) |layer, index| {
        const pool = layer.exl3.?.pool.?;
        pool.counts[pool.opt.layer] = counts.decoder[index];
    }
    std.log.info("EXL3 placement profile rank {d}: imported decoder lifetime counts, decayed once by 2 (nonzero minimum 1)", .{state.identity.rank});
    var warmed: usize = 0;
    for (layers) |layer| warmed += try layer.exl3.?.pool.?.warmProfile();
    std.log.info("EXL3 placement profile rank {d}: warmed {d} expert lines", .{ state.identity.rank, warmed });
    return true;
}

fn available(layers: anytype) bool {
    if (layers.len != 48) return false;
    for (layers) |layer| {
        const experts = layer.exl3 orelse return false;
        if (experts.bank == null or experts.routed.len != codec.experts) return false;
        if (experts.pool) |pool| if (pool.opt.layer >= 48) return false;
    }
    return true;
}

fn decay(counts: *codec.Counts) void {
    for (&counts.decoder) |*layer| decayRow(layer);
}

fn decayRow(row: *[512]u32) void {
    for (row) |*count| {
        if (count.* != 0) count.* = @max(1, count.* / 2);
    }
}

fn snapshot(own: anytype) !codec.Counts {
    const layers = own.engine.layers[0..own.engine.n_layers];
    if (!available(layers)) return error.ProfileUnavailable;
    try own.stream.synchronize();
    var counts: codec.Counts = .{};
    for (layers, 0..) |layer, index| {
        const pool = layer.exl3.?.pool.?;
        try pool.lines.drain();
        counts.decoder[index] = pool.counts[pool.opt.layer];
    }
    return counts;
}

fn hashBytes(hash: *Sha256, bytes: []const u8) void {
    var length: [8]u8 = undefined;
    std.mem.writeInt(u64, &length, bytes.len, .little);
    hash.update(&length);
    hash.update(bytes);
}

fn hashRouted(hash: *Sha256, routed: []const [3][]const u8) void {
    for (routed) |expert| for (expert) |projection| hashBytes(hash, projection);
}

fn fingerprint(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, layers: anytype) ![32]u8 {
    var hash: Sha256 = .init(.{});
    hashBytes(&hash, namespace);
    for ([_][]const u8{ "config.json", "model.safetensors.index.json" }) |name| {
        const path = try std.fs.path.join(gpa, &.{ dir, name });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 26));
        defer gpa.free(bytes);
        hashBytes(&hash, name);
        hashBytes(&hash, bytes);
    }
    hashBytes(&hash, "decoder-routed-trellis");
    for (layers) |layer| hashRouted(&hash, expertRouted(layer.exl3.?));
    return hash.finalResult();
}

fn expertRouted(experts: anytype) []const [3][]const u8 {
    if (experts.pool) |pool| return pool.src.routed;
    return experts.routed;
}

const MockPool = struct {
    counts: *[48][512]u32,
    opt: struct { layer: u8, lines: usize = 10 },
    src: struct { routed: []const [3][]const u8 = &.{.{ "gate", "up", "down" }} } = .{},
    lines: struct {
        drained: bool = false,
        pub fn drain(copies: *@This()) !void {
            copies.drained = true;
        }
    } = .{},
    warm_calls: usize = 0,
    fail_warm: bool = false,

    pub fn warmProfile(pool: *MockPool) !usize {
        pool.warm_calls += 1;
        if (pool.fail_warm) return error.MockCudaWarmFailure;
        return 1;
    }
};

const mock_routed: [512][3][]const u8 = @splat(.{ "gate", "up", "down" });
const MockExperts = struct { pool: ?*MockPool, bank: ?[]const u8 = "bank", routed: []const [3][]const u8 = &mock_routed };
const MockLayer = struct { exl3: ?*MockExperts };
const Mock = struct {
    decoder: [48][512]u32 = @splat(@splat(0)),
    pools: [48]MockPool = undefined,
    experts: [48]MockExperts = undefined,
    engine: struct { layers: [48]MockLayer, n_layers: usize = 48 } = undefined,

    fn init(mock: *Mock) void {
        mock.* = .{};
        mock.engine.n_layers = 48;
        for (&mock.pools, &mock.experts, 0..) |*pool, *experts, index| {
            pool.* = .{ .counts = &mock.decoder, .opt = .{ .layer = @intCast(index) }, .src = .{ .routed = &mock_routed } };
            experts.* = .{ .pool = pool };
            mock.engine.layers[index] = .{ .exl3 = experts };
        }
    }

    const Stream = struct {
        syncs: usize = 0,
        fail: bool = false,
        pub fn synchronize(stream: *@This()) !void {
            if (stream.fail) return error.MockCudaSyncFailure;
            stream.syncs += 1;
        }
    };
    fn owner(mock: *Mock) struct { engine: *@TypeOf(mock.engine), stream: Stream } {
        return .{ .engine = &mock.engine, .stream = .{} };
    }
};

const test_identity: codec.Identity = .{ .model = @splat(42), .world = 2, .rank = 0 };

test "exl3: placement state lifetime lock conflicts, rank separation and private storage" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const root = try temporary.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    var state = try open(allocator, io, root, test_identity);
    var held = true;
    defer if (held) state.deinit();
    try std.testing.expectError(error.WouldBlock, open(allocator, io, root, test_identity));
    const empty: codec.Counts = .{};
    try codec.write(allocator, io, state.path, state.identity, &empty);
    try std.testing.expectError(error.WouldBlock, open(allocator, io, root, test_identity));
    var rank_identity = test_identity;
    rank_identity.rank = 1;
    var other_rank = try open(allocator, io, root, rank_identity);
    defer other_rank.deinit();
    try std.testing.expect(!std.mem.eql(u8, state.path, other_rank.path));
    const lock_stat = try state.lock.stat(io);
    try std.testing.expectEqual(@as(u32, 0o600), lock_stat.permissions.toMode() & 0o777);
    const directory = try std.Io.Dir.cwd().openDir(io, std.fs.path.dirname(state.path).?, .{});
    defer directory.close(io);
    const directory_stat = try directory.stat(io);
    try std.testing.expectEqual(@as(u32, 0o700), directory_stat.permissions.toMode() & 0o777);
    state.deinit();
    held = false;
    var reopened = try open(allocator, io, root, test_identity);
    defer reopened.deinit();
}

test "exl3: placement state cold IO never imports or warms; CUDA warm errors propagate" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const root = try temporary.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    var state = try open(allocator, io, root, test_identity);
    defer state.deinit();
    var mock: Mock = undefined;
    mock.init();
    var own = mock.owner();
    try std.testing.expect(!try restore(&state, &own));
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = state.path, .data = "corrupt" });
    try std.testing.expect(!try restore(&state, &own));
    var counts: codec.Counts = .{};
    counts.decoder[0][1] = 123;
    try codec.write(allocator, io, state.path, state.identity, &counts);
    mock.pools[0].fail_warm = true;
    try std.testing.expectError(error.MockCudaWarmFailure, restore(&state, &own));
    try std.testing.expectEqual(@as(u32, 61), mock.decoder[0][1]);
    mock.engine.n_layers = 47;
    try std.testing.expect(!available(mock.engine.layers[0..mock.engine.n_layers]));
}

test "exl3: placement state decay and replacement snapshot do not add imported prior twice" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const root = try temporary.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    var state = try open(allocator, io, root, test_identity);
    defer state.deinit();
    var counts: codec.Counts = .{};
    counts.decoder[0][0..5].* = .{ 0, 1, 2, 3, std.math.maxInt(u32) };
    try codec.write(allocator, io, state.path, state.identity, &counts);
    var mock: Mock = undefined;
    mock.init();
    var own = mock.owner();
    try std.testing.expect(try restore(&state, &own));
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 1, 1, 2147483647 }, mock.decoder[0][0..5]);
    for (mock.pools) |pool| try std.testing.expectEqual(@as(usize, 1), pool.warm_calls);
    mock.decoder[0][1] += 8;
    const saved = try snapshot(&own);
    try codec.write(allocator, io, state.path, state.identity, &saved);
    const disk = try codec.read(allocator, io, state.path, state.identity);
    try std.testing.expectEqual(@as(u32, 9), disk.decoder[0][1]);
    mock.init();
    own = mock.owner();
    try std.testing.expect(try restore(&state, &own));
    try std.testing.expectEqual(@as(u32, 4), mock.decoder[0][1]);
    const missing = try std.fs.path.join(allocator, &.{ root, "missing", "profile" });
    defer allocator.free(missing);
    try std.testing.expectError(error.FileNotFound, codec.write(allocator, io, missing, state.identity, &saved));
}

test "exl3: placement state snapshot synchronizes once and drains decoder counts" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    var mock: Mock = undefined;
    mock.init();
    var own = mock.owner();
    for (&mock.pools, 0..) |*pool, index| pool.counts[pool.opt.layer][7] = @intCast(100 + index);
    const counts = try snapshot(&own);
    try std.testing.expectEqual(@as(usize, 1), own.stream.syncs);
    try std.testing.expectEqual(@as(u32, 100), counts.decoder[0][7]);
    try std.testing.expectEqual(@as(u32, 147), counts.decoder[47][7]);
    for (mock.pools) |pool| try std.testing.expect(pool.lines.drained);
    own.stream.fail = true;
    try std.testing.expectError(error.MockCudaSyncFailure, snapshot(&own));
}

test "exl3: prepared placement counts import and warm only after pools attach" {
    var mock: Mock = undefined;
    mock.init();
    var own = mock.owner();
    var counts: codec.Counts = .{};
    for (&counts.decoder, 0..) |*row, layer| row[7] = @intCast(layer + 1);

    try finish(&own, &counts);

    for (mock.pools[0..48], 0..) |pool, layer| {
        try std.testing.expectEqual(@as(u32, @intCast(layer + 1)), mock.decoder[layer][7]);
        try std.testing.expectEqual(@as(usize, 1), pool.warm_calls);
    }
}

test "exl3: placement fingerprint reads routed and metadata bytes and ignores pool size" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const root = try temporary.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    try temporary.dir.writeFile(io, .{ .sub_path = "config.json", .data = "config" });
    try temporary.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "index" });
    var mock: Mock = undefined;
    mock.init();
    const original = try fingerprint(allocator, io, root, &mock.engine.layers);
    for (&mock.experts) |*experts| experts.pool = null;
    try std.testing.expectEqual(original, try fingerprint(allocator, io, root, &mock.engine.layers));
    for (&mock.experts, &mock.pools) |*experts, *pool| experts.pool = pool;
    mock.pools[0].opt.lines = 23;
    try std.testing.expectEqual(original, try fingerprint(allocator, io, root, &mock.engine.layers));
    mock.pools[47].src.routed = &.{.{ "changed gate", "up", "down" }};
    try std.testing.expect(!std.mem.eql(u8, &original, &(try fingerprint(allocator, io, root, &mock.engine.layers))));
    mock.pools[47].src.routed = &.{.{ "gate", "up", "down" }};
    try temporary.dir.writeFile(io, .{ .sub_path = "config.json", .data = "changed config" });
    try std.testing.expect(!std.mem.eql(u8, &original, &(try fingerprint(allocator, io, root, &mock.engine.layers))));
    try temporary.dir.writeFile(io, .{ .sub_path = "config.json", .data = "config" });
    try temporary.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "changed index" });
    try std.testing.expect(!std.mem.eql(u8, &original, &(try fingerprint(allocator, io, root, &mock.engine.layers))));
    try temporary.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "index" });
}

test "exl3: placement lock rejects nonregular storage without waiting for FIFO readers" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const path = try temporary.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(path);
    try std.testing.expectError(error.ProfileLockNotRegular, openLock(io, "/dev/null"));
    const fifo = try std.fs.path.join(allocator, &.{ path, "profile.lock" });
    defer allocator.free(fifo);
    const terminated = try allocator.dupeSentinel(u8, fifo, 0);
    defer allocator.free(terminated);
    switch (@import("builtin").os.tag) {
        .linux => {
            const result = std.os.linux.mknodat(temporary.dir.handle, "profile.lock", std.posix.S.IFIFO | 0o600, 0);
            if (std.posix.errno(result) != .SUCCESS) return error.FifoCreateFailed;
        },
        .macos => {
            const Native = struct {
                extern "c" fn mkfifo(pathname: [*:0]const u8, mode: std.posix.mode_t) c_int;
            };
            if (Native.mkfifo(terminated, 0o600) != 0) return error.FifoCreateFailed;
        },
        else => return error.SkipZigTest,
    }
    try std.testing.expectError(error.ProfileLockNotRegular, openLock(io, fifo));
}
