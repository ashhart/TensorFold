//! The Qwen3.5 CUDA disk image's tests: it covers a host copy, goes to disk and back byte-equal, refuses what differs.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const disk = @import("cuda_disk.zig");
const reuse = @import("cuda_reuse.zig");
const engine = @import("cuda_engine.zig");
const geometry = @import("cuda_geometry.zig");
const pc = api.prompt_cache;
const rows = pc.rows;
const Engine = engine.Engine;
const Copy = reuse.Copy;

/// Pinned host memory from the C heap, as the driver would hand it out.
const HostFake = struct {
    fn alloc(p: *?*anyopaque, len: usize, _: c_uint) callconv(.c) cuda.abi.Result {
        p.* = std.c.malloc(len);
        return if (p.* == null) 2 else cuda.abi.success;
    }
    fn free(p: ?*anyopaque) callconv(.c) cuda.abi.Result {
        std.c.free(p);
        return cuda.abi.success;
    }
    var fail_sync = false;
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return if (fail_sync) cuda.abi.error_not_ready else cuda.abi.success;
    }
};

/// Three linear layers then attention, twice: the 27B's pattern at a test's size.
fn testGeometry() geometry.Geometry {
    var g: geometry.Geometry = undefined;
    g.hidden = 256;
    g.k_heads = 2;
    g.v_heads = 4;
    g.q_heads = 4;
    g.kv_heads = 2;
    g.layers = 8;
    @memset(&g.linear, false);
    for (0..g.layers) |i| g.linear[i] = i % 4 != 3;
    return g;
}

const Fixture = struct {
    d: cuda.Driver,
    e: Engine,
    caches: std.AutoHashMapUnmanaged(*@import("lanes").Stream, *engine.Cache) = .empty,
    seen: std.AutoHashMapUnmanaged(*@import("lanes").Stream, []reuse.Seen) = .empty,
    t: reuse.Target = undefined,

    fn init(f: *Fixture, kv8: bool) void {
        var a: cuda.abi.Api = undefined;
        a.cuMemHostAlloc = HostFake.alloc;
        a.cuMemFreeHost = HostFake.free;
        a.cuStreamSynchronize = HostFake.sync;
        f.d = .{ .lib = undefined, .api = a };
        f.e = undefined;
        f.e.kit = null;
        f.e.gpa = std.testing.allocator;
        f.e.d = &f.d;
        f.e.g = testGeometry();
        f.e.stream = .{ .d = &f.d, .handle = null };
        if (comptime @hasField(Engine, "kv8")) f.e.kv8 = kv8;
        f.t = .{ .e = &f.e, .caches = &f.caches, .seen = &f.seen };
    }

    /// A kept state at `at` whose bytes are a function of position and piece: a longer state extends a shorter one.
    fn kept(f: *Fixture, at: u32) !*Copy {
        const c = try disk.blankCopy(&f.e, at);
        const l = reuse.layoutOf(&f.e, at);
        var off: usize = 0;
        for (0..l.count()) |i| {
            const p = l.piece(i);
            const row = if (i == 0) p.len else p.len / at;
            for (c.host.bytes[off..][0..p.len], 0..) |*b, j| b.* = @truncate(i *% 131 +% (j / row) *% 7 +% j % row);
            if (i == 0) @memset(c.host.bytes[off..][0..p.len], @truncate(at)); // the recurrences: the position's own
            off += p.len;
        }
        return c;
    }

    fn snapshots(f: *Fixture) pc.Snapshots {
        return .{ .ptr = &f.t, .vtable = &vtable };
    }

    const vtable: pc.Snapshots.VTable = .{ .bytes = reuse.bytes, .save = reuse.save, .restore = reuse.restore, .drop = reuse.drop, .rows = &disk.image };
};

fn imageOf(a: std.mem.Allocator, f: *Fixture, c: *Copy) ![]u8 {
    var parts: std.ArrayList(rows.image.Part) = .empty;
    defer parts.deinit(a);
    try disk.partsOf(&f.e, @intCast(c.at), &parts, a);
    const out = try a.alloc(u8, @intCast(rows.image.imageBytes(parts.items)));
    try disk.image.gather(&f.t, c, &.{.{ .off = 0, .len = out.len }}, out);
    return out;
}

test "the image is the host copy and 16 bytes of words, its rows split into 256-token blocks, bf16 and fp8" {
    const a = std.testing.allocator;
    for ([_]bool{ false, true }) |kv8| {
        var f: Fixture = .{ .d = undefined, .e = undefined };
        f.init(kv8);
        for ([_]u32{ 1, 255, 256, 4096, 4100 }) |at| {
            var parts: std.ArrayList(rows.image.Part) = .empty;
            defer parts.deinit(a);
            try disk.partsOf(&f.e, at, &parts, a);
            try std.testing.expect(rows.image.valid(parts.items, at, rows.block_tokens));
            try std.testing.expectEqual(16 + reuse.layoutOf(&f.e, at).total(), rows.image.imageBytes(parts.items));
            try std.testing.expectEqual(f.e.g.stateBytes(), parts.items[1].bytes); // the recurrences and tails, own
        }
    }
}

test "gather and scatter give back the copy byte for byte; a copy kept with images stays off disk" {
    const a = std.testing.allocator;
    var f: Fixture = .{ .d = undefined, .e = undefined };
    f.init(true);
    const c = try f.kept(512);
    defer reuse.dropCopy(&f.e, c);
    c.delta = -3;
    const img = try imageOf(a, &f, c);
    defer a.free(img);
    const back: *Copy = @ptrCast(@alignCast(try disk.image.blank(&f.t, 512)));
    defer reuse.dropCopy(&f.e, back);
    // two pieces, split inside the words: the store reads blocks and a state's own ranges apart
    try disk.image.scatter(&f.t, back, &.{ .{ .off = 0, .len = 9 }, .{ .off = 9, .len = img.len - 9 } }, img);
    try std.testing.expectEqualSlices(u8, c.host.bytes[0..c.len], back.host.bytes[0..back.len]);
    try std.testing.expectEqual(@as(i32, -3), back.delta);
    try std.testing.expect(disk.image.whole.?(&f.t, c));
    var images = [_]reuse.Seen{.{ .at = 4, .id = 9 }};
    const own = c.images;
    c.images = &images;
    try std.testing.expect(!disk.image.whole.?(&f.t, c));
    c.images = own;
    // a state read back into another position or format restores nowhere
    const other: *Copy = @ptrCast(@alignCast(try disk.image.blank(&f.t, 512)));
    defer reuse.dropCopy(&f.e, other);
    other.format = .bf16;
    try std.testing.expectError(error.KeptWordsDiffer, disk.image.scatter(&f.t, other, &.{.{ .off = 0, .len = img.len }}, img));
}

/// Every file under `path` whose name ends in `ext`: cut to 100 bytes, or byte 40 overwritten.
fn spoil(a: std.mem.Allocator, io: std.Io, path: []const u8, ext: []const u8, cut: bool) !void {
    var d = try std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true });
    defer d.close(io);
    var w = try d.walk(a);
    defer w.deinit();
    while (try w.next(io)) |e| {
        if (e.kind != .file or !std.mem.endsWith(u8, e.basename, ext)) continue;
        const f = try e.dir.openFile(io, e.basename, .{ .mode = .read_write });
        defer f.close(io);
        if (cut) try f.setLength(io, 100) else try f.writePositionalAll(io, "x", 40);
    }
}

fn entry(a: std.mem.Allocator, c: *Copy, prompt: []u32) !*pc.Entry {
    const e = try a.create(pc.Entry);
    e.* = .{ .tokens = prompt[0..c.at], .at = @intCast(c.at), .saved = c, .bytes = c.len, .born = @intCast(prompt.len), .used = 0, .last = prompt };
    return e;
}

/// The short and long states written under `o.root` by a store that closes after.
fn writeBoth(a: std.mem.Allocator, f: *Fixture, o: rows.Options, prompt: []u32, short: *Copy, long: *Copy) !void {
    const r = try rows.Rows.open(std.testing.allocator, f.snapshots(), o);
    defer r.deinit();
    rows.write.queued(r, try entry(a, short, prompt), &.{ 0, 512 });
    rows.write.queued(r, try entry(a, long, prompt), &.{ 0, 512 });
    _ = rows.write.drain(r, rows.write.nowNs() + 3 * std.time.ns_per_s); // the write worker publishes the states
    try std.testing.expectEqual(@as(usize, 2), r.states.items.len);
    try std.testing.expectEqual(@as(usize, 4), r.blocks.count()); // 512 and 1024 tokens on a 512-row chunk grid: two blocks shared
}

test "kept states written to disk read back byte-equal after a restart, sharing blocks; a cut or flipped file is refused" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fmt.allocPrint(a, "{s}/restart", .{try tmp.dir.realPathFileAlloc(io, ".", a)});
    const prompt = try a.alloc(u32, 1200);
    for (prompt, 0..) |*x, i| x.* = @intCast(1000 + i * 7 % 997);
    const o: rows.Options = .{ .root = root, .identity = 5, .cap = 1 << 34, .floor = 0, .daily = 1 << 40, .planned = true };
    for ([_]bool{ false, true }) |kv8| {
        try tmp.dir.deleteTree(io, "restart");
        var f: Fixture = .{ .d = undefined, .e = undefined };
        f.init(kv8);
        const short = try f.kept(512);
        defer reuse.dropCopy(&f.e, short);
        const long = try f.kept(1024);
        defer reuse.dropCopy(&f.e, long);
        try writeBoth(a, &f, o, prompt, short, long);
        const r = try rows.Rows.open(gpa, f.snapshots(), o);
        defer r.deinit();
        try std.testing.expectEqual(@as(usize, 2), r.states.items.len);
        for (r.states.items, 0..) |s, i| {
            const want = if (s.at == 512) short else long;
            const got: *Copy = @ptrCast(@alignCast(try rows.read.read(r, i)));
            defer reuse.dropCopy(&f.e, got);
            try std.testing.expectEqualSlices(u8, want.host.bytes[0..want.len], got.host.bytes[0..got.len]);
        }
    }
    // a block cut short, then a state's byte flipped: each read fails its check, those states go, nothing crashes
    for ([_][]const u8{ ".blk", ".st" }) |ext| {
        try tmp.dir.deleteTree(io, "restart");
        var f: Fixture = .{ .d = undefined, .e = undefined };
        f.init(true);
        const short = try f.kept(512);
        defer reuse.dropCopy(&f.e, short);
        const long = try f.kept(1024);
        defer reuse.dropCopy(&f.e, long);
        try writeBoth(a, &f, o, prompt, short, long);
        const r = try rows.Rows.open(gpa, f.snapshots(), o);
        defer r.deinit();
        try spoil(a, io, r.dir, ext, std.mem.eql(u8, ext, ".blk"));
        while (r.states.items.len > 0) {
            const before = r.states.items.len;
            if (rows.read.read(r, 0)) |got| {
                reuse.dropCopy(&f.e, @ptrCast(@alignCast(got)));
                return error.CorruptStateRead;
            } else |err| try std.testing.expect(err == error.RowsRead or err == error.RowsHash or err == error.RowsMissing);
            try std.testing.expect(r.states.items.len < before);
        }
    }
}

/// A target whose prompt pass fails or not, counting its releases.
const ProbeTarget = struct {
    fail: bool = false,
    releases: usize = 0,

    fn of(p: *anyopaque) *ProbeTarget {
        return @ptrCast(@alignCast(p));
    }
    fn prefill(p: *anyopaque, _: *Stream) anyerror!void {
        if (of(p).fail) return error.Injected;
    }
    fn release(p: *anyopaque, _: *Stream) void {
        of(p).releases += 1;
    }
    fn first(_: *anyopaque, _: *Stream, _: u64) anyerror!u64 {
        return error.Unexpected;
    }
    fn queue(_: *anyopaque, _: *Stream, _: backend.Feed, _: u64) anyerror!u64 {
        return error.Unexpected;
    }
    fn read(_: *anyopaque, _: u64) anyerror!u32 {
        return error.Unexpected;
    }
    fn verify(_: *anyopaque, _: []const backend.Window, _: []backend.Verified) anyerror!void {
        return error.Unexpected;
    }
    fn keep(_: *anyopaque, _: []const backend.Window, _: []const []const u32) anyerror!void {
        return error.Unexpected;
    }
    fn draft(_: *anyopaque, _: []const backend.DraftRequest) anyerror!void {
        return error.Unexpected;
    }
    const vtable: backend.Backend.VTable = .{ .prefill = prefill, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release };
};
const Stream = @import("lanes").Stream;
const backend = @import("lanes").backend;

test "the probe releases its stream once whether its prompt pass, its wait or its mark fails" {
    var f: Fixture = undefined;
    f.init(false);
    const io = std.testing.io;
    for ([_]struct { fail: bool, sync: bool, err: anyerror }{
        .{ .fail = true, .sync = false, .err = error.Injected },
        .{ .fail = false, .sync = true, .err = error.NotReady },
        .{ .fail = false, .sync = false, .err = error.ProbeUnmarked }, // the pass kept no state at the mark
    }) |c| {
        var target: ProbeTarget = .{ .fail = c.fail };
        HostFake.fail_sync = c.sync;
        defer HostFake.fail_sync = false;
        try std.testing.expectError(c.err, disk.probe(std.testing.allocator, io, .{ .ptr = &target, .vtable = &ProbeTarget.vtable }, &f.t));
        try std.testing.expectEqual(@as(usize, 1), target.releases);
    }
}
