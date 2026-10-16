//! A rank's checkpoint for the ranks' agreement: its identity before any weight loads, then every byte as it loads.
const std = @import("std");
const builtin = @import("builtin");
const Sha256 = std.crypto.hash.sha2.Sha256;
const Io = std.Io;

/// Before any weight loads: config.json (a split's rank left out), the index, each shard's name, size and header.
pub fn identity(gpa: std.mem.Allocator, io: Io, dir: []const u8) ![32]u8 {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var h = Sha256.init(.{});
    const config = try Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "config.json" }), a, .limited(1 << 26));
    hashValue(&h, try std.json.parseFromSliceLeaky(std.json.Value, a, config, .{}), false);
    for (try shards(a, io, dir, &h)) |name| {
        h.update(name);
        const file = try Io.Dir.cwd().openFile(io, try std.fs.path.join(a, &.{ dir, name }), .{});
        defer file.close(io);
        const size = try file.length(io);
        h.update(std.mem.asBytes(&size));
        h.update(try header(a, io, file, size));
    }
    return h.finalResult();
}

/// The index's shard names (or the one model.safetensors) in name order, each once; the index text goes into `h`.
fn shards(a: std.mem.Allocator, io: Io, dir: []const u8, h: ?*Sha256) ![]const []const u8 {
    var names: std.ArrayList([]const u8) = .empty;
    const index = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
    if (Io.Dir.cwd().readFileAlloc(io, index, a, .limited(1 << 26))) |text| {
        if (h) |x| x.update(text);
        const root = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        const map = if (root == .object) root.object.get("weight_map") else null;
        if (map == null or map.? != .object) return error.BadIndex;
        for (map.?.object.values()) |v| if (v == .string) try names.append(a, v.string);
    } else |err| switch (err) {
        error.FileNotFound => try names.append(a, "model.safetensors"),
        else => return err,
    }
    std.mem.sort([]const u8, names.items, {}, lessThan);
    var out: std.ArrayList([]const u8) = .empty;
    for (names.items) |n| if (out.items.len == 0 or !std.mem.eql(u8, out.items[out.items.len - 1], n)) try out.append(a, n);
    return out.items;
}

/// A shard's JSON header, checked against its size.
fn header(a: std.mem.Allocator, io: Io, file: Io.File, size: u64) ![]u8 {
    var head: [8]u8 = undefined;
    if (try file.readPositionalAll(io, &head, 0) != 8) return error.BadHeader;
    const n = std.mem.readInt(u64, &head, .little);
    if (n > 1 << 30 or 8 + n > size) return error.BadHeader;
    const text = try a.alloc(u8, @intCast(n));
    if (try file.readPositionalAll(io, text, 8) != text.len) return error.BadHeader;
    return text;
}

/// A JSON value in a fixed form; our split's record (tensorfold_tp) is digested without its rank, which differs.
fn hashValue(h: *Sha256, v: std.json.Value, in_split: bool) void {
    switch (v) {
        .null => h.update("n"),
        .bool => |b| h.update(if (b) "t" else "f"),
        .integer => |i| {
            h.update("i");
            h.update(std.mem.asBytes(&i));
        },
        .float => |f| {
            h.update("d");
            h.update(std.mem.asBytes(&f));
        },
        .number_string, .string => |s| {
            h.update("s");
            h.update(std.mem.asBytes(&s.len));
            h.update(s);
        },
        .array => |x| {
            h.update("[");
            for (x.items) |item| hashValue(h, item, false);
            h.update("]");
        },
        .object => |o| {
            h.update("{");
            for (o.keys(), o.values()) |k, item| {
                if (in_split and std.mem.eql(u8, k, "rank")) continue;
                hashValue(h, .{ .string = k }, false);
                hashValue(h, item, std.mem.eql(u8, k, "tensorfold_tp"));
            }
            h.update("}");
        },
    }
}

/// An expert tensor (routed or shared): a split checkpoint's halves hold different shares of it.
fn expert(name: []const u8) bool {
    return std.mem.indexOf(u8, name, ".experts.") != null or std.mem.indexOf(u8, name, ".shared_experts.") != null;
}

fn lessThan(_: void, x: []const u8, y: []const u8) bool {
    return std.mem.order(u8, x, y) == .lt;
}

/// A byte range of a shard: a non-expert tensor's payload.
const Range = struct { lo: u64, hi: u64 };

fn byLo(_: void, x: Range, y: Range) bool {
    return x.lo < y.lo;
}

/// Every byte of every shard, hashed in a few threads while the weights load (whole: all of it; split: see finish).
pub const Content = struct {
    gpa: std.mem.Allocator,
    io: Io,
    arena: std.heap.ArenaAllocator,
    dir: []const u8,
    split: bool,
    names: []const []const u8,
    files: [][32]u8, // each file's sha256, every byte
    plain: [][32]u8, // each file's non-expert tensor bytes (a split's shared part)
    next: std.atomic.Value(usize) = .init(0),
    failed: std.atomic.Value(usize) = .init(std.math.maxInt(usize)),
    stop: std.atomic.Value(bool) = .init(false),
    bytes: std.atomic.Value(u64) = .init(0),
    threads: [workers]?std.Thread = @splat(null),
    t0: Io.Timestamp,

    const workers = 8;
    const block = 8 << 20;

    pub fn start(gpa: std.mem.Allocator, io: Io, dir: []const u8, split: bool) !*Content {
        const c = try gpa.create(Content);
        errdefer gpa.destroy(c);
        c.* = .{ .gpa = gpa, .io = io, .arena = .init(gpa), .dir = undefined, .split = split, .names = undefined, .files = undefined, .plain = undefined, .t0 = Io.Clock.awake.now(io) };
        errdefer c.arena.deinit();
        const a = c.arena.allocator();
        c.dir = try a.dupe(u8, dir);
        c.names = try shards(a, io, dir, null);
        c.files = try a.alloc([32]u8, c.names.len);
        c.plain = try a.alloc([32]u8, c.names.len);
        for (&c.threads) |*t| t.* = std.Thread.spawn(.{}, work, .{c}) catch null;
        if (c.threads[0] == null) c.work(); // no thread could start: hash here
        return c;
    }

    fn work(c: *Content) void {
        const buf = std.heap.page_allocator.alloc(u8, block) catch return c.fail(c.names.len);
        defer std.heap.page_allocator.free(buf);
        while (!c.stop.load(.acquire)) {
            const i = c.next.fetchAdd(1, .monotonic);
            if (i >= c.names.len) return;
            c.file(i, buf) catch return c.fail(i);
        }
    }

    fn fail(c: *Content, i: usize) void {
        _ = c.failed.fetchMin(i, .monotonic);
        c.stop.store(true, .release);
    }

    /// File `i`: its sha256 over every byte; for a split, its non-expert tensors' bytes too, in offset order.
    fn file(c: *Content, i: usize, buf: []u8) !void {
        var arena: std.heap.ArenaAllocator = .init(c.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const f = try Io.Dir.cwd().openFile(c.io, try std.fs.path.join(a, &.{ c.dir, c.names[i] }), .{});
        defer f.close(c.io);
        const size = try f.length(c.io);
        var ranges: std.ArrayList(Range) = .empty;
        if (c.split) {
            const text = try header(a, c.io, f, size);
            const root = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
            if (root != .object) return error.BadHeader;
            const start_at = 8 + @as(u64, text.len);
            for (root.object.keys(), root.object.values()) |k, t| {
                if (expert(k) or std.mem.eql(u8, k, "__metadata__")) continue;
                const offs = if (t == .object) t.object.get("data_offsets") else null;
                if (offs == null or offs.? != .array or offs.?.array.items.len != 2) return error.BadHeader;
                const b0: u64 = @intCast(offs.?.array.items[0].integer);
                const b1: u64 = @intCast(offs.?.array.items[1].integer);
                if (b1 < b0 or start_at + b1 > size) return error.BadHeader;
                try ranges.append(a, .{ .lo = start_at + b0, .hi = start_at + b1 });
            }
            std.mem.sort(Range, ranges.items, {}, byLo);
        }
        var all = Sha256.init(.{});
        var part = Sha256.init(.{});
        var at: u64 = 0;
        var r: usize = 0;
        while (at < size) {
            if (c.stop.load(.acquire)) return error.Stopped;
            const n = try f.readPositionalAll(c.io, buf[0..@intCast(@min(buf.len, size - at))], at);
            if (n == 0) return error.FileShrank;
            all.update(buf[0..n]);
            const end = at + n;
            while (r < ranges.items.len and ranges.items[r].lo < end) {
                const lo = @max(ranges.items[r].lo, at);
                const hi = @min(ranges.items[r].hi, end);
                if (hi > lo) part.update(buf[@intCast(lo - at)..@intCast(hi - at)]);
                if (ranges.items[r].hi > end) break;
                r += 1;
            }
            drop(f, at, n);
            at = end;
            _ = c.bytes.fetchAdd(n, .monotonic);
        }
        c.files[i] = all.finalResult();
        c.plain[i] = part.finalResult();
    }

    /// The pages just read leave the page cache: on unified memory they would take the device's.
    fn drop(f: Io.File, at: u64, n: usize) void {
        if (builtin.os.tag != .linux) return;
        _ = std.os.linux.fadvise(f.handle, @intCast(at), @intCast(n), std.os.linux.POSIX_FADV.DONTNEED);
    }

    fn join(c: *Content) void {
        for (&c.threads) |*t| if (t.*) |x| {
            x.join();
            t.* = null;
        };
    }

    /// Stops the threads and frees everything (a load that failed first).
    pub fn abandon(c: *Content) void {
        c.stop.store(true, .release);
        c.join();
        c.arena.deinit();
        c.gpa.destroy(c);
    }

    /// Each file checked against its manifest or blob name; the digest: a whole one's bytes, a split's non-experts.
    pub fn finish(c: *Content) ![32]u8 {
        defer c.abandon();
        c.join();
        const bad = c.failed.load(.monotonic);
        if (bad != std.math.maxInt(usize)) {
            std.log.err("hashing the checkpoint failed at {s}", .{if (bad < c.names.len) c.names[bad] else "a buffer"});
            return error.CheckpointUnreadable;
        }
        const a = c.arena.allocator();
        const want = if (c.split) try manifest(a, c.io, c.dir, c.names) else try blobNames(a, c.io, c.dir, c.names);
        for (c.names, c.files, want) |name, got, w| if (w) |x| if (!std.mem.eql(u8, &got, &x)) {
            if (!builtin.is_test) std.log.err("{s}'s bytes differ from its recorded sha256 ({s}): the copy is damaged or another revision", .{ name, if (c.split) "the export manifest" else "its Hugging Face blob name" });
            return error.ShardBytesDiffer;
        };
        var h = Sha256.init(.{});
        for (c.names, c.files, c.plain) |name, all, part| {
            h.update(name);
            h.update(if (c.split) &part else &all);
        }
        const s = @as(f64, @floatFromInt(Io.Clock.awake.now(c.io).toNanoseconds() - c.t0.toNanoseconds())) / 1e9;
        std.log.info("checkpoint bytes hashed: {d:.1} GiB in {d:.1} s, every byte of {d} files", .{ @as(f64, @floatFromInt(c.bytes.load(.monotonic))) / (1 << 30), s, c.names.len });
        return h.finalResult();
    }
};

/// A split's export manifest: each file's sha256 (every file must be listed).
fn manifest(a: std.mem.Allocator, io: Io, dir: []const u8, names: []const []const u8) ![]?[32]u8 {
    const path = try std.fs.path.join(a, &.{ dir, "tensorfold-export-manifest.json" });
    const text = Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 26)) catch |err| {
        std.log.err("a split checkpoint needs its tensorfold-export-manifest.json to check its expert bytes ({t})", .{err});
        return error.NoManifest;
    };
    const root = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
    const files = if (root == .object) root.object.get("files") else null;
    if (files == null or files.? != .object) return error.NoManifest;
    const out = try a.alloc(?[32]u8, names.len);
    for (names, out) |n, *o| {
        const e = files.?.object.get(n) orelse return error.NoManifest;
        const sha = if (e == .object) e.object.get("sha256") else null;
        if (sha == null or sha.? != .string) return error.NoManifest;
        o.* = hex(sha.?.string) orelse return error.NoManifest;
    }
    return out;
}

/// Each file's sha256 from its Hugging Face blob name (a snapshot's links into blobs/), or null.
fn blobNames(a: std.mem.Allocator, io: Io, dir: []const u8, names: []const []const u8) ![]?[32]u8 {
    const out = try a.alloc(?[32]u8, names.len);
    for (names, out) |n, *o| {
        o.* = null;
        const real = Io.Dir.cwd().realPathFileAlloc(io, try std.fs.path.join(a, &.{ dir, n }), a) catch continue;
        const parent = std.fs.path.dirname(real) orelse continue;
        if (std.mem.eql(u8, std.fs.path.basename(parent), "blobs")) o.* = hex(std.fs.path.basename(real));
    }
    return out;
}

fn hex(s: []const u8) ?[32]u8 {
    if (s.len != 64) return null;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, s) catch return null;
    return out;
}

/// A test checkpoint: config.json, a dense and a 1 MiB expert tensor, a manifest of `manifest_sha` (else the shard's).
fn write(io: Io, dir: Io.Dir, sub: []const u8, config: []const u8, dense: u8, flip: ?usize, manifest_sha: ?[32]u8) !void {
    const gpa = std.testing.allocator;
    const mib = 1 << 20;
    const head = "{\"m.layers.0.d.weight\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]}," ++
        "\"m.layers.1.mixer.experts.0.up_proj.weight\":{\"dtype\":\"U8\",\"shape\":[1048576],\"data_offsets\":[4,1048580]}}";
    const bytes = try gpa.alloc(u8, 8 + head.len + 4 + mib);
    defer gpa.free(bytes);
    std.mem.writeInt(u64, bytes[0..8], head.len, .little);
    @memcpy(bytes[8..][0..head.len], head);
    const body = bytes[8 + head.len ..];
    for (body, 0..) |*b, i| b.* = @truncate(i *% 13);
    body[0] = dense;
    if (flip) |at| body[4 + at] ^= 1;
    try dir.createDirPath(io, sub);
    var d = try dir.openDir(io, sub, .{});
    defer d.close(io);
    try d.writeFile(io, .{ .sub_path = "config.json", .data = config });
    try d.writeFile(io, .{ .sub_path = "model.safetensors", .data = bytes });
    var sha: [32]u8 = undefined;
    Sha256.hash(bytes, &sha, .{});
    const m = try std.fmt.allocPrint(gpa, "{{\"files\": {{\"model.safetensors\": {{\"sha256\": \"{s}\"}}}}}}", .{&std.fmt.bytesToHex(manifest_sha orelse sha, .lower)});
    defer gpa.free(m);
    try d.writeFile(io, .{ .sub_path = "tensorfold-export-manifest.json", .data = m });
}

fn contentOf(io: Io, path: []const u8, split: bool) ![32]u8 {
    const c = try Content.start(std.testing.allocator, io, path, split);
    return c.finish();
}

test "every byte counts: one bit at 8 KiB into a 1 MiB expert payload changes the digest, or fails a split's manifest" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const cfg = "{\"hidden_size\": 8, \"tensorfold_tp\": {\"rank\": 0, \"world\": 2}}";
    try write(io, tmp.dir, "a", cfg, 9, null, null);
    try write(io, tmp.dir, "b", "{\"hidden_size\": 8, \"tensorfold_tp\": {\"rank\": 1, \"world\": 2}}", 9, null, null);
    try write(io, tmp.dir, "c", cfg, 9, 8192, null); // the expert byte past every old sample, manifest updated
    try write(io, tmp.dir, "d", cfg, 10, null, null); // a dense byte
    var good: [32]u8 = undefined;
    Sha256.hash("not this shard", &good, .{});
    try write(io, tmp.dir, "e", cfg, 9, 8192, good); // the expert byte, the manifest still the old file's
    var path: [5][]u8 = undefined;
    for ([_][]const u8{ "a", "b", "c", "d", "e" }, &path) |s, *p| p.* = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/{s}", .{ tmp.sub_path, s });
    defer for (path) |p| gpa.free(p);
    // before loading: the split's rank leaves the identity alone, its bytes are not read
    try std.testing.expectEqual(try identity(gpa, io, path[0]), try identity(gpa, io, path[1]));
    try std.testing.expectEqual(try identity(gpa, io, path[0]), try identity(gpa, io, path[2]));
    // whole: every byte, so the unsampled expert byte changes the digest
    try std.testing.expect(!std.mem.eql(u8, &try contentOf(io, path[0], false), &try contentOf(io, path[2], false)));
    // split: the halves share their non-expert bytes; a dense byte differs; a damaged expert byte fails the manifest
    try std.testing.expectEqual(try contentOf(io, path[0], true), try contentOf(io, path[1], true));
    try std.testing.expect(!std.mem.eql(u8, &try contentOf(io, path[0], true), &try contentOf(io, path[3], true)));
    try std.testing.expectError(error.ShardBytesDiffer, contentOf(io, path[4], true));
}

test "a Hugging Face snapshot's shard is checked against its blob name: the right sha256 passes, another refuses" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try write(io, tmp.dir, "src", "{\"hidden_size\": 8}", 9, null, null);
    const bytes = try tmp.dir.readFileAlloc(io, "src/model.safetensors", gpa, .limited(1 << 24));
    defer gpa.free(bytes);
    var sha: [32]u8 = undefined;
    Sha256.hash(bytes, &sha, .{});
    var other: [32]u8 = undefined;
    Sha256.hash("another revision", &other, .{});
    for ([_][]const u8{ "good", "bad" }, [_][32]u8{ sha, other }) |repo, name| {
        var buf: [128]u8 = undefined;
        const blobs = try tmp.dir.createDirPathOpen(io, try std.fmt.bufPrint(&buf, "{s}/blobs", .{repo}), .{});
        defer blobs.close(io);
        const hexname = std.fmt.bytesToHex(name, .lower);
        try blobs.writeFile(io, .{ .sub_path = &hexname, .data = bytes });
        const snap = try tmp.dir.createDirPathOpen(io, try std.fmt.bufPrint(&buf, "{s}/snapshots/r1", .{repo}), .{});
        defer snap.close(io);
        try snap.writeFile(io, .{ .sub_path = "config.json", .data = "{\"hidden_size\": 8}" });
        snap.symLink(io, try std.fmt.bufPrint(&buf, "../../blobs/{s}", .{&hexname}), "model.safetensors", .{}) catch |e|
            return if (e == error.PermissionDenied and @import("builtin").os.tag == .windows) error.SkipZigTest else e; // symlinks need privilege on Windows
    }
    const good = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/good/snapshots/r1", .{tmp.sub_path});
    defer gpa.free(good);
    const bad = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/bad/snapshots/r1", .{tmp.sub_path});
    defer gpa.free(bad);
    _ = try contentOf(io, good, false);
    try std.testing.expectError(error.ShardBytesDiffer, contentOf(io, bad, false));
}
