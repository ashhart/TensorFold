//! A rank's checkpoint for the ranks' agreement: its identity before any weight loads, then every byte as it loads.
const std = @import("std");
const builtin = @import("builtin");
const Sha256 = std.crypto.hash.sha2.Sha256;
const Io = std.Io;
const digest_cache = @import("digest_cache.zig");
const runtime_sha = @import("runtime_sha256.zig");
const index_format = @import("checkpoint_index.zig");

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
        const map = try index_format.weightMap(root);
        for (map.values()) |v| try names.append(a, v.string);
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
    const length = @import("safetensors.zig").headerSize(n, size) catch return error.BadHeader;
    const text = try a.alloc(u8, length);
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
    cache_root: ?[]const u8 = null,
    keys: []digest_cache.Key,
    errors: []?anyerror,
    hits: std.atomic.Value(usize) = .init(0),
    files: [][32]u8, // each file's sha256, every byte
    plain: [][32]u8, // each file's non-expert tensor bytes (a split's shared part)
    next: std.atomic.Value(usize) = .init(0),
    failed: std.atomic.Value(usize) = .init(std.math.maxInt(usize)),
    stop: std.atomic.Value(bool) = .init(false),
    bytes: std.atomic.Value(u64) = .init(0),
    threads: [max_workers]?std.Thread = @splat(null),
    hash_backend: runtime_sha.Backend,
    t0: Io.Timestamp,

    pub const max_workers = 8;
    const block = 8 << 20;
    pub const Options = struct { serving_threads: usize = 2, storage_workers: usize = max_workers, hash_mode: runtime_sha.Mode = .auto };

    pub fn workerCount(cpus: usize, serving: usize, files: usize, storage: usize) usize {
        return @min(files, max_workers, @max(1, cpus -| serving), @max(1, storage));
    }

    pub fn start(gpa: std.mem.Allocator, io: Io, dir: []const u8, split: bool) !*Content {
        const root = try digest_cache.environmentRoot(gpa);
        defer if (root) |path| gpa.free(path);
        return startWithCache(gpa, io, dir, split, root);
    }

    pub fn startWithCache(gpa: std.mem.Allocator, io: Io, dir: []const u8, split: bool, root: ?[]const u8) !*Content {
        return startWithOptions(gpa, io, dir, split, root, .{});
    }

    pub fn startWithOptions(gpa: std.mem.Allocator, io: Io, dir: []const u8, split: bool, root: ?[]const u8, options: Options) !*Content {
        const backend = try runtime_sha.select(options.hash_mode);
        const c = try gpa.create(Content);
        errdefer gpa.destroy(c);
        c.* = .{ .gpa = gpa, .io = io, .arena = .init(gpa), .dir = undefined, .split = split, .names = undefined, .files = undefined, .plain = undefined, .keys = undefined, .errors = undefined, .hash_backend = backend, .t0 = Io.Clock.awake.now(io) };
        errdefer c.arena.deinit();
        const a = c.arena.allocator();
        c.dir = try a.dupe(u8, dir);
        c.cache_root = if (root) |path| try a.dupe(u8, path) else null;
        c.names = try shards(a, io, dir, null);
        c.files = try a.alloc([32]u8, c.names.len);
        c.plain = try a.alloc([32]u8, c.names.len);
        c.keys = try a.alloc(digest_cache.Key, c.names.len);
        c.errors = try a.alloc(?anyerror, c.names.len);
        @memset(c.errors, null);
        const workers = workerCount(std.Thread.getCpuCount() catch 1, options.serving_threads, c.names.len, options.storage_workers);
        var started: usize = 0;
        for (c.threads[0..workers]) |*thread| {
            thread.* = std.Thread.spawn(.{}, work, .{c}) catch null;
            if (thread.* != null) started += 1;
        }
        if (workers > 0 and started == 0) c.work();
        return c;
    }

    fn work(c: *Content) void {
        const buf = std.heap.page_allocator.alloc(u8, block) catch return c.fail(c.names.len, error.OutOfMemory);
        defer std.heap.page_allocator.free(buf);
        while (!c.stop.load(.acquire)) {
            const i = c.next.fetchAdd(1, .monotonic);
            if (i >= c.names.len) return;
            c.file(i, buf) catch |err| return c.fail(i, err);
        }
    }

    fn fail(c: *Content, i: usize, err: anyerror) void {
        if (i < c.errors.len) c.errors[i] = err;
        _ = c.failed.fetchMin(i, .monotonic);
        c.stop.store(true, .release);
    }

    /// Reuse both the whole-file and shared-tensor digests under the same descriptor identity.
    fn file(c: *Content, i: usize, buf: []u8) !void {
        const path = try std.fs.path.join(c.gpa, &.{ c.dir, c.names[i] });
        defer c.gpa.free(path);
        const Reader = struct {
            owner: *Content,
            buffer: []u8,
            pub fn read(reader: @This(), f: Io.File, size: u64) !digest_cache.Value {
                return reader.owner.read(f, size, reader.buffer);
            }
        };
        const got = try (digest_cache.Cache{ .root = c.cache_root }).get(c.gpa, c.io, path, true, Reader{ .owner = c, .buffer = buf });
        c.files[i] = got.value.sha256;
        c.plain[i] = got.value.shared.?;
        c.keys[i] = got.key;
        if (got.hit) _ = c.hits.fetchAdd(1, .monotonic);
    }

    fn read(c: *Content, f: Io.File, size: u64, buf: []u8) !digest_cache.Value {
        var arena: std.heap.ArenaAllocator = .init(c.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        var ranges: std.ArrayList(Range) = .empty;
        const text = try header(a, c.io, f, size);
        const root = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        if (root != .object) return error.BadHeader;
        const start_at = 8 + @as(u64, text.len);
        for (root.object.keys(), root.object.values()) |k, t| {
            if (expert(k) or std.mem.eql(u8, k, "__metadata__")) continue;
            const offs = if (t == .object) t.object.get("data_offsets") else null;
            if (offs == null or offs.? != .array or offs.?.array.items.len != 2) return error.BadHeader;
            const pair = offs.?.array.items;
            if (pair[0] != .integer or pair[1] != .integer or pair[0].integer < 0 or pair[1].integer < 0)
                return error.BadHeader;
            const b0: u64 = @intCast(pair[0].integer);
            const b1: u64 = @intCast(pair[1].integer);
            if (b1 < b0 or b1 > size - start_at) return error.BadHeader;
            try ranges.append(a, .{ .lo = start_at + b0, .hi = start_at + b1 });
        }
        std.mem.sort(Range, ranges.items, {}, byLo);
        for (ranges.items, 0..) |range, i| if (i > 0 and range.lo < ranges.items[i - 1].hi) return error.BadHeader;
        var all = runtime_sha.Hash.init(c.hash_backend);
        var part = runtime_sha.Hash.init(c.hash_backend);
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
        return .{ .sha256 = all.finalResult(), .shared = part.finalResult() };
    }

    /// The pages just read leave the page cache: on unified memory they would take the device's.
    fn drop(f: Io.File, at: u64, n: usize) void {
        if (builtin.os.tag != .linux) return;
        _ = std.os.linux.fadvise(f.handle, @intCast(at), @intCast(n), std.os.linux.POSIX_FADV.DONTNEED);
    }

    pub fn wait(c: *Content) void {
        for (&c.threads) |*t| if (t.*) |x| {
            x.join();
            t.* = null;
        };
    }

    /// Stops the threads and frees everything (a load that failed first).
    pub fn abandon(c: *Content) void {
        c.stop.store(true, .release);
        c.wait();
        c.arena.deinit();
        c.gpa.destroy(c);
    }

    /// Each file checked against its manifest or blob name; the digest: a whole one's bytes, a split's non-experts.
    pub fn finish(c: *Content) ![32]u8 {
        return (try c.finishWithStats()).digest;
    }

    pub const Stats = struct { digest: [32]u8, bytes_read: u64, cache_hits: usize };

    pub fn finishWithStats(c: *Content) !Stats {
        defer c.abandon();
        c.wait();
        const bad = c.failed.load(.monotonic);
        if (bad != std.math.maxInt(usize)) {
            if (!builtin.is_test) std.log.err("hashing the checkpoint failed at {s}", .{if (bad < c.names.len) c.names[bad] else "a buffer"});
            return if (bad < c.errors.len) c.errors[bad].? else error.OutOfMemory;
        }
        const a = c.arena.allocator();
        for (c.names, c.keys) |name, key| {
            const f = try Io.Dir.cwd().openFile(c.io, try std.fs.path.join(a, &.{ c.dir, name }), .{});
            defer f.close(c.io);
            if (!key.equal(try digest_cache.Key.of(f, c.io))) return error.DigestFileChanged;
        }
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
        std.log.info("checkpoint digest: {d:.1} GiB read in {d:.1} s, {d}/{d} cached files", .{ @as(f64, @floatFromInt(c.bytes.load(.monotonic))) / (1 << 30), s, c.hits.load(.monotonic), c.names.len });
        return .{ .digest = h.finalResult(), .bytes_read = c.bytes.load(.monotonic), .cache_hits = c.hits.load(.monotonic) };
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

/// Metadata identity plus every shard byte, with no folder-name dependency.
pub fn fullIdentity(gpa: std.mem.Allocator, io: Io, dir: []const u8) ![32]u8 {
    const root = try digest_cache.environmentRoot(gpa);
    defer if (root) |path| gpa.free(path);
    return fullWithCache(gpa, io, dir, root);
}

pub fn fullWithCache(gpa: std.mem.Allocator, io: Io, dir: []const u8, root: ?[]const u8) ![32]u8 {
    const meta = try identity(gpa, io, dir);
    const bytes = try (try Content.startWithCache(gpa, io, dir, false, root)).finish();
    var h = Sha256.init(.{});
    h.update("TensorFold full checkpoint v1");
    h.update(&meta);
    h.update(&bytes);
    return h.finalResult();
}

test {
    _ = @import("checkpoint_identity_test.zig");
}
