const std = @import("std");
const mx = @import("mlx.zig");
const files = @import("snapshot_file.zig");
const session = @import("session.zig");
const policy = @import("prompt_cache.zig");
const memory = @import("memory_budget.zig");
const runtime = @import("memory_runtime.zig");
const Hash = std.crypto.hash.sha2.Sha256;
const a = std.heap.c_allocator;
pub const Cache = policy.Store(session.Snapshot);

fn hashFile(io: std.Io, hash: *Hash, path: []const u8) !void {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var buffer: [65536]u8 = undefined;
    var offset: u64 = 0;
    while (true) {
        const count = try file.readPositional(io, &.{&buffer}, offset);
        if (count == 0) break;
        hash.update(buffer[0..count]);
        offset += count;
    }
}

fn checkpointIdentity(io: std.Io, hash: *Hash, path: []const u8) !void {
    const absolute = try std.Io.Dir.cwd().realPathFileAlloc(io, path, a);
    defer a.free(absolute);
    hash.update(absolute);
    var dir = try std.Io.Dir.cwd().openDir(io, absolute, .{ .iterate = true });
    defer dir.close(io);
    var names: std.ArrayList([]const u8) = .empty;
    defer {
        for (names.items) |name| a.free(name);
        names.deinit(a);
    }
    var iterator = dir.iterate();
    while (try iterator.next(io)) |entry| {
        if (!std.mem.endsWith(u8, entry.name, ".json") and !std.mem.endsWith(u8, entry.name, ".safetensors") and !std.mem.endsWith(u8, entry.name, ".jinja")) continue;
        const name = try a.dupe(u8, entry.name);
        errdefer a.free(name);
        try names.append(a, name);
    }
    std.mem.sort([]const u8, names.items, {}, struct {
        fn less(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.less);
    for (names.items) |name| {
        const stat = try dir.statFile(io, name, .{});
        // ctime catches in-place weight replacement even when mtime is preserved.
        const stamp = try std.fmt.allocPrint(a, "\x00{s}\x00{d}:{d}:{d}:{d}", .{ name, stat.inode, stat.size, stat.mtime.toNanoseconds(), stat.ctime.toNanoseconds() });
        defer a.free(stamp);
        hash.update(stamp);
    }
}

pub fn identity(s: *session.Session) ![]u8 {
    var base = Hash.init(.{});
    try checkpointIdentity(s.io, &base, s.directory);
    var base_digest: [32]u8 = undefined;
    base.final(&base_digest);
    var hash = Hash.init(.{});
    hash.update(&base_digest);
    const executable = try std.process.executablePathAlloc(s.io, a);
    defer a.free(executable);
    try hashFile(s.io, &hash, executable);
    if (s.draft_options.enabled) if (s.draft_options.directory) |path| try checkpointIdentity(s.io, &hash, path);
    if (s.draft_options.calibration) |path| try hashFile(s.io, &hash, path);
    const form = if (s.backend == .qwen) s.backend.qwen.weights.bonsai_form else null;
    const options = try std.json.Stringify.valueAlloc(a, .{ .draft = s.draft_options, .tensor = mx.tensor_units, .bonsai = form, .buffers = @import("kv_buffer.zig").enabled }, .{});
    defer a.free(options);
    hash.update(options);
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    return std.fmt.allocPrint(a, "{s}|{s}", .{ std.fmt.bytesToHex(base_digest, .lower), std.fmt.bytesToHex(digest, .lower) });
}

pub const Entry = struct {
    path: []u8,
    tokens: []i32,
    bytes: u64,
    load_bytes: u64,
    modified: i96,
    pinned: bool,
    compatible: bool,
    fn deinit(e: *Entry) void {
        a.free(e.path);
        a.free(e.tokens);
    }
};

pub const Store = struct {
    io: std.Io,
    identity: []u8,
    directory: []u8,
    conversations: []u8,
    budget: u64,
    spill_bytes: u64,
    max_snapshots: usize,
    reads: u64 = 0,
    writes: u64 = 0,

    pub fn init(s: *session.Session, directory: []const u8, budget: u64, spill_bytes: u64, max_snapshots: usize) !Store {
        const id = try identity(s);
        defer a.free(id);
        return initIdentity(s.io, directory, id, budget, spill_bytes, max_snapshots);
    }

    fn initIdentity(io: std.Io, directory: []const u8, identity_value: []const u8, budget: u64, spill_bytes: u64, max_snapshots: usize) !Store {
        const id = try a.dupe(u8, identity_value);
        errdefer a.free(id);
        const dir = try a.dupe(u8, directory);
        errdefer a.free(dir);
        const conversations = try std.fs.path.join(a, &.{ std.fs.path.dirname(directory) orelse ".", "native-session-snapshots" });
        return .{ .io = io, .identity = id, .directory = dir, .conversations = conversations, .budget = budget, .spill_bytes = spill_bytes, .max_snapshots = max_snapshots };
    }
    pub fn deinit(s: *Store) void {
        a.free(s.identity);
        a.free(s.directory);
        a.free(s.conversations);
    }

    fn ours(s: *Store, id: []const u8) bool {
        const base = std.mem.indexOfScalar(u8, s.identity, '|') orelse s.identity.len;
        return id.len > base and id[base] == '|' and std.mem.eql(u8, id[0..base], s.identity[0..base]);
    }

    fn list(s: *Store, pinned: bool, exact: bool) !std.ArrayList(Entry) {
        const directory = if (pinned) s.directory else s.conversations;
        var result: std.ArrayList(Entry) = .empty;
        errdefer freeEntries(&result);
        var dir = std.Io.Dir.cwd().openDir(s.io, directory, .{ .iterate = true }) catch |err| return if (err == error.FileNotFound) result else err;
        defer dir.close(s.io);
        var iterator = dir.iterate();
        while (try iterator.next(s.io)) |item| {
            if (!std.mem.endsWith(u8, item.name, ".safetensors") or std.mem.endsWith(u8, item.name, ".partial.safetensors") or item.kind == .sym_link) continue;
            const path = try std.fs.path.join(a, &.{ directory, item.name });
            defer a.free(path);
            var reader = (if (exact) files.Reader.open(s.io, path, s.identity) else files.Reader.inspect(s.io, path)) catch |err| {
                if (err == error.OutOfMemory) return err;
                continue;
            };
            defer reader.deinit();
            if (!s.ours(reader.metadata.value.identity) or reader.metadata.value.tokens.len == 0) continue;
            const stat = try reader.file.file.stat(s.io);
            const owned = try a.dupe(u8, path);
            errdefer a.free(owned);
            const tokens = try a.dupe(i32, reader.metadata.value.tokens);
            errdefer a.free(tokens);
            const compatible = std.mem.eql(u8, reader.metadata.value.identity, s.identity) and std.mem.eql(u8, reader.metadata.value.dependencies, @embedFile("dependencies.json")) and reader.metadata.value.tensor_backend == mx.tensor_units;
            try result.append(a, .{ .path = owned, .tokens = tokens, .bytes = stat.size, .load_bytes = reader.loadBytes(), .modified = stat.mtime.toNanoseconds(), .pinned = pinned, .compatible = compatible });
        }
        std.mem.sort(Entry, result.items, {}, struct {
            fn less(_: void, x: Entry, y: Entry) bool {
                return if (x.modified == y.modified) std.mem.lessThan(u8, x.path, y.path) else x.modified > y.modified;
            }
        }.less);
        return result;
    }

    fn prune(s: *Store, pinned: bool, keep: usize, limit: u64) !void {
        var entries = try s.list(pinned, false);
        defer freeEntries(&entries);
        var total: u64 = 0;
        for (entries.items, 0..) |entry, i| {
            if (i >= keep or entry.bytes > limit -| total) {
                try std.Io.Dir.cwd().deleteFile(s.io, entry.path);
            } else total += entry.bytes;
        }
    }

    pub fn save(s: *Store, entry: *const Cache.Entry, pinned: bool, keep: usize, limit: u64) !void {
        if (entry.nbytes > limit) return;
        var hash = Hash.init(.{});
        hash.update(s.identity);
        hash.update(std.mem.sliceAsBytes(entry.tokens));
        var digest: [32]u8 = undefined;
        hash.final(&digest);
        const path = try std.fmt.allocPrint(a, "{s}/{s}.safetensors", .{ if (pinned) s.directory else s.conversations, std.fmt.bytesToHex(digest, .lower) });
        defer a.free(path);
        if (files.Reader.open(s.io, path, s.identity)) |loaded| {
            var reader = loaded;
            defer reader.deinit();
            try reader.file.file.setTimestampsNow(s.io);
        } else |err| {
            if (err == error.OutOfMemory) return err;
            if ((try runtime.activeBytes()) +| entry.nbytes > s.budget) return;
            try entry.cache.save(s.io, path, s.identity, entry.tokens);
            s.writes +|= 1;
        }
        try s.prune(pinned, keep, limit);
    }

    pub fn evicted(raw: ?*anyopaque, entry: *const Cache.Entry) void {
        const s: *Store = @ptrCast(@alignCast(raw.?));
        if (entry.pinned or s.spill_bytes == 0) return;
        s.save(entry, false, std.math.maxInt(usize), s.spill_bytes) catch |err| report("spill", err);
    }

    pub fn persist(s: *Store, entry: *const Cache.Entry) void {
        s.save(entry, true, 8, std.math.maxInt(u64)) catch |err| report("save", err);
    }

    fn load(s: *Store, entry: Entry, cache: *Cache, tag: std.meta.Tag(session.Backend), extra: u64, keep: ?[]const i32) !bool {
        if (cache.budget_bytes) |budget| if (entry.bytes > budget) return false;
        while ((try runtime.activeBytes()) +| entry.load_bytes +| extra > s.budget) {
            if (((try runtime.activeBytes()) -| cache.nbytes()) +| entry.load_bytes +| extra > s.budget) return false;
            if (!cache.evictOne(keep)) return false;
            try mx.check(mx.c.mlx_clear_cache());
        }
        var reader = try files.Reader.open(s.io, entry.path, s.identity);
        defer reader.deinit();
        // Revalidate after selection and admission: another process may atomically replace the file.
        if (!std.mem.eql(i32, entry.tokens, reader.metadata.value.tokens) or (try runtime.activeBytes()) +| reader.loadBytes() +| extra > s.budget) return false;
        const restored = try session.Snapshot.loadReader(&reader, entry.tokens, tag);
        try cache.insertOwned(entry.tokens, restored, entry.tokens, entry.pinned);
        reader.file.file.setTimestampsNow(s.io) catch {};
        s.reads +|= 1;
        return true;
    }

    pub fn startup(s: *Store, cache: *Cache, tag: std.meta.Tag(session.Backend), extra: u64) !void {
        var entries = try s.list(true, true);
        defer freeEntries(&entries);
        var index = @min(entries.items.len, s.max_snapshots);
        while (index > 0) {
            index -= 1;
            _ = s.load(entries.items[index], cache, tag, extra, null) catch |err| {
                report("load", err);
                continue;
            };
        }
    }

    pub fn blocksToWarm(s: *Store) ![][]i32 {
        var entries = try s.list(true, false);
        defer freeEntries(&entries);
        var wanted: std.ArrayList([]i32) = .empty;
        errdefer {
            for (wanted.items) |tokens| a.free(tokens);
            wanted.deinit(a);
        }
        outer: for (entries.items) |entry| {
            if (entry.compatible) continue;
            for (entries.items) |have| if (have.compatible and std.mem.eql(i32, have.tokens, entry.tokens)) continue :outer;
            for (wanted.items) |tokens| if (std.mem.startsWith(i32, tokens, entry.tokens)) continue :outer;
            var i: usize = 0;
            while (i < wanted.items.len) {
                if (std.mem.startsWith(i32, entry.tokens, wanted.items[i])) a.free(wanted.orderedRemove(i)) else i += 1;
            }
            const tokens = try a.dupe(i32, entry.tokens);
            errdefer a.free(tokens);
            try wanted.append(a, tokens);
        }
        return wanted.toOwnedSlice(a);
    }

    pub fn readBest(s: *Store, cache: *Cache, tag: std.meta.Tag(session.Backend), prompt: []const i32, boundary: policy.Boundary, extra: u64) !void {
        var best: ?Entry = null;
        defer if (best) |*entry| entry.deinit();
        const retained = cache.longest(prompt, boundary);
        var length = retained;
        for ([_]bool{ true, false }) |pinned| {
            var entries = try s.list(pinned, true);
            defer freeEntries(&entries);
            for (entries.items) |entry| {
                const n = entry.tokens.len;
                if (n <= length or n >= prompt.len or !boundary.allows(n) or !std.mem.eql(i32, entry.tokens, prompt[0..n])) continue;
                const path = try a.dupe(u8, entry.path);
                errdefer a.free(path);
                const tokens = try a.dupe(i32, entry.tokens);
                if (best) |*old| old.deinit();
                best = entry;
                best.?.path = path;
                best.?.tokens = tokens;
                length = n;
            }
        }
        if (best) |entry| _ = try s.load(entry, cache, tag, extra, if (retained > 0) prompt[0..retained] else null);
    }

    pub fn shutdown(s: *Store, cache: *Cache) void {
        const limit = if (s.spill_bytes > 0) s.spill_bytes else 10 * memory.gib;
        const keep = if (s.spill_bytes > 0) std.math.maxInt(usize) else @as(usize, 2);
        var saved: usize = 0;
        var total: u64 = 0;
        // Longest conversations win, with current LRU order breaking ties.
        var previous: usize = std.math.maxInt(usize);
        var previous_index: usize = std.math.maxInt(usize);
        while (saved < keep) {
            var next: ?usize = null;
            for (cache.entries.items, 0..) |entry, i| {
                if (entry.pinned or entry.tokens.len > previous or (entry.tokens.len == previous and i <= previous_index)) continue;
                if (next == null or entry.tokens.len > cache.entries.items[next.?].tokens.len) next = i;
            }
            const entry = if (next) |i| &cache.entries.items[i] else break;
            if (entry.nbytes > limit -| total) break;
            s.save(entry, false, keep, limit) catch |err| {
                report("shutdown save", err);
                break;
            };
            total += entry.nbytes;
            previous = entry.tokens.len;
            previous_index = next.?;
            saved += 1;
        }
    }
};

fn freeEntries(entries: *std.ArrayList(Entry)) void {
    for (entries.items) |*entry| entry.deinit();
    entries.deinit(a);
}

pub fn checkWarming(io: std.Io, path: []const u8) !void {
    const backend = mx.tensor_units;
    mx.tensor_units = true;
    defer mx.tensor_units = backend;
    const source = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(8 * 1024 * 1024));
    defer a.free(source);
    const cases = try std.json.parseFromSlice([]const struct { directory: []const u8, identity: []const u8, expected: []const []const i32 }, a, source, .{});
    defer cases.deinit();
    for (cases.value) |case| {
        var disk = try Store.initIdentity(io, case.directory, case.identity, 0, 0, 0);
        defer disk.deinit();
        const blocks = try disk.blocksToWarm();
        defer {
            for (blocks) |tokens| a.free(tokens);
            a.free(blocks);
        }
        try std.testing.expectEqual(case.expected.len, blocks.len);
        for (case.expected, blocks) |expected, actual| try std.testing.expectEqualSlices(i32, expected, actual);
    }
    std.debug.print("PASS: {d} upstream snapshot warming selections: model isolation, newest/longest prefixes, covered blocks and invalid/partial files\n", .{cases.value.len});
}
pub fn report(action: []const u8, err: anyerror) void {
    @import("server_live.zig").print("Native snapshot {s} failed: {s}\n", .{ action, @errorName(err) });
}

pub fn check(io: std.Io) !void {
    const root = "build/native-checks/snapshot-store";
    std.Io.Dir.cwd().deleteTree(io, root) catch |err| if (err != error.FileNotFound) return err;
    var disk = try Store.initIdentity(io, root ++ "/system", "model-a|revision-1", std.math.maxInt(u64), memory.gib, 3);
    defer disk.deinit();
    var other = try Store.initIdentity(io, root ++ "/system", "model-b|revision-1", std.math.maxInt(u64), memory.gib, 3);
    defer other.deinit();
    var revision = try Store.initIdentity(io, root ++ "/system", "model-a|revision-2", std.math.maxInt(u64), memory.gib, 3);
    defer revision.deinit();
    var cache = try Cache.init(mx.allocator, 1, memory.gib);
    defer cache.deinit();
    cache.on_evict = Store.evicted;
    cache.eviction_context = &disk;
    var scope = mx.Scope{};
    defer scope.deinit();
    const values = try scope.ints(&.{ 31, 67 });
    const layers = try mx.allocator.alloc(@import("model.zig").Cache, 1);
    @memset(layers, .{});
    var snapshot = session.Snapshot{ .qwen = .{ .cache = layers, .position = 3 } };
    defer snapshot.deinit();
    snapshot.qwen.cache[0].a = try mx.retain(values);
    try cache.insertOwned(&.{ 1, 2, 3 }, try snapshot.clone(), &.{ 1, 2, 3, 4 }, true);
    disk.persist(&cache.entries.items[0]);
    other.persist(&cache.entries.items[0]);
    revision.persist(&cache.entries.items[0]);
    try std.testing.expectEqual(@as(u64, 1), disk.writes);
    try std.testing.expect(cache.evictOne(null));
    var shorter = try snapshot.clone();
    shorter.qwen.position = 2;
    try cache.insertOwned(&.{ 1, 2 }, shorter, &.{ 1, 2, 3, 4 }, true);
    disk.budget = try runtime.activeBytes();
    try disk.readBest(&cache, .qwen, &.{ 1, 2, 3, 4 }, .{}, 0);
    try std.testing.expectEqual(@as(usize, 1), cache.entries.items.len);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2 }, cache.entries.items[0].tokens);
    try std.testing.expect(cache.evictOne(null));
    disk.budget = 0;
    try disk.startup(&cache, .qwen, 0);
    try std.testing.expectEqual(@as(usize, 0), cache.entries.items.len);
    disk.budget = std.math.maxInt(u64);
    try disk.startup(&cache, .qwen, 0);
    try std.testing.expectEqual(@as(usize, 1), cache.entries.items.len);
    try std.testing.expect(cache.entries.items[0].pinned);
    try std.testing.expect(cache.evictOne(null));
    try disk.readBest(&cache, .qwen, &.{ 1, 2, 3 }, .{}, 0);
    try std.testing.expectEqual(@as(usize, 0), cache.entries.items.len);
    try disk.readBest(&cache, .qwen, &.{ 1, 2, 3, 4 }, .{ .step = 2 }, 0);
    try std.testing.expectEqual(@as(usize, 0), cache.entries.items.len);
    try disk.readBest(&cache, .qwen, &.{ 1, 2, 3, 4 }, .{}, 0);
    try std.testing.expectEqual(@as(usize, 1), cache.entries.items.len);
    try std.testing.expectEqual(@as(u64, 2), disk.reads);
    try cache.insertOwned(&.{ 5, 6, 7 }, try snapshot.clone(), &.{ 5, 6, 7, 8 }, false);
    try cache.insertOwned(&.{ 9, 10, 11 }, try snapshot.clone(), &.{ 9, 10, 11, 12 }, false);
    try std.testing.expectEqual(@as(u64, 2), disk.writes);
    try disk.readBest(&cache, .qwen, &.{ 5, 6, 7, 8 }, .{}, 0);
    try std.testing.expectEqualSlices(i32, &.{ 5, 6, 7 }, cache.entries.items[0].tokens);
    try std.testing.expect(!cache.entries.items[0].pinned);
    try @import("sampling_checks.zig").equal(&scope, values, cache.entries.items[0].cache.qwen.cache[0].a);
    disk.shutdown(&cache);
    try disk.prune(true, 0, 0);
    var ours = try disk.list(true, true);
    defer freeEntries(&ours);
    var theirs = try other.list(true, true);
    defer freeEntries(&theirs);
    var older = try revision.list(true, true);
    defer freeEntries(&older);
    try std.testing.expectEqual(@as(usize, 0), ours.items.len);
    try std.testing.expectEqual(@as(usize, 0), older.items.len);
    try std.testing.expectEqual(@as(usize, 1), theirs.items.len);
    try disk.prune(false, std.math.maxInt(usize), 0);
    var sessions = try disk.list(false, true);
    defer freeEntries(&sessions);
    try std.testing.expectEqual(@as(usize, 0), sessions.items.len);
    std.debug.print("PASS: disk snapshots isolate models/revisions, obey memory and chunk boundaries, spill on eviction, reload exact tensors and prune only their model\n", .{});
}
