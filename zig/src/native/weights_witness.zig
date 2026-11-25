//! An engine witness's backend-neutral half: stamps before the load, every weight byte once ready, one hash thread.
const std = @import("std");
const api = @import("engine_api");
const core = @import("core");
const Key = core.digest_cache.Key;
const Content = core.checkpoint_identity.Content;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Allocator = std.mem.Allocator;
const Facts = api.execution_witness.Facts;

/// Shard workers beside live serving: one, so a hash takes one core and one stream of reads from the decode.
pub const workers = 1;

/// Whether `o` may take a witness path before the weights load: the flag, one rank, text, nothing learned or drafted.
pub fn wanted(o: api.Open) bool {
    return o.execution_witness and o.tp == 1 and o.vision == null and o.drafter == null and !o.slide and
        o.record == null and o.learn == null and !o.recall and (o.mtp_drafts orelse 0) == 0;
}

/// One file the load reads, stamped before it reads it (null: absent then).
const Stamp = struct { path: []const u8, key: ?Key };

/// One engine's attestation over backend `B`: its `domain`, the files it `stamp`s, what it `keep`s and its `tail`.
pub fn Witness(comptime B: type) type {
    return struct {
        const Self = @This();

        gpa: Allocator,
        io: std.Io,
        arena: std.heap.ArenaAllocator,
        dir: []const u8,
        stamps: []const Stamp,
        path: ?[]const u8 = null,
        backend: B = .{},
        /// Set once the host owns it: `release` then leaves it to the host's close.
        adopted: bool = false,
        mutex: std.Io.Mutex = .init,
        state: Facts = .pending,
        thread: ?std.Thread = null,
        begun: bool = false,
        stopping: bool = false,
        content: ?*Content = null,

        /// Stamps config.json, the index, every shard it names and the backend's own files, before the load.
        pub fn open(gpa: Allocator, io: std.Io, dir: []const u8, backend: B) !*Self {
            const w = try gpa.create(Self);
            errdefer gpa.destroy(w);
            w.* = .{ .gpa = gpa, .io = io, .arena = .init(gpa), .dir = undefined, .stamps = undefined, .backend = backend };
            errdefer w.arena.deinit();
            const a = w.arena.allocator();
            w.dir = try a.dupe(u8, dir);
            var paths: std.ArrayList([]const u8) = .empty;
            for ([_][]const u8{ "config.json", "model.safetensors.index.json" }) |name| try paths.append(a, try std.fs.path.join(a, &.{ dir, name }));
            for (try core.checkpoint_identity.shardNames(a, io, dir)) |name| try paths.append(a, try std.fs.path.join(a, &.{ dir, name }));
            try w.backend.stamp(a, io, &paths);
            const stamps = try a.alloc(Stamp, paths.items.len);
            for (paths.items, stamps) |path, *s| s.* = .{ .path = path, .key = try keyOf(io, path) };
            w.stamps = stamps;
            return w;
        }

        /// `open`, or when a file cannot be stamped a witness that reads hash_failed; only OutOfMemory fails the load.
        pub fn openOrFailed(gpa: Allocator, io: std.Io, dir: []const u8, backend: B) error{OutOfMemory}!*Self {
            return open(gpa, io, dir, backend) catch |e| {
                if (e == error.OutOfMemory) return error.OutOfMemory;
                std.log.warn("execution witness: the checkpoint's files cannot be stamped ({s}); the witness reads hash_failed", .{@errorName(e)});
                const w = try gpa.create(Self);
                w.* = .{ .gpa = gpa, .io = io, .arena = .init(gpa), .dir = "", .stamps = &.{}, .state = .{ .unknown = .hash_failed }, .begun = true };
                return w;
            };
        }

        pub fn source(w: *Self) api.execution_witness.Source {
            return .{ .ctx = w, .read = read, .begin = begin };
        }

        fn read(ctx: *anyopaque) Facts {
            const w: *Self = @ptrCast(@alignCast(ctx));
            w.mutex.lockUncancelable(w.io);
            defer w.mutex.unlock(w.io);
            return w.state;
        }

        /// The server is ready (or this engine was reloaded into it): hash on a thread of its own, once.
        fn begin(ctx: *anyopaque) void {
            const w: *Self = @ptrCast(@alignCast(ctx));
            w.mutex.lockUncancelable(w.io);
            defer w.mutex.unlock(w.io);
            if (w.begun or w.stopping) return;
            w.begun = true;
            w.thread = std.Thread.spawn(.{}, work, .{w}) catch {
                w.state = .{ .unknown = .hash_failed };
                return;
            };
        }

        fn work(w: *Self) void {
            const got: Facts = if (w.digest()) |d| .{ .known = .{ .path = w.path.?, .digest = d } } else |e| .{ .unknown = reasonOf(e) };
            w.mutex.lockUncancelable(w.io);
            defer w.mutex.unlock(w.io);
            w.state = got;
        }

        /// The engine's digest: the path, the checkpoint's structure and every shard byte, what the backend keeps.
        pub fn digest(w: *Self) ![32]u8 {
            var arena: std.heap.ArenaAllocator = .init(w.gpa);
            defer arena.deinit();
            const a = arena.allocator();
            try w.unchanged();
            var h = Sha256.init(.{});
            field(&h, B.domain);
            field(&h, w.path.?);
            field(&h, &try core.checkpoint_identity.identity(a, w.io, w.dir));
            field(&h, &try w.weights());
            var kept = Sha256.init(.{});
            try w.backend.keep(&kept, a, w.io);
            try w.unchanged(); // after the shards and the backend's files are read: none moved since the load
            field(&h, &kept.finalResult());
            field(&h, w.backend.tail());
            return h.finalResult();
        }

        /// Every shard byte, read in full (no digest cache), on `workers` threads; a close stops it at its next block.
        fn weights(w: *Self) ![32]u8 {
            const c = try Content.startWithOptions(w.gpa, w.io, w.dir, false, null, .{ .storage_workers = workers });
            w.mutex.lockUncancelable(w.io);
            if (w.stopping) {
                w.mutex.unlock(w.io);
                c.abandon();
                return error.Stopped;
            }
            w.content = c;
            w.mutex.unlock(w.io);
            c.wait();
            w.mutex.lockUncancelable(w.io);
            w.content = null;
            w.mutex.unlock(w.io);
            return c.finish();
        }

        /// Each file as the load found it: present with the same stamps, or still absent (else FilesChanged).
        fn unchanged(w: *const Self) !void {
            for (w.stamps) |s| {
                const now = try keyOf(w.io, s.path);
                if ((now == null) != (s.key == null)) return error.FilesChanged;
                if (now) |k| if (!k.equal(s.key.?)) return error.FilesChanged;
            }
        }

        /// Waits for a begun hash to publish its result.
        pub fn wait(w: *Self) void {
            w.mutex.lockUncancelable(w.io);
            const thread = w.thread;
            w.thread = null;
            w.mutex.unlock(w.io);
            if (thread) |t| t.join();
        }

        /// Stops a hash in progress at its next block, waits for it and frees everything.
        pub fn close(w: *Self) void {
            w.mutex.lockUncancelable(w.io);
            w.stopping = true;
            if (w.content) |c| c.stop.store(true, .release);
            w.mutex.unlock(w.io);
            w.wait();
            w.arena.deinit();
            w.gpa.destroy(w);
        }

        /// A witness the host did not adopt (a refusal or failure after `open`), freed.
        pub fn release(w: ?*Self) void {
            if (w) |x| if (!x.adopted) x.close();
        }

        /// `Info.witness` for an adopted witness on its path; null otherwise (unsupported_path).
        pub fn sourceOf(w: ?*Self) ?api.execution_witness.Source {
            const x = w orelse return null;
            return if (x.path != null) x.source() else null;
        }
    };
}

fn keyOf(io: std.Io, path: []const u8) !?Key {
    const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch |e| return if (e == error.FileNotFound) null else e;
    defer file.close(io);
    return try Key.of(file, io);
}

fn reasonOf(e: anyerror) api.execution_witness.Reason {
    return switch (e) {
        error.FilesChanged, error.DigestFileChanged, error.ShardBytesDiffer, error.FileShrank => .files_changed,
        else => .hash_failed,
    };
}

/// `bytes` behind its little-endian u64 length, so no two field lists hash the same.
pub fn field(h: *Sha256, bytes: []const u8) void {
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, bytes.len, .little);
    h.update(&len);
    h.update(bytes);
}
