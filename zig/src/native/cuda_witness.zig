//! The execution witness's first CUDA path: Qwen3.5-2B, one rank, MLX affine 4-bit, text only, every weight byte.
const std = @import("std");
const api = @import("engine_api");
const cuda = @import("cuda");
const core = @import("core");
const cuda_disk = @import("cuda_disk.zig");
const disk_identity = @import("disk_identity.zig");
const Key = core.digest_cache.Key;
const Content = core.checkpoint_identity.Content;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Allocator = std.mem.Allocator;
const Facts = api.execution_witness.Facts;

/// Whether `o` may take the first path before the weights load: the flag, dense Qwen, one rank, text, nothing learned.
pub fn wanted(comptime F: type, o: api.Open) bool {
    if (comptime F != @import("qwen35").native) return false;
    return o.execution_witness and o.tp == 1 and o.vision == null and o.drafter == null and !o.slide and
        o.record == null and o.learn == null and !o.recall and (o.mtp_drafts orelse 0) == 0;
}

/// Shard workers beside live serving: one, so a hash takes one core and one stream of reads from the decode.
pub const workers = 1;

/// One file the load reads, stamped before it reads it (null: absent then).
const Stamp = struct { name: []const u8, key: ?Key };

/// One engine's attestation: stamps from before its load, facts from after it, the hash once the server is ready.
pub const Witness = struct {
    gpa: Allocator,
    io: std.Io,
    arena: std.heap.ArenaAllocator,
    dir: []const u8,
    stamps: []const Stamp,
    path: ?[]const u8 = null,
    kernels: ?[]const u8 = null,
    device: cuda_disk.Device = undefined,
    facts: []const u64 = &.{},
    profile: disk_identity.Profile = .{ .family = "qwen3_5", .backend = "cuda" },
    /// The SM count the kernels' tiles are weighed against, the window, the streams and the staggered segments.
    words: [4]u64 = @splat(0),
    /// Set once the host owns it: `release` then leaves it to the host's close.
    adopted: bool = false,
    mutex: std.Io.Mutex = .init,
    state: Facts = .pending,
    thread: ?std.Thread = null,
    begun: bool = false,
    stopping: bool = false,
    content: ?*Content = null,

    /// Stamps config.json, the index and every shard the index names, before the load reads any of them.
    pub fn open(gpa: Allocator, io: std.Io, dir: []const u8) !*Witness {
        const w = try gpa.create(Witness);
        errdefer gpa.destroy(w);
        w.* = .{ .gpa = gpa, .io = io, .arena = .init(gpa), .dir = undefined, .stamps = undefined };
        errdefer w.arena.deinit();
        const a = w.arena.allocator();
        w.dir = try a.dupe(u8, dir);
        const shards = try core.checkpoint_identity.shardNames(a, io, dir);
        const stamps = try a.alloc(Stamp, shards.len + 2);
        stamps[0].name = "config.json";
        stamps[1].name = "model.safetensors.index.json";
        for (shards, stamps[2..]) |name, *s| s.name = name;
        for (stamps) |*s| s.key = try keyOf(a, io, dir, s.name);
        w.stamps = stamps;
        return w;
    }

    /// After the load: the path the loaded geometry takes, this GPU's facts and the family's; false: no path.
    pub fn bind(w: *Witness, comptime K: type, target: anytype, ctx: *const cuda.Context, kernels: ?[]const u8, window: u64, streams: u64, segments: u64) bool {
        const path = K.witnessPath(target) orelse return false;
        const a = w.arena.allocator();
        const name_buf = a.alloc(u8, 256) catch return false;
        const device = cuda_disk.deviceOf(ctx, name_buf) catch return false;
        const sms = ctx.attribute(.multiprocessor_count) catch return false;
        const facts = K.facts(target);
        w.set(path, if (kernels) |k| a.dupe(u8, k) catch return false else null, device, a.dupe(u64, &facts) catch return false, .{ @intCast(sms), window, streams, segments });
        return true;
    }

    /// The facts `bind` reads from the device and family, given directly.
    pub fn set(w: *Witness, path: []const u8, kernels: ?[]const u8, device: cuda_disk.Device, facts: []const u64, words: [4]u64) void {
        w.path = path;
        w.kernels = kernels;
        w.device = device;
        w.facts = facts;
        w.words = words;
    }

    pub fn source(w: *Witness) api.execution_witness.Source {
        return .{ .ctx = w, .read = read, .begin = begin };
    }

    fn read(ctx: *anyopaque) Facts {
        const w: *Witness = @ptrCast(@alignCast(ctx));
        w.mutex.lockUncancelable(w.io);
        defer w.mutex.unlock(w.io);
        return w.state;
    }

    /// The server is ready (or this engine was reloaded into it): hash on a thread of its own, once.
    fn begin(ctx: *anyopaque) void {
        const w: *Witness = @ptrCast(@alignCast(ctx));
        w.mutex.lockUncancelable(w.io);
        defer w.mutex.unlock(w.io);
        if (w.begun or w.stopping) return;
        w.begun = true;
        w.thread = std.Thread.spawn(.{}, work, .{w}) catch {
            w.state = .{ .unknown = .hash_failed };
            return;
        };
    }

    fn work(w: *Witness) void {
        const got: Facts = if (w.digest()) |d| .{ .known = .{ .path = w.path.?, .digest = d } } else |e| .{ .unknown = reasonOf(e) };
        w.mutex.lockUncancelable(w.io);
        defer w.mutex.unlock(w.io);
        w.state = got;
    }

    /// The engine's digest: the path, the checkpoint's structure and every shard byte, the kept identity's components.
    pub fn digest(w: *Witness) ![32]u8 {
        var arena: std.heap.ArenaAllocator = .init(w.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        try w.unchanged(a);
        var h = Sha256.init(.{});
        field(&h, "tensorfold cuda engine witness v1");
        field(&h, w.path.?);
        field(&h, &try core.checkpoint_identity.identity(a, w.io, w.dir));
        field(&h, &try w.weights());
        try w.unchanged(a);
        var kept = Sha256.init(.{});
        try cuda_disk.components(&kept, a, w.io, .{ .binary = null, .checkpoint = null, .kernels = w.kernels, .device = w.device, .facts = w.facts, .probed = null, .profile = w.profile });
        field(&h, &kept.finalResult());
        field(&h, std.mem.sliceAsBytes(&w.words));
        return h.finalResult();
    }

    /// Every shard byte, read in full (no digest cache), on `workers` threads; a close stops it at its next block.
    fn weights(w: *Witness) ![32]u8 {
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
    fn unchanged(w: *const Witness, a: Allocator) !void {
        for (w.stamps) |s| {
            const now = try keyOf(a, w.io, w.dir, s.name);
            if ((now == null) != (s.key == null)) return error.FilesChanged;
            if (now) |k| if (!k.equal(s.key.?)) return error.FilesChanged;
        }
    }

    /// Waits for a begun hash to publish its result.
    pub fn wait(w: *Witness) void {
        w.mutex.lockUncancelable(w.io);
        const thread = w.thread;
        w.thread = null;
        w.mutex.unlock(w.io);
        if (thread) |t| t.join();
    }

    /// Stops a hash in progress at its next block, waits for it and frees everything.
    pub fn close(w: *Witness) void {
        w.mutex.lockUncancelable(w.io);
        w.stopping = true;
        if (w.content) |c| c.stop.store(true, .release);
        w.mutex.unlock(w.io);
        w.wait();
        w.arena.deinit();
        w.gpa.destroy(w);
    }
};

/// The witness `F`'s load took onto its path, or null (any other family, geometry or format: unsupported_path).
pub fn bound(comptime F: type, w: ?*Witness, loaded: anytype, ctx: *const cuda.Context, kernels: ?[]const u8, words: [3]u64) ?*Witness {
    if (comptime F != @import("qwen35").native) return null;
    const x = w orelse return null;
    return if (x.bind(F.kept_disk, loaded.target, ctx, kernels, words[0], words[1], words[2])) x else null;
}

/// A witness the host did not adopt (a refusal or failure after `open`), freed.
pub fn release(w: ?*Witness) void {
    if (w) |x| if (!x.adopted) x.close();
}

/// `Info.witness` for an adopted witness on its path; null otherwise (unsupported_path).
pub fn source(w: ?*Witness) ?api.execution_witness.Source {
    const x = w orelse return null;
    return if (x.path != null) x.source() else null;
}

fn keyOf(a: Allocator, io: std.Io, dir: []const u8, name: []const u8) !?Key {
    const path = try std.fs.path.join(a, &.{ dir, name });
    defer a.free(path);
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

fn field(h: *Sha256, bytes: []const u8) void {
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, bytes.len, .little);
    h.update(&len);
    h.update(bytes);
}

test {
    _ = @import("cuda_witness_test.zig");
}
