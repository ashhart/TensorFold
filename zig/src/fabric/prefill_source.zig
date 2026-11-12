//! Where a stream's prompt state comes from: its own prefill (native), or another machine's over the fabric (remote).
const std = @import("std");
const manifest = @import("manifest.zig");
const mailbox = @import("mailbox.zig");
const decoder = @import("decoder.zig");
const words = @import("words.zig");

/// macOS and glibc both provide it; Zig's std.c declares it on some systems only.
extern "c" fn getentropy(buf: [*]u8, len: usize) c_int;

/// A remote prefix never counts as the engine's own exact state.
pub const Provenance = enum { native, remote };

/// The engine may keep a prompt prefix in its exact cache only when it computed it itself.
pub fn cacheable(p: Provenance) bool {
    return p == .native;
}

/// What a source fills: prompt tokens [0, upto); the backend prefills [upto, prompt_len) and draws the first token.
pub const Offer = struct { upto: u64, provenance: Provenance };

/// Opt-in and experimental: off unless the server was started with the fabric, and outside the exactness contract.
pub const Policy = struct {
    enabled: bool = false,
    min_prefix: u64 = 1024,
    chunk: u64 = 512,
    /// Prompt tokens left to the backend (not 0): the prefix is `prompt_len - tail` and `chunk` is not used.
    tail: u64 = 0,

    /// The remote prefix: all but `tail` tokens, else whole chunks and one token left local; nothing below `cached`.
    pub fn upto(p: Policy, prompt_len: u64, cached: u64) ?u64 {
        if (!p.enabled or prompt_len < p.min_prefix) return null;
        if (p.tail != 0) return if (prompt_len > p.tail and prompt_len - p.tail > cached) prompt_len - p.tail else null;
        if (p.chunk == 0) return null;
        const n = (prompt_len - 1) / p.chunk * p.chunk;
        return if (n > cached) n else null;
    }
};

/// Where filled state lands: the stream's caches in the backend; nothing is visible to rounds before `commit`.
pub const StateSink = struct {
    ptr: *anyopaque,
    put_fn: *const fn (ptr: *anyopaque, layer: *const manifest.Layer, row_start: u64, rows: u64, bytes: []const u8) anyerror!void,
    commit_fn: *const fn (ptr: *anyopaque, upto: u64, provenance: Provenance) void,
    abort_fn: *const fn (ptr: *anyopaque) void,
    /// Shard `k`'s sink when the state arrives from several producers at once (each `put_fn` of a shard may run on its
    shard_fn: ?*const fn (ptr: *anyopaque, shard: u32) decoder.Sink = null,
};

/// A provider the lane engine asks before Backend.prefill; the engine owns the stream and the caches throughout.
pub const PrefillSource = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// How much of this prompt the source fills now; null leaves the whole prompt to the backend.
        offer: *const fn (ptr: *anyopaque, prompt: []const u32, cached: u64) ?Offer,
        /// Fill prompt[0..offer.upto] through `sink`, then commit; on error the sink is aborted and nothing is kept.
        fill: *const fn (ptr: *anyopaque, prompt: []const u32, offer: Offer, sink: StateSink) anyerror!void,
    };

    pub fn offer(s: PrefillSource, prompt: []const u32, cached: u64) ?Offer {
        return s.vtable.offer(s.ptr, prompt, cached);
    }

    pub fn fill(s: PrefillSource, prompt: []const u32, o: Offer, sink: StateSink) !void {
        s.vtable.fill(s.ptr, prompt, o, sink) catch |err| {
            sink.abort_fn(sink.ptr);
            return err;
        };
    }
};

/// Asks a producer to prefill `tokens` under a handoff id: its prefix and one more token (all but it is exported).
pub const Trigger = struct {
    ptr: *anyopaque,
    start_fn: *const fn (ptr: *anyopaque, handoff: [16]u8, tokens: []const u32) anyerror!void,
};

/// The fabric's source: one handoff at a time per link, pulled with the decoder into the stream's caches.
pub const Remote = struct {
    gpa: std.mem.Allocator,
    policy: Policy,
    client: *mailbox.Client,
    /// The other producers of a KV_SPLIT export.
    peers: []const *mailbox.Client = &.{},
    trigger: Trigger,
    model: []const u8,
    protocol: u16 = 1, // 2 to take a producer's recurrent state (MKVH protocol 2); 1 refuses such an export
    busy: bool = false,
    next_id: u64 = 1,
    reason: decoder.Reason = .{},
    /// the pull's request is gone (set before each fill, which one gate serialises); the decoder stops between calls
    abort: ?*const std.atomic.Value(bool) = null,
    /// the last fill's trigger exchange (the producer's prefill) in ns, and the payload bytes it pulled
    trigger_ns: u64 = 0,
    bytes: u64 = 0,

    pub fn source(r: *Remote) PrefillSource {
        return .{ .ptr = r, .vtable = &.{ .offer = offerFn, .fill = fillFn } };
    }

    fn offerFn(ptr: *anyopaque, prompt: []const u32, cached: u64) ?Offer {
        const r: *Remote = @ptrCast(@alignCast(ptr));
        if (r.busy or r.client.poisoned) return null;
        for (r.peers) |c| if (c.poisoned) return null;
        const n = r.policy.upto(prompt.len, cached) orelse return null;
        return .{ .upto = n, .provenance = .remote };
    }

    fn fillFn(ptr: *anyopaque, prompt: []const u32, o: Offer, sink: StateSink) anyerror!void {
        const r: *Remote = @ptrCast(@alignCast(ptr));
        r.busy = true;
        defer r.busy = false;
        var id: [16]u8 = undefined;
        if (getentropy(&id, 8) != 0) return error.NoEntropy;
        std.mem.writeInt(u64, id[8..16], r.next_id, .little);
        r.next_id += 1;
        if (o.upto >= prompt.len) return error.PrefixTooLong; // a token past the prefix goes too
        const tokens = prompt[0..@intCast(o.upto)];
        const began = words.nowNs();
        try r.trigger.start_fn(r.trigger.ptr, id, prompt[0..@intCast(o.upto + 1)]);
        r.trigger_ns = words.nowNs() - began;
        if (r.peers.len > 0) {
            try r.pullShards(id, tokens, sink);
        } else {
            var arena: std.heap.ArenaAllocator = .init(r.gpa);
            defer arena.deinit();
            const pass: decoder.Sink = .{ .ptr = sink.ptr, .put_fn = sink.put_fn };
            const got = try decoder.pull(arena.allocator(), r.client, id, .{ .model = r.model, .tokens = tokens, .protocol = r.protocol, .abort = r.abort }, pass, &r.reason);
            if (got.manifest.first_token != 0) return error.PartialPrefix;
            r.bytes = got.bytes;
        }
        sink.commit_fn(sink.ptr, o.upto, .remote);
    }

    /// One shard's pull on its own link; the thread's result and the producer's reason stay in the job.
    const Job = struct {
        client: *mailbox.Client,
        id: [16]u8,
        expect: decoder.Expect,
        sink: decoder.Sink,
        arena: std.heap.ArenaAllocator,
        reason: decoder.Reason = .{},
        result: anyerror!decoder.Pulled = error.NotRun,

        fn run(j: *Job) void {
            j.result = decoder.pull(j.arena.allocator(), j.client, j.id, j.expect, j.sink, &j.reason);
        }
    };

    /// Every shard's export pulled at once, each over its own link into its own sink; the first failure is returned.
    fn pullShards(r: *Remote, id: [16]u8, tokens: []const u32, sink: StateSink) !void {
        const shard_fn = sink.shard_fn orelse return error.SinkNotSharded;
        const n: u32 = @intCast(1 + r.peers.len);
        const jobs = try r.gpa.alloc(Job, n);
        defer r.gpa.free(jobs);
        var made: usize = 0;
        defer for (jobs[0..made]) |*j| j.arena.deinit();
        for (jobs, 0..) |*j, k| {
            const expect: decoder.Expect = .{ .model = r.model, .tokens = tokens, .protocol = r.protocol, .tp_rank = k, .tp_size = n, .abort = r.abort };
            j.* = .{ .client = if (k == 0) r.client else r.peers[k - 1], .id = id, .expect = expect, .sink = shard_fn(sink.ptr, @intCast(k)), .arena = .init(r.gpa) };
            made += 1;
        }
        const threads = try r.gpa.alloc(?std.Thread, n);
        defer r.gpa.free(threads);
        @memset(threads, null);
        for (jobs[1..], threads[1..]) |*j, *t| t.* = std.Thread.spawn(.{}, Job.run, .{j}) catch null;
        jobs[0].run();
        for (jobs[1..], threads[1..]) |*j, t| if (t) |th| th.join() else j.run();
        r.bytes = 0;
        for (jobs) |*j| {
            const got = j.result catch |err| {
                r.reason = j.reason;
                return err;
            };
            if (got.manifest.first_token != 0) return error.PartialPrefix;
            r.bytes += got.bytes;
        }
    }
};

test "only native prefixes are cacheable, and a remote prefix leaves a local tail on a chunk boundary" {
    try std.testing.expect(cacheable(.native) and !cacheable(.remote));
    const p: Policy = .{ .enabled = true, .min_prefix = 1024, .chunk = 512 };
    try std.testing.expectEqual(@as(?u64, 1536), p.upto(2000, 0));
    try std.testing.expectEqual(@as(?u64, 1536), p.upto(2048, 0));
    try std.testing.expectEqual(@as(?u64, 2048), p.upto(2049, 0));
    try std.testing.expectEqual(@as(?u64, null), p.upto(1000, 0));
    try std.testing.expectEqual(@as(?u64, null), p.upto(2000, 1536));
    try std.testing.expectEqual(@as(?u64, null), (Policy{}).upto(4096, 0));
}

test "a tail policy leaves exactly its tokens local, whatever the chunk, and nothing below what is cached" {
    const p: Policy = .{ .enabled = true, .min_prefix = 1024, .chunk = 512, .tail = 1 };
    try std.testing.expectEqual(@as(?u64, 1999), p.upto(2000, 0));
    try std.testing.expectEqual(@as(?u64, 2047), p.upto(2048, 0));
    try std.testing.expectEqual(@as(?u64, 8191), p.upto(8192, 0));
    try std.testing.expectEqual(@as(?u64, null), p.upto(1000, 0)); // under min_prefix
    try std.testing.expectEqual(@as(?u64, null), p.upto(2000, 1999)); // the cache already holds all but the tail
    try std.testing.expectEqual(@as(?u64, 1999), p.upto(2000, 1998));
    var t64 = p;
    t64.tail = 64;
    try std.testing.expectEqual(@as(?u64, 1936), t64.upto(2000, 0));
    t64.min_prefix = 0;
    try std.testing.expectEqual(@as(?u64, null), t64.upto(64, 0)); // nothing would be pulled
    try std.testing.expectEqual(@as(?u64, 1), t64.upto(65, 0));
    var off = p;
    off.enabled = false;
    try std.testing.expectEqual(@as(?u64, null), off.upto(4096, 0));
}
