//! Where a stream's prompt state comes from: its own prefill (native), or another machine's over the fabric (remote).
const std = @import("std");
const manifest = @import("manifest.zig");
const mailbox = @import("mailbox.zig");
const decoder = @import("decoder.zig");

/// macOS and glibc both provide it; Zig's std.c declares it on some systems only.
extern "c" fn getentropy(buf: [*]u8, len: usize) c_int;

/// A remote prefix never counts as the engine's own exact state.
pub const Provenance = enum { native, remote };

/// The engine may keep a prompt prefix in its exact cache only when it computed it itself.
pub fn cacheable(p: Provenance) bool {
    return p == .native;
}

/// What a source fills: prompt tokens [0, upto); the backend prefills [upto, prompt_len) and draws the first token itself.
pub const Offer = struct { upto: u64, provenance: Provenance };

/// Opt-in and experimental: off unless the server was started with the fabric, and outside the exactness contract.
pub const Policy = struct {
    enabled: bool = false,
    min_prefix: u64 = 1024,
    chunk: u64 = 512,

    /// The remote prefix for a prompt: whole chunks only, at least one token left local, nothing below `cached`.
    pub fn upto(p: Policy, prompt_len: u64, cached: u64) ?u64 {
        if (!p.enabled or prompt_len < p.min_prefix or p.chunk == 0) return null;
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

/// Asks a producer to prefill tokens under a handoff id (an in-band request kind once MCDMA's protocol has one).
pub const Trigger = struct {
    ptr: *anyopaque,
    start_fn: *const fn (ptr: *anyopaque, handoff: [16]u8, tokens: []const u32) anyerror!void,
};

/// The fabric's source: one handoff at a time per link, pulled with the decoder into the stream's caches.
pub const Remote = struct {
    gpa: std.mem.Allocator,
    policy: Policy,
    client: *mailbox.Client,
    trigger: Trigger,
    model: []const u8,
    busy: bool = false,
    next_id: u64 = 1,
    reason: decoder.Reason = .{},

    pub fn source(r: *Remote) PrefillSource {
        return .{ .ptr = r, .vtable = &.{ .offer = offerFn, .fill = fillFn } };
    }

    fn offerFn(ptr: *anyopaque, prompt: []const u32, cached: u64) ?Offer {
        const r: *Remote = @ptrCast(@alignCast(ptr));
        if (r.busy or r.client.poisoned) return null;
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
        const tokens = prompt[0..@intCast(o.upto)];
        try r.trigger.start_fn(r.trigger.ptr, id, tokens);
        var arena: std.heap.ArenaAllocator = .init(r.gpa);
        defer arena.deinit();
        const pass: decoder.Sink = .{ .ptr = sink.ptr, .put_fn = sink.put_fn };
        const got = try decoder.pull(arena.allocator(), r.client, id, .{ .model = r.model, .tokens = tokens }, pass, &r.reason);
        if (got.manifest.first_token != 0) return error.PartialPrefix;
        sink.commit_fn(sink.ptr, o.upto, .remote);
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
