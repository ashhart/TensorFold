//! The decoder's side of a handoff: OPEN until the export is recorded, PULL every frame into a sink, then CLOSE.
const std = @import("std");
const wire = @import("wire.zig");
const manifest = @import("manifest.zig");
const mailbox = @import("mailbox.zig");
const words = @import("words.zig");
const Allocator = std.mem.Allocator;

pub const Error = error{ Refused, Protocol, Mismatch, Corrupt, Expired, Aborted } || mailbox.Error || manifest.Error;

/// What the decoder knows the export must be: the same checkpoint, prompt and tensor-parallel rank.
pub const Expect = struct {
    model: []const u8,
    tokens: []const u32,
    tp_rank: u64 = 0,
    tp_size: u64 = 1,
    checksum: bool = true,
    open_timeout_ns: u64 = 30 * std.time.ns_per_s,
    /// The newest protocol this decoder reads: 2 takes recurrent layers; 1 refuses a protocol 2 export as foreign.
    protocol: u16 = wire.version,
    /// Ask a producer that is still prefilling for its export now: frames not yet converted are awaited, any order.
    stream: bool = false,
    /// While it reads true the producer works (the trigger is out): no wait for the export or a frame times out.
    working: ?*const std.atomic.Value(bool) = null,
    /// Set by the caller when the producer failed: the pull stops between calls, after a CLOSE if it had opened.
    failed: ?*const std.atomic.Value(bool) = null,

    pub fn aborted(e: Expect) bool {
        return if (e.failed) |f| f.load(.acquire) else false;
    }

    fn busy(e: Expect) bool {
        return if (e.working) |w| w.load(.acquire) else false;
    }
};

/// Where pulled rows go: rows [row_start, row_start + rows) of `layer`, C-contiguous in the manifest's shape.
pub const Sink = struct {
    ptr: *anyopaque,
    put_fn: *const fn (ptr: *anyopaque, layer: *const manifest.Layer, row_start: u64, rows: u64, bytes: []const u8) anyerror!void,
};

/// The producer's ERROR text, kept for the caller's message.
pub const Reason = struct {
    buf: [256]u8 = undefined,
    len: usize = 0,

    pub fn text(r: *const Reason) []const u8 {
        return r.buf[0..r.len];
    }

    fn set(r: *Reason, s: []const u8) void {
        r.len = @min(s.len, r.buf.len);
        @memcpy(r.buf[0..r.len], s[0..r.len]);
    }
};

pub const Pulled = struct { manifest: manifest.Manifest, frames: u64, bytes: u64 };

/// Pull one handoff over `client`; any failure after OPEN still asks the producer to CLOSE it.
pub fn pull(arena: Allocator, client: *mailbox.Client, handoff: [16]u8, expect: Expect, sink: Sink, reason: *Reason) Error!Pulled {
    const m = try open(arena, client, handoff, expect, reason);
    const got = frames(arena, client, handoff, m, expect, sink, reason) catch |err| {
        if (!client.poisoned) close(client, handoff) catch {};
        return err;
    };
    try close(client, handoff);
    return got;
}

fn request(client: *mailbox.Client, kind: wire.Kind, handoff: [16]u8, frame: u32, payload: []const u8, most: u16) Error!wire.Header {
    var msg: [wire.header_bytes + 64]u8 = undefined;
    msg[0..wire.header_bytes].* = wire.pack(.{ .kind = kind, .handoff = handoff, .frame = frame, .nbytes = payload.len });
    @memcpy(msg[wire.header_bytes..][0..payload.len], payload);
    const reply = try client.call(msg[0 .. wire.header_bytes + payload.len]);
    const h = switch (wire.unpackUpTo(reply, most)) {
        .ok => |h| h,
        .bad => return error.Protocol,
    };
    if (!std.mem.eql(u8, &h.handoff, &handoff) and h.kind != .@"error") return error.Protocol;
    return h;
}

fn open(arena: Allocator, client: *mailbox.Client, handoff: [16]u8, expect: Expect, reason: *Reason) Error!manifest.Manifest {
    const options = if (expect.stream) (if (expect.checksum) "{\"checksum\": true, \"stream\": true}" else "{\"checksum\": false, \"stream\": true}") else if (expect.checksum) "{\"checksum\": true}" else "{\"checksum\": false}";
    var deadline = words.nowNs() + expect.open_timeout_ns;
    var nap: u64 = std.time.ns_per_ms;
    while (true) {
        const h = try request(client, .open, handoff, 0, options, expect.protocol);
        const reply = client.box.replyPayload();
        switch (h.kind) {
            .wait => {
                if (expect.aborted()) return error.Aborted;
                if (expect.busy()) deadline = words.nowNs() + expect.open_timeout_ns;
                if (words.nowNs() >= deadline) return error.Expired;
                words.pause(nap);
                nap = @min(nap * 2, 20 * std.time.ns_per_ms);
            },
            .@"error" => {
                reason.set(wire.body(reply, h));
                return error.Refused;
            },
            .manifest => {
                const m = try manifest.parseUpTo(arena, try arena.dupe(u8, wire.body(reply, h)), expect.protocol);
                if (m.protocol != h.version) return error.Protocol;
                try check(m, h, handoff, expect);
                return m;
            },
            else => return error.Protocol,
        }
    }
}

/// The manifest must be for this prompt, checkpoint and rank, and its layers must be ones the sink can place.
fn check(m: manifest.Manifest, h: wire.Header, handoff: [16]u8, expect: Expect) Error!void {
    if (!std.mem.eql(u8, &m.handoff, &handoff) or m.frames != h.frames) return error.Protocol;
    if (!std.mem.eql(u8, m.model, expect.model) or m.prompt_tokens != expect.tokens.len) return error.Mismatch;
    if (!std.mem.eql(u8, &m.token_sha256, &wire.tokenSha256(expect.tokens))) return error.Mismatch;
    if (m.tp_rank != expect.tp_rank or m.tp_size != expect.tp_size or m.first_token % m.block_size != 0) return error.Mismatch;
    for (m.layers, 0..) |l, i| {
        for (m.layers[0..i]) |o| if (o.index == l.index) return error.Protocol;
    }
}

/// Every frame pulled; one a streaming producer answers WAIT (not converted yet) is asked again next round.
fn frames(arena: Allocator, client: *mailbox.Client, handoff: [16]u8, m: manifest.Manifest, expect: Expect, sink: Sink, reason: *Reason) Error!Pulled {
    var next: [512]u64 = @splat(0);
    if (m.layers.len > next.len) return error.Protocol;
    var bytes: u64 = 0;
    const todo = try arena.alloc(u32, m.frames);
    for (todo, 0..) |*t, i| t.* = @intCast(i);
    var left: usize = todo.len;
    var nap: u64 = std.time.ns_per_ms / 4;
    var idle_since = words.nowNs();
    while (left > 0) {
        var kept: usize = 0;
        for (todo[0..left]) |frame| {
            if (expect.aborted()) return error.Aborted;
            const h = try request(client, .pull, handoff, frame, "", m.protocol);
            const reply = client.box.replyPayload();
            if (h.kind == .wait) {
                todo[kept] = frame;
                kept += 1;
                continue;
            }
            if (h.kind == .@"error") {
                reason.set(wire.body(reply, h));
                return error.Refused;
            }
            if (h.kind != .data or h.frame != frame or h.frames != m.frames or (h.kind == .data and h.version != m.protocol)) return error.Protocol;
            const at = for (m.layers, 0..) |l, j| {
                if (l.index == h.layer) break j;
            } else return error.Protocol;
            const layer = &m.layers[at];
            // a streamed export's frames arrive in any order (a deferred one lands late): the rows total checks
            if (!rowsFit(h.row_start, h.rows, layer.rows()) or (layer.kind != .recurrent and !expect.stream and h.row_start != next[at])) return error.Protocol;
            if (h.nbytes != layer.frameBytes(h.row_start, h.rows) orelse return error.Protocol) return error.Protocol;
            const payload = wire.body(reply, h);
            if (expect.checksum and (h.flags & wire.checked == 0 or wire.crc32(payload) != h.crc)) return error.Corrupt;
            sink.put_fn(sink.ptr, layer, h.row_start, h.rows, payload) catch return error.Protocol;
            next[at] += h.rows;
            bytes += h.nbytes;
            idle_since = words.nowNs();
            nap = std.time.ns_per_ms / 4;
        }
        if (kept > 0 and kept == left) { // a whole round of WAIT: the producer has nothing more yet
            if (expect.busy()) idle_since = words.nowNs();
            if (words.nowNs() - idle_since >= expect.open_timeout_ns) return error.Expired;
            words.pause(nap);
            nap = @min(nap * 2, 4 * std.time.ns_per_ms);
        }
        left = kept;
    }
    for (m.layers, 0..) |l, j| if (next[j] != l.rows()) return error.Protocol;
    return .{ .manifest = m, .frames = m.frames, .bytes = bytes };
}

/// Rows [start, start + count) inside a layer of `total`, without the sum wrapping for a peer's huge values.
fn rowsFit(start: u64, count: u64, total: u64) bool {
    return count > 0 and count <= total and start <= total - count;
}

test "a frame's rows fit its layer only when their end does, however large a peer's values" {
    try std.testing.expect(rowsFit(0, 3, 3) and rowsFit(2, 1, 3));
    try std.testing.expect(!rowsFit(3, 1, 3) and !rowsFit(0, 0, 3) and !rowsFit(0, 4, 3));
    try std.testing.expect(!rowsFit(std.math.maxInt(u64), 1, 3) and !rowsFit(1, std.math.maxInt(u64), 3));
}

fn close(client: *mailbox.Client, handoff: [16]u8) Error!void {
    const h = try request(client, .close, handoff, 0, "", wire.recurrent_version);
    if (h.kind != .ack) return error.Protocol;
}
