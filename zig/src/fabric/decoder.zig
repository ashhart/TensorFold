//! The decoder's side of a handoff: OPEN until the export is recorded, PULL every frame into a sink, then CLOSE.
const std = @import("std");
const wire = @import("wire.zig");
const manifest = @import("manifest.zig");
const mailbox = @import("mailbox.zig");
const words = @import("words.zig");
const Allocator = std.mem.Allocator;

pub const Error = error{ Refused, Protocol, Mismatch, Corrupt, Expired } || mailbox.Error || manifest.Error;

/// What the decoder knows the export must be: the same checkpoint, prompt and tensor-parallel rank.
pub const Expect = struct {
    model: []const u8,
    tokens: []const u32,
    tp_rank: u64 = 0,
    tp_size: u64 = 1,
    checksum: bool = true,
    open_timeout_ns: u64 = 30 * std.time.ns_per_s,
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
    const got = frames(client, handoff, m, expect, sink, reason) catch |err| {
        if (!client.poisoned) close(client, handoff) catch {};
        return err;
    };
    try close(client, handoff);
    return got;
}

fn request(client: *mailbox.Client, kind: wire.Kind, handoff: [16]u8, frame: u32, payload: []const u8) Error!wire.Header {
    var msg: [wire.header_bytes + 32]u8 = undefined;
    msg[0..wire.header_bytes].* = wire.pack(.{ .kind = kind, .handoff = handoff, .frame = frame, .nbytes = payload.len });
    @memcpy(msg[wire.header_bytes..][0..payload.len], payload);
    const reply = try client.call(msg[0 .. wire.header_bytes + payload.len]);
    const h = switch (wire.unpack(reply)) {
        .ok => |h| h,
        .bad => return error.Protocol,
    };
    if (!std.mem.eql(u8, &h.handoff, &handoff) and h.kind != .@"error") return error.Protocol;
    return h;
}

fn open(arena: Allocator, client: *mailbox.Client, handoff: [16]u8, expect: Expect, reason: *Reason) Error!manifest.Manifest {
    const options = if (expect.checksum) "{\"checksum\": true}" else "{\"checksum\": false}";
    const deadline = words.nowNs() + expect.open_timeout_ns;
    var nap: u64 = std.time.ns_per_ms;
    while (true) {
        const h = try request(client, .open, handoff, 0, options);
        const reply = client.box.replyPayload();
        switch (h.kind) {
            .wait => {
                if (words.nowNs() >= deadline) return error.Expired;
                words.pause(nap);
                nap = @min(nap * 2, 20 * std.time.ns_per_ms);
            },
            .@"error" => {
                reason.set(wire.body(reply, h));
                return error.Refused;
            },
            .manifest => {
                const m = try manifest.parse(arena, try arena.dupe(u8, wire.body(reply, h)));
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

fn frames(client: *mailbox.Client, handoff: [16]u8, m: manifest.Manifest, expect: Expect, sink: Sink, reason: *Reason) Error!Pulled {
    var next: [512]u64 = @splat(0);
    if (m.layers.len > next.len) return error.Protocol;
    var bytes: u64 = 0;
    for (0..m.frames) |i| {
        const frame: u32 = @intCast(i);
        const h = try request(client, .pull, handoff, frame, "");
        const reply = client.box.replyPayload();
        if (h.kind == .@"error") {
            reason.set(wire.body(reply, h));
            return error.Refused;
        }
        if (h.kind != .data or h.frame != frame or h.frames != m.frames) return error.Protocol;
        const at = for (m.layers, 0..) |l, j| {
            if (l.index == h.layer) break j;
        } else return error.Protocol;
        const layer = &m.layers[at];
        if (h.row_start != next[at] or h.rows == 0 or h.row_start + h.rows > layer.rows()) return error.Protocol;
        if (h.nbytes != h.rows * layer.rowBytes()) return error.Protocol;
        const payload = wire.body(reply, h);
        if (expect.checksum and (h.flags & wire.checked == 0 or wire.crc32(payload) != h.crc)) return error.Corrupt;
        sink.put_fn(sink.ptr, layer, h.row_start, h.rows, payload) catch return error.Protocol;
        next[at] += h.rows;
        bytes += h.nbytes;
    }
    for (m.layers, 0..) |l, j| if (next[j] != l.rows()) return error.Protocol;
    return .{ .manifest = m, .frames = m.frames, .bytes = bytes };
}

fn close(client: *mailbox.Client, handoff: [16]u8) Error!void {
    const h = try request(client, .close, handoff, 0, "");
    if (h.kind != .ack) return error.Protocol;
}
