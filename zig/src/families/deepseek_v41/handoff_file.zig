//! A hand-off as a byte stream (file or HTTP body): the producer's MKVH protocol 2 replies to OPEN and each PULL.
const std = @import("std");
const fabric = @import("fabric");
const wire = fabric.wire;
const manifest = fabric.manifest;
const producer = fabric.producer;
const decoder = fabric.decoder;

pub const extension = ".mkvh2";
/// The reply area of a link of 4 MiB segments; frames are planned and cut for it, so no message is longer.
pub const reply_area = 4 << 20;

/// `<id as 32 hex digits>.mkvh2`.
pub fn fileName(buf: *[32 + extension.len]u8, id: [16]u8) []const u8 {
    return std.fmt.bufPrint(buf, "{x}{s}", .{ &id, extension }) catch unreachable;
}

const Capture = struct {
    len: usize = 0,
    fn publish(ptr: *anyopaque, _: u32, len: usize) anyerror!void {
        const c: *Capture = @ptrCast(@alignCast(ptr));
        c.len = len;
    }
};

/// One export's stream, a message at a time; the export is only read, so many streams may read one export at once.
pub const Replies = struct {
    gpa: std.mem.Allocator,
    exp: producer.Export,
    table: producer.Table,
    area: []u8,
    cap: Capture = .{},
    responder: producer.Responder,
    /// messages given so far (the MANIFEST is the first, waiting in `area` from `open`)
    at: u64 = 0,
    /// the stream's whole length in bytes
    size: u64 = 0,

    pub fn open(gpa: std.mem.Allocator, exp: *const producer.Export, model: []const u8, tp_rank: u64, tp_size: u64) !*Replies {
        const r = try gpa.create(Replies);
        errdefer gpa.destroy(r);
        r.* = .{ .gpa = gpa, .exp = exp.*, .table = .{ .gpa = gpa, .ttl_ns = std.math.maxInt(u64), .clock = fabric.words.nowNs }, .area = try gpa.alloc(u8, reply_area), .responder = undefined };
        errdefer gpa.free(r.area);
        r.exp.frames = &.{};
        errdefer r.table.deinit();
        try r.table.add(exp.handoff, "replies", .{ .ready = &r.exp });
        r.responder = .{ .gpa = gpa, .table = &r.table, .out = .{ .area = r.area, .ptr = &r.cap, .publish_fn = Capture.publish }, .model = model, .tp_rank = tp_rank, .tp_size = tp_size };
        errdefer r.responder.deinit();
        errdefer if (r.exp.frames.len > 0) gpa.free(r.exp.frames);

        const options = "{\"checksum\": true}";
        var request: [wire.header_bytes + options.len]u8 = undefined;
        request[0..wire.header_bytes].* = wire.pack(.{ .kind = .open, .handoff = exp.handoff, .nbytes = options.len });
        @memcpy(request[wire.header_bytes..], options);
        try r.responder.handle(1, &request);
        const head = switch (wire.unpackUpTo(r.area[0..r.cap.len], wire.recurrent_version)) {
            .ok => |h| h,
            .bad => return error.Protocol,
        };
        if (head.kind != .manifest) return error.Protocol;
        r.size = r.cap.len;
        for (r.exp.frames) |f| r.size += wire.header_bytes + (r.exp.layers[f.position].frameBytes(f.row_start, f.rows) orelse return error.Protocol);
        return r;
    }

    pub fn deinit(r: *Replies) void {
        const gpa = r.gpa;
        if (r.exp.frames.len > 0) gpa.free(r.exp.frames);
        r.responder.deinit();
        r.table.deinit();
        gpa.free(r.area);
        gpa.destroy(r);
    }

    /// The next message, valid until the next call; null after the last.
    pub fn next(r: *Replies) !?[]const u8 {
        const i = r.at;
        if (i == 0) {
            r.at = 1;
            return r.area[0..r.cap.len];
        }
        if (i - 1 >= r.exp.frames.len) return null;
        const request = wire.pack(.{ .kind = .pull, .handoff = r.exp.handoff, .frame = @intCast(i - 1) });
        try r.responder.handle(@intCast(1 + i), &request);
        r.at = i + 1;
        return r.area[0..r.cap.len];
    }
};

/// `exp` as the replies a responder gives its decoder; `exp` itself is left unchanged.
pub fn write(gpa: std.mem.Allocator, w: *std.Io.Writer, exp: *const producer.Export, model: []const u8, tp_rank: u64, tp_size: u64) !void {
    const replies = try Replies.open(gpa, exp, model, tp_rank, tp_size);
    defer replies.deinit();
    while (try replies.next()) |message| try w.writeAll(message);
}

/// What a decoder can meet beyond the link decoder's own refusals: a stream that ends early, or a read that fails.
pub const Error = decoder.Error || error{ Truncated, TransportFailed };

const Message = struct { h: wire.Header, payload: []const u8 };

fn readFailure(err: std.Io.Reader.Error) Error {
    return switch (err) {
        error.EndOfStream => error.Truncated,
        error.ReadFailed => error.TransportFailed,
    };
}

/// The next message: its header, then exactly its payload, in the reader's buffer (valid until the next call).
fn next(r: *std.Io.Reader, most: u16) Error!Message {
    const head = r.peek(wire.header_bytes) catch |err| return readFailure(err);
    const nbytes = std.mem.readInt(u64, head[56..64], .little);
    if (nbytes > reply_area - wire.header_bytes) return error.Protocol;
    const msg = r.take(wire.header_bytes + @as(usize, @intCast(nbytes))) catch |err| return readFailure(err);
    const h = switch (wire.unpackUpTo(msg, most)) {
        .ok => |h| h,
        .bad => return error.Protocol,
    };
    // every message of a stream carries a payload: manifest, data or error
    if (h.kind != .manifest and h.kind != .data and h.kind != .@"error") return error.Protocol;
    return .{ .h = h, .payload = wire.body(msg, h) };
}

/// Pulls hand-off `r` into `sink` with `fabric.decoder.pull`'s checks; `r` needs a `reply_area` buffer.
pub fn pullFrom(arena: std.mem.Allocator, r: *std.Io.Reader, expect: decoder.Expect, sink: decoder.Sink, reason: *decoder.Reason) Error!decoder.Pulled {
    const first = try next(r, expect.protocol);
    switch (first.h.kind) {
        .@"error" => {
            reason.len = @min(first.payload.len, reason.buf.len);
            @memcpy(reason.buf[0..reason.len], first.payload[0..reason.len]);
            return error.Refused;
        },
        .manifest => {},
        else => return error.Protocol,
    }
    const m = try manifest.parseUpTo(arena, try arena.dupe(u8, first.payload), expect.protocol);
    if (m.protocol != first.h.version) return error.Protocol;
    const handoff = first.h.handoff;
    if (!std.mem.eql(u8, &m.handoff, &handoff) or m.frames != first.h.frames) return error.Protocol;
    if (!std.mem.eql(u8, m.model, expect.model) or m.prompt_tokens != expect.tokens.len) return error.Mismatch;
    if (!std.mem.eql(u8, &m.token_sha256, &wire.tokenSha256(expect.tokens))) return error.Mismatch;
    if (m.tp_rank != expect.tp_rank or m.tp_size != expect.tp_size or m.first_token % m.block_size != 0) return error.Mismatch;
    for (m.layers, 0..) |l, i| for (m.layers[0..i]) |o| if (o.index == l.index) return error.Protocol;

    var rows_next: [512]u64 = @splat(0);
    if (m.layers.len > rows_next.len) return error.Protocol;
    var total: u64 = 0;
    for (0..m.frames) |i| {
        const frame: u32 = @intCast(i);
        const msg = try next(r, m.protocol);
        const h = msg.h;
        if (h.kind == .@"error") {
            reason.len = @min(msg.payload.len, reason.buf.len);
            @memcpy(reason.buf[0..reason.len], msg.payload[0..reason.len]);
            return error.Refused;
        }
        if (h.kind != .data or h.frame != frame or h.frames != m.frames or h.version != m.protocol or !std.mem.eql(u8, &h.handoff, &handoff)) return error.Protocol;
        const idx = for (m.layers, 0..) |l, j| {
            if (l.index == h.layer) break j;
        } else return error.Protocol;
        const layer = &m.layers[idx];
        if (!rowsFit(h.row_start, h.rows, layer.rows()) or h.row_start != rows_next[idx]) return error.Protocol;
        if (h.nbytes != layer.frameBytes(h.row_start, h.rows) orelse return error.Protocol) return error.Protocol;
        if (expect.checksum and (h.flags & wire.checked == 0 or wire.crc32(msg.payload) != h.crc)) return error.Corrupt;
        sink.put_fn(sink.ptr, layer, h.row_start, h.rows, msg.payload) catch return error.Protocol;
        rows_next[idx] += h.rows;
        total += h.nbytes;
    }
    if (r.peekByte()) |_| return error.Protocol else |err| switch (err) { // nothing may follow the last frame
        error.EndOfStream => {},
        error.ReadFailed => return error.TransportFailed,
    }
    for (m.layers, 0..) |l, j| if (rows_next[j] != l.rows()) return error.Protocol;
    return .{ .manifest = m, .frames = m.frames, .bytes = total };
}

/// `pullFrom` over a hand-off held in memory (a file read whole); a short file is a protocol error, as it always was.
pub fn pull(arena: std.mem.Allocator, bytes: []const u8, expect: decoder.Expect, sink: decoder.Sink, reason: *decoder.Reason) decoder.Error!decoder.Pulled {
    var r = std.Io.Reader.fixed(bytes);
    return pullFrom(arena, &r, expect, sink, reason) catch |err| switch (err) {
        error.Truncated, error.TransportFailed => error.Protocol,
        else => |e| e,
    };
}

fn rowsFit(start: u64, count: u64, total: u64) bool {
    return count > 0 and count <= total and start <= total - count;
}
