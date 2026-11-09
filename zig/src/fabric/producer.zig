//! The producer's side of a handoff: exports held until closed or expired, and answers to OPEN, PULL and CLOSE.
const std = @import("std");
const wire = @import("wire.zig");
const manifest = @import("manifest.zig");
const frames = @import("frames.zig");
const mailbox = @import("mailbox.zig");
const words = @import("words.zig");
const Allocator = std.mem.Allocator;

/// One request's exported pages; `sources[i]` holds `layers[i]`'s rows.
pub const Export = struct {
    handoff: [16]u8,
    tokens: []const u32,
    first_token: u64,
    tokens_per_row: u64,
    layers: []const manifest.Layer,
    sources: []const frames.Source,
    frames: []frames.Frame = &.{},

    /// Protocol 2 when a layer is recurrent, so a protocol 1 decoder refuses it; else protocol 1, byte for byte.
    pub fn protocol(e: *const Export) u16 {
        for (e.layers) |l| if (l.kind == .recurrent) return wire.recurrent_version;
        return wire.version;
    }

    pub fn toManifest(e: *const Export, model: []const u8, tp_rank: u64, tp_size: u64) manifest.Manifest {
        return .{ .protocol = e.protocol(), .handoff = e.handoff, .model = model, .prompt_tokens = e.tokens.len, .first_token = e.first_token, .token_sha256 = wire.tokenSha256(e.tokens), .block_size = e.tokens_per_row, .tp_rank = tp_rank, .tp_size = tp_size, .layers = e.layers, .frames = e.frames.len };
    }
};

/// An export, or the reason one could not be made, which the decoder is told.
pub const Entry = union(enum) { ready: *Export, failed: []const u8 };

const Held = struct { entry: Entry, request: []const u8, added: u64, bytes: u64 = 0 };

/// Exports by handoff id, released when the decoder closes them or after `ttl_ns`.
pub const Table = struct {
    gpa: Allocator,
    ttl_ns: u64,
    clock: *const fn () u64,
    cap_bytes: u64 = 0, // the most export bytes held at once (0: no cap); an export past it is refused
    held_bytes: u64 = 0,
    held: std.AutoArrayHashMapUnmanaged([16]u8, Held) = .empty,
    finished: std.ArrayList([]const u8) = .empty,
    lock: std.atomic.Mutex = .unlocked, // a server's engine thread adds; its serving thread reads and releases

    fn hold(t: *Table) void {
        while (!t.lock.tryLock()) std.atomic.spinLoopHint();
    }

    pub fn deinit(t: *Table) void {
        t.held.deinit(t.gpa);
        t.finished.deinit(t.gpa);
    }

    pub fn add(t: *Table, handoff: [16]u8, request: []const u8, entry: Entry) !void {
        return t.addSized(handoff, request, entry, 0);
    }

    /// `add` for an export of `bytes`, refused with error.HeldExportsFull when it would pass `cap_bytes`.
    pub fn addSized(t: *Table, handoff: [16]u8, request: []const u8, entry: Entry, bytes: u64) !void {
        t.hold();
        defer t.lock.unlock();
        if (t.cap_bytes > 0 and t.held_bytes + bytes > t.cap_bytes) return error.HeldExportsFull;
        const slot = try t.held.getOrPut(t.gpa, handoff);
        if (slot.found_existing) return error.HandoffHeld; // the export it holds stays reachable by CLOSE and its TTL
        slot.value_ptr.* = .{ .entry = entry, .request = request, .added = t.clock(), .bytes = bytes };
        t.held_bytes += bytes;
    }

    /// An entry stays valid until the serving thread itself releases it (only it finishes and expires them).
    pub fn get(t: *Table, handoff: [16]u8) ?Entry {
        t.hold();
        defer t.lock.unlock();
        return if (t.held.get(handoff)) |h| h.entry else null;
    }

    /// Release a handoff; its request is reported finished so its pages can be freed.
    pub fn finish(t: *Table, handoff: [16]u8) !void {
        t.hold();
        defer t.lock.unlock();
        try t.finishHeld(handoff);
    }

    fn finishHeld(t: *Table, handoff: [16]u8) !void {
        const h = t.held.fetchOrderedRemove(handoff) orelse return;
        t.held_bytes -= h.value.bytes;
        try t.finished.append(t.gpa, h.value.request);
    }

    pub fn expire(t: *Table) !void {
        t.hold();
        defer t.lock.unlock();
        const now = t.clock();
        var i: usize = 0;
        while (i < t.held.count()) {
            if (now - t.held.values()[i].added > t.ttl_ns) {
                try t.finishHeld(t.held.keys()[i]);
            } else i += 1;
        }
    }

    /// Requests released since the last call (the caller frees their pages).
    pub fn takeFinished(t: *Table, out: *std.ArrayList([]const u8)) !void {
        t.hold();
        defer t.lock.unlock();
        try out.appendSlice(t.gpa, t.finished.items);
        t.finished.clearRetainingCapacity();
    }
};

/// Where replies go: the service's reply area and its publish (or a test capture).
pub const Outbox = struct {
    area: []u8,
    ptr: *anyopaque,
    publish_fn: *const fn (ptr: *anyopaque, seq: u32, len: usize) anyerror!void,

    fn publish(o: Outbox, seq: u32, len: usize) !void {
        return o.publish_fn(o.ptr, seq, len);
    }
};

const no_handoff: [16]u8 = @splat(0);

/// Answers one link's requests exactly as MCDMA's responder.py does, byte for byte.
pub const Responder = struct {
    gpa: Allocator,
    table: *Table,
    out: Outbox,
    model: []const u8,
    tp_rank: u64 = 0,
    tp_size: u64 = 1,
    checksums: std.AutoHashMapUnmanaged([16]u8, bool) = .empty,

    pub fn deinit(r: *Responder) void {
        r.checksums.deinit(r.gpa);
    }

    fn reply(r: *Responder, seq: u32, kind: wire.Kind, handoff: [16]u8, payload: []const u8, frames_n: u32) !void {
        return r.replyAs(seq, wire.version, kind, handoff, payload, frames_n);
    }

    fn replyAs(r: *Responder, seq: u32, v: u16, kind: wire.Kind, handoff: [16]u8, payload: []const u8, frames_n: u32) !void {
        if (wire.header_bytes + payload.len > r.out.area.len) return error.ReplyTooLarge;
        const h = wire.pack(.{ .version = v, .kind = kind, .handoff = handoff, .frames = frames_n, .nbytes = payload.len });
        @memcpy(r.out.area[0..wire.header_bytes], &h);
        @memcpy(r.out.area[wire.header_bytes..][0..payload.len], payload);
        try r.out.publish(seq, wire.header_bytes + payload.len);
    }

    /// Answer one request; an error here becomes the producer-failed ERROR that `failed` sends.
    pub fn handle(r: *Responder, seq: u32, msg: []const u8) !void {
        const h = switch (wire.unpackUpTo(msg, wire.recurrent_version)) {
            .ok => |h| h,
            .bad => |b| {
                var buf: [64]u8 = undefined;
                return r.reply(seq, .@"error", no_handoff, b.why.text(b.kind, &buf), 0);
            },
        };
        const entry = r.table.get(h.handoff);
        if (entry) |e| if (e == .failed) return r.reply(seq, .@"error", h.handoff, e.failed, 0);
        const exp: ?*Export = if (entry) |e| e.ready else null;
        switch (h.kind) {
            .open => try r.open(seq, h, msg, exp),
            .pull => try r.pull(seq, h, exp),
            .close => {
                try r.table.finish(h.handoff);
                _ = r.checksums.remove(h.handoff);
                try r.reply(seq, .ack, h.handoff, "", 0);
            },
            else => {
                var buf: [48]u8 = undefined;
                const text = try std.fmt.bufPrint(&buf, "unexpected request kind {d}", .{@backingInt(h.kind)});
                try r.reply(seq, .@"error", h.handoff, text, 0);
            },
        }
    }

    /// The ERROR responder.py's run() sends when answering failed.
    pub fn failed(r: *Responder, seq: u32, err: anyerror) !void {
        var buf: [128]u8 = undefined;
        const text = std.fmt.bufPrint(&buf, "the producer failed: {s}", .{@errorName(err)}) catch "the producer failed";
        try r.reply(seq, .@"error", no_handoff, text, 0);
    }

    fn open(r: *Responder, seq: u32, h: wire.Header, msg: []const u8, exp: ?*Export) !void {
        const e = exp orelse return r.reply(seq, .wait, h.handoff, "", 0);
        try r.checksums.put(r.gpa, h.handoff, try checksumOption(r.gpa, wire.body(msg, h)));
        if (e.frames.len == 0) {
            const rows = try r.gpa.alloc(frames.LayerRows, e.layers.len);
            defer r.gpa.free(rows);
            var sizes: std.ArrayList(u64) = .empty;
            defer sizes.deinit(r.gpa);
            for (e.layers) |l| for (l.parts) |p| try sizes.append(r.gpa, p.bytes());
            var at: usize = 0;
            for (e.layers, rows) |l, *x| {
                x.* = .{ .rows = l.rows(), .row_bytes = l.rowBytes(), .sizes = sizes.items[at..][0..l.parts.len] };
                at += l.parts.len;
            }
            e.frames = try frames.plan(r.gpa, rows, r.out.area.len - wire.header_bytes);
        }
        var text: std.Io.Writer.Allocating = .init(r.gpa);
        defer text.deinit();
        try manifest.write(e.toManifest(r.model, r.tp_rank, r.tp_size), &text.writer);
        try r.replyAs(seq, e.protocol(), .manifest, h.handoff, text.written(), @intCast(e.frames.len));
    }

    fn pull(r: *Responder, seq: u32, h: wire.Header, exp: ?*Export) !void {
        const e = exp orelse return r.noFrame(seq, h);
        if (h.frame >= e.frames.len) return r.noFrame(seq, h);
        const f = e.frames[h.frame];
        const area = r.out.area[wire.header_bytes..];
        const nbytes = try e.sources[f.position].gather(f.row_start, f.rows, area);
        const checked = r.checksums.get(h.handoff) orelse true;
        const crc = if (checked) wire.crc32(area[0..@intCast(nbytes)]) else 0;
        const head = wire.pack(.{ .version = e.protocol(), .kind = .data, .handoff = h.handoff, .frame = h.frame, .frames = @intCast(e.frames.len), .layer = e.layers[f.position].index, .flags = if (checked) wire.checked else 0, .row_start = f.row_start, .rows = f.rows, .nbytes = nbytes, .crc = crc });
        @memcpy(r.out.area[0..wire.header_bytes], &head);
        try r.out.publish(seq, wire.header_bytes + @as(usize, @intCast(nbytes)));
    }

    fn noFrame(r: *Responder, seq: u32, h: wire.Header) !void {
        var buf: [48]u8 = undefined;
        try r.reply(seq, .@"error", h.handoff, try std.fmt.bufPrint(&buf, "no frame {d} in this handoff", .{h.frame}), 0);
    }
};

/// One link's serving thread: each request on the listen end answered, until `stop`; expired exports released.
pub const Serving = struct {
    service: mailbox.Service,
    responder: Responder,
    stop: std.atomic.Value(bool) = .init(false),

    /// `s` must stay at its address while it serves: its responder publishes through it.
    pub fn init(s: *Serving, gpa: Allocator, box: *const mailbox.Mailbox, ws: words.Words, table: *Table, model: []const u8) void {
        s.* = .{ .service = mailbox.Service.init(box, ws), .responder = undefined };
        s.responder = .{ .gpa = gpa, .table = table, .out = .{ .area = s.service.replyArea(), .ptr = s, .publish_fn = publish }, .model = model };
    }

    pub fn deinit(s: *Serving) void {
        s.responder.deinit();
    }

    fn publish(ptr: *anyopaque, seq: u32, len: usize) anyerror!void {
        const s: *Serving = @ptrCast(@alignCast(ptr));
        try s.service.publish(seq, len);
    }

    pub fn run(s: *Serving) void {
        while (!s.stop.load(.acquire)) {
            s.responder.table.expire() catch {};
            const req = (s.service.next(std.time.ns_per_ms) catch return) orelse continue;
            s.responder.handle(req.seq, req.payload) catch |err| s.responder.failed(req.seq, err) catch return;
        }
    }
};

/// bool(json.loads(body or "{}").get("checksum", True)), with unparsable options read as {} as responder.py does.
fn checksumOption(gpa: Allocator, body: []const u8) !bool {
    const text = if (body.len == 0) "{}" else body;
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, text, .{}) catch return true;
    defer parsed.deinit();
    if (parsed.value != .object) return error.OptionsNotObject;
    const v = parsed.value.object.get("checksum") orelse return true;
    return switch (v) {
        .null => false,
        .bool => |b| b,
        .integer => |i| i != 0,
        .float => |x| x != 0,
        .number_string, .string => |s| s.len > 0 and !std.mem.eql(u8, s, "0"),
        .array => |a| a.items.len > 0,
        .object => |o| o.count() > 0,
    };
}

test "the table refuses a hand-off id it already holds, so the entry it holds is the one served" {
    const clock = struct {
        fn now() u64 {
            return 0;
        }
    }.now;
    var t: Table = .{ .gpa = std.testing.allocator, .ttl_ns = std.time.ns_per_s, .clock = clock };
    defer t.deinit();
    const id: [16]u8 = @splat(3);
    try t.add(id, "first", .{ .failed = "first" });
    try std.testing.expectError(error.HandoffHeld, t.add(id, "second", .{ .failed = "second" }));
    try std.testing.expectEqualStrings("first", t.get(id).?.failed);
    try t.finish(id);
    try t.add(id, "third", .{ .failed = "third" }); // released: the id is free again
}

test "the table holds export bytes up to its cap, refuses the export past it by name, and frees room on release" {
    const clock = struct {
        fn now() u64 {
            return 0;
        }
    }.now;
    var t: Table = .{ .gpa = std.testing.allocator, .ttl_ns = std.time.ns_per_s, .clock = clock, .cap_bytes = 100 };
    defer t.deinit();
    try t.addSized(@splat(1), "one", .{ .failed = "one" }, 60);
    try std.testing.expectError(error.HeldExportsFull, t.addSized(@splat(2), "two", .{ .failed = "two" }, 50));
    try t.addSized(@splat(2), "two", .{ .failed = "two" }, 40); // exactly at the cap
    try t.add(@splat(3), "failed", .{ .failed = "a failure holds no bytes" });
    try t.finish(@splat(1));
    try std.testing.expectEqual(@as(u64, 40), t.held_bytes);
    try t.addSized(@splat(4), "four", .{ .failed = "four" }, 60);
}

test "checksum options follow Python truthiness" {
    const gpa = std.testing.allocator;
    try std.testing.expect(try checksumOption(gpa, ""));
    try std.testing.expect(try checksumOption(gpa, "{\"checksum\": true}"));
    try std.testing.expect(!try checksumOption(gpa, "{\"checksum\": false}"));
    try std.testing.expect(!try checksumOption(gpa, "{\"checksum\": 0}"));
    try std.testing.expect(try checksumOption(gpa, "not json"));
    try std.testing.expectError(error.OptionsNotObject, checksumOption(gpa, "[1]"));
}
