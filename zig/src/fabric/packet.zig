//! Typed full-span packets over sender-owned double buffers, with credit returned only after the receiver releases its view.
const std = @import("std");
const rma = @import("rdma.zig");
const words = @import("words.zig");

pub const header_bytes = 64;
pub const alignment = 16384;
pub const Error = rma.Error || error{ InvalidLayout, TooLarge, InvalidFrame, WrongVersion, WrongRank, WrongShape, Outstanding, Timeout, Stale, SequenceOverflow };
pub const Kind = enum(u16) { boot_rows = 1, boot_stage_state = 2, boot_taps = 3, boot_logits = 4, boot_keep = 5, boot_release = 6, boot_stop = 7, control = 8 };
pub const Dtype = enum(u16) {
    bf16 = 1,
    f32 = 2,
    u8 = 3,
    u32 = 4,

    pub fn bytes(self: Dtype) usize {
        return switch (self) {
            .bf16 => 2,
            .f32, .u32 => 4,
            .u8 => 1,
        };
    }
};

pub const Frame = struct {
    version: u16 = 1,
    kind: Kind,
    dtype: Dtype = .bf16,
    source: u32,
    destination: u32,
    round: u64,
    rows: u32,
    columns: u32,
    planes: u32 = 1,
    credit: u64 = 0, // the sender's released steps from the destination: a credit riding the frame

    pub fn payloadBytes(self: Frame) Error!usize {
        const count = std.math.mul(usize, self.rows, self.columns) catch return error.WrongShape;
        const planes = std.math.mul(usize, count, self.planes) catch return error.WrongShape;
        return std.math.mul(usize, planes, self.dtype.bytes()) catch error.WrongShape;
    }

    pub fn encode(self: Frame, payload_bytes: usize) Error![header_bytes]u8 {
        if (self.version != 1) return error.WrongVersion;
        if (try self.payloadBytes() != payload_bytes) return error.WrongShape;
        var out: [header_bytes]u8 = @splat(0);
        @memcpy(out[0..4], "TFP1");
        std.mem.writeInt(u16, out[4..6], self.version, .little);
        std.mem.writeInt(u16, out[6..8], header_bytes, .little);
        std.mem.writeInt(u16, out[8..10], @backingInt(self.kind), .little);
        std.mem.writeInt(u16, out[10..12], @backingInt(self.dtype), .little);
        std.mem.writeInt(u32, out[12..16], self.source, .little);
        std.mem.writeInt(u32, out[16..20], self.destination, .little);
        std.mem.writeInt(u32, out[20..24], self.rows, .little);
        std.mem.writeInt(u32, out[24..28], self.columns, .little);
        std.mem.writeInt(u32, out[28..32], self.planes, .little);
        std.mem.writeInt(u64, out[32..40], self.round, .little);
        std.mem.writeInt(u64, out[40..48], payload_bytes, .little);
        std.mem.writeInt(u64, out[48..56], self.credit, .little);
        return out;
    }

    pub fn decode(bytes: []const u8, payload_bytes: usize) Error!Frame {
        if (bytes.len != header_bytes or !std.mem.eql(u8, bytes[0..4], "TFP1") or std.mem.readInt(u16, bytes[6..8], .little) != header_bytes) return error.InvalidFrame;
        if (std.mem.readInt(u16, bytes[4..6], .little) != 1) return error.WrongVersion;
        for (bytes[56..]) |b| if (b != 0) return error.InvalidFrame;
        const kind = std.mem.readInt(u16, bytes[8..10], .little);
        const dtype = std.mem.readInt(u16, bytes[10..12], .little);
        if (kind < 1 or kind > 8 or dtype < 1 or dtype > 4) return error.InvalidFrame;
        const frame: Frame = .{
            .kind = @fromBackingInt(@intCast(kind)),
            .dtype = @fromBackingInt(@intCast(dtype)),
            .source = std.mem.readInt(u32, bytes[12..16], .little),
            .destination = std.mem.readInt(u32, bytes[16..20], .little),
            .rows = std.mem.readInt(u32, bytes[20..24], .little),
            .columns = std.mem.readInt(u32, bytes[24..28], .little),
            .planes = std.mem.readInt(u32, bytes[28..32], .little),
            .round = std.mem.readInt(u64, bytes[32..40], .little),
            .credit = std.mem.readInt(u64, bytes[48..56], .little),
        };
        if (std.mem.readInt(u64, bytes[40..48], .little) != payload_bytes or try frame.payloadBytes() != payload_bytes) return error.WrongShape;
        return frame;
    }
};

pub const Layout = struct {
    ranks: u32,
    capacity: usize,
    stride: usize,
    total: usize,

    pub fn init(ranks: u32, capacity: usize) Error!Layout {
        if (ranks == 0 or ranks > 64 or capacity > std.math.maxInt(u32) - header_bytes) return error.InvalidLayout;
        const slot_bytes = std.mem.alignForward(usize, header_bytes + capacity, 64);
        const stride = std.mem.alignForward(usize, 64 + 2 * slot_bytes, alignment);
        const total = std.math.mul(usize, ranks, stride) catch return error.InvalidLayout;
        return .{ .ranks = ranks, .capacity = capacity, .stride = stride, .total = total };
    }

    fn base(self: Layout, sender: u32) usize {
        return sender * self.stride;
    }

    fn slot(self: Layout, sender: u32, step: u64) usize {
        return self.base(sender) + 64 + (step & 1) * std.mem.alignForward(usize, header_bytes + self.capacity, 64);
    }
};

pub const Received = struct { frame: Frame, payload: []const u8 };

pub const Peer = struct {
    ep: rma.Rdma,
    peer: u32,
    layout: Layout,
    sent: u64 = 0,
    received: u64 = 0,
    seen_credit: u64 = 0, // the peer's credit for our steps, as its frames carry it
    held: bool = false,
    timeout_ns: u64 = 10 * std.time.ns_per_s,
    wait_ns: u64 = 0, // the last receive's wait for its flag
    wait_gap_ns: u64 = 0, // the largest gap between that wait's clock reads (this thread off the CPU)

    pub const Snapshot = struct { sent: u64, received: u64, expected: u64, ready: u64, credit: u64, held: bool };
    pub fn snapshot(self: *const Peer) Snapshot {
        const step = self.received + 1;
        return .{ .sent = self.sent, .received = self.received, .expected = step, .ready = self.ep.local(self.layout.base(self.peer) + (step & 1) * 8), .credit = self.ep.local(self.layout.base(self.peer) + 16), .held = self.held };
    }

    pub fn init(ep: rma.Rdma, peer: u32, layout: Layout) Error!Peer {
        if (ep.size() != layout.ranks or peer >= ep.size() or peer == ep.rank() or ep.window().len < layout.total) return error.InvalidLayout;
        return .{ .ep = ep, .peer = peer, .layout = layout };
    }

    fn next(step: u64) Error!u64 {
        if (step >= std.math.maxInt(u32)) return error.SequenceOverflow;
        return step + 1;
    }

    /// One full typed span; the sender stages header and payload separately, then signals after both writes.
    pub fn send(self: *Peer, frame: Frame, payload: []const u8) Error!void {
        try self.stage(frame, payload);
        try self.ep.flush();
    }

    /// `send` without the flush: one write of header and payload, then the flag; the frame carries our released steps.
    pub fn stage(self: *Peer, frame: Frame, payload: []const u8) Error!void {
        if (frame.source != self.ep.rank() or frame.destination != self.peer) return error.WrongRank;
        if (payload.len > self.layout.capacity) return error.TooLarge;
        var carried = frame;
        carried.credit = self.received;
        const header = try carried.encode(payload.len);
        const step = try next(self.sent);
        const deadline = try words.checkedNowNs() + self.timeout_ns;
        while (step > 2 and @max(self.ep.local(self.layout.base(self.peer) + 16), self.seen_credit) < step - 2) {
            if (try words.checkedNowNs() > deadline) return error.Timeout;
            std.atomic.spinLoopHint();
        }
        const slot = self.layout.slot(self.ep.rank(), step);
        try self.ep.write2Signal(self.peer, slot, &header, payload, self.layout.base(self.ep.rank()) + (step & 1) * 8, step << 32 | (header_bytes + payload.len));
        self.sent = step;
    }

    /// The payload aliases the registered receive window and remains stable until release.
    pub fn receive(self: *Peer, timeout_ns: u64) Error!Received {
        if (self.held) return error.Outstanding;
        const step = try next(self.received);
        const began = try words.checkedNowNs();
        const deadline = began + timeout_ns;
        const flag = self.layout.base(self.peer) + (step & 1) * 8;
        var word = self.ep.local(flag);
        var prev = began;
        self.wait_gap_ns = 0;
        while (word >> 32 != step) : (word = self.ep.local(flag)) {
            if (word >> 32 > step) return error.Stale;
            const now = try words.checkedNowNs();
            self.wait_gap_ns = @max(self.wait_gap_ns, now - prev);
            prev = now;
            if (now > deadline) return error.Timeout;
            std.atomic.spinLoopHint();
        }
        self.wait_ns = (try words.checkedNowNs()) - began;
        const len: usize = @intCast(word & 0xFFFF_FFFF);
        if (len < header_bytes or len - header_bytes > self.layout.capacity) return error.TooLarge;
        const slot = self.layout.slot(self.peer, step);
        const bytes = self.ep.window()[slot..][0..len];
        const frame = try Frame.decode(bytes[0..header_bytes], len - header_bytes);
        if (frame.source != self.peer or frame.destination != self.ep.rank()) return error.WrongRank;
        self.seen_credit = @max(self.seen_credit, frame.credit);
        self.held = true;
        return .{ .frame = frame, .payload = bytes[header_bytes..] };
    }

    pub fn release(self: *Peer) Error!void {
        try self.unhold();
        try self.ep.flush();
    }

    /// `release` without the flush: the credit rides the next flush on the same link.
    pub fn unhold(self: *Peer) Error!void {
        try self.unholdQuiet();
        try self.ep.signal(self.peer, self.layout.base(self.ep.rank()) + 16, self.received);
    }

    /// Release the view with no credit message: our next frame to this peer carries it (peers that always answer).
    pub fn unholdQuiet(self: *Peer) Error!void {
        if (!self.held) return error.Outstanding;
        self.received = try next(self.received);
        self.held = false;
    }

    pub fn post(self: *Peer, frame: Frame, payload: []const u8) Error!void {
        return self.send(frame, payload);
    }

    pub fn wait(self: *Peer, timeout_ns: u64) Error!Received {
        return self.receive(timeout_ns);
    }

    pub fn done(self: *Peer) Error!void {
        return self.release();
    }

    pub fn fence(self: *Peer) Error!void {
        return self.ep.flush();
    }
};

pub const Link = Peer;

test "frames require a whole typed span and version" {
    const frame: Frame = .{ .kind = .boot_stage_state, .source = 0, .destination = 1, .round = 7, .rows = 4, .columns = 8, .planes = 3 };
    const bytes = try frame.encode(192);
    try std.testing.expectEqualDeep(frame, try Frame.decode(&bytes, 192));
    try std.testing.expectError(error.WrongShape, frame.encode(190));
    var bad = bytes;
    bad[4] = 2;
    try std.testing.expectError(error.WrongVersion, Frame.decode(&bad, 192));
    bad = bytes;
    bad[56] = 1;
    try std.testing.expectError(error.InvalidFrame, Frame.decode(&bad, 192));
    var credited = frame;
    credited.credit = 41;
    try std.testing.expectEqual(@as(u64, 41), (try Frame.decode(&try credited.encode(192), 192)).credit);
}
