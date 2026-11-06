//! Four-rank ring: two half flags a source, alternating completed relays, and two-slot custody.
const std = @import("std");
const guard = @import("roce_multi_guard.zig");
pub const Error = error{ BadRelay, WrongSequence, Timeout };
pub const Half = struct { offset: u32, bytes: u32 };
pub fn half(padded: u32, direction: u32) Error!Half {
    if (padded == 0 or padded % 16 != 0 or padded > guard.max_write or direction > 1) return error.BadRelay;
    const first = ((padded / 16 + 1) / 2) * 16;
    return if (direction == 0) .{ .offset = 0, .bytes = first } else .{ .offset = first, .bytes = padded - first };
}
pub fn neighbour(rank: u32, direction: u32) u32 {
    return (rank + if (direction == 0) @as(u32, 1) else 3) % 4;
}
pub fn newer(a: u32, b: u32) bool {
    return @as(i32, @bitCast(a -% b)) > 0;
}
pub const Segment = struct { dst: u32, source: u32, seq: u32, direction: u32, total: u32 };
/// The same guards before a shared-memory copy or a verbs post; length lives at flag + 4, before the flag's release.
pub fn plan(l: guard.Layout, me: u32, s: Segment) Error!guard.Stripe {
    if (l.world != 4 or me >= 4 or s.source >= 4 or s.dst >= 4 or s.direction > 1) return error.BadRelay;
    if (s.dst != neighbour(me, 0) and s.dst != neighbour(me, 1)) return error.BadRelay;
    if (s.source != me and (s.source != neighbour(me, 1 - s.direction) or s.dst != neighbour(me, s.direction)))
        return error.BadRelay;
    if (l.slot == 0 or l.slot > guard.max_write or !std.meta.eql(l, guard.Layout.of(4, l.slot)) or s.total > l.slot) return error.BadRelay;
    const h = try half(s.total, s.direction);
    return .{ .local = (if (s.source == me) l.sendAt(s.seq & 1) else l.recvAt(s.source, s.seq & 1)) + h.offset, .remote = l.recvAt(s.source, s.seq & 1) + h.offset, .len = h.bytes, .flag = l.flagAt(s.source, s.seq & 1, s.direction) };
}
pub const Fault = struct {
    rank: u32 = 0,
    sequence: u32 = 0,
    seen: u32 = 0,
    stage: enum { producer, custody, receive, forwarding } = .producer,
    pub fn describe(f: Fault, out: []u8) []const u8 {
        return std.fmt.bufPrint(out, "rank {d} sequence {d} ({t}, seen {d})", .{ f.rank, f.sequence, f.stage, f.seen }) catch "relay failure";
    }
};
pub const Relay = struct {
    me: u32,
    layout: guard.Layout,
    region: []u8,
    timeout_ns: u64,
    last: u32 = 0,
    started: bool = false,
    sent: [2]bool = .{ false, false },
    direction: u32 = 0,
    waiting_at: ?u64 = null,
    fault: Fault = .{},

    pub fn init(me: u32, l: guard.Layout, region: []u8, timeout_ns: u64) Error!Relay {
        if (l.world != 4 or me >= 4 or region.len < l.len or l.slot == 0 or l.slot > guard.max_write or timeout_ns == 0)
            return error.BadRelay;
        if (!std.meta.eql(l, guard.Layout.of(4, l.slot))) return error.BadRelay;
        return .{ .me = me, .layout = l, .region = region, .timeout_ns = timeout_ns };
    }
    pub fn word(r: *const Relay, offset: u64) *std.atomic.Value(u32) {
        return @ptrCast(@alignCast(r.region[offset..].ptr));
    }
    fn flag(r: *const Relay, source: u32, seq: u32, direction: u32) u32 {
        return r.word(r.layout.flagAt(source, seq & 1, direction)).load(.acquire);
    }
    fn wait(r: *Relay, now: u64, fault: Fault) Error!bool {
        r.fault = fault;
        if (r.waiting_at == null) r.waiting_at = now;
        if (now -| r.waiting_at.? >= r.timeout_ns) return error.Timeout;
        return false;
    }
    pub fn allSeen(r: *Relay, seq: u32, now: u64) Error!bool {
        for (0..4) |src| if (src != r.me) for (0..2) |h| {
            const seen = r.flag(@intCast(src), seq, @intCast(h));
            if (seen == seq) continue;
            r.fault = .{ .rank = @intCast(src), .sequence = seq, .seen = seen, .stage = .receive };
            if (newer(seen, seq)) return error.WrongSequence;
            return r.wait(now, r.fault);
        };
        return true;
    }
    /// The producer starts a collective only after its preceding consumer returns (the existing CUDA stream contract).
    pub fn progress(r: *Relay, wire: anytype, doorbell: u32, consumed: u32, padded: u32, now: u64) !bool {
        r.fault = .{ .rank = r.me, .sequence = r.last, .stage = .forwarding };
        try wire.reap();
        if (!wire.idle()) return r.wait(now, r.fault);
        // Both forwards of the previous op complete before any new op posts. Only this owner touches queue pairs.
        if (r.started) {
            if (!r.sent[r.direction]) {
                const source = neighbour(r.me, 1 - r.direction);
                const seen = r.flag(source, r.last, r.direction);
                r.fault = .{ .rank = source, .sequence = r.last, .seen = seen, .stage = .forwarding };
                if (seen != r.last) {
                    if (newer(seen, r.last)) return error.WrongSequence;
                    return r.wait(now, r.fault);
                }
                const total = r.word(r.layout.flagAt(source, r.last & 1, r.direction) + 4).load(.acquire);
                try wire.segment(.{ .dst = neighbour(r.me, r.direction), .source = source, .seq = r.last, .direction = r.direction, .total = total });
                r.sent[r.direction] = true;
                r.direction = 1 - r.direction;
                r.waiting_at = null;
                return true;
            }
            if (!try r.allSeen(r.last, now)) return false;
        }
        if (doorbell == r.last) {
            r.waiting_at = null;
            return false;
        }
        const seq = r.last +% 1;
        r.fault = .{ .rank = r.me, .sequence = seq, .seen = doorbell, .stage = .producer };
        if (doorbell -% r.last > 2) return error.WrongSequence;
        // A published s-1 proves that producer consumed s-2. Waiting for every s-1 protects all receive-slot readers.
        if (r.started and newer(seq -% 2, consumed))
            return r.wait(now, .{ .rank = r.me, .sequence = seq -% 2, .seen = consumed, .stage = .custody });
        for (0..2) |d| for (0..2) |h| {
            const dst = neighbour(r.me, @intCast(d));
            r.fault = .{ .rank = dst, .sequence = seq, .stage = .forwarding };
            try wire.segment(.{ .dst = dst, .source = r.me, .seq = seq, .direction = @intCast(h), .total = padded });
        };
        r.last = seq;
        r.started = true;
        r.sent = .{ false, false };
        r.waiting_at = null;
        return true;
    }
};
