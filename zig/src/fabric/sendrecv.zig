//! Two-sided links, as Apple's Thunderbolt RDMA runs every request: ordered messages per peer, received into rings.
const std = @import("std");
const words = @import("words.zig");
const collective = @import("collective.zig");

pub const Error = error{ TooLarge, Outstanding, PeerTimeout, PeerDown, NoSuchRank, Protocol };

pub const Link = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        rank: *const fn (ptr: *anyopaque) u32,
        size: *const fn (ptr: *anyopaque) u32,
        /// Send `bytes` as `peer`'s next message (empty allowed); `bytes` may be reused once this returns.
        send: *const fn (ptr: *anyopaque, peer: u32, bytes: []const u8) Error!void,
        /// The oldest whole message from `peer` not yet returned, or null; it stays valid until released.
        next: *const fn (ptr: *anyopaque, peer: u32) Error!?[]const u8,
        /// Free the oldest returned message from `peer`.
        release: *const fn (ptr: *anyopaque, peer: u32) Error!void,
        /// Return once every posted send has left this rank.
        flush: *const fn (ptr: *anyopaque) Error!void,
    };

    pub fn rank(l: Link) u32 {
        return l.vtable.rank(l.ptr);
    }
    pub fn size(l: Link) u32 {
        return l.vtable.size(l.ptr);
    }
    pub fn send(l: Link, peer: u32, bytes: []const u8) Error!void {
        return l.vtable.send(l.ptr, peer, bytes);
    }
    pub fn next(l: Link, peer: u32) Error!?[]const u8 {
        return l.vtable.next(l.ptr, peer);
    }
    pub fn release(l: Link, peer: u32) Error!void {
        return l.vtable.release(l.ptr, peer);
    }
    pub fn flush(l: Link) Error!void {
        return l.vtable.flush(l.ptr);
    }
};

const max = collective.max_ranks;

/// Lockstep steps over a link: each step one message to and from every peer; a step's views live until the next post.
pub const Exchange = struct {
    link: Link,
    step: u64 = 0,
    completed: u64 = 0,
    held: bool = false,
    own: []const u8 = &.{},
    timeout_ns: u64 = 10 * std.time.ns_per_s,

    pub fn init(link: Link) Exchange {
        std.debug.assert(link.size() <= max);
        return .{ .link = link };
    }

    /// Release the last step's messages, then send chunks[p] to each rank p; returns the step.
    pub fn post(x: *Exchange, chunks: []const []const u8) Error!u64 {
        if (x.completed != x.step) return error.Outstanding;
        const me = x.link.rank();
        std.debug.assert(chunks.len == x.link.size());
        if (x.held) for (0..x.link.size()) |i| {
            if (i != me) try x.link.release(@intCast(i));
        };
        x.held = false;
        x.step += 1;
        for (chunks, 0..) |c, i| {
            if (i != me) try x.link.send(@intCast(i), c);
        }
        x.own = chunks[me];
        return x.step;
    }

    /// Wait for every peer's message of `step`; out[p] views it until the next post.
    pub fn complete(x: *Exchange, step: u64, out: [][]const u8) Error!void {
        std.debug.assert(step == x.step and out.len == x.link.size());
        const me = x.link.rank();
        const deadline = words.nowNs() + x.timeout_ns;
        for (out, 0..) |*o, i| {
            const p: u32 = @intCast(i);
            if (p == me) {
                o.* = x.own;
                continue;
            }
            while (true) {
                if (try x.link.next(p)) |m| {
                    o.* = m;
                    break;
                }
                if (words.nowNs() > deadline) return error.PeerTimeout;
                std.atomic.spinLoopHint();
            }
        }
        x.held = true;
        try x.link.flush();
        x.completed = step;
    }

    pub fn exchange(x: *Exchange, chunks: []const []const u8, out: [][]const u8) Error!void {
        try x.complete(try x.post(chunks), out);
    }

    pub fn ranks(x: *const Exchange) u32 {
        return x.link.size();
    }

    pub fn myRank(x: *const Exchange) u32 {
        return x.link.rank();
    }
};
