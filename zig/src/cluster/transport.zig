//! What the control plane needs from a link layer: ordered messages per RDMA link, and each link's state; never TCP.
const std = @import("std");
const node = @import("node.zig");

/// A local port index; messages are addressed to links, and membership learns who is at the other end.
pub const LinkId = u8;
pub const max_links = 16;

pub const Error = error{ LinkDown, TooLarge };

pub const Link = struct {
    id: LinkId,
    up: bool,
    kind: node.LinkKind = .tb5,
    gbps: u32 = 0,
};

pub const Received = struct { link: LinkId, len: usize };

/// MCDMA implements it with one mailbox pair per link direction; tests with fake_net's queues.
pub const Transport = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Queue `bytes` for the node at the other end of `link`; messages on one link arrive in order or not at all.
        send: *const fn (ptr: *anyopaque, link: LinkId, bytes: []const u8) Error!void,
        /// The next message from any link, copied into `buf`; null when none waits.
        recv: *const fn (ptr: *anyopaque, buf: []u8) ?Received,
        /// Every local link with its state.
        links: *const fn (ptr: *anyopaque, out: []Link) usize,
        /// The largest message a link carries (the mailbox payload).
        limit: *const fn (ptr: *anyopaque) usize,
    };

    pub fn send(t: Transport, link: LinkId, bytes: []const u8) Error!void {
        return t.vtable.send(t.ptr, link, bytes);
    }
    pub fn recv(t: Transport, buf: []u8) ?Received {
        return t.vtable.recv(t.ptr, buf);
    }
    pub fn links(t: Transport, out: []Link) usize {
        return t.vtable.links(t.ptr, out);
    }
    pub fn limit(t: Transport) usize {
        return t.vtable.limit(t.ptr);
    }
};
