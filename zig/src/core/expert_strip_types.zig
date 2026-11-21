//! Shared immutable identity, source and device-handoff contracts for striped expert reads.
const std = @import("std");

pub const Digest = [32]u8;

pub const Identity = struct {
    tensor: Digest,
    format: Digest,
    checkpoint: Digest,

    pub fn eql(a: Identity, b: Identity) bool {
        return std.mem.eql(u8, &a.tensor, &b.tensor) and
            std.mem.eql(u8, &a.format, &b.format) and
            std.mem.eql(u8, &a.checkpoint, &b.checkpoint);
    }
};

pub const Candidate = struct {
    replica: usize,
    disk: u32,
    offset: u64,
};

pub const Cancellation = struct {
    flag: std.atomic.Value(bool) = .init(false),
    parent: ?*const Cancellation = null,

    pub fn reset(c: *Cancellation, parent: ?*const Cancellation) void {
        c.parent = parent;
        c.flag.store(false, .release);
    }

    pub fn cancel(c: *Cancellation) void {
        c.flag.store(true, .release);
    }

    pub fn isCancelled(c: *const Cancellation) bool {
        var current: ?*const Cancellation = c;
        while (current) |token| {
            if (token.flag.load(.acquire)) return true;
            current = token.parent;
        }
        return false;
    }
};

pub const Host = struct {
    ctx: ?*anyopaque = null,
    artifact: *const fn (ctx: ?*anyopaque, candidate: Candidate) anyerror!Identity,
    read: *const fn (ctx: ?*anyopaque, candidate: Candidate, offset: u64, destination: []u8, cancel: ?*const Cancellation) anyerror!usize,
};

pub const Sink = struct {
    ctx: ?*anyopaque = null,
    stage: ?*const fn (ctx: ?*anyopaque, device: u32, stripe: usize, bytes: []const u8) anyerror!void = null,
    commit: ?*const fn (ctx: ?*anyopaque) anyerror!void = null,
    abort: ?*const fn (ctx: ?*anyopaque) void = null,
};
