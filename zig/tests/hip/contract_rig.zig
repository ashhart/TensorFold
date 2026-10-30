//! The contract's HIP buffers: inputs uploaded, outputs poisoned, every payload between guard bytes.
const std = @import("std");
const hip = @import("hip");

pub const guard = 256;
const pattern = 0xa5;

const Kind = enum { input, output, scratch };
const Slot = struct { buf: hip.DeviceBuffer, len: usize, kind: Kind };

/// One call's device buffers on one stream: each payload its own allocation, guarded on both sides.
pub const Rig = struct {
    a: std.mem.Allocator,
    r: *const hip.Runtime,
    s: hip.Stream,
    slots: std.ArrayList(Slot) = .empty,
    seen: std.ArrayList(u8) = .empty,
    outs: std.ArrayList(u8) = .empty,
    host: std.ArrayList(u8) = .empty,

    pub fn deinit(r: *Rig) void {
        r.reset();
        r.slots.deinit(r.a);
        r.seen.deinit(r.a);
        r.outs.deinit(r.a);
        r.host.deinit(r.a);
    }

    pub fn reset(r: *Rig) void {
        r.s.synchronize() catch {};
        for (r.slots.items) |*s| s.buf.free();
        r.slots.clearRetainingCapacity();
    }

    fn slot(r: *Rig, data: ?[]const u8, len: usize, fill: u8, kind: Kind) !u64 {
        const all = try r.a.alloc(u8, len + 2 * guard);
        defer r.a.free(all);
        @memset(all, pattern);
        if (data) |d| @memcpy(all[guard..][0..len], d) else @memset(all[guard..][0..len], fill);
        var buf = try hip.DeviceBuffer.fromHost(r.r, all);
        errdefer buf.free();
        try r.slots.append(r.a, .{ .buf = buf, .len = len, .kind = kind });
        return buf.base() + guard;
    }

    pub fn input(r: *Rig, data: []const u8) !u64 {
        return r.slot(data, data.len, 0, .input);
    }

    /// An output of `len` bytes, all 0xff: an element no launch writes reads as NaN.
    pub fn output(r: *Rig, len: usize) !u64 {
        return r.slot(null, len, 0xff, .output);
    }

    /// Working memory between a call's launches: neither an input nor an output.
    pub fn scratch(r: *Rig, len: usize) !u64 {
        return r.slot(null, len, 0, .scratch);
    }

    /// The slot a device address of this rig's lies in, and the offset of the address in it.
    fn find(r: *Rig, ptr: u64) !struct { hip.DeviceBuffer, usize } {
        for (r.slots.items) |s| {
            if (ptr >= s.buf.base() and ptr < s.buf.base() + s.buf.len) return .{ s.buf, ptr - s.buf.base() };
        }
        return error.Invalid;
    }

    /// `len` bytes at a device address of this rig's, downloaded after the stream drains.
    pub fn bytes(r: *Rig, ptr: u64, len: usize) ![]u8 {
        try r.s.synchronize();
        try r.host.resize(r.a, len);
        const b, const off = try r.find(ptr);
        try b.download(off, r.host.items);
        return r.host.items;
    }

    /// Writes `data` at a device address of this rig's (a run restoring a state it updates in place).
    pub fn put(r: *Rig, ptr: u64, data: []const u8) !void {
        try r.s.synchronize();
        const b, const off = try r.find(ptr);
        try b.upload(off, data);
    }

    fn gather(r: *Rig, kind: Kind, out: *std.ArrayList(u8), whole: bool) ![]const u8 {
        try r.s.synchronize();
        out.clearRetainingCapacity();
        for (r.slots.items) |s| if (s.kind == kind or whole) {
            const at = out.items.len;
            const n = if (whole) s.len + 2 * guard else s.len;
            try out.resize(r.a, at + n);
            try s.buf.download(if (whole) 0 else guard, out.items[at..][0..n]);
        };
        return out.items;
    }

    pub fn outputBytes(r: *Rig) ![]const u8 {
        return r.gather(.output, &r.outs, false);
    }

    pub fn inputBytes(r: *Rig) ![]const u8 {
        return r.gather(.input, &r.seen, false);
    }

    /// Every guard of every buffer still holds its pattern.
    pub fn intact(r: *Rig) !bool {
        const all = try r.gather(.input, &r.seen, true);
        var at: usize = 0;
        for (r.slots.items) |s| {
            for (all[at..][0..guard]) |v| if (v != pattern) return false;
            for (all[at + guard + s.len ..][0..guard]) |v| if (v != pattern) return false;
            at += s.len + 2 * guard;
        }
        return true;
    }

    /// Each payload is its own allocation, so an overlap means one buffer bound as both input and output.
    pub fn overlap(r: *Rig) bool {
        for (r.slots.items, 0..) |s, i| for (r.slots.items[i + 1 ..]) |t| {
            if (s.buf.base() == t.buf.base() and s.kind != t.kind) return true;
        };
        return false;
    }
};
