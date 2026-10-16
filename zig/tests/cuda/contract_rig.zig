//! The contract's CUDA buffers: inputs uploaded, outputs poisoned, every payload between guard bytes.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");

pub const guard = 256;
const pattern = 0xa5;

const Kind = enum { input, output, scratch };
const Slot = struct { buf: cuda.DeviceBuffer, len: usize, kind: Kind };

/// One call's device buffers on one stream: each payload its own allocation, guarded on both sides.
pub const Rig = struct {
    gpu: check.Gpu,
    s: cuda.Stream,
    slots: std.ArrayList(Slot) = .empty,
    seen: std.ArrayList(u8) = .empty,
    outs: std.ArrayList(u8) = .empty,
    host: std.ArrayList(u8) = .empty,

    pub fn deinit(r: *Rig) void {
        r.reset();
        r.slots.deinit(r.gpu.gpa);
        r.seen.deinit(r.gpu.gpa);
        r.outs.deinit(r.gpu.gpa);
        r.host.deinit(r.gpu.gpa);
    }

    pub fn reset(r: *Rig) void {
        r.s.synchronize() catch {};
        for (r.slots.items) |*s| s.buf.free();
        r.slots.clearRetainingCapacity();
    }

    fn slot(r: *Rig, data: ?[]const u8, len: usize, fill: u8, kind: Kind) !u64 {
        const a = r.gpu.gpa;
        const all = try a.alloc(u8, len + 2 * guard);
        defer a.free(all);
        @memset(all, pattern);
        if (data) |d| @memcpy(all[guard..][0..len], d) else @memset(all[guard..][0..len], fill);
        var buf = try cuda.DeviceBuffer.fromHost(r.gpu.d, all);
        errdefer buf.free();
        try r.slots.append(a, .{ .buf = buf, .len = len, .kind = kind });
        return buf.ptr + guard;
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

    /// `len` bytes at a device address of this rig's, downloaded after the stream drains.
    pub fn bytes(r: *Rig, ptr: u64, len: usize) ![]u8 {
        try r.s.synchronize();
        try r.host.resize(r.gpu.gpa, len);
        const b: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = ptr, .len = len };
        try b.download(0, r.host.items);
        return r.host.items;
    }

    /// Zeroes an output's payload before a launch that adds into it.
    pub fn zero(r: *Rig, ptr: u64, len: usize) !void {
        const b: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = ptr, .len = len };
        try b.fill8(0, r.s.handle);
    }

    fn gather(r: *Rig, kind: Kind, out: *std.ArrayList(u8), whole: bool) ![]const u8 {
        try r.s.synchronize();
        out.clearRetainingCapacity();
        for (r.slots.items) |s| if (s.kind == kind or whole) {
            const at = out.items.len;
            const n = if (whole) s.len + 2 * guard else s.len;
            try out.resize(r.gpu.gpa, at + n);
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
            if (s.buf.ptr == t.buf.ptr and s.kind != t.kind) return true;
        };
        return false;
    }
};
