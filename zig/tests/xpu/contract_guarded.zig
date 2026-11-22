//! The XPU contract's guarded buffers: inputs, outputs and scratch on one stream, poisoned outputs, intact guards.
const std = @import("std");
const xpu = @import("xpu");

pub const guard = 256;
const pattern = 0xa5;

const Kind = enum { input, output, scratch };
const Slot = struct { buf: xpu.DeviceBuffer, len: usize, kind: Kind };

/// One call's buffers on `Dev`'s stream (its `ctx` and `s`): each payload its own allocation, guarded both sides.
pub fn Rig(comptime Dev: type) type {
    return struct {
        const Self = @This();

        a: std.mem.Allocator,
        dev: *Dev,
        slots: std.ArrayList(Slot) = .empty,
        seen: std.ArrayList(u8) = .empty,
        outs: std.ArrayList(u8) = .empty,
        host: std.ArrayList(u8) = .empty,

        pub fn deinit(r: *Self) void {
            r.reset();
            r.slots.deinit(r.a);
            r.seen.deinit(r.a);
            r.outs.deinit(r.a);
            r.host.deinit(r.a);
        }

        pub fn reset(r: *Self) void {
            r.dev.s.synchronize() catch {};
            for (r.slots.items) |*s| s.buf.free();
            r.slots.clearRetainingCapacity();
        }

        fn slot(r: *Self, data: ?[]const u8, len: usize, fill: u8, kind: Kind) !?*anyopaque {
            const all = try r.a.alloc(u8, len + 2 * guard);
            defer r.a.free(all);
            @memset(all, pattern);
            if (data) |d| @memcpy(all[guard..][0..len], d) else @memset(all[guard..][0..len], fill);
            var buf = try xpu.DeviceBuffer.alloc(&r.dev.ctx, all.len);
            errdefer buf.free();
            try buf.upload(r.dev.s, 0, all);
            try r.slots.append(r.a, .{ .buf = buf, .len = len, .kind = kind });
            return try buf.at(guard);
        }

        pub fn input(r: *Self, data: []const u8) !?*anyopaque {
            return r.slot(data, data.len, 0, .input);
        }

        /// An output of `len` bytes, all 0xff: an element no launch writes reads as NaN.
        pub fn output(r: *Self, len: usize) !?*anyopaque {
            return r.slot(null, len, 0xff, .output);
        }

        /// Working memory between a call's launches: neither an input nor an output.
        pub fn scratch(r: *Self, len: usize) !?*anyopaque {
            return r.slot(null, len, 0, .scratch);
        }

        fn find(r: *Self, p: ?*anyopaque) !struct { xpu.DeviceBuffer, usize } {
            const at = @intFromPtr(p);
            for (r.slots.items) |s| {
                const base = @intFromPtr(s.buf.ptr);
                if (at >= base and at < base + s.buf.len) return .{ s.buf, at - base };
            }
            return error.Invalid;
        }

        /// `len` bytes at a device address of this rig's, downloaded after the stream drains.
        pub fn bytes(r: *Self, p: ?*anyopaque, len: usize) ![]u8 {
            try r.dev.s.synchronize();
            try r.host.resize(r.a, len);
            const b, const off = try r.find(p);
            try b.download(r.dev.s, off, r.host.items);
            return r.host.items;
        }

        /// Writes `data` at a device address of this rig's (a run restoring a state it updates in place).
        pub fn put(r: *Self, p: ?*anyopaque, data: []const u8) !void {
            try r.dev.s.synchronize();
            const b, const off = try r.find(p);
            try b.upload(r.dev.s, off, data);
        }

        /// Sets the arguments in order and queues the kernel; the call is waited for.
        pub fn launch(r: *Self, k: xpu.Kernel, groups: [3]u32, args: anytype) !void {
            try xpu.launch.dispatch(k, r.dev.s, groups, args);
            try r.dev.s.synchronize();
        }

        fn gather(r: *Self, kind: Kind, out: *std.ArrayList(u8), whole: bool) ![]const u8 {
            try r.dev.s.synchronize();
            out.clearRetainingCapacity();
            for (r.slots.items) |s| if (s.kind == kind or whole) {
                const at = out.items.len;
                const n = if (whole) s.len + 2 * guard else s.len;
                try out.resize(r.a, at + n);
                try s.buf.download(r.dev.s, if (whole) 0 else guard, out.items[at..][0..n]);
            };
            return out.items;
        }

        pub fn outputBytes(r: *Self) ![]const u8 {
            return r.gather(.output, &r.outs, false);
        }

        pub fn inputBytes(r: *Self) ![]const u8 {
            return r.gather(.input, &r.seen, false);
        }

        /// Every guard of every buffer still holds its pattern.
        pub fn intact(r: *Self) !bool {
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
        pub fn overlap(r: *Self) bool {
            for (r.slots.items, 0..) |s, i| for (r.slots.items[i + 1 ..]) |t| {
                if (s.buf.ptr == t.buf.ptr and s.kind != t.kind) return true;
            };
            return false;
        }
    };
}
