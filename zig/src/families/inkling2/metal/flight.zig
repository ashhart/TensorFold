//! Token handle ring, async flight metadata, and in-flight command-buffer completion tracking.
const std = @import("std");
const mtl = @import("metal");
const st = @import("state.zig");
const ap = @import("attention_plan.zig");
const attention = @import("attention.zig");

pub const ring = 4096;
const max_flights = 32;
pub const Flight = struct { end: u64, cb: mtl.CommandBuffer };
pub const FlightMeta = extern struct { pos: i32, seg_of: i32, seg: ap.Seg };

pub const Ring = struct {
    gpa: std.mem.Allocator,
    tokens: mtl.Buffer,
    meta: mtl.Buffer,
    next: u64 = 0,
    landed: u64 = 0,
    flights: std.ArrayList(Flight) = .empty,

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device) !Ring {
        const tokens = try device.buffer(ring * 4, mtl.ResourceOptions.shared);
        errdefer tokens.deinit();
        const meta = try device.buffer(ring * @sizeOf(FlightMeta), mtl.ResourceOptions.shared);
        return .{ .gpa = gpa, .tokens = tokens, .meta = meta };
    }

    pub fn deinit(r: *Ring) void {
        r.drain() catch {};
        r.flights.deinit(r.gpa);
        r.meta.deinit();
        r.tokens.deinit();
    }

    pub fn land(r: *Ring) !void {
        const f = r.flights.orderedRemove(0);
        defer f.cb.release();
        f.cb.wait();
        if (f.cb.failure()) |msg| {
            std.log.err("inkling2_loop step: {s}", .{msg});
            return error.GpuFailed;
        }
        r.landed = @max(r.landed, f.end);
    }

    pub fn drain(r: *Ring) !void {
        while (r.flights.items.len > 0) try r.land();
        r.landed = r.next;
    }

    pub fn take(r: *Ring) !u64 {
        while ((r.next - r.landed >= ring or r.flights.items.len >= max_flights) and r.flights.items.len > 0) try r.land();
        const h = r.next;
        r.next += 1;
        return h;
    }

    pub fn slotOf(r: *const Ring, h: u64) st.Ref {
        return .{ .buf = r.tokens, .off = @intCast((h % ring) * 4) };
    }

    pub fn setMeta(r: *Ring, h: u64, cache_len: usize) attention.RowMeta {
        r.meta.slice(FlightMeta, ring)[@intCast(h % ring)] = .{
            .pos = @intCast(cache_len),
            .seg_of = 0,
            .seg = .{ .row0 = 0, .n = 1, .p0 = @intCast(cache_len), .off = 0 },
        };
        const base = @as(usize, @intCast(h % ring)) * @sizeOf(FlightMeta);
        return .{
            .pos = .{ .buf = r.meta, .off = base },
            .seg_of = .{ .buf = r.meta, .off = base + @offsetOf(FlightMeta, "seg_of") },
            .segs = .{ .buf = r.meta, .off = base + @offsetOf(FlightMeta, "seg") },
        };
    }

    pub fn write(r: *Ring, h: u64, token: u32) void {
        r.tokens.slice(u32, ring)[@intCast(h % ring)] = token;
    }

    pub fn put(r: *Ring, token: u32) !u64 {
        const h = try r.take();
        r.write(h, token);
        if (r.flights.items.len == 0) r.landed = r.next;
        return h;
    }

    pub fn read(r: *Ring, handle: u64, vocab: usize) !u32 {
        if (handle >= r.next or r.next - handle > ring) return error.InvalidInklingTokenHandle;
        while (handle >= r.landed and r.flights.items.len > 0) try r.land();
        const token = r.tokens.slice(u32, ring)[@intCast(handle % ring)];
        if (token >= vocab) return error.InvalidInklingToken;
        return token;
    }

    pub fn slice(r: *const Ring) []const u32 {
        return r.tokens.slice(u32, ring);
    }
};

/// The token behind `handle`: only the last `ring` handles drawn are still held.
pub fn ringToken(tokens: []const u32, next: u64, handle: u64) !u32 {
    if (handle >= next or next - handle > ring) return error.InvalidInklingTokenHandle;
    return tokens[@intCast(handle % ring)];
}

test "a token handle expires once the ring has wrapped past it" {
    var tokens: [ring]u32 = @splat(0);
    tokens[0] = 7;
    try std.testing.expectEqual(@as(u32, 7), try ringToken(&tokens, ring, 0));
    try std.testing.expectError(error.InvalidInklingTokenHandle, ringToken(&tokens, ring + 1, 0));
    try std.testing.expectEqual(@as(u32, 7), try ringToken(&tokens, ring + 1, ring));
    try std.testing.expectError(error.InvalidInklingTokenHandle, ringToken(&tokens, ring, ring));
}
