//! Token handle ring, async flight metadata, and in-flight command-buffer completion tracking.
const std = @import("std");
const builtin = @import("builtin");
const mtl = @import("metal");
const st = @import("state.zig");
const ap = @import("attention_plan.zig");
const attention = @import("attention.zig");

pub const ring = 4096;
pub const max_flights = 32;
pub const Flight = struct { end: u64, cb: mtl.CommandBuffer };
pub const FlightMeta = extern struct { pos: i32, seg_of: i32, seg: ap.Seg };

pub fn RingOf(comptime Mtl: type) type {
    return struct {
        const Self = @This();
        pub const Entry = if (Mtl == mtl) Flight else struct { end: u64, cb: Mtl.CommandBuffer };

        gpa: std.mem.Allocator,
        tokens: Mtl.Buffer,
        meta: Mtl.Buffer,
        next: u64 = 0,
        landed: u64 = 0,
        failed: bool = false,
        flights: std.ArrayList(Entry) = .empty,

        pub fn init(gpa: std.mem.Allocator, device: Mtl.Device) !Self {
            const tokens = try device.buffer(ring * 4, Mtl.ResourceOptions.shared);
            errdefer tokens.deinit();
            const meta = try device.buffer(ring * @sizeOf(FlightMeta), Mtl.ResourceOptions.shared);
            return .{ .gpa = gpa, .tokens = tokens, .meta = meta };
        }

        /// Waits for and releases every outstanding flight once, without advancing `landed` after a failure.
        pub fn cleanup(r: *Self) void {
            while (r.flights.items.len > 0) _ = r.retire();
        }

        pub fn deinit(r: *Self) void {
            r.cleanup();
            r.flights.deinit(r.gpa);
            r.meta.deinit();
            r.tokens.deinit();
        }

        fn retire(r: *Self) bool {
            const f = r.flights.orderedRemove(0);
            defer f.cb.release();
            f.cb.wait();
            if (f.cb.failure()) |msg| {
                if (!builtin.is_test) std.log.err("inkling2_loop step: {s}", .{msg});
                r.failed = true;
                return false;
            }
            if (!r.failed) r.landed = @max(r.landed, f.end);
            return !r.failed;
        }

        pub fn check(r: *const Self) !void {
            if (r.failed) return error.GpuFailed;
        }

        pub fn land(r: *Self) !void {
            try r.check();
            if (!r.retire()) return error.GpuFailed;
        }

        pub fn drain(r: *Self) !void {
            r.cleanup();
            try r.check();
            r.landed = r.next;
        }

        /// Ensures room for `count` new slots and one flight entry without advancing `next`.
        pub fn prepare(r: *Self, count: u64) !void {
            try r.check();
            while ((r.next - r.landed + count > ring or r.flights.items.len >= max_flights) and r.flights.items.len > 0) try r.land();
            try r.check();
            try r.flights.ensureUnusedCapacity(r.gpa, 1);
        }

        pub fn take(r: *Self) !u64 {
            try r.check();
            while ((r.next - r.landed >= ring or r.flights.items.len >= max_flights) and r.flights.items.len > 0) try r.land();
            try r.check();
            const h = r.next;
            r.next += 1;
            return h;
        }

        pub fn takeWrite(r: *Self, token: u32) !u64 {
            const h = try r.take();
            r.write(h, token);
            return h;
        }

        /// Runs a synchronous GPU draw into the next slot after draining; failure latches without landing.
        pub fn drawSync(r: *Self, ctx: anytype, comptime run: anytype) !u64 {
            try r.drain();
            const h = r.next;
            run(ctx, h) catch |err| {
                r.failed = true;
                return err;
            };
            r.next = h + 1;
            r.landed = r.next;
            return h;
        }

        pub fn slotOf(r: *const Self, h: u64) st.Ref {
            return .{ .buf = r.tokens, .off = @intCast((h % ring) * 4) };
        }

        pub fn setMeta(r: *Self, h: u64, cache_len: usize) attention.RowMeta {
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

        pub fn write(r: *Self, h: u64, token: u32) void {
            r.tokens.slice(u32, ring)[@intCast(h % ring)] = token;
        }

        pub fn put(r: *Self, token: u32) !u64 {
            try r.drain();
            const h = try r.take();
            r.write(h, token);
            r.landed = r.next;
            return h;
        }

        pub fn read(r: *Self, handle: u64, vocab: usize) !u32 {
            try r.check();
            if (handle >= r.next or r.next - handle > ring) return error.InvalidInklingTokenHandle;
            while (handle >= r.landed and r.flights.items.len > 0) try r.land();
            if (r.failed or handle >= r.landed) return error.GpuFailed;
            const token = r.tokens.slice(u32, ring)[@intCast(handle % ring)];
            if (token >= vocab) return error.InvalidInklingToken;
            return token;
        }

        pub fn slice(r: *const Self) []const u32 {
            return r.tokens.slice(u32, ring);
        }
    };
}

pub const Ring = RingOf(mtl);

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
