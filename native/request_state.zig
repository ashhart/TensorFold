const std = @import("std");
const mx = @import("mlx.zig");

const DFlash = struct {
    cache: []@import("dflash.zig").Cache,
    position: i32 = 0,
    projected_position: i32 = 0,
    pending: mx.Array = mx.empty,
    started: bool = false,

    fn init(d: anytype) !DFlash {
        const cache = try mx.allocator.alloc(@import("dflash.zig").Cache, d.cache.len);
        @memset(cache, .{});
        return .{ .cache = cache };
    }
    fn swap(s: *DFlash, d: anytype) void {
        inline for (.{ "cache", "position", "projected_position", "pending", "started" }) |field| std.mem.swap(@FieldType(DFlash, field), &@field(s, field), &@field(d, field));
    }
    fn deinit(s: *DFlash) void {
        for (s.cache) |cache| {
            mx.free(cache.keys);
            mx.free(cache.values);
        }
        mx.allocator.free(s.cache);
        mx.free(s.pending);
    }
};

const DSpark = struct {
    keys: []mx.Array,
    position: i32 = 0,
    fn init(d: anytype) !DSpark {
        const keys = try mx.allocator.alloc(mx.Array, d.keys.len);
        @memset(keys, mx.empty);
        return .{ .keys = keys };
    }
    fn swap(s: *DSpark, d: anytype) void {
        std.mem.swap([]mx.Array, &s.keys, &d.keys);
        std.mem.swap(i32, &s.position, &d.position);
    }
    fn deinit(s: *DSpark) void {
        for (s.keys) |key| mx.free(key);
        mx.allocator.free(s.keys);
    }
};

/// Owns request caches while model weights, compiled operations and kernels stay shared.
pub fn State(comptime M: type) type {
    const Cache = switch (@typeInfo(@FieldType(M, "cache"))) {
        .array => |info| info.child,
        .pointer => |info| info.child,
        else => @compileError("Expected model cache array or slice"),
    };
    return struct {
        const Self = @This();
        cache: []Cache,
        position: i32 = 0,
        rope_delta: i32 = 0,
        generation: u64 = 0,
        mtp_cache: if (@hasField(M, "mtp_cache")) @FieldType(M, "mtp_cache") else void = if (@hasField(M, "mtp_cache")) .{} else {},
        mtp_position: i32 = 0,
        mtp_generation: u64 = 0,
        draft: ?DFlash = null,
        dspark: ?DSpark = null,

        pub fn init(m: *const M) !Self {
            const cache = try mx.allocator.alloc(Cache, m.cache.len);
            @memset(cache, .{});
            var state = Self{ .cache = cache };
            errdefer state.deinit();
            if (@hasField(M, "draft")) if (m.draft) |d| {
                state.draft = try DFlash.init(d);
            };
            if (@hasField(M, "dspark")) if (m.dspark) |d| {
                state.dspark = try DSpark.init(d);
            };
            return state;
        }

        pub fn deinit(s: *Self) void {
            for (s.cache) |*cache| cache.deinit();
            mx.allocator.free(s.cache);
            if (@hasField(M, "mtp_cache")) s.mtp_cache.deinit();
            if (s.draft) |*draft| draft.deinit();
            if (s.dspark) |*draft| draft.deinit();
            s.* = undefined;
        }

        /// Every pass must be committed or destroyed before switching requests.
        pub fn swap(s: *Self, m: *M) void {
            for (s.cache, m.cache[0..]) |*saved, *active| std.mem.swap(Cache, saved, active);
            inline for (.{ "position", "rope_delta", "generation", "mtp_cache", "mtp_position", "mtp_generation" }) |field| if (@hasField(M, field)) {
                std.mem.swap(@FieldType(M, field), &@field(s, field), &@field(m, field));
            };
            if (@hasField(M, "draft")) if (s.draft) |*draft| draft.swap(&m.draft.?);
            if (@hasField(M, "dspark")) if (s.dspark) |*draft| draft.swap(&m.dspark.?);
        }
    };
}
