const std = @import("std");
const mx = @import("mlx.zig");

fn retained(array: mx.Array) !mx.Array {
    return if (array.ctx == null) mx.empty else mx.retain(array);
}

fn arrayBytes(value: anytype) u64 {
    const T = @TypeOf(value);
    if (T == mx.Array) return if (value.ctx == null) 0 else mx.c.mlx_array_nbytes(value);
    if (T == @import("kv_buffer.zig").Write) return 0;
    var total: u64 = 0;
    switch (@typeInfo(T)) {
        .@"struct" => inline for (comptime std.meta.fieldNames(T)) |name| {
            total +|= arrayBytes(@field(value, name));
        },
        .array => for (value) |element| {
            total +|= arrayBytes(element);
        },
        else => {},
    }
    return total;
}

fn cacheBytes(cache: anytype) u64 {
    const T = @TypeOf(cache);
    var total: u64 = 0;
    inline for (comptime std.meta.fieldNames(T)) |name| {
        const view = comptime if (std.mem.eql(u8, name, "a")) "keys" else if (std.mem.eql(u8, name, "b")) "values" else if (std.mem.eql(u8, name, "raw")) "index_keys" else "";
        const borrowed_view = if (comptime @hasField(T, view) and @FieldType(T, view) == @import("kv_buffer.zig").Buffer) @field(cache, view).current.ctx != null else false;
        if (!borrowed_view) total +|= arrayBytes(@field(cache, name));
    }
    return total;
}

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
    fn clone(s: DFlash) !DFlash {
        var out = try DFlash.init(&s);
        errdefer out.deinit();
        out.position = s.position;
        out.projected_position = s.projected_position;
        out.started = s.started;
        out.pending = try retained(s.pending);
        for (s.cache, out.cache) |cache, *copy| {
            copy.keys = try retained(cache.keys);
            copy.values = try retained(cache.values);
        }
        return out;
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
    fn clone(s: DSpark) !DSpark {
        var out = try DSpark.init(&s);
        errdefer out.deinit();
        out.position = s.position;
        for (s.keys, out.keys) |key, *copy| copy.* = try retained(key);
        return out;
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
        draft_hidden: mx.Array = mx.empty,
        head_cache: if (@hasDecl(M, "DraftCache")) M.DraftCache else void = if (@hasDecl(M, "DraftCache")) .{} else {},
        dflash_cache: [5]@import("model.zig").Cache = @splat(.{}),
        dflash_offset: i32 = 0,
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
            mx.free(s.draft_hidden);
            if (@hasDecl(M, "DraftCache")) s.head_cache.deinit();
            for (&s.dflash_cache) |*cache| cache.deinit();
            if (s.draft) |*draft| draft.deinit();
            if (s.dspark) |*draft| draft.deinit();
            s.* = undefined;
        }

        pub fn clone(s: *const Self) !Self {
            const cache = try mx.allocator.alloc(Cache, s.cache.len);
            @memset(cache, .{});
            var out = Self{ .cache = cache, .position = s.position, .rope_delta = s.rope_delta, .generation = s.generation, .mtp_position = s.mtp_position, .mtp_generation = s.mtp_generation };
            errdefer out.deinit();
            out.draft_hidden = try retained(s.draft_hidden);
            if (@hasDecl(M, "DraftCache")) out.head_cache = try s.head_cache.clone();
            out.dflash_offset = s.dflash_offset;
            for (s.dflash_cache, &out.dflash_cache) |source, *copy| copy.* = try source.clone();
            for (s.cache, out.cache) |source, *copy| copy.* = try source.clone();
            if (@hasField(M, "mtp_cache")) out.mtp_cache = try s.mtp_cache.clone();
            if (s.draft) |draft| out.draft = try draft.clone();
            if (s.dspark) |draft| out.dspark = try draft.clone();
            return out;
        }

        pub fn nbytes(s: *const Self) u64 {
            var total: u64 = 0;
            for (s.cache) |cache| total +|= cacheBytes(cache);
            if (@hasField(M, "mtp_cache")) total +|= cacheBytes(s.mtp_cache);
            total +|= arrayBytes(s.draft_hidden);
            if (@hasDecl(M, "DraftCache")) total +|= cacheBytes(s.head_cache);
            for (s.dflash_cache) |cache| total +|= cacheBytes(cache);
            if (s.draft) |draft| {
                for (draft.cache) |cache| total +|= cacheBytes(cache);
                total +|= arrayBytes(draft.pending);
            }
            if (s.dspark) |draft| for (draft.keys) |key| {
                total +|= arrayBytes(key);
            };
            return total;
        }

        /// Every pass must be committed or destroyed before switching requests.
        pub fn swapDFlash(s: *Self, d: *@import("drafter.zig").Drafter) void {
            std.mem.swap(@TypeOf(s.dflash_cache), &s.dflash_cache, &d.cache);
            std.mem.swap(i32, &s.dflash_offset, &d.offset);
        }

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
