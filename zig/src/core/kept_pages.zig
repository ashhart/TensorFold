//! Immutable snapshot pages: shared full prefixes, private tails, and the core pool's reference counts.
const std = @import("std");
const kv = @import("kv_pool.zig");

pub const Image = struct {
    ids: []u32,
    at: u32,
    fresh: usize,
};

pub const Store = struct {
    gpa: std.mem.Allocator,
    pool: kv.Pool,
    tokens: u32,
    refs: ?Refs = null,
    const Refs = struct {
        ptr: *anyopaque,
        retain: *const fn (*anyopaque, u32) void,
        release: *const fn (*anyopaque, u32) void,
        holders: *const fn (*anyopaque, u32) u32,
    };

    pub fn init(gpa: std.mem.Allocator, capacity: usize, bytes: u64, tokens: u32) !Store {
        if (tokens == 0) return error.PageTokens;
        return .{ .gpa = gpa, .pool = try kv.Pool.init(gpa, capacity, bytes), .tokens = tokens };
    }

    /// Bind the prompt cache's Pages interface without coupling this pool to a particular engine module.
    pub fn bind(s: *Store, api: anytype) void {
        s.refs = .{ .ptr = api.ptr, .retain = api.vtable.retain, .release = api.vtable.release, .holders = api.vtable.holders };
    }

    fn references(s: *Store) Refs {
        const V = struct {
            fn of(p: *anyopaque) *kv.Pool {
                return @ptrCast(@alignCast(p));
            }
            fn retain(p: *anyopaque, id: u32) void {
                of(p).retain(id);
            }
            fn release(p: *anyopaque, id: u32) void {
                of(p).release(id);
            }
            fn holders(p: *anyopaque, id: u32) u32 {
                return of(p).holders(id);
            }
        };
        return s.refs orelse .{ .ptr = &s.pool, .retain = V.retain, .release = V.release, .holders = V.holders };
    }

    pub fn count(s: *const Store, at: u32) usize {
        return std.math.divCeil(usize, at, s.tokens) catch unreachable;
    }

    pub fn shared(s: *const Store, old: ?Image, at: u32) usize {
        const p = old orelse return 0;
        return @min(p.at, at) / s.tokens;
    }

    pub fn added(s: *const Store, old: ?Image, at: u32) u64 {
        return (s.count(at) - s.shared(old, at)) * s.pool.page_bytes;
    }

    pub fn keep(s: *Store, old: ?Image, at: u32) !Image {
        const n = s.count(at);
        const shared_ = s.shared(old, at);
        const ids = try s.gpa.alloc(u32, n);
        errdefer s.gpa.free(ids);
        var lease: kv.Lease = .{};
        defer s.pool.finish(&lease);
        try s.pool.promise(&lease, n - shared_);
        try s.pool.cover(&lease, n - shared_, null);
        if (old) |p| @memcpy(ids[0..shared_], p.ids[0..shared_]);
        @memcpy(ids[shared_..], lease.ids.items);
        const api = s.references();
        for (ids) |id| api.retain(api.ptr, id);
        return .{ .ids = ids, .at = at, .fresh = shared_ };
    }

    pub fn drop(s: *Store, image: Image) void {
        const api = s.references();
        for (image.ids) |id| api.release(api.ptr, id);
        s.gpa.free(image.ids);
    }

    /// Bytes only this state holds: siblings' common pages are counted once by sharedBytes.
    pub fn exclusive(s: *Store, image: Image) u64 {
        const api = s.references();
        var bytes: u64 = 0;
        for (image.ids) |id| if (api.holders(api.ptr, id) == 1) {
            bytes += s.pool.page_bytes;
        };
        return bytes;
    }

    pub fn sharedBytes(s: *const Store) u64 {
        var n: u64 = 0;
        for (s.pool.refs) |refs| if (refs > 1) {
            n += s.pool.page_bytes;
        };
        return n;
    }

    pub fn deinit(s: *Store) void {
        std.debug.assert(s.pool.available() == s.pool.total());
        s.pool.deinit();
    }
};

/// The legacy image is fixed state, head-major KV, then private drafter taps.
pub const Layout = struct {
    fixed: usize,
    groups: usize,
    row: usize,
    tokens: usize,
    pub const Span = struct { page: ?usize, offset: usize, len: usize, output: usize };

    pub fn pageBytes(l: Layout) usize {
        return l.groups * l.row * l.tokens;
    }

    pub fn walk(l: Layout, at: u32, off: usize, len: usize, ctx: anytype, comptime emit: anytype) !void {
        const kv_end = l.fixed + l.groups * @as(usize, at) * l.row;
        var pos = off;
        const end = try std.math.add(usize, off, len);
        while (pos < end) {
            var span: Span = .{ .page = null, .offset = pos, .len = end - pos, .output = pos - off };
            if (pos < l.fixed) {
                span.len = @min(span.len, l.fixed - pos);
            } else if (pos >= kv_end) {
                span.offset = l.fixed + pos - kv_end;
            } else {
                const within = (pos - l.fixed) % (@as(usize, at) * l.row);
                const group = (pos - l.fixed) / (@as(usize, at) * l.row);
                const token = within / l.row;
                span.page = token / l.tokens;
                span.offset = group * l.tokens * l.row + within % (l.tokens * l.row);
                span.len = @min(span.len, @min((l.tokens - token % l.tokens) * l.row - within % l.row, @as(usize, at) * l.row - within));
            }
            try emit(ctx, span);
            pos += span.len;
        }
    }
};
