//! Bounded resident-weight views for the optional decode prefetch. No scratch or mapped host addresses.
const std = @import("std");
const cuda = @import("root.zig");
const pf = @import("l2_prefetch.zig");
pub const Range = struct { ptr: u64, len: usize };

pub fn Settings(comptime prefix: [:0]const u8) type {
    return struct {
        on: bool = false,
        knobs: pf.Knobs = .{},
        pub fn read(get: *const fn ([:0]const u8) ?[]const u8, problem: *[]const u8) !@This() {
            var o: @This() = .{};
            const value = get(prefix) orelse return o;
            if (std.mem.eql(u8, value, "0")) return o;
            if (!std.mem.eql(u8, value, "1")) {
                problem.* = prefix ++ ": 0 or 1";
                return error.BadSetting;
            }
            o.on = true;
            const knobs = .{ .{ "_MB", "mb", 1 }, .{ "_CTAS", "ctas", 1 }, .{ "_PACE_GBPS", "pace_gbps", 0 } };
            inline for (knobs) |k| if (get(prefix ++ k[0])) |v| {
                const n = std.fmt.parseInt(u32, v, 10) catch null;
                if (n == null or n.? < k[2]) {
                    problem.* = prefix ++ k[0];
                    return error.BadSetting;
                }
                @field(o.knobs, k[1]) = n.?;
            };
            return o;
        }
    };
}

pub const Sources = struct {
    ptrs: [32]u64 = undefined,
    len: usize = 0,
    pub fn add(s: *Sources, p: u64) void {
        if (p == 0) return;
        for (s.ptrs[0..s.len]) |old| if (old == p) return;
        std.debug.assert(s.len < s.ptrs.len);
        s.ptrs[s.len] = p;
        s.len += 1;
    }
    pub fn affine(s: *Sources, q: anytype) void {
        s.add(q.s);
        s.add(q.b);
        s.add(q.w);
    }
    pub fn lane(s: *Sources, q: cuda.qmmf.Weight) void {
        s.add(q.scales);
        s.add(q.codes);
    }
    pub fn exl3(s: *Sources, q: cuda.exl3.Weight) void {
        s.add(q.suh);
        s.add(q.svh);
        s.add(q.bias);
        s.add(q.trellis);
    }
    pub fn linear(s: *Sources, q: cuda.qlinear.Weight) void {
        switch (q) {
            .affine4 => |a| s.affine(a),
            .exl3 => |a| s.exl3(a),
            .fp8g, .mxfp8, .nvfp4, .fp8, .bf16 => |a| s.lane(a),
            .q8_0, .iq2xxs => |a| s.add(a.w),
        }
    }
};

/// A source may be an interior packed view. Clip it to its actual allocation and the next selected view.
pub fn append(gpa: std.mem.Allocator, src: Sources, ranges: []const Range, budget: u64, rows: *std.ArrayList([2]i64), sp: *pf.Span) !void {
    sp.* = .{ .at = @intCast(rows.items.len) };
    var sizes: [32]u64 = undefined;
    for (src.ptrs[0..src.len], sizes[0..src.len]) |p, *size| {
        size.* = for (ranges) |b| {
            if (p >= b.ptr and p - b.ptr < b.len) break b.len - (p - b.ptr);
        } else return error.PrefetchWeightRange;
        for (src.ptrs[0..src.len]) |next| if (next > p) {
            size.* = @min(size.*, next - p);
        };
    }
    var got: [32]u64 = undefined;
    pf.take(sizes[0..src.len], budget, got[0..src.len]);
    for (src.ptrs[0..src.len], got[0..src.len]) |p, bytes| if (bytes > 0) {
        if (p % 16 != 0) return error.PrefetchWeightAlignment;
        try rows.append(gpa, .{ @intCast(p), @intCast(bytes) });
        sp.rows += 1;
        sp.bytes += bytes;
    };
}

/// OFF returns before touching views, the allocator or the driver.
pub fn start(gpa: std.mem.Allocator, d: *const cuda.Driver, on: bool, knobs: pf.Knobs, views: []const Sources, ranges: []const Range) !?pf.Runtime {
    if (!on) return null;
    const spans = try gpa.alloc(pf.Span, views.len);
    errdefer gpa.free(spans);
    var rows: std.ArrayList([2]i64) = .empty;
    defer rows.deinit(gpa);
    for (views, spans) |src, *sp| try append(gpa, src, ranges, @as(u64, knobs.mb) << 20, &rows, sp);
    if (rows.items.len == 0) return error.NoPrefetchRows;
    return try pf.Runtime.initRows(gpa, d, knobs, rows.items, spans);
}
