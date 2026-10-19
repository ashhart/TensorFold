//! TF_DSV41_FORMS_INPLACE=1: the dense forms (s.<w>.lanes / .T) written over their trellis, as dense3.to_lanes does.

const std = @import("std");
const buffers = @import("buffers.zig");
const prepare = @import("prepare.zig");

const log = std.log.scoped(.dsv41);

pub fn enabled() bool {
    const v = std.c.getenv("TF_DSV41_FORMS_INPLACE") orelse return false;
    return std.mem.eql(u8, std.mem.trim(u8, std.mem.span(v), " "), "1");
}

pub const Bound = struct { roles: u32 = 0, bytes: u64 = 0 };

/// The source weight of a dense form role (`<w>` of "s.<w>.lanes" / "s.<w>.T"), null for any other role.
pub fn sourceOf(role: []const u8) ?[]const u8 {
    if (!std.mem.startsWith(u8, role, "s.")) return null;
    for ([_][]const u8{ ".lanes", ".T" }) |suf| if (std.mem.endsWith(u8, role, suf)) return role[2 .. role.len - suf.len];
    return null;
}

/// Binds `plans[0]`'s forms (r: run.Runner, weights: load.Weights) unless read, a source's two, or too large.
pub fn bind(gpa: std.mem.Allocator, r: anytype, weights: anytype, plans: []const *const buffers.Plan) !Bound {
    if (plans.len == 0) return error.EmptyPlans;
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    defer seen.deinit(gpa);
    // Count distinct roles across all plans; repeated row buckets do not create a second form.
    var forms: std.StringHashMapUnmanaged(u32) = .empty;
    defer forms.deinit(gpa);
    for (plans) |p| for (p.sizes.keys()) |role| if (sourceOf(role)) |w| {
        const role_entry = try seen.getOrPut(gpa, role);
        if (role_entry.found_existing) continue;
        const g = try forms.getOrPut(gpa, w);
        g.value_ptr.* = if (g.found_existing) g.value_ptr.* + 1 else 1;
    };
    var out: Bound = .{};
    var name: [192]u8 = undefined;
    for (plans[0].sizes.keys(), plans[0].sizes.values()) |role, size| {
        const w = sourceOf(role) orelse continue;
        if (forms.get(w).? != 1 or r.addressOf(role) != null) continue;
        const tr = std.fmt.bufPrint(&name, "{s}.trellis", .{w}) catch continue;
        const t = weights.get(tr) orelse continue;
        if (size > t.len) continue;
        const read = for (plans) |p| {
            if (p.weights.contains(tr)) break true;
        } else false;
        if (read) continue;
        try r.external(role, t.ptr);
        out.roles += 1;
        out.bytes += t.len;
    }
    log.info("forms in place (TF_DSV41_FORMS_INPLACE): {d} dense forms over their trellis, {d:.2} GiB not allocated", .{ out.roles, @as(f64, @floatFromInt(out.bytes)) / (1 << 30) });
    return out;
}

const testing = std.testing;

/// The weights' and runner's parts `bind` uses (load.Weights.get, run.Runner.addressOf / external).
const Span = struct { ptr: u64, len: usize };
const Weights = struct {
    map: std.StringHashMapUnmanaged(Span) = .empty,
    fn get(w: *const Weights, name: []const u8) ?Span {
        return w.map.get(name);
    }
};
const Roles = struct {
    gpa: std.mem.Allocator,
    addrs: std.StringHashMapUnmanaged(u64) = .empty,
    fn addressOf(r: *const Roles, role: []const u8) ?u64 {
        return r.addrs.get(role);
    }
    fn external(r: *Roles, role: []const u8, addr: u64) !void {
        try r.addrs.put(r.gpa, role, addr);
    }
};

test "forms in place: each dense form bound to its own trellis; read, shared, larger or other roles keep their own" {
    const gpa = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var w: Weights = .{};
    defer w.map.deinit(gpa);
    const ws = [_]struct { []const u8, u64, usize }{
        .{ "L0.attn.wq_b.trellis", 0x10000, 4096 },
        .{ "head.trellis", 0x20000, 1024 },
        .{ "L1.attn.wo_b.trellis", 0x30000, 512 },
        .{ "L2.attn.wq_a.trellis", 0x40000, 256 },
        .{ "L3.attn.wkv.trellis", 0x50000, 256 },
    };
    for (ws) |x| try w.map.put(gpa, x[0], .{ .ptr = x[1], .len = x[2] });
    var p: buffers.Plan = .{ .a = a };
    for ([_]struct { []const u8, u64 }{
        .{ "s.L0.attn.wq_b.lanes", 4096 }, // bound
        .{ "s.head.T", 1000 }, // bound (smaller than its trellis)
        .{ "s.L1.attn.wo_b.lanes", 1024 }, // larger than its trellis: own buffer
        .{ "s.L2.attn.wq_a.lanes", 256 }, // its trellis is read by a 4K program: own buffer
        .{ "s.L3.attn.wkv.lanes", 256 }, // two forms of one source: both their own
        .{ "s.L3.attn.wkv.T", 256 },
        .{ "s.L4.attn.wq_b.lanes", 64 }, // no trellis on this rank: own buffer
        .{ "s.L0.hc_attn.fn16", 64 }, // not a dense form
        .{ "w.x", 64 },
    }) |x| try p.sizes.put(a, x[0], x[1]);
    var k4: buffers.Plan = .{ .a = a };
    try k4.weights.put(a, "L2.attn.wq_a.trellis", {});
    var r: Roles = .{ .gpa = gpa };
    defer r.addrs.deinit(gpa);
    const got = try bind(gpa, &r, &w, &.{ &p, &k4 });
    try testing.expectEqual(@as(u32, 2), got.roles);
    try testing.expectEqual(@as(u64, 4096 + 1024), got.bytes);
    try testing.expectEqual(@as(?u64, 0x10000), r.addressOf("s.L0.attn.wq_b.lanes"));
    try testing.expectEqual(@as(?u64, 0x20000), r.addressOf("s.head.T"));
    for ([_][]const u8{ "s.L1.attn.wo_b.lanes", "s.L2.attn.wq_a.lanes", "s.L3.attn.wkv.lanes", "s.L3.attn.wkv.T", "s.L4.attn.wq_b.lanes", "s.L0.hc_attn.fn16", "w.x" }) |role|
        try testing.expectEqual(@as(?u64, null), r.addressOf(role));
    try testing.expectEqualStrings("L0.attn.wq_b", sourceOf("s.L0.attn.wq_b.lanes").?);
    try testing.expectEqual(@as(?[]const u8, null), sourceOf("L.x.lanes"));
}

test "forms in place: strips and lanes are the trellis's own length, so a form fits over it" {
    const gpa = testing.allocator;
    // kt 2 K tiles, nt 8 N tiles, 32 int16 words a tile (K2 4): one strip, one lanes group pair
    const src = try gpa.alloc(u8, 2 * 8 * 32 * 2);
    defer gpa.free(src);
    for (src, 0..) |*b, i| b.* = @truncate(i *% 37 +% 11);
    const st = try prepare.strips(gpa, src, 2, 8, 32);
    defer gpa.free(st);
    const ln = try prepare.lanes(gpa, st, 2, 4);
    defer gpa.free(ln);
    try testing.expectEqual(src.len, st.len);
    try testing.expectEqual(src.len, ln.len);
}

test "forms across all plans preserve the original source for both conversions" {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var first: buffers.Plan = .{ .a = arena.allocator() };
    var second: buffers.Plan = .{ .a = arena.allocator() };
    try first.sizes.put(first.a, "s.L0.attn.wq_b.lanes", 1024);
    try second.sizes.put(second.a, "s.L0.attn.wq_b.T", 1024);
    var weights: Weights = .{};
    defer weights.map.deinit(a);
    var source: [1024]u8 = undefined;
    for (&source, 0..) |*v, i| v.* = @truncate(i *% 37 +% 11);
    const original = source;
    try weights.map.put(a, "L0.attn.wq_b.trellis", .{ .ptr = @intFromPtr(&source), .len = source.len });
    var roles: Roles = .{ .gpa = a };
    defer roles.addrs.deinit(a);
    const bound = try bind(a, &roles, &weights, &.{ &first, &second });
    try testing.expectEqual(@as(u32, 0), bound.roles);
    const strip = try prepare.strips(a, &source, 2, 8, 32);
    defer a.free(strip);
    const lanes = try prepare.lanes(a, strip, 2, 4);
    defer a.free(lanes);
    if (roles.addressOf("s.L0.attn.wq_b.lanes") != null) @memcpy(&source, lanes);
    try testing.expectEqualSlices(u8, &original, &source);
    const second_strip = try prepare.strips(a, &source, 2, 8, 32);
    defer a.free(second_strip);
    try testing.expectEqualSlices(u8, strip, second_strip);
    try testing.expectEqual(@as(u32, 1), (try bind(a, &roles, &weights, &.{ &first, &first })).roles);
    try testing.expectError(error.EmptyPlans, bind(a, &roles, &weights, &.{}));
}
