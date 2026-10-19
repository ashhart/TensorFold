//! TF_DSV41_EX_STACKS=1: routed and shared expert suh / svh uploaded back to back, so the forms' stacks are views.

const std = @import("std");
const buffers = @import("buffers.zig");

const log = std.log.scoped(.dsv41);

pub fn enabled() bool {
    const v = std.c.getenv("TF_DSV41_EX_STACKS") orelse return false;
    return std.mem.eql(u8, std.mem.trim(u8, std.mem.span(v), " "), "1");
}

/// The shared expert's tensor for a routed suh / svh ("L3.moe.w1.suh" -> "L3.moe.shared.0.w1.suh"), else null.
pub fn sharedOf(buf: []u8, name: []const u8) ?[]const u8 {
    const at = std.mem.indexOf(u8, name, ".moe.w") orelse return null;
    const rest = name[at + ".moe.".len ..];
    if (rest.len != "w1.suh".len or rest[1] < '1' or rest[1] > '3') return null;
    if (!std.mem.endsWith(u8, rest, ".suh") and !std.mem.endsWith(u8, rest, ".svh")) return null;
    return std.fmt.bufPrint(buf, "{s}.moe.shared.0.{s}", .{ name[0..at], rest }) catch null;
}

/// The routed weight starting a stack role ("s.L3.ex.suh_g" -> "L3.moe.w1.suh"; g w1, u w3, d w2), else null.
pub fn stackWeight(buf: []u8, role: []const u8) ?[]const u8 {
    if (!std.mem.startsWith(u8, role, "s.L")) return null;
    const ex = std.mem.indexOf(u8, role, ".ex.") orelse return null;
    const what = role[ex + 4 ..];
    if (what.len != 5 or what[3] != '_') return null;
    const kind = what[0..3];
    if (!std.mem.eql(u8, kind, "suh") and !std.mem.eql(u8, kind, "svh")) return null;
    const w: []const u8 = switch (what[4]) {
        'g' => "w1",
        'u' => "w3",
        'd' => "w2",
        else => return null,
    };
    return std.fmt.bufPrint(buf, "{s}.moe.{s}.{s}", .{ role[2..ex], w, kind }) catch null;
}

/// Index pairs (routed, shared) of `names` to upload together: both present and the routed one's length 16-aligned.
pub fn pairs(gpa: std.mem.Allocator, names: []const []const u8, lens: []const usize) ![][2]usize {
    var at: std.StringHashMapUnmanaged(usize) = .empty;
    defer at.deinit(gpa);
    for (names, 0..) |n, i| try at.put(gpa, n, i);
    var out: std.ArrayList([2]usize) = .empty;
    errdefer out.deinit(gpa);
    var buf: [192]u8 = undefined;
    for (names, 0..) |n, i| {
        const s = sharedOf(&buf, n) orelse continue;
        const j = at.get(s) orelse continue;
        if (lens[i] % 16 != 0) continue; // the shared half must start where a kernel may read it as a tensor
        try out.append(gpa, .{ i, j });
    }
    return out.toOwnedSlice(gpa);
}

pub const Bound = struct { roles: u32 = 0, bytes: u64 = 0 };

/// Binds `plan`'s stacks to their routed weight where the shared half follows (r: run.Runner, weights: load.Weights).
pub fn bind(r: anytype, weights: anytype, plan: *const buffers.Plan) !Bound {
    var out: Bound = .{};
    var nb: [192]u8 = undefined;
    var sb: [192]u8 = undefined;
    for (plan.sizes.keys(), plan.sizes.values()) |role, size| {
        const wn = stackWeight(&nb, role) orelse continue;
        if (r.addressOf(role) != null) continue;
        const t = weights.get(wn) orelse continue;
        const sn = sharedOf(&sb, wn) orelse continue;
        const s = weights.get(sn) orelse continue;
        if (s.ptr != t.ptr + t.len or size > t.len + s.len) continue;
        try r.external(role, t.ptr);
        out.roles += 1;
        out.bytes += t.len + s.len;
    }
    log.info("expert stacks (TF_DSV41_EX_STACKS): {d} stacks read in place, {d:.2} GiB not allocated", .{ out.roles, @as(f64, @floatFromInt(out.bytes)) / (1 << 30) });
    return out;
}

const testing = std.testing;

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

test "expert stacks: names, pairs, and a stack bound only where its shared half follows the routed one" {
    const gpa = testing.allocator;
    var b: [192]u8 = undefined;
    try testing.expectEqualStrings("L3.moe.shared.0.w1.suh", sharedOf(&b, "L3.moe.w1.suh").?);
    try testing.expectEqualStrings("L40.moe.shared.0.w2.svh", sharedOf(&b, "L40.moe.w2.svh").?);
    try testing.expectEqual(@as(?[]const u8, null), sharedOf(&b, "L3.moe.w1.trellis"));
    try testing.expectEqual(@as(?[]const u8, null), sharedOf(&b, "L3.moe.shared.0.w1.suh"));
    try testing.expectEqualStrings("L3.moe.w1.suh", stackWeight(&b, "s.L3.ex.suh_g").?);
    try testing.expectEqualStrings("L3.moe.w3.svh", stackWeight(&b, "s.L3.ex.svh_u").?);
    try testing.expectEqualStrings("L3.moe.w2.suh", stackWeight(&b, "s.L3.ex.suh_d").?);
    try testing.expectEqual(@as(?[]const u8, null), stackWeight(&b, "s.L3.ex.tp_g"));
    try testing.expectEqual(@as(?[]const u8, null), stackWeight(&b, "s.L3.ex.k2_d"));
    // the builder's order: a pair is found wherever its halves sit; an odd routed length is left alone
    const names = [_][]const u8{ "L3.moe.shared.0.w1.suh", "L3.attn.wq_b.trellis", "L3.moe.w1.suh", "L3.moe.w2.svh", "L3.moe.shared.0.w2.svh" };
    const p = try pairs(gpa, &names, &.{ 64, 8, 4096, 4090, 64 });
    defer gpa.free(p);
    try testing.expectEqual(@as(usize, 1), p.len);
    try testing.expectEqual([2]usize{ 2, 0 }, p[0]);
    // binding: suh_g's halves adjacent (bound), suh_u's apart (its own buffer), a stack larger than both (its own)
    var w: Weights = .{};
    defer w.map.deinit(gpa);
    try w.map.put(gpa, "L3.moe.w1.suh", .{ .ptr = 0x1000, .len = 4096 });
    try w.map.put(gpa, "L3.moe.shared.0.w1.suh", .{ .ptr = 0x2000, .len = 64 });
    try w.map.put(gpa, "L3.moe.w3.suh", .{ .ptr = 0x9000, .len = 4096 });
    try w.map.put(gpa, "L3.moe.shared.0.w3.suh", .{ .ptr = 0x20000, .len = 64 });
    try w.map.put(gpa, "L3.moe.w2.suh", .{ .ptr = 0x30000, .len = 4096 });
    try w.map.put(gpa, "L3.moe.shared.0.w2.suh", .{ .ptr = 0x31000, .len = 64 });
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var plan: buffers.Plan = .{ .a = arena.allocator() };
    try plan.sizes.put(arena.allocator(), "s.L3.ex.suh_g", 4096 + 64);
    try plan.sizes.put(arena.allocator(), "s.L3.ex.suh_u", 4096 + 64);
    try plan.sizes.put(arena.allocator(), "s.L3.ex.suh_d", 4096 + 128);
    try plan.sizes.put(arena.allocator(), "s.L3.ex.tp_g", 64);
    var r: Roles = .{ .gpa = gpa };
    defer r.addrs.deinit(gpa);
    const got = try bind(&r, &w, &plan);
    try testing.expectEqual(@as(u32, 1), got.roles);
    try testing.expectEqual(@as(?u64, 0x1000), r.addressOf("s.L3.ex.suh_g"));
    try testing.expectEqual(@as(?u64, null), r.addressOf("s.L3.ex.suh_u"));
    try testing.expectEqual(@as(?u64, null), r.addressOf("s.L3.ex.suh_d"));
    try testing.expectEqual(@as(?u64, null), r.addressOf("s.L3.ex.tp_g"));
}

test "paired list repeated growth releases earlier storage on OOM and retries" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const names = try a.alloc([]const u8, 160);
    var lens: [160]usize = @splat(16);
    for (0..80) |i| {
        names[2 * i] = try std.fmt.allocPrint(a, "L{d}.moe.w1.suh", .{i});
        names[2 * i + 1] = try std.fmt.allocPrint(a, "L{d}.moe.shared.0.w1.suh", .{i});
    }
    const C = struct {
        fn run(gpa: std.mem.Allocator, ns: []const []const u8, ls: []const usize) !void {
            var proxy: NoResize = .{ .child = gpa };
            const allocator = proxy.allocator();
            const out = try pairs(allocator, ns, ls);
            defer allocator.free(out);
            try testing.expectEqual(@as(usize, 80), out.len);
            for (out, 0..) |pair, i| try testing.expectEqual([2]usize{ 2 * i, 2 * i + 1 }, pair);
        }
    };
    try testing.checkAllAllocationFailures(testing.allocator, C.run, .{ names, &lens });
    try C.run(testing.allocator, names, &lens);
}

const NoResize = struct {
    child: std.mem.Allocator,
    fn allocator(self: *@This()) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn alloc(ctx: *anyopaque, n: usize, alignment: std.mem.Alignment, ret: usize) ?[*]u8 {
        const self: *@This() = @ptrCast(@alignCast(ctx));
        return self.child.rawAlloc(n, alignment, ret);
    }
    fn free(ctx: *anyopaque, b: []u8, alignment: std.mem.Alignment, ret: usize) void {
        const self: *@This() = @ptrCast(@alignCast(ctx));
        self.child.rawFree(b, alignment, ret);
    }
    fn resize(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) bool {
        return false;
    }
    fn remap(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) ?[*]u8 {
        return null;
    }
};
