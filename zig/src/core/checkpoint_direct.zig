//! A layer loader's tensors read past the page cache (direct_gather), the next layer on a thread while one packs.

const std = @import("std");
const builtin = @import("builtin");
const gather = @import("direct_gather.zig");
const Checkpoint = @import("checkpoint.zig").Checkpoint;
const Tensor = @import("checkpoint.zig").Tensor;

/// One prefix's tensors: their names, their views (repointed at the copies), and the copies by name.
const Slot = struct {
    prefix: [320]u8 = undefined,
    prefix_len: usize = 0,
    names: std.ArrayList([]const u8) = .empty,
    views: std.ArrayList([]const u8) = .empty,
    ptrs: std.ArrayList(*[]const u8) = .empty,
    copies: std.StringHashMapUnmanaged([]const u8) = .empty,
    mapped: std.ArrayList(usize) = .empty, // each view's address in the mapping, before the read repointed it
    moved: std.AutoHashMapUnmanaged(usize, []const u8) = .empty, // the copies by mapped address (repoint)

    fn deinit(s: *Slot, gpa: std.mem.Allocator) void {
        s.mapped.deinit(gpa);
        s.moved.deinit(gpa);
        s.names.deinit(gpa);
        s.views.deinit(gpa);
        s.ptrs.deinit(gpa);
        s.copies.deinit(gpa);
    }
};

/// Off (every call a no-op) with TF_LOAD_DIRECT=0, no readers, off Linux, or more shards than one read draws on.
pub const Layers = struct {
    gpa: std.mem.Allocator,
    ck: ?*Checkpoint = null,
    sources: [gather.max_sources]gather.Source = undefined,
    n: usize = 0,
    pair: gather.Pair,
    // each read fills the next slot and the pair's next arena: a slot's views point into its own arena
    slots: [2]Slot = .{ .{}, .{} },
    at: usize = 0, // the slot the next read fills
    ahead: bool = false, // slots[at ^ 1] is being read by the pair's thread
    current: ?*Slot = null, // the slot `enter` handed out last

    pub fn init(gpa: std.mem.Allocator, ck: *Checkpoint, readers: usize) Layers {
        var l: Layers = .{ .gpa = gpa, .pair = .init(gpa, readers) };
        const files = ck.files.items;
        if (builtin.os.tag != .linux or readers == 0 or !gather.enabled() or files.len > gather.max_sources) return l;
        for (files, l.sources[0..files.len]) |*f, *s| s.* = .{ .path = f.path, .memory = f.map.memory };
        l.n = files.len;
        l.ck = ck;
        return l;
    }

    pub fn on(l: *const Layers) bool {
        return l.ck != null;
    }

    /// `prefix`'s tensors as the copies `get` returns (read now unless read ahead), and `next`'s read begun.
    pub fn enter(l: *Layers, prefix: []const u8, next: ?[]const u8) !void {
        const ck = l.ck orelse return;
        ck.copies = null;
        var want: [320]u8 = undefined;
        const name = try std.fmt.bufPrint(&want, "{s}{s}", .{ ck.scope, prefix });
        try l.pair.wait();
        var s = &l.slots[1 - l.at];
        if (!l.ahead or !std.mem.eql(u8, s.prefix[0..s.prefix_len], name)) {
            s = try l.collect(prefix);
            try l.pair.read(l.sources[0..l.n], s.ptrs.items);
        }
        l.ahead = false;
        try s.copies.ensureTotalCapacity(l.gpa, @intCast(s.names.items.len));
        for (s.names.items, s.views.items) |k, v| s.copies.putAssumeCapacity(k, v);
        try s.moved.ensureTotalCapacity(l.gpa, @intCast(s.views.items.len));
        for (s.mapped.items, s.views.items) |a, v| s.moved.putAssumeCapacity(a, v);
        ck.copies = &s.copies;
        l.current = s;
        // the slot read ahead into held the layer before this one: its bytes are on the device
        const t = try l.collect(next orelse return);
        try l.pair.start(l.sources[0..l.n], t.ptrs.items, null);
        l.ahead = true;
    }

    /// `enter` on `fmt`'s prefix for layer `i`, with layer `next`'s read begun (none on the last).
    pub fn layerAt(l: *Layers, comptime fmt: []const u8, i: usize, next: ?usize) !void {
        if (l.ck == null) return;
        var b: [2][160]u8 = undefined;
        const n = if (next) |j| try std.fmt.bufPrint(&b[1], fmt, .{j}) else null;
        try l.enter(try std.fmt.bufPrint(&b[0], fmt, .{i}), n);
    }

    /// The tensors in `v` (a pointer: fields, unions, optionals, arrays, slices) moved onto the current layer's copies.
    pub fn repoint(l: *const Layers, v: anytype) void {
        if (l.current) |s| walk(&s.moved, v);
    }

    /// `v.*` (layer `i` of `n`, mapped before the loop) on its copies: layerAt, then repoint.
    pub fn moved(l: *Layers, v: anytype, comptime fmt: []const u8, i: usize, n: usize) !@TypeOf(v.*) {
        try l.layerAt(fmt, i, if (i + 1 < n) i + 1 else null);
        l.repoint(v);
        return v.*;
    }

    pub fn deinit(l: *Layers) void {
        l.pair.deinit();
        if (l.ck) |ck| ck.copies = null;
        for (&l.slots) |*s| s.deinit(l.gpa);
    }

    /// The next slot holding `prefix`'s names and mapped views; only the checkpoint's indexes are read.
    fn collect(l: *Layers, prefix: []const u8) !*Slot {
        const gpa = l.gpa;
        const ck = l.ck.?;
        const s = &l.slots[l.at];
        l.at = 1 - l.at;
        s.prefix_len = (try std.fmt.bufPrint(&s.prefix, "{s}{s}", .{ ck.scope, prefix })).len;
        s.names.clearRetainingCapacity();
        s.views.clearRetainingCapacity();
        s.copies.clearRetainingCapacity();
        s.mapped.clearRetainingCapacity();
        s.moved.clearRetainingCapacity();
        const full = s.prefix[0..s.prefix_len];
        for (ck.files.items) |*f| for (f.names.keys()) |k| {
            if (!std.mem.startsWith(u8, k, full)) continue;
            try s.names.append(gpa, k);
            try s.views.append(gpa, f.get(k).?.bytes);
        };
        try s.ptrs.resize(gpa, s.views.items.len);
        try s.mapped.resize(gpa, s.views.items.len);
        for (s.views.items, s.ptrs.items, s.mapped.items) |*v, *p, *a| {
            p.* = v;
            a.* = @intFromPtr(v.ptr);
        }
        return s;
    }
};

/// Each Tensor reachable from pointer `v` takes its copy's bytes; one not in the layer keeps the mapping.
fn walk(moved: *const std.AutoHashMapUnmanaged(usize, []const u8), v: anytype) void {
    const T = @typeInfo(@TypeOf(v)).pointer.child;
    if (T == Tensor) {
        if (v.bytes.len != 0) if (moved.get(@intFromPtr(v.bytes.ptr))) |c| {
            v.bytes = c;
        };
        return;
    }
    switch (@typeInfo(T)) {
        .@"struct" => |st| inline for (st.field_names) |f| walk(moved, &@field(v.*, f)),
        .@"union" => switch (v.*) {
            inline else => |*x| walk(moved, x),
        },
        .optional => if (v.*) |*x| walk(moved, x),
        .array => |a| if (holds(a.child)) for (v) |*x| walk(moved, x),
        .pointer => |p| if (p.size == .slice and !p.attrs.@"const" and holds(p.child)) for (v.*) |*x| walk(moved, x),
        else => {},
    }
}

/// Whether a value of `T` can hold a Tensor (walk skips byte and number arrays).
fn holds(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .@"struct", .@"union", .optional => true,
        .array => |a| holds(a.child),
        else => false,
    };
}

/// A safetensors image of U8 tensors, one per name, each `bytes` long and filled from its index.
fn testImage(a: std.mem.Allocator, names: []const []const u8, bytes: usize) ![]u8 {
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(a);
    try header.append(a, '{');
    for (names, 0..) |n, i| {
        if (i > 0) try header.append(a, ',');
        const entry = "\"{s}\":{{\"dtype\":\"U8\",\"shape\":[{d}],\"data_offsets\":[{d},{d}]}}";
        try header.print(a, entry, .{ n, bytes, i * bytes, (i + 1) * bytes });
    }
    try header.append(a, '}');
    const out = try a.alloc(u8, 8 + header.items.len + names.len * bytes);
    std.mem.writeInt(u64, out[0..8], header.items.len, .little);
    @memcpy(out[8..][0..header.items.len], header.items);
    for (out[8 + header.items.len ..], 0..) |*b, i| b.* = @truncate(i * 31 + i / bytes);
    return out;
}

test "each layer's tensors come back as copies off the mapping, the same bytes, read ahead or not" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const shards = [_][]const []const u8{
        &.{ "layers.0.a", "layers.1.a", "head" },
        &.{ "layers.0.b", "layers.1.b", "layers.10.a" },
    };
    for (shards, 0..) |names, i| {
        const image = try testImage(gpa, names, 3 * 4096 + 17);
        defer gpa.free(image);
        var file: [32]u8 = undefined;
        const sub = try std.fmt.bufPrint(&file, "model-{d}.safetensors", .{i});
        try tmp.dir.writeFile(io, .{ .sub_path = sub, .data = image });
    }
    var path: [64]u8 = undefined;
    var ck = try Checkpoint.openModel(gpa, io, try std.fmt.bufPrint(&path, ".zig-cache/tmp/{s}", .{tmp.sub_path}));
    defer ck.close();
    var mapped: [6][]const u8 = undefined;
    const all = [_][]const u8{ "layers.0.a", "layers.0.b", "layers.1.a", "layers.1.b", "layers.10.a", "head" };
    for (all, &mapped) |n, *m| {
        for (ck.files.items) |*f| if (f.get(n)) |t| {
            m.* = t.bytes;
        };
    }
    var l = Layers.init(gpa, &ck, 3);
    defer l.deinit();
    if (!l.on()) return error.SkipZigTest;
    // layer 0 read now, 1 read ahead, 10 named out of order (its read ahead is for another prefix)
    const steps = [_]struct { []const u8, ?[]const u8, []const []const u8, []const usize }{
        .{ "layers.0.", "layers.1.", &.{ "layers.0.a", "layers.0.b" }, &.{ 0, 1 } },
        .{ "layers.1.", "layers.2.", &.{ "layers.1.a", "layers.1.b" }, &.{ 2, 3 } },
        .{ "layers.10.", null, &.{"layers.10.a"}, &.{4} },
    };
    for (steps) |st| {
        try l.enter(st[0], st[1]);
        for (st[2], st[3]) |n, m| {
            const t = try ck.get(n);
            try std.testing.expect(t.bytes.ptr != mapped[m].ptr);
            try std.testing.expectEqualSlices(u8, mapped[m], t.bytes);
        }
        // a name outside the layer stays on the mapping
        try std.testing.expectEqual(mapped[5].ptr, (try ck.get("head")).bytes.ptr);
    }
    try std.testing.expectEqual(@as(usize, 0), ck.unused());
}

test "layerAt names a layer and the next; repoint moves a map-first loader's views onto the same copies" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const names = [_][]const u8{ "layers.0.a", "layers.0.b", "layers.1.a", "layers.1.b", "head" };
    const image = try testImage(gpa, &names, 2 * 4096 + 5);
    defer gpa.free(image);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = image });
    var path: [64]u8 = undefined;
    var ck = try Checkpoint.openModel(gpa, io, try std.fmt.bufPrint(&path, ".zig-cache/tmp/{s}", .{tmp.sub_path}));
    defer ck.close();
    const Held = struct { a: Tensor, more: []Tensor, maybe: ?Tensor };
    var more: [2][1]Tensor = undefined;
    var map: [2]Held = undefined;
    for (&map, &more, 0..) |*m, *x, i| {
        var b: [32]u8 = undefined;
        x[0] = try ck.get(try std.fmt.bufPrint(&b, "layers.{d}.b", .{i}));
        const a = try ck.get(try std.fmt.bufPrint(&b, "layers.{d}.a", .{i}));
        m.* = .{ .a = a, .more = x, .maybe = try ck.get("head") };
    }
    const mapped = [2][2][]const u8{ .{ map[0].a.bytes, more[0][0].bytes }, .{ map[1].a.bytes, more[1][0].bytes } };
    var l = Layers.init(gpa, &ck, 2);
    defer l.deinit();
    if (!l.on()) return error.SkipZigTest;
    for (&map, 0..) |*m, i| {
        try l.layerAt("layers.{d}.", i, if (i == 0) 1 else null);
        l.repoint(m);
        try std.testing.expect(m.a.bytes.ptr != mapped[i][0].ptr and m.more[0].bytes.ptr != mapped[i][1].ptr);
        try std.testing.expectEqualSlices(u8, mapped[i][0], m.a.bytes);
        try std.testing.expectEqualSlices(u8, mapped[i][1], m.more[0].bytes);
        // the same copies `get` hands out, and a tensor outside the layer stays on the mapping
        var b: [32]u8 = undefined;
        const got = try ck.get(try std.fmt.bufPrint(&b, "layers.{d}.a", .{i}));
        try std.testing.expectEqual(got.bytes.ptr, m.a.bytes.ptr);
        try std.testing.expectEqual((try ck.get("head")).bytes.ptr, m.maybe.?.bytes.ptr);
    }
}

test "TF_LOAD_DIRECT off or no readers leaves every call a no-op" {
    const gpa = std.testing.allocator;
    var ck: Checkpoint = .{ .gpa = gpa, .io = std.testing.io };
    var l = Layers.init(gpa, &ck, 0);
    defer l.deinit();
    try std.testing.expect(!l.on());
    try l.enter("layers.0.", "layers.1.");
    try l.layerAt("layers.{d}.", 0, 1);
    var t: Tensor = .{ .dtype = .u8, .rank = 0, .shape = undefined, .bytes = "x" };
    l.repoint(&t);
    try std.testing.expectEqualStrings("x", t.bytes);
    try std.testing.expectEqual(@as(?*const std.StringHashMapUnmanaged([]const u8), null), ck.copies);
}
