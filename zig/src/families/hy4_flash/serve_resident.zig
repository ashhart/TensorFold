//! Every Metal buffer a served Hy4-flash model holds, gathered once for its residency set.
const std = @import("std");
const mtl = @import("metal");
const target = @import("metal_model.zig");
const dfly = @import("metal_draft.zig");
const Metal = @import("serve_metal.zig").Metal;

pub fn buffers(self: *Metal, list: *std.ArrayList(mtl.Buffer)) !void {
    try gather(target.Model, &self.model, list, self.gpa);
    try gather(dfly.Draft, &self.draft, list, self.gpa);
    try gather(target.Frame, &self.frame, list, self.gpa);
    try gather(dfly.Frame, &self.dframe, list, self.gpa);
    for ([_]mtl.Buffer{ self.positions, self.writes, self.slide_keys, self.slide_counts, self.sparse_keys, self.sparse_counts, self.slide_ranges, self.sparse_ranges, self.tiles, self.scores, self.d_ids, self.d_tokens, self.d_positions, self.d_writes }) |b| try list.append(self.gpa, b);
    try list.appendSlice(self.gpa, self.d_keys);
    try list.appendSlice(self.gpa, self.d_counts);
    for (self.ix_tokens, self.ix_counts) |x, y| {
        if (x) |b| try list.append(self.gpa, b);
        if (y) |b| try list.append(self.gpa, b);
    }
    if (self.kept) |c| {
        try list.appendSlice(self.gpa, c.kv);
        for (c.token_keys, c.group_keys) |x, y| {
            if (x) |b| try list.append(self.gpa, b);
            if (y) |b| try list.append(self.gpa, b);
        }
    }
    // one entry a buffer
    std.mem.sort(mtl.Buffer, list.items, {}, struct {
        fn less(_: void, x: mtl.Buffer, y: mtl.Buffer) bool {
            return @intFromPtr(x.id) < @intFromPtr(y.id);
        }
    }.less);
    var n: usize = 0;
    for (list.items) |b| {
        if (n > 0 and list.items[n - 1].id == b.id) continue;
        list.items[n] = b;
        n += 1;
    }
    list.items.len = n;
}

/// Whether a value of T can hold a Metal buffer (structs, optionals, arrays, slices and unions of them).
fn holds(comptime T: type, comptime depth: u32) bool {
    if (T == mtl.Buffer) return true;
    if (depth == 0) return false;
    return switch (@typeInfo(T)) {
        .@"struct" => |st| blk: {
            for (st.field_types, st.field_attrs) |F, attrs| if (!attrs.@"comptime" and holds(F, depth - 1)) break :blk true;
            break :blk false;
        },
        .optional => |o| holds(o.child, depth - 1),
        .array => |ar| holds(ar.child, depth - 1),
        .pointer => |ptr| ptr.size == .slice and holds(ptr.child, depth - 1),
        .@"union" => |u| blk: {
            if (u.tag_type == null) break :blk false;
            for (u.field_types) |F| if (holds(F, depth - 1)) break :blk true;
            break :blk false;
        },
        else => false,
    };
}

fn gather(comptime T: type, v: *const T, list: *std.ArrayList(mtl.Buffer), gpa: std.mem.Allocator) !void {
    if (T == mtl.Buffer) return list.append(gpa, v.*);
    if (comptime !holds(T, 8)) return;
    switch (@typeInfo(T)) {
        .@"struct" => |st| inline for (st.field_names, st.field_types, st.field_attrs) |f, F, attrs| {
            if (!attrs.@"comptime") try gather(F, &@field(v.*, f), list, gpa);
        },
        .optional => |o| if (v.*) |*x| try gather(o.child, x, list, gpa),
        .array => |ar| for (v) |*x| try gather(ar.child, x, list, gpa),
        .pointer => |ptr| if (ptr.size == .slice) for (v.*) |*x| try gather(ptr.child, x, list, gpa),
        .@"union" => switch (v.*) {
            inline else => |*x| try gather(@TypeOf(x.*), x, list, gpa),
        },
        else => {},
    }
}
