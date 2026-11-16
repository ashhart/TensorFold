//! A safetensors file's header read into tensor entries: name, dtype and shape, the dtypes this family stores.
const std = @import("std");

pub const Dtype = enum {
    bf16,
    f32,
    f8_e4m3,
    u8,

    /// The dtype a safetensors tag names, or null for a tag this family does not store.
    pub fn of(text: []const u8) ?Dtype {
        for (std.enums.values(Dtype)) |d| if (std.mem.eql(u8, d.tag(), text)) return d;
        return null;
    }

    pub fn tag(d: Dtype) []const u8 {
        return switch (d) {
            .bf16 => "BF16",
            .f32 => "F32",
            .f8_e4m3 => "F8_E4M3",
            .u8 => "U8",
        };
    }
};

pub const max_rank = 4;

pub const Entry = struct {
    name: []const u8,
    dtype: Dtype,
    rank: u8,
    dims: [max_rank]u64,

    pub fn shape(e: *const Entry) []const u64 {
        return e.dims[0..e.rank];
    }
};

/// The JSON text of a file's header: the 8-byte little-endian length, then that many bytes.
pub fn jsonOf(file: []const u8) error{BadHeader}![]const u8 {
    if (file.len < 8) return error.BadHeader;
    const n = std.mem.readInt(u64, file[0..8], .little);
    if (n > file.len - 8) return error.BadHeader;
    return file[8..][0..@intCast(n)];
}

/// Every tensor of the header JSON (the `__metadata__` block skipped), sorted by name; text and list live in `arena`.
pub fn parse(arena: std.mem.Allocator, json: []const u8) ![]Entry {
    const root = std.json.parseFromSliceLeaky(std.json.Value, arena, json, .{}) catch return error.BadHeader;
    if (root != .object) return error.BadHeader;
    var out: std.ArrayList(Entry) = .empty;
    var it = root.object.iterator();
    while (it.next()) |kv| {
        if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) continue;
        const v = kv.value_ptr.*;
        if (v != .object) return error.BadHeader;
        const tag = v.object.get("dtype") orelse return error.BadHeader;
        const shape = v.object.get("shape") orelse return error.BadHeader;
        if (tag != .string or shape != .array or shape.array.items.len > max_rank) return error.BadHeader;
        var e: Entry = .{
            .name = kv.key_ptr.*,
            .dtype = Dtype.of(tag.string) orelse return error.UnknownDtype,
            .rank = @intCast(shape.array.items.len),
            .dims = @splat(0),
        };
        for (shape.array.items, 0..) |d, i| {
            if (d != .integer or d.integer < 0) return error.BadHeader;
            e.dims[i] = @intCast(d.integer);
        }
        try out.append(arena, e);
    }
    sortEntries(out.items);
    return out.items;
}

/// Orders entries by name, as `parse` returns them.
pub fn sortEntries(entries: []Entry) void {
    std.mem.sort(Entry, entries, {}, lessThan);
}

fn lessThan(_: void, a: Entry, b: Entry) bool {
    return std.mem.lessThan(u8, a.name, b.name);
}
