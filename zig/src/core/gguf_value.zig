//! GGUF v3 metadata values borrow their serialized bytes; walking an array never allocates.
const std = @import("std");

pub const Limits = struct {
    metadata_bytes: usize = 64 << 20,
    metadata_entries: usize = 1_000_000,
    tensors: usize = 1_000_000,
    array_elements: usize = 16 << 20,
    depth: u8 = 16,
};
pub const Kind = enum(u32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    boolean = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,

    pub fn minimum(k: Kind) usize {
        return switch (k) {
            .uint8, .int8, .boolean => 1,
            .uint16, .int16 => 2,
            .uint32, .int32, .float32 => 4,
            .uint64, .int64, .float64, .string => 8,
            .array => 12,
        };
    }
};
pub const Value = union(Kind) {
    uint8: u8,
    int8: i8,
    uint16: u16,
    int16: i16,
    uint32: u32,
    int32: i32,
    float32: f32,
    boolean: bool,
    string: []const u8,
    array: Array,
    uint64: u64,
    int64: i64,
    float64: f64,

    pub fn unsigned(v: Value) !u64 {
        return switch (v) {
            .uint8 => |x| x,
            .uint16 => |x| x,
            .uint32 => |x| x,
            .uint64 => |x| x,
            .int8 => |x| std.math.cast(u64, x) orelse error.MetadataTypeMismatch,
            .int16 => |x| std.math.cast(u64, x) orelse error.MetadataTypeMismatch,
            .int32 => |x| std.math.cast(u64, x) orelse error.MetadataTypeMismatch,
            .int64 => |x| std.math.cast(u64, x) orelse error.MetadataTypeMismatch,
            else => error.MetadataTypeMismatch,
        };
    }
    pub fn float(v: Value) !f64 {
        return switch (v) {
            .float32 => |x| x,
            .float64 => |x| x,
            else => error.MetadataTypeMismatch,
        };
    }
};
pub const Array = struct {
    kind: Kind,
    len: usize,
    bytes: []const u8,
    limits: Limits,

    pub fn iterator(a: Array) Iterator {
        return .{ .cursor = .{ .bytes = a.bytes, .ceiling = a.bytes.len, .limits = a.limits }, .kind = a.kind, .left = a.len };
    }
    pub fn at(a: Array, index: usize) !Value {
        if (index >= a.len) return error.ArrayIndexOutOfBounds;
        var it = a.iterator();
        for (0..index) |_| _ = try it.next();
        return (try it.next()).?;
    }
};
pub const Iterator = struct {
    cursor: Cursor,
    kind: Kind,
    left: usize,
    pub fn next(it: *Iterator) !?Value {
        if (it.left == 0) return null;
        const v = try it.cursor.value(it.kind, 0);
        it.left -= 1;
        return v;
    }
};
pub const UnsupportedTensor = struct { name: []const u8 = "", kind: u32 = 0 };
pub const Cursor = struct {
    bytes: []const u8,
    pos: usize = 0,
    ceiling: usize,
    limits: Limits,
    unsupported: ?*UnsupportedTensor = null,

    pub fn take(c: *Cursor, n: u64) ![]const u8 {
        if (c.pos > c.bytes.len or n > c.bytes.len - c.pos) return error.TruncatedInput;
        if (c.pos > c.ceiling or n > c.ceiling - c.pos) return error.MetadataTooLarge;
        const len: usize = @intCast(n);
        const out = c.bytes[c.pos..][0..len];
        c.pos += len;
        return out;
    }
    pub fn int(c: *Cursor, comptime T: type) !T {
        return std.mem.readInt(T, (try c.take(@sizeOf(T)))[0..@sizeOf(T)], .little);
    }
    pub fn string(c: *Cursor) ![]const u8 {
        const n = try c.int(u64);
        return c.take(n);
    }
    pub fn kind(c: *Cursor) !Kind {
        return std.enums.fromInt(Kind, try c.int(u32)) orelse error.UnknownMetadataType;
    }
    pub fn value(c: *Cursor, k: Kind, depth: u8) anyerror!Value {
        return switch (k) {
            .uint8 => .{ .uint8 = try c.int(u8) },
            .int8 => .{ .int8 = try c.int(i8) },
            .uint16 => .{ .uint16 = try c.int(u16) },
            .int16 => .{ .int16 = try c.int(i16) },
            .uint32 => .{ .uint32 = try c.int(u32) },
            .int32 => .{ .int32 = try c.int(i32) },
            .float32 => .{ .float32 = @bitCast(try c.int(u32)) },
            .uint64 => .{ .uint64 = try c.int(u64) },
            .int64 => .{ .int64 = try c.int(i64) },
            .float64 => .{ .float64 = @bitCast(try c.int(u64)) },
            .boolean => blk: {
                const b = try c.int(u8);
                if (b > 1) return error.InvalidBoolean;
                break :blk .{ .boolean = b == 1 };
            },
            .string => .{ .string = try c.string() },
            .array => blk: {
                if (depth >= c.limits.depth) return error.ArrayNestingTooDeep;
                const element = try c.kind();
                const count = try c.int(u64);
                if (count > (c.bytes.len - c.pos) / element.minimum()) return error.ArrayCountExceedsFile;
                if (count > c.limits.array_elements) return error.TooManyArrayElements;
                const n: usize = @intCast(count);
                const start = c.pos;
                for (0..n) |_| _ = try c.value(element, depth + 1);
                break :blk .{ .array = .{ .kind = element, .len = n, .bytes = c.bytes[start..c.pos], .limits = c.limits } };
            },
        };
    }
};
