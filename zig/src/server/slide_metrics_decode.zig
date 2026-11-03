//! Decode fixed metric receipts without the typed JSON number parser's wide floating conversions.
const std = @import("std");
const json = @import("json");
const Error = error{ InvalidMetricReceipt, OutOfMemory };

pub fn decode(comptime T: type, a: std.mem.Allocator, v: json.Value) Error!T {
    return switch (@typeInfo(T)) {
        .int => if (v == .int) std.fmt.parseInt(T, v.int, 10) catch error.InvalidMetricReceipt else blk: {
            if (v != .float or !std.math.isFinite(v.float) or v.float < 0 or
                v.float > std.math.maxInt(T) or @floor(v.float) != v.float) return error.InvalidMetricReceipt;
            break :blk @intFromFloat(v.float);
        },
        .bool => if (v == .bool) v.bool else error.InvalidMetricReceipt,
        .@"enum" => if (v == .string) std.meta.stringToEnum(T, v.string) orelse error.InvalidMetricReceipt else error.InvalidMetricReceipt,
        .optional => |info| if (v == .null) null else try decode(info.child, a, v),
        .array => |info| blk: {
            if (v != .array or v.array.len != info.len) return error.InvalidMetricReceipt;
            var out: T = undefined;
            for (&out, v.array) |*slot, item| slot.* = try decode(info.child, a, item);
            break :blk out;
        },
        .pointer => |info| blk: {
            if (info.size != .slice or !info.attrs.@"const") @compileError("metric receipts use immutable slices");
            if (info.child == u8) break :blk if (v == .string) try a.dupe(u8, v.string) else error.InvalidMetricReceipt;
            if (v != .array) return error.InvalidMetricReceipt;
            const out = try a.alloc(info.child, v.array.len);
            // The caller supplies its receipt arena, which owns nested allocations on error too.
            for (out, v.array) |*slot, item| slot.* = try decode(info.child, a, item);
            break :blk out;
        },
        .@"struct" => |info| blk: {
            if (v != .object) return error.InvalidMetricReceipt;
            for (v.object.keys()) |key| {
                var known = false;
                inline for (info.field_names) |name| known = known or std.mem.eql(u8, key, name);
                if (!known) return error.InvalidMetricReceipt;
            }
            var out: T = undefined;
            inline for (info.field_names, info.field_types) |name, Field| {
                const attributes = std.meta.fieldInfo(T, @field(std.meta.FieldEnum(T), name)).attrs;
                @field(out, name) = if (v.get(name)) |item| try decode(Field, a, item) else attributes.defaultValue(Field) orelse return error.InvalidMetricReceipt;
            }
            break :blk out;
        },
        else => @compileError("unsupported metric receipt field"),
    };
}
