//! Direct buffer fields share one inventory for initial enrollment and paired lifetime callbacks.
const std = @import("std");

pub fn Fields(comptime Metal: type) type {
    return struct {
        pub fn append(gpa: std.mem.Allocator, list: *std.ArrayList(Metal.Buffer), owner: anytype) !void {
            const info = @typeInfo(@TypeOf(owner.*)).@"struct";
            inline for (info.field_names, info.field_types) |name, T| {
                const value = @field(owner.*, name);
                switch (@typeInfo(T)) {
                    .array => |a| if (a.child == Metal.Buffer) try list.appendSlice(gpa, &value),
                    .optional => |o| if (o.child == Metal.Buffer) if (value) |b| try list.append(gpa, b),
                    else => if (T == Metal.Buffer) try list.append(gpa, value),
                }
            }
        }

        pub fn apply(owner: anytype, hook: Metal.ResidencySet.Follow, joining: bool) void {
            const info = @typeInfo(@TypeOf(owner.*)).@"struct";
            inline for (info.field_names, info.field_types) |name, T| {
                const value = @field(owner.*, name);
                switch (@typeInfo(T)) {
                    .array => |a| if (a.child == Metal.Buffer) {
                        for (value) |b| one(b, hook, joining);
                    },
                    .optional => |o| if (o.child == Metal.Buffer) if (value) |b| one(b, hook, joining),
                    else => if (T == Metal.Buffer) one(value, hook, joining),
                }
            }
        }

        pub fn one(buffer: Metal.Buffer, hook: Metal.ResidencySet.Follow, joining: bool) void {
            if (joining) hook.join(hook.ctx, buffer) else hook.leave(hook.ctx, buffer);
        }
    };
}
