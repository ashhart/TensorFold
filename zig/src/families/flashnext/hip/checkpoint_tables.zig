//! The checkpoint as core/quant reads it for the HIP loader: tensors by name, code widths from the config.
const std = @import("std");
const core = @import("core");
const Config = @import("../config.zig").Config;

/// Each side passes its own core's checkpoint and width types (one by path, one as a module).
pub fn Tables(comptime Checkpoint: type, comptime Width: type) type {
    const Got = @typeInfo(@TypeOf(Checkpoint.get)).@"fn".return_type.?;
    return struct {
        ck: *Checkpoint,
        cfg: *const Config,

        const Self = @This();

        pub fn has(t: *const Self, key: []const u8) bool {
            return t.ck.has(key);
        }

        pub fn get(t: *const Self, key: []const u8) Got {
            return t.ck.get(key);
        }

        /// The width core/quant reads `key`'s words and group tables with.
        pub fn width(t: *const Self, key: []const u8) !Width {
            const s = (try t.cfg.declared(key, core.quant.keys.canonical)) orelse return error.UnsupportedQuantization;
            return .{ .bits = s.bits, .group = @intCast(s.group) };
        }
    };
}

test "a projection's width comes from the config's table" {
    var c = try @import("../config.zig").parse(std.testing.allocator, @embedFile("../fixtures/config.json"));
    defer c.deinit();
    const Fake = struct {
        pub fn has(_: *@This(), _: []const u8) bool {
            return false;
        }
        pub fn get(_: *@This(), _: []const u8) !u8 {
            return error.NotFound;
        }
    };
    var f: Fake = .{};
    const t: Tables(Fake, struct { bits: u8, group: u16 }) = .{ .ck = &f, .cfg = &c };
    const w = try t.width("language_model.model.layers.0.linear_attn.in_proj_qkv");
    try std.testing.expectEqual(@as(u8, 6), w.bits);
    try std.testing.expectEqual(@as(u16, 32), w.group);
    try std.testing.expect(!t.has("x"));
}
