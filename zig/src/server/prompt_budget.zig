//! Opt-in request prompt limits, checked before joining, decoding or reply fan-out.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const limits = @import("core").resource_limits;
const Value = json.Value;
const Cx = errors.Cx;

pub const max_bytes: usize = 1 << 20;
pub const max_tokens: usize = 32768;
const max_depth = 64;

pub fn check(cx: *Cx, value: ?Value, extra: []const u8) errors.Refused!void {
    if (!limits.enabled()) return;
    var total = extra.len;
    if (value) |v| total = try add(cx, total, try bytes(cx, v, 0));
    if (total > max_bytes) return cx.refuse("strict prompt source exceeds 1048576 bytes");
}

fn bytes(cx: *Cx, value: Value, depth: usize) errors.Refused!usize {
    if (depth >= max_depth) return cx.refuse("strict prompt nesting exceeds 64 levels");
    return switch (value) {
        .string => |s| s.len,
        .array => |a| {
            var total: usize = 0;
            for (a) |item| total = try add(cx, total, try bytes(cx, item, depth + 1));
            return total;
        },
        .object => |o| {
            var total: usize = 0;
            for (o.keys(), o.values()) |key, item| {
                total = try add(cx, total, key.len);
                if (std.mem.eql(u8, key, "image_url") or std.mem.eql(u8, key, "video_url") or
                    std.mem.eql(u8, key, "file") or std.mem.eql(u8, key, "input_file") or
                    std.mem.eql(u8, key, "file_data") or std.mem.eql(u8, key, "data") or std.mem.eql(u8, key, "url")) continue;
                total = try add(cx, total, try bytes(cx, item, depth + 1));
            }
            return total;
        },
        else => 0,
    };
}

fn add(cx: *Cx, a: usize, b: usize) errors.Refused!usize {
    const total = std.math.add(usize, a, b) catch return cx.refuse("strict prompt source exceeds 1048576 bytes");
    if (total > max_bytes) return cx.refuse("strict prompt source exceeds 1048576 bytes");
    return total;
}

pub fn ids(cx: *Cx, count: usize) errors.Refused!void {
    if (limits.enabled() and count > max_tokens) return cx.refuse("strict prompt token count exceeds 32768");
}
