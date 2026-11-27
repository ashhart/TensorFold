//! Stable source metadata names keys; deterministic pack attestations strengthen admission when locally verified.
const std = @import("std");
fn stableLine(line: []const u8) bool {
    return line.len > 0 and !std.mem.startsWith(u8, line, "payload ");
}
pub fn matches(recorded: []const u8, current: []const u8) bool {
    var left = std.mem.splitScalar(u8, recorded, '\n');
    var right = std.mem.splitScalar(u8, current, '\n');
    var saved: ?[]const u8 = null;
    while (right.next()) |line| {
        if (line.len == 0) continue;
        if (std.mem.startsWith(u8, line, "payload ")) {
            if (saved == null or !std.mem.eql(u8, saved.?, line)) return false;
            continue;
        }
        var found = false;
        while (left.next()) |other| {
            if (!stableLine(other)) continue;
            if (!std.mem.eql(u8, other, line)) return false;
            saved = null;
            if (left.peek()) |next| if (std.mem.startsWith(u8, next, "payload ")) {
                saved = left.next();
            };
            found = true;
            break;
        }
        if (!found) return false;
    }
    while (left.next()) |line| if (stableLine(line)) return false;
    return true;
}

/// Cache readiness already compares its supplied metadata key; only available source content evidence is added here.
pub fn matchesKnown(recorded: []const u8, current: []const u8) bool {
    var lines = std.mem.splitScalar(u8, current, '\n');
    var shard: ?[]const u8 = null;
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "shard ")) shard = line;
        if (!std.mem.startsWith(u8, line, "payload ")) continue;
        const name = shard orelse return false;
        var saved = std.mem.splitScalar(u8, recorded, '\n');
        var found = false;
        while (saved.next()) |candidate| {
            if (!std.mem.eql(u8, candidate, name)) continue;
            const digest = saved.next() orelse return false;
            if (!std.mem.eql(u8, digest, line)) return false;
            found = true;
            break;
        }
        if (!found) return false;
    }
    return true;
}
