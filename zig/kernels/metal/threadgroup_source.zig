//! Keep one equal threadgroup reservation per declaration and refuse conflicting reservations.
const std = @import("std");
const name = "max_total_threads_per_threadgroup";
const Reservation = struct { value: []const u8, start: usize, end: usize, standalone: bool };

fn skipQuoted(source: []const u8, start: usize) usize {
    var i = start + 1;
    while (i < source.len) : (i += 1) {
        if (source[i] == '\\') i += @intFromBool(i + 1 < source.len) else if (source[i] == source[start]) return i + 1;
    }
    return source.len;
}

fn same(a: []const u8, b: []const u8) bool {
    var i: usize = 0;
    var j: usize = 0;
    while (true) {
        while (i < a.len and std.ascii.isWhitespace(a[i])) : (i += 1) {}
        while (j < b.len and std.ascii.isWhitespace(b[j])) : (j += 1) {}
        if (i == a.len or j == b.len) return i == a.len and j == b.len;
        if (a[i] != b[j]) return false;
        i += 1;
        j += 1;
    }
}

fn append(out: []u8, written: *usize, bytes: []const u8) void {
    @memcpy(out[written.*..][0..bytes.len], bytes);
    written.* += bytes.len;
}

/// Only duplicate attribute spans are removed; comments, declarations and kernel bodies retain their bytes.
pub fn write(source: []const u8, out: []u8) !usize {
    if (out.len < source.len) return error.NoSpaceLeft;
    var cursor: usize = 0;
    var copied: usize = 0;
    var written: usize = 0;
    var previous: ?Reservation = null;
    while (cursor < source.len) {
        if (std.mem.startsWith(u8, source[cursor..], "//")) {
            cursor = if (std.mem.indexOfScalarPos(u8, source, cursor, '\n')) |i| i + 1 else source.len;
            continue;
        }
        if (std.mem.startsWith(u8, source[cursor..], "/" ++ "*")) {
            cursor = (std.mem.indexOfPos(u8, source, cursor + 2, "*/") orelse return error.BadMetalSource) + 2;
            continue;
        }
        if (source[cursor] == '"' or source[cursor] == '\'') {
            cursor = skipQuoted(source, cursor);
            continue;
        }
        if (std.mem.startsWith(u8, source[cursor..], "[[")) {
            const end = (std.mem.indexOfPos(u8, source, cursor + 2, "]]") orelse return error.BadMetalSource) + 2;
            const attrs = source[cursor + 2 .. end - 2];
            const at = std.mem.indexOf(u8, attrs, name) orelse {
                cursor = end;
                continue;
            };
            if (std.mem.count(u8, attrs, name) != 1) return error.RepeatedCombinedReservation;
            var open = at + name.len;
            while (open < attrs.len and std.ascii.isWhitespace(attrs[open])) : (open += 1) {}
            if (open == attrs.len or attrs[open] != '(') return error.BadMetalSource;
            var close = open + 1;
            var depth: usize = 1;
            while (close < attrs.len and depth != 0) : (close += 1) {
                if (attrs[close] == '(') depth += 1;
                if (attrs[close] == ')') depth -= 1;
            }
            if (depth != 0 or close == open + 2) return error.BadMetalSource;
            const value = attrs[open + 1 .. close - 1];
            const standalone = same(attrs, attrs[at..close]);
            const stop = end + @intFromBool(end < source.len and source[end] == '\n');
            append(out, &written, source[copied..cursor]);
            if (previous) |p| {
                if (!same(p.value, value)) return error.ConflictingThreadgroupReservation;
                if (standalone) {
                    copied = stop;
                    cursor = stop;
                    continue;
                }
                if (!p.standalone) return error.RepeatedCombinedReservation;
                std.mem.copyForwards(u8, out[p.start..], out[p.end..written]);
                written -= p.end - p.start;
            }
            const start = written;
            append(out, &written, source[cursor..stop]);
            previous = .{ .value = value, .start = start, .end = written, .standalone = standalone };
            copied = stop;
            cursor = stop;
            continue;
        }
        if (std.mem.indexOfScalar(u8, "{};", source[cursor]) != null) previous = null;
        cursor += 1;
    }
    append(out, &written, source[copied..]);
    return written;
}

/// Source assembly fails at build time when reservations disagree, before any runtime Metal compiler is called.
pub fn normalize(comptime source: []const u8) []const u8 {
    return comptime blk: {
        @setEvalBranchQuota(20_000_000);
        var out: [source.len]u8 = undefined;
        const len = write(source, &out) catch |err| @compileError("Metal source: " ++ @errorName(err));
        const fixed = out;
        break :blk fixed[0..len];
    };
}
