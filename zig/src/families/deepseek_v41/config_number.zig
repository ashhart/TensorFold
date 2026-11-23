const std = @import("std");

pub fn numberText(v: std.json.Value) ![]const u8 {
    return switch (v) {
        .number_string, .string => |s| s,
        else => error.BadConfig,
    };
}

pub fn unsigned(comptime T: type, value: std.json.Value) !T {
    const text = try numberText(value);
    if (text.len == 0) return error.InvalidCharacter;
    if (std.fmt.parseInt(T, text, 10)) |n| return n else |_| {}
    var start: usize = 0;
    const negative = text[0] == '-';
    if (negative or text[0] == '+') start = 1;
    const exp_pos = std.mem.indexOfAnyPos(u8, text, start, "eE") orelse text.len;
    const exponent: i64 = if (exp_pos < text.len)
        try std.fmt.parseInt(i32, text[exp_pos + 1 ..], 10)
    else
        0;
    var digits: usize = 0;
    var fractional: usize = 0;
    var dot = false;
    var nonzero = false;
    for (text[start..exp_pos]) |ch| {
        if (ch == '.' and !dot) {
            dot = true;
            continue;
        }
        if (ch < '0' or ch > '9') return error.InvalidCharacter;
        digits += 1;
        if (dot) fractional += 1;
        nonzero = nonzero or ch != '0';
    }
    if (digits == 0) return error.InvalidCharacter;
    if (!nonzero) return 0;
    if (negative) return error.Overflow;
    const scale = exponent - @as(i64, @intCast(fractional));
    const discarded: usize = if (scale < 0) @intCast(-scale) else 0;
    if (discarded >= digits) return error.InvalidNumber;
    const kept = digits - discarded;
    var n: T = 0;
    var index: usize = 0;
    for (text[start..exp_pos]) |ch| {
        if (ch == '.') continue;
        if (index < kept) {
            const digit: T = @intCast(ch - '0');
            if (n > (std.math.maxInt(T) - digit) / 10) return error.Overflow;
            n = n * 10 + digit;
        } else if (ch != '0') return error.InvalidNumber;
        index += 1;
    }
    // Any nonzero ordinary integer overflows after at most 20 decimal shifts.
    if (scale > 20) return error.Overflow;
    var shift: i64 = 0;
    while (shift < scale) : (shift += 1) {
        if (n > std.math.maxInt(T) / 10) return error.Overflow;
        n *= 10;
    }
    return n;
}
