//! str.format() for templates: "{}" and "{N}" fields and the "{{" "}}" escapes; every other field is refused.
const std = @import("std");
const v = @import("value.zig");
const Value = v.Value;
const Rt = v.Rt;
const Error = v.Error;

const Numbering = enum { unset, auto, manual };

/// Python's str.format on `s` with positional `args`; a keyword, attribute, conversion or spec field fails by name.
pub fn format(rt: *Rt, s: v.Str, args: []const Value, keywords: usize) Error!Value {
    if (keywords > 0) return rt.fail("str.format() keyword fields are not supported", .{});
    const t = s.s;
    var out: v.Buffer = .{};
    var numbering: Numbering = .unset;
    var next: usize = 0;
    var from: usize = 0;
    var i: usize = 0;
    while (i < t.len) : (i += 1) {
        if (t[i] != '{' and t[i] != '}') continue;
        try out.append(rt.a, try s.part(rt.a, t[from..i]));
        if (i + 1 < t.len and t[i + 1] == t[i]) {
            try out.append(rt.a, try s.part(rt.a, t[i .. i + 1]));
            i += 1;
            from = i + 1;
            continue;
        }
        if (t[i] == '}') return rt.fail("Single '}}' encountered in format string", .{});
        const close = std.mem.indexOfScalarPos(u8, t, i + 1, '}') orelse return rt.fail("Single '{{' encountered in format string", .{});
        const field = t[i + 1 .. close];
        var at = next;
        if (field.len == 0) {
            if (numbering == .manual) return rt.fail("cannot switch from manual field specification to automatic field numbering", .{});
            numbering = .auto;
            next += 1;
        } else {
            // Python's index is decimal digits alone: '+0', '-0' and '0_0' are keyword names (parseInt takes them)
            for (field) |ch| if (!std.ascii.isDigit(ch)) return rt.fail("str.format() field '{s}' is not supported", .{field});
            at = std.fmt.parseInt(usize, field, 10) catch return rt.fail("Too many decimal digits in format string", .{});
            if (numbering == .auto) return rt.fail("cannot switch from automatic field numbering to manual field specification", .{});
            numbering = .manual;
        }
        if (at >= args.len) return rt.fail("Replacement index {d} out of range for positional args tuple", .{at});
        try out.append(rt.a, try v.asStr(rt, args[at]));
        i = close;
        from = close + 1;
    }
    try out.append(rt.a, try s.part(rt.a, t[from..]));
    return .{ .str = out.str() };
}
