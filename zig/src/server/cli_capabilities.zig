//! The native capabilities document: the table's native flags and the honoured variables.
const std = @import("std");
const cli = @import("cli.zig");

pub fn capabilities(w: *std.Io.Writer, e: cli.Engines) !void {
    try w.print("{{\"schema\": 1, \"engine\": \"zig\", \"version\": \"{s}\", \"chip\": ", .{e.version});
    if (e.chip) |c| try w.print("\"{s}\"", .{c}) else try w.writeAll("null");
    try w.writeAll(", \"backends\": [");
    for (e.backends, 0..) |b, i| try w.print("{s}\"{s}\"", .{ if (i > 0) ", " else "", b });
    try w.writeAll("], \"families\": {");
    for (e.families, 0..) |f, i| {
        try w.print("{s}\"{s}\": [", .{ if (i > 0) ", " else "", f.model_type });
        for (f.formats, 0..) |fmt, j| try w.print("{s}\"{s}\"", .{ if (j > 0) ", " else "", fmt });
        try w.writeAll("]");
    }
    try w.writeAll("}, \"serve\": {\"flags\": {");
    var first = true;
    for (cli.flags) |f| {
        if (!f.native) continue;
        try w.print("{s}\"{s}\": {{", .{ if (first) "" else ", ", f.name });
        first = false;
        const values = f.native_values orelse f.choices;
        if (values.len > 0) {
            try w.writeAll("\"values\": [");
            for (values, 0..) |v, j| try w.print("{s}\"{s}\"", .{ if (j > 0) ", " else "", v });
            try w.writeAll("]");
        }
        try w.writeAll("}");
    }
    try w.writeAll("}}, \"env\": [");
    for (cli.env, 0..) |name, i| try w.print("{s}\"{s}\"", .{ if (i > 0) ", " else "", name });
    try w.writeAll("], \"optional_libraries\": ");
    try std.json.Stringify.value(e.optional_libraries, .{}, w);
    try w.writeAll("}\n");
}
