//! ``capabilities --json``: engine facts, native flags and selected documented variables.
const std = @import("std");
const api = @import("engine_api");
const cli = @import("cli.zig");

/// What ``capabilities --json`` reports about the engine side: its version, chip, backends and families.
pub const Engines = struct {
    version: []const u8,
    chip: ?[]const u8 = null,
    optional_libraries: []const @import("sys").optional_libraries.Status = &.{},
    backends: []const []const u8 = &.{},
    /// model_type to the weight formats it reads, as gate entries name them; `decisions` lists the decision-only ones.
    families: []const api.Family = &.{},
    /// A CUDA host's NVML P2P summary as one JSON object, only with two or more GPUs (p2p_report.compact).
    p2p: ?[]const u8 = null,
};

/// Native flags and selected supported variables; env is non-exhaustive, profiles lists registered switches.
pub fn capabilities(w: *std.Io.Writer, e: Engines) !void {
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
    try w.writeAll("}, \"decisions\": {"); // formats served only for decisions (a readout head), never chat
    var listed = false;
    for (e.families) |f| if (f.decision_formats.len > 0) {
        try w.print("{s}\"{s}\": [", .{ if (listed) ", " else "", f.model_type });
        listed = true;
        for (f.decision_formats, 0..) |fmt, j| try w.print("{s}\"{s}\"", .{ if (j > 0) ", " else "", fmt });
        try w.writeAll("]");
    };
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
    try w.writeAll(", \"profiles\": ");
    try std.json.Stringify.value(.{
        .default = @tagName(api.profile.Profile.stable),
        .available = [_][]const u8{"stable"},
        .registered_switches = api.profile.entries,
    }, .{}, w);
    if (e.p2p) |p| try w.print(", \"p2p\": {s}", .{p});
    try w.writeAll("}\n");
}

test "capability audit: environment list is a subset beside the complete profile registry" {
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try capabilities(&out.writer, .{ .version = "test" });
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, out.written(), .{});
    defer parsed.deinit();
    const env = parsed.value.object.get("env").?.array.items;
    const entries = parsed.value.object.get("profiles").?.object.get("registered_switches").?.array.items;
    try std.testing.expectEqual(api.profile.entries.len, entries.len);
    var additional: usize = 0;
    for (entries, api.profile.entries) |value, entry| {
        const name = value.object.get("id").?.string;
        try std.testing.expectEqualStrings(entry.name(), name);
        const listed = for (env) |v| {
            if (std.mem.eql(u8, name, v.string)) break true;
        } else false;
        if (!listed) additional += 1;
    }
    try std.testing.expect(additional > 0);
}

test "capability audit: decision-only formats are listed apart from the formats a family chats in" {
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    const families = [_]api.Family{
        .{ .model_type = "plain", .formats = &.{"mlx-q4g64"} },
        .{ .model_type = "decides", .formats = &.{"mlx-q4g64"}, .decision_formats = &.{"bf16"} },
    };
    try capabilities(&out.writer, .{ .version = "test", .families = &families });
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, out.written(), .{});
    defer parsed.deinit();
    const chat = parsed.value.object.get("families").?.object.get("decides").?.array.items;
    try std.testing.expectEqual(@as(usize, 1), chat.len);
    try std.testing.expectEqualStrings("mlx-q4g64", chat[0].string);
    const decisions = parsed.value.object.get("decisions").?.object;
    try std.testing.expectEqual(@as(usize, 1), decisions.count());
    try std.testing.expectEqualStrings("bf16", decisions.get("decides").?.array.items[0].string);
    var empty: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer empty.deinit();
    try capabilities(&empty.writer, .{ .version = "test" });
    const none = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, empty.written(), .{});
    defer none.deinit();
    try std.testing.expectEqual(@as(usize, 0), none.value.object.get("decisions").?.object.count());
}

test "capability audit: environment table omits dead readers and slide metadata names its members" {
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try capabilities(&out.writer, .{ .version = "test" });
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, out.written(), .{});
    defer parsed.deinit();
    const env = parsed.value.object.get("env").?.array.items;
    for (env) |value| {
        try std.testing.expect(!std.mem.eql(u8, value.string, "TENSORFOLD_NO_UPDATE_CHECK"));
        try std.testing.expect(!std.mem.eql(u8, value.string, "HF_HUB_OFFLINE"));
    }
    const rows = parsed.value.object.get("profiles").?.object.get("registered_switches").?.array.items;
    var groups: usize = 0;
    for (rows) |row| {
        if (!std.mem.eql(u8, row.object.get("id").?.string, "slide_learn_mode")) continue;
        groups += 1;
        try std.testing.expectEqualStrings("experimental", row.object.get("class").?.string);
        try std.testing.expectEqual(@as(usize, 4), row.object.get("members").?.array.items.len);
        try std.testing.expectEqual(@as(usize, 2), row.object.get("scopes").?.array.items.len);
        try std.testing.expect(row.object.get("stable").? == .null);
    }
    try std.testing.expectEqual(@as(usize, 1), groups);
}
