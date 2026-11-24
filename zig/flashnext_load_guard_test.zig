//! Source wiring guard: all load entries reach the final selected-settings admission before construction.
const std = @import("std");

const Sources = struct {
    loader: []const u8 = @embedFile("src/families/flashnext/engine_load.zig"),
    engine: []const u8 = @embedFile("src/families/flashnext/engine.zig"),
    single: []const u8 = @embedFile("src/native/flashnext_host.zig"),
    batch: []const u8 = @embedFile("src/native/flashnext_batch_host.zig"),
};

fn has(source: []const u8, text: []const u8) bool {
    return std.mem.indexOf(u8, source, text) != null;
}

/// A delegation-only entry must not gain a second statement that can bypass the guarded callee.
fn delegates(source: []const u8, entry: []const u8, call: []const u8) bool {
    const start = std.mem.indexOf(u8, source, entry) orelse return false;
    const sig = std.mem.indexOfPos(u8, source, start, "!*Engine {") orelse return false;
    const begin = sig + "!*Engine {".len;
    const end = @min(std.mem.indexOfPos(u8, source, begin, "\n    }") orelse source.len, std.mem.indexOfPos(u8, source, begin, "\n}") orelse source.len);
    if (end == source.len) return false;
    return std.mem.eql(u8, std.mem.trim(u8, source[begin..end], " \r\n\t"), call);
}

pub fn guarded(s: Sources) bool {
    if (!delegates(s.engine, "pub fn load(", "return loadWith(gpa, io, model_dir, dump_dir, null);")) return false;
    if (!has(s.engine, "pub const loadWith = @import(\"engine_load.zig\").loadWith;")) return false;
    if (!has(s.engine, "pub const loadConfigured = @import(\"engine_load.zig\").loadConfigured;")) return false;
    if (!delegates(s.loader, "pub fn loadWith(", "return loadConfigured(gpa, io, model_dir, dump_dir, speed_up, .{});")) return false;
    const start = std.mem.indexOf(u8, s.loader, "pub fn loadConfigured(") orelse return false;
    const source = s.loader[start..];
    const priced = std.mem.indexOf(u8, source, "const startup = try memory.budgetLiveFor(weights, selected.capacity, inventory, extra, existing, selected.sibling(paired));") orelse return false;
    const guard = std.mem.indexOf(u8, source, "const load_limit = memory.admitBudget(limit, startup, route.needs_build) catch |err| {") orelse return false;
    if (priced >= guard or !has(source[0..priced], "var selected = window;")) return false;
    if (has(source[0..guard], "return ")) return false;
    if (!has(source[guard..], "return err;")) return false;
    if (!has(source[guard..], ".window = selected, .load_limit = load_limit")) return false;
    for ([_][]const u8{ "gpa.create(Engine)", "device.queue()", "r.compile(", "r.sel =", "pack_cache.ensure(" }) |name| {
        const at = std.mem.indexOf(u8, source, name) orelse return false;
        if (guard >= at) return false;
    }
    const pack_at = std.mem.indexOf(u8, source, "pack_cache.ensure(").?;
    if (pack_at >= std.mem.indexOf(u8, source, "gpa.create(Engine)").?) return false;
    if (pack_at >= std.mem.indexOf(u8, source, "device.queue()").?) return false;
    for ([_][]const u8{ s.single, s.batch }) |body| {
        if (!has(body, "withCache(settings, cache_gib)")) return false;
        if (!has(body, "const eng = try fx.Engine.loadConfigured(gpa, io, dir, dump,")) return false;
        if (!has(body, "budget_settings);")) return false;
    }
    return true;
}

test "Flash Next memory: guard controls reject missing admission, early construction and entry bypasses" {
    const a = std.testing.allocator;
    try std.testing.expect(guarded(.{}));
    const Source = struct { field: []const u8, before: []const u8, after: []const u8 };
    const cases = [_]Source{
        .{ .field = "loader", .before = "memory.admitBudget(", .after = "uncheckedBudget(" },
        .{ .field = "loader", .before = "const startup = try memory.budgetLiveFor(weights, selected.capacity,", .after = "const startup = try memory.budgetLiveFor(weights, window.capacity," },
        .{ .field = "loader", .before = "var selected = window;", .after = "return finishLoadUnchecked();\n    var selected = window;" },
        .{ .field = "loader", .before = "const extra = try", .after = "const early = try device.queue();\n    const extra = try" },
        .{ .field = "loader", .before = "return err;", .after = "return finishLoadUnchecked();" },
        .{ .field = "loader", .before = ".window = selected, .load_limit = load_limit", .after = ".window = window, .load_limit = 0" },
        .{ .field = "loader", .before = "return loadConfigured(", .after = "return unguardedLoad(" },
        .{ .field = "engine", .before = "return loadWith(", .after = "return unguardedLoad(" },
        .{ .field = "engine", .before = ").loadWith;", .after = ").unguardedLoad;" },
        .{ .field = "engine", .before = ").loadConfigured;", .after = ").unguardedLoad;" },
        .{ .field = "single", .before = "fx.Engine.loadConfigured(", .after = "fx.Engine.loadWith(" },
        .{ .field = "batch", .before = "fx.Engine.loadConfigured(", .after = "fx.Engine.loadWith(" },
    };
    inline for (cases) |c| {
        var inputs: Sources = .{};
        const original = @field(inputs, c.field);
        try std.testing.expect(has(original, c.before));
        const changed = try std.mem.replaceOwned(u8, a, original, c.before, c.after);
        defer a.free(changed);
        @field(inputs, c.field) = changed;
        try std.testing.expect(!guarded(inputs));
    }
}
