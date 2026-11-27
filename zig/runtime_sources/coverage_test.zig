//! Every exported Metal source string must be assembled into a runtime library or explicitly excluded.
const std = @import("std");
const sources = @import("kernel_sources");
const inventory = @import("root.zig");

const Exclusion = struct { path: []const u8, reason: []const u8 };
const exclusions = [_]Exclusion{
    .{ .path = "kernel_sources.core_contract_controls", .reason = "planted kernel-contract violations, not engine runtime libraries" },
    .{ .path = "kernel_sources.drama3", .reason = "the engine loads prebuilt Drama3 metallibs; these texts are for kernel contracts" },
};

const Checker = struct {
    sets: []const inventory.Set,
    failures: usize = 0,
    fn fingerprint(text: []const u8) []const u8 {
        var best: []const u8 = "";
        var lines = std.mem.splitScalar(u8, text, '\n');
        while (lines.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \t\r");
            if (std.mem.startsWith(u8, trimmed, "//")) continue;
            if (trimmed.len > best.len) best = trimmed;
        }
        if (best.len != 0) return best;
        return std.mem.trim(u8, text, " \t\r\n");
    }

    fn assembled(source_text: []const u8, library: []const u8) bool {
        if (std.mem.indexOf(u8, library, source_text) != null) return true;
        var total: usize = 0;
        var matched: usize = 0;
        var lines = std.mem.splitScalar(u8, source_text, '\n');
        while (lines.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \t\r");
            if (trimmed.len < 8 or std.mem.startsWith(u8, trimmed, "//") or std.mem.startsWith(u8, trimmed, "#include")) continue;
            total += 1;
            if (std.mem.indexOf(u8, library, trimmed) != null) matched += 1;
        }
        return total != 0 and matched * 4 >= total * 3;
    }

    fn registered(self: *const Checker, text: []const u8) bool {
        const fingerprint_ = fingerprint(text);
        if (fingerprint_.len == 0) return true;
        for (self.sets) |set| for (set.entries.items) |entry| {
            if (std.mem.indexOf(u8, entry.text, fingerprint_) != null and assembled(text, entry.text)) return true;
        };
        return false;
    }

    fn excluded(path: []const u8) bool {
        for (exclusions) |entry| {
            if (std.mem.eql(u8, entry.path, path)) return true;
            if (path.len > entry.path.len and std.mem.startsWith(u8, path, entry.path) and path[entry.path.len] == '.') return true;
        }
        return false;
    }

    fn source(self: *Checker, path: []const u8, text: []const u8) void {
        if (self.registered(text) or excluded(path)) return;
        self.failures += 1;
        std.debug.print("unregistered Metal source export: {s} ({d} bytes)\n", .{ path, text.len });
    }

    fn value(self: *Checker, comptime item_value: anytype, path: []const u8) !void {
        const T = @TypeOf(item_value);
        if (comptime T == type) return self.namespace(item_value, path);
        switch (comptime @typeInfo(T)) {
            .pointer => |p| {
                if (comptime isString(T)) {
                    const text: []const u8 = item_value;
                    self.source(path, text);
                } else if (comptime p.size == .slice and isString(p.child)) {
                    for (item_value) |item| {
                        const text: []const u8 = item;
                        self.source(path, text);
                    }
                }
            },
            .array => |a| {
                if (comptime isString(a.child)) {
                    inline for (item_value) |item| {
                        const text: []const u8 = item;
                        self.source(path, text);
                    }
                }
            },
            else => {},
        }
    }

    fn namespace(self: *Checker, comptime module: type, prefix: []const u8) !void {
        inline for (comptime std.meta.declarations(module)) |decl| {
            const path = try std.fmt.allocPrint(std.testing.allocator, "{s}.{s}", .{ prefix, decl });
            defer std.testing.allocator.free(path);
            try self.value(@field(module, decl), path);
        }
    }
};

fn isString(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .pointer => |p| p.size == .one and switch (@typeInfo(p.child)) {
            .array => |a| a.child == u8,
            else => false,
        },
        else => false,
    };
}

test "Metal runtime source coverage: every export is registered or explicitly excluded" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const sets = [_]inventory.Set{
        try inventory.generate(a, false, 13),
        try inventory.generate(a, true, 17),
    };
    var checker = Checker{ .sets = &sets };
    try checker.namespace(sources, "kernel_sources");
    try checker.namespace(@import("inkling2_metal"), "inkling2_metal");
    if (checker.failures != 0) return error.UnregisteredMetalSources;
}
