//! Source coverage for literal CUDA/HIP lookups, including macro exports and Itanium C++ names.
const std = @import("std");
const Source = struct { path: []const u8, bytes: [:0]const u8 };
const test_only_probes = [_][]const u8{ "tf_probe_missing", "tf_ptx_add_one" };

fn identifier(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
}

fn word(text: []const u8, name: []const u8) bool {
    if (name.len == 0) return false;
    var start: usize = 0;
    while (std.mem.indexOfPos(u8, text, start, name)) |at| {
        const end = at + name.len;
        if ((at == 0 or !identifier(text[at - 1])) and (end == text.len or !identifier(text[end]))) return true;
        start = at + 1;
    }
    return false;
}

fn defined(name: []const u8, sources: []const Source) bool {
    for (sources) |s| if (word(s.bytes, name)) return true;
    if (!std.mem.startsWith(u8, name, "_Z")) return false;
    const nested = name.len > 2 and name[2] == 'N';
    var at: usize = if (nested) 3 else 2;
    var parts: [16][]const u8 = undefined;
    var count: usize = 0;
    while (at < name.len and std.ascii.isDigit(name[at])) {
        const begin = at;
        while (at < name.len and std.ascii.isDigit(name[at])) : (at += 1) {}
        const len = std.fmt.parseInt(usize, name[begin..at], 10) catch return false;
        if (len == 0 or len > name.len - at or count == parts.len) return false;
        parts[count] = name[at..][0..len];
        count += 1;
        at += len;
        if (!nested) break;
    }
    if (count < (if (nested) @as(usize, 2) else 1)) return false;
    for (sources) |s| {
        for (parts[0..count]) |part| {
            if (!word(s.bytes, part)) break;
        } else return true;
    }
    return false;
}

fn probe(name: []const u8) bool {
    for (test_only_probes) |allowed| if (std.mem.eql(u8, name, allowed)) return true;
    return false;
}

fn collect(a: std.mem.Allocator, dir: std.Io.Dir, prefix: []const u8, zig: *std.ArrayList(Source), kernels: *std.ArrayList(Source)) !void {
    const io = std.testing.io;
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ prefix, entry.name });
        if (entry.kind == .directory) {
            var child = try dir.openDir(io, entry.name, .{ .iterate = true });
            defer child.close(io);
            try collect(a, child, path, zig, kernels);
        } else if (entry.kind == .file) {
            const ext = std.fs.path.extension(entry.name);
            const is_zig = std.mem.eql(u8, ext, ".zig");
            var is_kernel = false;
            for ([_][]const u8{ ".cu", ".cuh", ".hip", ".hpp" }) |e| is_kernel = is_kernel or std.mem.eql(u8, ext, e);
            if (!is_zig and !is_kernel) continue;
            const data = try dir.readFileAlloc(io, entry.name, a, .limited(64 * 1024 * 1024));
            const source: Source = .{ .path = path, .bytes = try a.dupeSentinel(u8, data, 0) };
            try (if (is_zig) zig else kernels).append(a, source);
        }
    }
}

fn calls(a: std.mem.Allocator, source: [:0]const u8) !std.ArrayList([]const u8) {
    var lexer = std.zig.Tokenizer.init(source);
    var tokens: std.ArrayList(std.zig.Token) = .empty;
    defer tokens.deinit(a);
    while (true) {
        const token = lexer.next();
        if (token.tag == .eof) break;
        try tokens.append(a, token);
    }
    var names: std.ArrayList([]const u8) = .empty;
    for (tokens.items, 0..) |token, i| {
        if (token.tag != .period or i + 4 >= tokens.items.len) continue;
        const t = tokens.items[i..];
        if (t[1].tag != .identifier or !std.mem.eql(u8, source[t[1].loc.start..t[1].loc.end], "function")) continue;
        if (t[2].tag != .l_paren or t[3].tag != .string_literal) continue;
        if (t[4].tag != .r_paren and !(t[4].tag == .comma and t.len > 5 and t[5].tag == .r_paren)) continue;
        try names.append(a, try std.zig.string_literal.parseAlloc(a, source[t[3].loc.start..t[3].loc.end]));
    }
    return names;
}

test "every literal device function lookup has a CUDA or HIP source token" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var zig: std.ArrayList(Source) = .empty;
    var kernels: std.ArrayList(Source) = .empty;
    var dir = try std.Io.Dir.cwd().openDir(std.testing.io, "zig", .{ .iterate = true });
    defer dir.close(std.testing.io);
    try collect(a, dir, "zig", &zig, &kernels);
    var missing: usize = 0;
    var checked: usize = 0;
    for (zig.items) |s| {
        const names = try calls(a, s.bytes);
        for (names.items) |name| {
            checked += 1;
            if (probe(name) or defined(name, kernels.items)) continue;
            std.debug.print("kernel lookup without source token: {s}: {s}\n", .{ s.path, name });
            missing += 1;
        }
    }
    try std.testing.expect(checked > 0 and kernels.items.len > 0);
    try std.testing.expectEqual(@as(usize, 0), missing);
}

test "source tokens reject prefixes and require a mangled namespace and base in the same file" {
    const sources = [_]Source{
        .{ .path = "macro.cu", .bytes = "MAKE(tf_tree_round) fn_copy_cols_suffix" },
        .{ .path = "mailbox.cu", .bytes = "tf_mailbox_wait" },
        .{ .path = "nested.cu", .bytes = "namespace example { void kernel(); }" },
        .{ .path = "other.hip", .bytes = "namespace elsewhere { void other(); }" },
    };
    try std.testing.expect(defined("tf_tree_round", &sources));
    try std.testing.expect(defined("tf_mailbox_wait", &sources));
    try std.testing.expect(!defined("fn_copy_cols", &sources));
    try std.testing.expect(defined("_ZN7example6kernelEv", &sources));
    try std.testing.expect(defined("_Z6kernelv", &sources));
    try std.testing.expect(!defined("_ZN7example5otherEv", &sources));
    try std.testing.expect(!defined("_ZN7missing6kernelEv", &sources));
    try std.testing.expect(probe("tf_probe_missing"));
    try std.testing.expect(probe("tf_ptx_add_one"));
    try std.testing.expect(!probe("fn_copy_cols"));
}

test "lookup scanner handles whitespace, comments and trailing commas but excludes expressions and string contents" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const names = try calls(arena.allocator(),
        \\const a = m.function // comment
        \\(
        \\  "tf_one",
        \\);
        \\const b = m.function("prefix_" ++ suffix);
        \\const c = ".function(\"not_a_call\")";
        \\const d = m.function("tf_two");
    );
    try std.testing.expectEqual(@as(usize, 2), names.items.len);
    try std.testing.expectEqualStrings("tf_one", names.items[0]);
    try std.testing.expectEqualStrings("tf_two", names.items[1]);
}
