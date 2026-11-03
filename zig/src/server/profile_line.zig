//! The selected profile and applicable registered overrides; arbitrary environment variables never enter this line.
const std = @import("std");
const p = @import("engine_api").profile;

/// Bounded switch text cannot inject terminal controls or another startup line.
fn shown(writer: *std.Io.Writer, raw: []const u8) !void {
    const limit = @min(raw.len, 64);
    for (raw[0..limit]) |byte| {
        if (byte < 0x20 or byte > 0x7e) try writer.print("\\x{x:0>2}", .{byte}) else try writer.writeByte(byte);
    }
    if (raw.len > limit) try writer.writeAll("...");
}

/// Registry coverage is explicit: unregistered variables and another family's switches are not advertised as active.
pub fn format(a: std.mem.Allocator, profile: p.Profile, family: []const u8, backend: []const u8, environment: anytype) ![]const u8 {
    var out: std.Io.Writer.Allocating = .init(a);
    errdefer out.deinit();
    try out.writer.print("profile: {s}; registered overrides:", .{@tagName(profile)});
    var found = false;
    for (p.entries) |e| {
        if (!e.applies(family, backend)) continue;
        const raw = environment.get(e.name()) orelse continue;
        if (!e.differs(raw)) continue;
        try out.writer.print("{s} {s}=", .{ if (found) ";" else "", e.name() });
        try shown(&out.writer, raw);
        try out.writer.print(" ({s}, env)", .{@tagName(e.class)});
        found = true;
    }
    if (!found) try out.writer.writeAll(" none");
    return try out.toOwnedSlice();
}

const TestEnv = struct {
    qwen: ?[]const u8 = null,
    all: ?[]const u8 = null,
    share: ?[]const u8 = null,
    glm_pages: ?[]const u8 = null,
    pub fn get(e: @This(), name: []const u8) ?[]const u8 {
        if (std.mem.eql(u8, name, "TF_QWEN_KV_PAGES")) return e.qwen;
        if (std.mem.eql(u8, name, "TF_QWEN_KV_PAGES_ALL")) return e.all;
        if (std.mem.eql(u8, name, "TF_QWEN_KV_SHARE")) return e.share;
        if (std.mem.eql(u8, name, "TF_KV_PAGES")) return e.glm_pages;
        if (std.mem.eql(u8, name, "TF_GLM_KV_SHARE")) return null;
        return "fixture-secret-must-not-be-read";
    }
};

test "profile line: only applicable allowlisted changes are shown, without claiming all environment coverage" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const env: TestEnv = .{ .qwen = "256", .all = "1", .share = "8" };
    const line = try format(a, .stable, "qwen3_5", "cuda", env);
    try std.testing.expectEqualStrings("profile: stable; registered overrides: TF_QWEN_KV_PAGES=256 (identical, env); TF_QWEN_KV_PAGES_ALL=1 (debug, env); TF_QWEN_KV_SHARE=8 (identical, env)", line);
    for ([_][2][]const u8{ .{ "unknown", "cuda" }, .{ "qwen3_5", "mlx" }, .{ "qwen3_5", "rocm" } }) |scope| {
        try std.testing.expectEqualStrings("profile: stable; registered overrides: none", try format(a, .stable, scope[0], scope[1], env));
    }
    try std.testing.expectEqualStrings("profile: stable; registered overrides: none", try format(a, .stable, "qwen3_5", "cuda", TestEnv{ .qwen = "00", .all = "true", .share = "+0" }));
}

test "profile line: GLM reports explicit page policies including off without treating auto as an override" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]?[]const u8{ null, "auto" }) |raw|
        try std.testing.expectEqualStrings("profile: stable; registered overrides: none", try format(a, .stable, "glm5_next", "cuda", TestEnv{ .glm_pages = raw }));
    for ([_][]const u8{ "off", "0", "on", "1", "all", "", "false" }) |raw| {
        const expected = try std.fmt.allocPrint(a, "profile: stable; registered overrides: TF_KV_PAGES={s} (identical, env)", .{raw});
        try std.testing.expectEqualStrings(expected, try format(a, .stable, "glm5_next", "cuda", TestEnv{ .glm_pages = raw }));
    }
    for ([_][2][]const u8{ .{ "qwen3_5", "cuda" }, .{ "glm5_next", "mlx" }, .{ "unknown", "cuda" } }) |scope|
        try std.testing.expectEqualStrings("profile: stable; registered overrides: none", try format(a, .stable, scope[0], scope[1], TestEnv{ .glm_pages = "off" }));
}

test "profile line: control bytes and overlong values stay bounded on a single line" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tail: [200]u8 = @splat('x');
    const raw = try std.mem.concat(a, u8, &.{ "4\n\x1b[31m\t", &tail });
    const line = try format(a, .stable, "qwen3_5", "cuda", TestEnv{ .qwen = raw });
    try std.testing.expect(std.mem.indexOf(u8, line, "4\\x0a\\x1b[31m\\x09") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "...") != null);
    try std.testing.expect(line.len < 200);
    for (line) |byte| try std.testing.expect(byte >= 0x20 and byte <= 0x7e);
}

test "profile line: Nemotron paging reports exact off and all, independently of GLM's auto default" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "nemotron_h", "nemotron_h_omni" }) |family| {
        for ([_][]const u8{ "0", "all" }) |raw| {
            const expected = try std.fmt.allocPrint(a, "profile: stable; registered overrides: TF_KV_PAGES={s} (identical, env)", .{raw});
            try std.testing.expectEqualStrings(expected, try format(a, .stable, family, "cuda", TestEnv{ .glm_pages = raw }));
        }
        for ([_]?[]const u8{ null, "", "1", "on", "off", "auto", "00", "+0", "false", "ALL", " all ", "0\n" }) |raw|
            try std.testing.expectEqualStrings("profile: stable; registered overrides: none", try format(a, .stable, family, "cuda", TestEnv{ .glm_pages = raw }));
        for ([_][]const u8{ "mlx", "rocm", "xpu" }) |backend|
            try std.testing.expectEqualStrings("profile: stable; registered overrides: none", try format(a, .stable, family, backend, TestEnv{ .glm_pages = "0" }));
    }
}
