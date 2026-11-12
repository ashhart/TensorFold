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
    const scope = if (std.mem.eql(u8, family, "deepseek_v4")) "deepseek_v41" else family;
    for (p.entries) |e| {
        if (!e.applies(scope, backend)) continue;
        if (e.members) |members| {
            for (members) |member| {
                if (!member.applies(scope)) continue;
                const raw = environment.get(member.name) orelse continue;
                if (!member.differs(raw)) continue;
                try append(&out.writer, member.name, raw, e.class, "env", &found);
            }
        } else {
            const own = environment.get(e.name());
            const raw = e.served(own, if (e.on_with) |w| environment.get(@tagName(w)) else null) orelse continue;
            if (e.id == .TF_DSV41_TOOL_GRAMMAR and environment.get("TENSORFOLD_TOOL_GRAMMAR") != null) continue;
            if (e.differs(raw)) try append(&out.writer, e.name(), if (e.redact) "<set>" else raw, e.class, if (own == null) @tagName(e.on_with.?) else "env", &found);
        }
    }
    if (!found) try out.writer.writeAll(" none");
    return try out.toOwnedSlice();
}

fn append(writer: *std.Io.Writer, name: []const u8, raw: []const u8, class: p.Class, from: []const u8, found: *bool) !void {
    try writer.print("{s} {s}=", .{ if (found.*) ";" else "", name });
    try shown(writer, raw);
    try writer.print(" ({s}, {s})", .{ @tagName(class), from });
    found.* = true;
}

const TestEnv = struct {
    qwen: ?[]const u8 = null,
    all: ?[]const u8 = null,
    share: ?[]const u8 = null,
    glm_pages: ?[]const u8 = null,
    glm_host: ?[]const u8 = null,
    glm_shared: ?[]const u8 = null,
    dory: ?[]const u8 = null,
    gemma4: ?[]const u8 = null,
    spark25: ?[]const u8 = null,
    nemotron_metal: ?[]const u8 = null,
    kv_share: ?[]const u8 = null,
    pub fn get(e: @This(), name: []const u8) ?[]const u8 {
        if (std.mem.eql(u8, name, "TF_QWEN_KV_PAGES")) return e.qwen;
        if (std.mem.eql(u8, name, "TF_QWEN_KV_PAGES_ALL")) return e.all;
        if (std.mem.eql(u8, name, "TF_QWEN_KV_SHARE")) return e.share;
        if (std.mem.eql(u8, name, "TF_KV_PAGES")) return e.glm_pages;
        if (std.mem.eql(u8, name, "TF_GLM_KV_HOST")) return e.glm_host;
        if (std.mem.eql(u8, name, "TF_GLM_KV_HOST_SHARED")) return e.glm_shared;
        if (std.mem.eql(u8, name, "TF_KV_SHARE")) return e.kv_share;
        if (std.mem.eql(u8, name, "TENSORFOLD_DORY_KV_PAGES")) return e.dory;
        if (std.mem.eql(u8, name, "TENSORFOLD_GEMMA4_KV_PAGES")) return e.gemma4;
        if (std.mem.eql(u8, name, "TENSORFOLD_SPARK25_KV_PAGES")) return e.spark25;
        if (std.mem.eql(u8, name, "TF_NEMOTRON_METAL_KV_PAGES")) return e.nemotron_metal;
        if (p.registered(name) != null) return null;
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
    for ([_][2][]const u8{ .{ "unknown", "cuda" }, .{ "qwen3_5", "metal" }, .{ "qwen3_5", "hip" } }) |scope| {
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
    for ([_][2][]const u8{ .{ "qwen3_5", "cuda" }, .{ "glm5_next", "metal" }, .{ "unknown", "cuda" } }) |scope|
        try std.testing.expectEqualStrings("profile: stable; registered overrides: none", try format(a, .stable, scope[0], scope[1], TestEnv{ .glm_pages = "off" }));
}

test "profile line: the GLM host KV tier is reported for its own family and backend only" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("profile: stable; registered overrides: TF_GLM_KV_HOST=1 (identical, env)", try format(a, .stable, "glm5_next", "cuda", TestEnv{ .glm_host = "1" }));
    try std.testing.expectEqualStrings("profile: stable; registered overrides: TF_GLM_KV_HOST=1 (identical, env); TF_GLM_KV_HOST_SHARED=1 (identical, env)", try format(a, .stable, "glm5_next", "cuda", TestEnv{ .glm_host = "1", .glm_shared = "1" }));
    for ([_][]const u8{ "0", "", "01", "true", " 1 " }) |raw|
        try std.testing.expectEqualStrings("profile: stable; registered overrides: none", try format(a, .stable, "glm5_next", "cuda", TestEnv{ .glm_host = raw, .glm_shared = raw }));
    for ([_][2][]const u8{ .{ "qwen3_5", "cuda" }, .{ "gemma4", "cuda" }, .{ "glm5_next", "mlx" }, .{ "unknown", "cuda" } }) |scope|
        try std.testing.expectEqualStrings("profile: stable; registered overrides: none", try format(a, .stable, scope[0], scope[1], TestEnv{ .glm_host = "1", .glm_shared = "1" }));
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
        for ([_][]const u8{ "metal", "hip", "xpu" }) |backend|
            try std.testing.expectEqualStrings("profile: stable; registered overrides: none", try format(a, .stable, family, backend, TestEnv{ .glm_pages = "0" }));
    }
}

const ExtraEnv = struct {
    name: []const u8,
    value: []const u8,
    dory: ?[]const u8 = null,
    gemma4: ?[]const u8 = null,
    spark25: ?[]const u8 = null,
    nemotron_metal: ?[]const u8 = null,
    pub fn get(e: @This(), name: []const u8) ?[]const u8 {
        return if (std.mem.eql(u8, name, e.name)) e.value else null;
    }
};

test "profile line: new rows and every grouped member report real names and scoped classes" {
    const a = std.testing.allocator;
    for ([_]struct { name: []const u8, raw: []const u8, family: []const u8, backend: []const u8, class: []const u8 }{
        .{ .name = "TENSORFOLD_KOLIBRI_KV_PAGES", .raw = "0", .family = "kolibri1", .backend = "cuda", .class = "arithmetic" },
        .{ .name = "TENSORFOLD_DORY_KV_PAGES", .raw = "", .family = "dory", .backend = "cuda", .class = "arithmetic" },
        .{ .name = "GEMMA_DRAFT_BF16", .raw = "self_attn.q_proj", .family = "gemma4", .backend = "metal", .class = "debug" },
        .{ .name = "TF_ANE_COLUMNS", .raw = "3072", .family = "qwen3_5", .backend = "metal", .class = "experimental" },
        .{ .name = "TF_PROMPT_ROWS", .raw = "128", .family = "qwen3_5", .backend = "metal", .class = "arithmetic" },
        .{ .name = "TF_SLIDE_POCKET_KEY", .raw = "both", .family = "nemotron_h", .backend = "metal", .class = "experimental" },
        .{ .name = "TF_SLIDE_EDIT", .raw = "0", .family = "nemotron_h", .backend = "metal", .class = "experimental" },
        .{ .name = "TF_SLIDE_PROBE", .raw = "", .family = "nemotron_h", .backend = "cuda", .class = "experimental" },
        .{ .name = "TF_SLIDE_ROUTED", .raw = "1", .family = "nemotron_h", .backend = "cuda", .class = "experimental" },
        .{ .name = "TF_QWEN27_METAL_KV_PAGES", .raw = "1", .family = "qwen3_5", .backend = "metal", .class = "arithmetic" },
        .{ .name = "TF_QWEN35_METAL_KV_PAGES", .raw = "1", .family = "qwen3_5", .backend = "metal", .class = "arithmetic" },
        .{ .name = "TF_SLIDE_FIT_REDUCE", .raw = "tree", .family = "qwen3_5", .backend = "cuda", .class = "arithmetic" },
    }) |c| {
        const env: ExtraEnv = .{ .name = c.name, .value = c.raw };
        const line = try format(a, .stable, c.family, c.backend, env);
        defer a.free(line);
        const expected = try std.fmt.allocPrint(a, "profile: stable; registered overrides: {s}={s} ({s}, env)", .{ c.name, c.raw, c.class });
        defer a.free(expected);
        try std.testing.expectEqualStrings(expected, line);
        const hidden = try format(a, .stable, "unknown", c.backend, env);
        defer a.free(hidden);
        try std.testing.expectEqualStrings("profile: stable; registered overrides: none", hidden);
        try std.testing.expect(std.mem.indexOf(u8, line, "slide_learn_mode=") == null);
    }
}

test "profile line: Qwen CUDA does not advertise Nemotron-only learning modes" {
    const a = std.testing.allocator;
    for ([_][]const u8{ "TF_SLIDE_EDIT", "TF_SLIDE_PROBE", "TF_SLIDE_ROUTED" }) |name| {
        const line = try format(a, .stable, "qwen3_5", "cuda", ExtraEnv{ .name = name, .value = "1" });
        defer a.free(line);
        try std.testing.expectEqualStrings("profile: stable; registered overrides: none", line);
    }
    const line = try format(a, .stable, "qwen3_5", "cuda", ExtraEnv{ .name = "TF_SLIDE_POCKET_KEY", .value = "both" });
    defer a.free(line);
    try std.testing.expectEqualStrings("profile: stable; registered overrides: TF_SLIDE_POCKET_KEY=both (experimental, env)", line);
}

test "profile line: the pool-page switches and Nemotron's Metal paging report for their own scopes" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("profile: stable; registered overrides: TENSORFOLD_DORY_KV_PAGES=1 (arithmetic, env)", try format(a, .stable, "dory", "cuda", TestEnv{ .dory = "1" }));
    try std.testing.expectEqualStrings("profile: stable; registered overrides: none", try format(a, .stable, "dory", "cuda", TestEnv{ .dory = "0" }));
    try std.testing.expectEqualStrings("profile: stable; registered overrides: TENSORFOLD_GEMMA4_KV_PAGES=0 (identical, env)", try format(a, .stable, "gemma4", "cuda", TestEnv{ .gemma4 = "0" }));
    try std.testing.expectEqualStrings("profile: stable; registered overrides: none", try format(a, .stable, "gemma4", "cuda", TestEnv{ .gemma4 = "1" }));
    try std.testing.expectEqualStrings("profile: stable; registered overrides: TENSORFOLD_SPARK25_KV_PAGES=0 (identical, env)", try format(a, .stable, "spark2_5", "cuda", TestEnv{ .spark25 = "0" }));
    try std.testing.expectEqualStrings("profile: stable; registered overrides: TF_NEMOTRON_METAL_KV_PAGES=1 (identical, env)", try format(a, .stable, "nemotron_h", "metal", TestEnv{ .nemotron_metal = "1" }));
    try std.testing.expectEqualStrings("profile: stable; registered overrides: none", try format(a, .stable, "nemotron_h", "cuda", TestEnv{ .nemotron_metal = "1" }));
}

const DeepEnv = struct {
    name: []const u8,
    raw: []const u8 = "0",
    canonical: ?[]const u8 = null,
    pub fn get(e: @This(), key: []const u8) ?[]const u8 {
        if (std.mem.eql(u8, key, "TENSORFOLD_TOOL_GRAMMAR")) return e.canonical;
        return if (std.mem.eql(u8, e.name, key)) e.raw else null;
    }
};

test "profile line: every independent DeepSeek override is scoped and paths are redacted" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var count: usize = 0;
    for (p.entries) |e| {
        if (!e.applies("deepseek_v41", "cuda")) continue;
        count += 1;
        const env: DeepEnv = .{ .name = e.name(), .raw = if (e.redact) "fixture-private-directory" else "0" };
        const expected = try std.fmt.allocPrint(a, "profile: stable; registered overrides: {s}={s} ({s}, env)", .{ e.name(), if (e.redact) "<set>" else "0", @tagName(e.class) });
        try std.testing.expectEqualStrings(expected, try format(a, .stable, "deepseek_v41", "cuda", env));
        for ([_][2][]const u8{ .{ "deepseek_v41", "metal" }, .{ "glm5_next", "cuda" } }) |scope|
            try std.testing.expectEqualStrings("profile: stable; registered overrides: none", try format(a, .stable, scope[0], scope[1], env));
    }
    try std.testing.expectEqual(@as(usize, 20), count);
    try std.testing.expectEqualStrings("profile: stable; registered overrides: none", try format(a, .stable, "deepseek_v41", "cuda", DeepEnv{ .name = "TF_DSV41_TOOL_GRAMMAR", .canonical = "0" }));
}

test "profile line: both DeepSeek model type names report the same CUDA registry rows" {
    const Environment = struct {
        pub fn get(_: @This(), name: []const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_DSV41_OWN_PREFILL")) "on" else null;
        }
    };
    const a = std.testing.allocator;
    const canonical = try format(a, .stable, "deepseek_v41", "cuda", Environment{});
    defer a.free(canonical);
    const alias = try format(a, .stable, "deepseek_v4", "cuda", Environment{});
    defer a.free(alias);
    try std.testing.expectEqualStrings(canonical, alias);
    try std.testing.expect(std.mem.indexOf(u8, alias, "TF_DSV41_OWN_PREFILL=on") != null);
    const metal = try format(a, .stable, "deepseek_v4", "metal", Environment{});
    defer a.free(metal);
    try std.testing.expectEqualStrings("profile: stable; registered overrides: none", metal);
}

test "profile line: Dory's pages count as on, from TF_KV_SHARE, when TF_KV_SHARE turns them on" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const on = "profile: stable; registered overrides: TF_KV_SHARE=4 (identical, env); TENSORFOLD_DORY_KV_PAGES=1 (arithmetic, TF_KV_SHARE)";
    try std.testing.expectEqualStrings(on, try format(a, .stable, "dory", "cuda", TestEnv{ .kv_share = "4" }));
    const both = "profile: stable; registered overrides: TF_KV_SHARE=4 (identical, env); TENSORFOLD_DORY_KV_PAGES=1 (arithmetic, env)";
    try std.testing.expectEqualStrings(both, try format(a, .stable, "dory", "cuda", TestEnv{ .kv_share = "4", .dory = "1" }));
    for ([_]TestEnv{ .{}, .{ .kv_share = "0" }, .{ .dory = "0" }, .{ .kv_share = "0", .dory = "0" } }) |env|
        try std.testing.expectEqualStrings("profile: stable; registered overrides: none", try format(a, .stable, "dory", "cuda", env));
    const gemma = "profile: stable; registered overrides: TF_KV_SHARE=4 (identical, env)";
    try std.testing.expectEqualStrings(gemma, try format(a, .stable, "gemma4", "cuda", TestEnv{ .kv_share = "4" }));
}

test "profile line: an empty or whitespace-only TF_KV_SHARE does not turn Dory's pages on" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const empty = "profile: stable; registered overrides: TF_KV_SHARE= (identical, env)";
    try std.testing.expectEqualStrings(empty, try format(a, .stable, "dory", "cuda", TestEnv{ .kv_share = "" }));
    for ([_][]const u8{ " ", "\t", " \t\r\n" }) |raw| {
        const line = try format(a, .stable, "dory", "cuda", TestEnv{ .kv_share = raw });
        try std.testing.expect(std.mem.indexOf(u8, line, "TENSORFOLD_DORY_KV_PAGES") == null);
    }
}

test "profile line: Qwen's Metal paging shows only for exact 1, and the slide fit only for tree" {
    const a = std.testing.allocator;
    for ([_][2][]const u8{ .{ "TF_QWEN27_METAL_KV_PAGES", "0" }, .{ "TF_QWEN35_METAL_KV_PAGES", "0" }, .{ "TF_SLIDE_FIT_REDUCE", "ordered" } }) |c| {
        const backend: []const u8 = if (std.mem.startsWith(u8, c[0], "TF_SLIDE")) "cuda" else "metal";
        const line = try format(a, .stable, "qwen3_5", backend, ExtraEnv{ .name = c[0], .value = c[1] });
        defer a.free(line);
        try std.testing.expectEqualStrings("profile: stable; registered overrides: none", line);
    }
}
