//! Missing switch groups describe their actual parser semantics without changing values or defaults.
const std = @import("std");
const p = @import("profile.zig");
const t = std.testing;

test "stable profile: missing family rows have the requested classes and exact scopes" {
    for ([_]struct { id: p.Switch, family: []const u8, backend: []const u8, class: p.Class }{
        .{ .id = .TENSORFOLD_KOLIBRI_KV_PAGES, .family = "kolibri1", .backend = "cuda", .class = .arithmetic },
        .{ .id = .TENSORFOLD_DORY_KV_PAGES, .family = "dory", .backend = "cuda", .class = .arithmetic },
        .{ .id = .GEMMA_DRAFT_BF16, .family = "gemma4", .backend = "metal", .class = .debug },
        .{ .id = .TF_ANE_COLUMNS, .family = "qwen3_5", .backend = "metal", .class = .experimental },
        .{ .id = .TF_PROMPT_ROWS, .family = "qwen3_5_moe", .backend = "metal", .class = .arithmetic },
    }) |c| {
        const row = p.entry(c.id);
        try t.expectEqual(c.class, row.class);
        try t.expect(row.applies(c.family, c.backend));
        try t.expect(!row.applies(c.family, if (std.mem.eql(u8, c.backend, "cuda")) "metal" else "cuda"));
        try t.expect(row.stable == null);
        try t.expect(p.resolve(.stable, c.id, null) == null);
    }
    const kolibri = p.entry(.TENSORFOLD_KOLIBRI_KV_PAGES);
    try t.expect(kolibri.differs("0"));
    for ([_][]const u8{ "", "1", "00", "on", "typo" }) |raw| try t.expect(!kolibri.differs(raw));
    const dory = p.entry(.TENSORFOLD_DORY_KV_PAGES);
    try t.expect(!dory.differs("0"));
    for ([_][]const u8{ "", "1", "00", "on", "typo" }) |raw| try t.expect(dory.differs(raw));
    const gemma = p.entry(.GEMMA_DRAFT_BF16);
    for ([_][]const u8{ "all", "fc", "self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj" }) |raw| try t.expect(gemma.differs(raw));
    for ([_][]const u8{ "1", "0", "", "self_attn.v_proj", "mlp.up_proj" }) |raw| try t.expect(!gemma.differs(raw));
    try t.expect(!p.entry(.TF_ANE_COLUMNS).differs("0"));
    try t.expect(p.entry(.TF_ANE_COLUMNS).differs("3072"));
    try t.expect(p.entry(.TF_PROMPT_ROWS).differs("128"));
}

test "stable profile: one slide row has four real names and no synthetic environment lookup" {
    const row = p.entry(.slide_learn_mode);
    try t.expectEqual(p.Class.experimental, row.class);
    try t.expectEqual(@as(usize, 4), row.members.?.len);
    var count: usize = 0;
    for (p.entries) |r| if (r.id == .slide_learn_mode) {
        count += 1;
    };
    try t.expectEqual(@as(usize, 1), count);
    try t.expect(p.registered("slide_learn_mode") == null);
    try t.expect(p.value(.stable, .slide_learn_mode) == null);
    for (row.members.?, 0..) |member, i| {
        try t.expectEqual(p.Switch.slide_learn_mode, p.registered(member.name).?);
        const actual = std.c.getenv(member.name);
        const got = p.lookup(member.name);
        if (actual) |raw| {
            try t.expectEqualStrings(std.mem.span(raw), got.?);
            try t.expectEqual(@as([*]const u8, raw), got.?.ptr);
        } else try t.expect(got == null);
        try t.expect(member.applies("nemotron_h"));
        try t.expectEqual(i == 0, member.applies("qwen3_5"));
        if (i == 0) {
            try t.expect(!member.differs("auto"));
            for ([_][]const u8{ "reply", "subject", "both" }) |raw| try t.expect(member.differs(raw));
        } else for ([_][]const u8{ "", "0", "1", "anything" }) |raw| try t.expect(member.differs(raw));
    }
    try t.expect(row.applies("nemotron_h", "metal"));
    try t.expect(row.applies("nemotron_h_omni", "cuda"));
    try t.expect(row.applies("qwen3_5", "cuda"));
    try t.expect(!row.applies("qwen3_5", "metal"));
    try t.expect(!row.applies("qwen3_5_moe", "cuda"));
    try t.expect(!p.entry(.TF_ANE_COLUMNS).applies("qwen3_5_moe", "metal"));
    try t.expect(!row.applies("qwen4_exp", "cuda"));
}

const c_env = struct {
    extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
    extern "c" fn unsetenv(name: [*:0]const u8) c_int;
};

test "stable profile: grouped lookups keep four simultaneous raw values separate" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    const members = p.entry(.slide_learn_mode).members.?;
    const values = [_][:0]const u8{ "subject", "0", "", "routed-fixture" };
    var saved: [4]?[:0]u8 = @splat(null);
    var ready: usize = 0;
    defer for (members[0..ready], saved[0..ready]) |member, old| {
        if (old) |value| {
            _ = c_env.setenv(member.name, value, 1);
            t.allocator.free(value);
        } else _ = c_env.unsetenv(member.name);
    };
    for (members, 0..) |member, i| {
        if (std.c.getenv(member.name)) |raw| saved[i] = try t.allocator.dupeSentinel(u8, std.mem.span(raw), 0);
        ready += 1;
    }
    for (members, values) |member, value| try t.expectEqual(@as(c_int, 0), c_env.setenv(member.name, value, 1));
    for (members, values) |member, value| {
        const raw = std.c.getenv(member.name).?;
        const got = p.lookup(member.name).?;
        try t.expectEqualStrings(value, got);
        try t.expectEqual(@as([*]const u8, raw), got.ptr);
    }
    try t.expect(p.value(.stable, .slide_learn_mode) == null);
}

test "stable profile: given reads empty or whitespace-only text as unset and hands every other value on byte for byte" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    const name = "TF_KV_SHARE";
    const before = if (std.c.getenv(name)) |v| try t.allocator.dupeSentinel(u8, std.mem.span(v), 0) else null;
    defer {
        if (before) |v| {
            _ = c_env.setenv(name, v, 1);
            t.allocator.free(v);
        } else _ = c_env.unsetenv(name);
    }
    _ = c_env.unsetenv(name);
    try t.expect(p.given(.TF_KV_SHARE) == null);
    for ([_][:0]const u8{ "", " ", "\t", "\r\n", " \t\r\n" }) |v| {
        try t.expectEqual(@as(c_int, 0), c_env.setenv(name, v, 1));
        try t.expectEqualStrings(v, p.get(.TF_KV_SHARE).?);
        try t.expect(p.given(.TF_KV_SHARE) == null);
    }
    for ([_][:0]const u8{ "0", "16", " 16", "x" }) |v| {
        try t.expectEqual(@as(c_int, 0), c_env.setenv(name, v, 1));
        const raw = std.c.getenv(name).?;
        const got = p.given(.TF_KV_SHARE).?;
        try t.expectEqualStrings(v, got);
        try t.expectEqual(@as([*]const u8, raw), got.ptr);
    }
}
