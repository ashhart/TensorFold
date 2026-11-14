//! Stable profiles preserve the caller's raw environment and make no new settings active.
const std = @import("std");
const p = @import("profile.zig");
const t = std.testing;

test "stable profile: live lookup is exactly the legacy libc lookup for every registered switch" {
    inline for (p.entries) |e| {
        if (comptime e.members != null) continue;
        const old = std.c.getenv(e.name());
        const current = p.get(e.id);
        if (old) |raw| {
            try t.expect(current != null);
            try t.expectEqualStrings(std.mem.span(raw), current.?);
            try t.expectEqual(@as([*]const u8, raw), current.?.ptr);
        } else try t.expect(current == null);
    }
}

test "stable profile: registry scopes metadata and never supplies a default" {
    for (p.entries) |e| {
        try t.expectEqual(e.id, p.entry(e.id).id);
        try t.expect(e.name().len > 0 and e.families.len > 0 and e.effect.len > 0 and e.evidence.len > 0);
        try t.expect(e.stable == null);
        try t.expect(p.resolve(.stable, e.id, null) == null);
        if (e.scopes) |scopes| {
            for (scopes) |scope| try t.expect(e.applies(scope.families[0], scope.backend));
        } else try t.expect(e.applies(e.families[0], e.backend));
        try t.expect(!e.applies(e.families[0], "unknown-backend"));
        try t.expect(!e.applies("unknown-model", "cuda"));
    }
    try t.expectEqual(p.Class.debug, p.entry(.TF_QWEN_KV_PAGES_ALL).class);
    try t.expectEqual(p.Class.experimental, p.entry(.TF_FLASHNEXT_KV_PAGES).class);
}

test "stable profile: shared paging variable has disjoint GLM and Nemotron reporting policies" {
    var glm: usize = 0;
    var nemotron: usize = 0;
    var omni: usize = 0;
    for (p.entries) |e| {
        if (e.id != .TF_KV_PAGES) continue;
        if (e.applies("glm5_next", "cuda")) {
            glm += 1;
            try t.expectEqual(p.Syntax.auto, e.syntax);
            try t.expect(e.differs("on"));
            try t.expect(e.differs("off"));
            try t.expect(!e.differs("auto"));
        }
        if (e.applies("nemotron_h", "cuda")) {
            nemotron += 1;
            try t.expectEqual(@as(usize, 2), e.report_values.?.len);
            try t.expectEqualStrings("0", e.report_values.?[0]);
            try t.expectEqualStrings("all", e.report_values.?[1]);
            for ([_][]const u8{ "0", "all" }) |raw| try t.expect(e.differs(raw));
            for ([_][]const u8{ "", "on", "off", "1", "auto", "00", "+0", "ALL", "false", " all ", "0\n" }) |raw|
                try t.expect(!e.differs(raw));
        }
        if (e.applies("nemotron_h_omni", "cuda")) omni += 1;
        try t.expectEqual(p.Class.identical, e.class);
        try t.expect(e.stable == null);
    }
    try t.expectEqual(@as(usize, 1), glm);
    try t.expectEqual(@as(usize, 1), nemotron);
    try t.expectEqual(@as(usize, 1), omni);
}

test "stable profile: explicit off, empty, malformed and ordinary values are returned without coercion" {
    for (p.entries) |e| for ([_][:0]const u8{ "", "0", "1", "0000", "+0", "256", "-1", "false", "nan", " 1 ", "bad\nvalue" }) |raw| {
        const got = p.resolve(.stable, e.id, raw).?;
        try t.expectEqualStrings(raw, got);
        try t.expectEqual(raw.ptr, got.ptr);
    };
}

test "stable profile: reporting follows the original count and exact-one flag semantics" {
    const count = p.entry(.TF_QWEN_KV_SHARE);
    for ([_][]const u8{ "0", "00", "+0" }) |v| try t.expect(!count.differs(v));
    for ([_][]const u8{ "1", "256", "", "-1", "not-a-count" }) |v| try t.expect(count.differs(v));
    const flag = p.entry(.TF_QWEN_KV_PAGES_ALL);
    try t.expect(flag.differs("1"));
    for ([_][]const u8{ "0", "", "01", "true", " 1 " }) |v| try t.expect(!flag.differs(v));
}

test "stable profile: GLM paging registers automatic policy without supplying or coercing its value" {
    const e = p.entry(.TF_KV_PAGES);
    try t.expectEqual(p.Syntax.auto, e.syntax);
    try t.expectEqual(p.Class.identical, e.class);
    try t.expect(e.applies("glm5_next", "cuda"));
    try t.expect(!e.applies("qwen3_5", "cuda"));
    try t.expect(!e.applies("glm5_next", "metal"));
    try t.expect(!e.differs("auto"));
    for ([_][]const u8{ "off", "0", "on", "1", "all", "", "AUTO", "false", " 0 " }) |raw|
        try t.expect(e.differs(raw));
}

test "stable profile: the GLM host KV tier registers as an exact-one switch a family, off unless set" {
    inline for (.{ .TF_GLM_KV_HOST, .TF_GLM_KV_HOST_SHARED }) |id| {
        const e = p.entry(id);
        try t.expectEqual(p.Class.identical, e.class);
        try t.expectEqual(p.Syntax.one, e.syntax);
        try t.expect(e.applies("glm5_next", "cuda"));
        try t.expect(!e.applies("qwen3_5", "cuda"));
        try t.expect(!e.applies("glm5_next", "mlx"));
        try t.expectEqualStrings("#1795", e.evidence);
        try t.expect(p.resolve(.stable, id, null) == null);
        try t.expect(e.differs("1"));
        for ([_][]const u8{ "0", "", "01", "true", " 1 " }) |raw| try t.expect(!e.differs(raw));
    }
}

test "stable profile: GLM's raw prompt tail registers as an exact-one switch, off unless set" {
    const e = p.entry(.TF_GLM_PROMPT_TAIL);
    try t.expectEqual(p.Class.identical, e.class);
    try t.expectEqual(p.Syntax.one, e.syntax);
    try t.expect(e.applies("glm5_next", "cuda") and !e.applies("qwen3_5", "cuda"));
    try t.expectEqualStrings("#2226", e.evidence);
    try t.expect(p.resolve(.stable, .TF_GLM_PROMPT_TAIL, null) == null);
    try t.expect(e.differs("1"));
    for ([_][]const u8{ "0", "", "false", " 1 " }) |raw| try t.expect(!e.differs(raw));
    try t.expectEqual(p.Switch.TF_GLM_PROMPT_TAIL, p.registered("TF_GLM_PROMPT_TAIL").?);
}

test "stable profile: the keep-others rule registers as an identical exact-one CUDA switch, off unless set" {
    const e = p.entry(.TENSORFOLD_PROMPT_CACHE_KEEP_OTHERS);
    try t.expectEqual(p.Class.identical, e.class);
    try t.expectEqual(p.Syntax.one, e.syntax);
    try t.expect(e.applies("glm5_next", "cuda") and e.applies("nemotron_h", "cuda") and !e.applies("glm5_next", "metal"));
    try t.expect(!e.applies("deepseek_v41", "cuda"));
    try t.expect(p.resolve(.stable, .TENSORFOLD_PROMPT_CACHE_KEEP_OTHERS, null) == null);
    try t.expect(e.differs("1"));
    for ([_][]const u8{ "0", "", "true", " 1 " }) |raw| try t.expect(!e.differs(raw));
    try t.expectEqual(p.Switch.TENSORFOLD_PROMPT_CACHE_KEEP_OTHERS, p.registered("TENSORFOLD_PROMPT_CACHE_KEEP_OTHERS").?);
}

test "stable profile: a runtime name lookup covers the registry and leaves every other name to libc" {
    try t.expectEqual(p.Switch.TF_GLM_KV_HOST, p.registered("TF_GLM_KV_HOST").?);
    try t.expectEqual(p.Switch.TF_QWEN_KV_PAGES, p.registered("TF_QWEN_KV_PAGES").?);
    for ([_][]const u8{ "TF_GLM_KV_HOST_GIB", "TF_UNREGISTERED_FIXTURE", "", "tf_glm_kv_host" }) |name|
        try t.expect(p.registered(name) == null);
    // An unset variable is null through both paths, so an unregistered name keeps the reader's old default.
    try t.expect(p.lookup("TF_PROFILE_FIXTURE_UNSET") == null);
    inline for (p.entries) |e| {
        if (comptime e.members == null) try t.expectEqual(p.get(e.id), p.lookup(e.name()));
    }
}

test {
    _ = @import("profile_more_test.zig");
}

test "stable profile: Dory paging is on unless the value is exactly zero" {
    const e = p.entry(.TENSORFOLD_DORY_KV_PAGES);
    try t.expectEqual(p.Syntax.not_zero, e.syntax);
    try t.expectEqual(p.Class.arithmetic, e.class);
    try t.expect(e.applies("dory", "cuda"));
    try t.expect(!e.applies("dory", "metal"));
    try t.expect(p.resolve(.stable, .TENSORFOLD_DORY_KV_PAGES, null) == null);
    for ([_][]const u8{ "1", "", "on", "00" }) |raw| try t.expect(e.differs(raw));
    try t.expect(!e.differs("0"));
}

test "stable profile: Gemma 4, Spark2.5 and LFM2.5 paging is off only for exactly zero" {
    inline for (.{ .TENSORFOLD_GEMMA4_KV_PAGES, .TENSORFOLD_SPARK25_KV_PAGES, .TENSORFOLD_LFM2_KV_PAGES }) |id| {
        const e = p.entry(id);
        try t.expect(e.report_values != null and e.report_values.?.len == 1 and std.mem.eql(u8, e.report_values.?[0], "0"));
        try t.expectEqual(p.Class.identical, e.class);
        try t.expect(p.resolve(.stable, id, null) == null);
        try t.expect(e.differs("0"));
        for ([_][]const u8{ "1", "", "00", "false", " 0 " }) |raw| try t.expect(!e.differs(raw));
    }
    try t.expect(p.entry(.TENSORFOLD_GEMMA4_KV_PAGES).applies("gemma4_unified", "cuda"));
    try t.expect(p.entry(.TENSORFOLD_SPARK25_KV_PAGES).applies("spark2_5", "cuda"));
    try t.expect(p.entry(.TENSORFOLD_LFM2_KV_PAGES).applies("lfm2_moe", "cuda"));
    try t.expect(!p.entry(.TENSORFOLD_LFM2_KV_PAGES).applies("lfm2_moe", "metal"));
}

test "stable profile: Nemotron's paging on Metal has its own scope" {
    const e = p.entry(.TF_NEMOTRON_METAL_KV_PAGES);
    try t.expectEqual(p.Syntax.one, e.syntax);
    try t.expectEqualStrings("metal", e.backend);
    try t.expect(e.applies("nemotron_h", "metal"));
    try t.expect(!e.applies("nemotron_h", "cuda"));
    try t.expect(!e.applies("qwen3_5", "metal"));
    try t.expect(e.differs("1"));
    for ([_][]const u8{ "0", "", "01", "true", " 1 " }) |raw| try t.expect(!e.differs(raw));
}

test "stable profile: every DeepSeek audit group has independent scoped metadata" {
    const Row = struct { id: p.Switch, class: p.Class };
    const rows = [_]Row{
        .{ .id = .TF_DSV41_EXPERT_TOPP, .class = .experimental },
        .{ .id = .TF_DSV41_NUCLEUS, .class = .arithmetic },
        .{ .id = .TF_DSV41_CONTEXT, .class = .arithmetic },
        .{ .id = .TF_DSV41_LAYERS, .class = .debug },
        .{ .id = .TF_DSV41_OWN_PREFILL, .class = .experimental },
        .{ .id = .TF_DSV41_PREFILL_PIECES, .class = .experimental },
        .{ .id = .TF_DSV41_SLOTS, .class = .experimental },
        .{ .id = .TF_DSV41_POOL_TOKENS, .class = .arithmetic },
        .{ .id = .TF_DSV41_KV_SPLIT, .class = .arithmetic },
        .{ .id = .TF_DSV41_KV_SPLIT_UNION_MIB, .class = .arithmetic },
        .{ .id = .TF_DSV41_NEEDS_POOL, .class = .debug },
        .{ .id = .TF_DSV41_MHC_DEFER, .class = .experimental },
        .{ .id = .TF_DSV41_DRAFTS, .class = .experimental },
        .{ .id = .TF_DSV41_SAMPLING, .class = .experimental },
        .{ .id = .TF_DSV41_IMAGES, .class = .experimental },
        .{ .id = .TF_DSV41_BIAS_VL, .class = .experimental },
        .{ .id = .TF_DSV41_ENGRAM_DIR, .class = .experimental },
        .{ .id = .TF_DSV41_SCHEMA_PROMPT, .class = .experimental },
        .{ .id = .TF_DSV41_ROWS_CAP, .class = .arithmetic },
        .{ .id = .TF_DSV41_TOOL_GRAMMAR, .class = .experimental },
    };
    var count: usize = 0;
    for (p.entries) |e| if (e.applies("deepseek_v41", "cuda")) {
        count += 1;
    };
    try t.expectEqual(rows.len, count);
    for (rows) |row| {
        const e = p.entry(row.id);
        try t.expectEqual(row.class, e.class);
        try t.expectEqual(p.Syntax.present, e.syntax);
        try t.expectEqualStrings("#2370", e.evidence);
        try t.expectEqual(@as(usize, 1), e.families.len);
        try t.expect(e.applies("deepseek_v41", "cuda"));
        try t.expect(!e.applies("glm5_next", "cuda") and !e.applies("deepseek_v41", "metal"));
        try t.expect(p.resolve(.stable, row.id, null) == null);
        try t.expectEqual(row.id, p.registered(e.name()).?);
        try t.expectEqual(row.id == .TF_DSV41_ENGRAM_DIR, e.redact);
        for ([_][:0]const u8{ "0", "1", "", " 1 " }) |raw| {
            try t.expect(e.differs(raw));
            try t.expectEqual(raw.ptr, p.resolve(.stable, row.id, raw).?.ptr);
        }
    }
}

test "stable profile: GLM's direct switches register their off and on directions without supplying a value" {
    inline for (.{ .TF_GLM_HOST_CACHE, .TF_GLM_VISION_HOST }) |id| {
        const e = p.entry(id);
        try t.expectEqual(p.Syntax.one, e.syntax);
        try t.expect(e.differs("1"));
        for ([_][]const u8{ "0", "", "01", "true", " 1 " }) |raw| try t.expect(!e.differs(raw));
    }
    inline for (.{ .TF_GLM_PDL, .TF_GLM_PROMPT_STEPS, .TF_GLM_SHARED }) |id| {
        const e = p.entry(id);
        try t.expectEqual(@as(usize, 1), e.report_values.?.len);
        try t.expect(e.differs("0"));
        for ([_][]const u8{ "1", "", "00", "false", " 0 " }) |raw| try t.expect(!e.differs(raw));
    }
    inline for (.{ .TF_GLM_HOST_CACHE, .TF_GLM_VISION_HOST, .TF_GLM_PDL, .TF_GLM_PROMPT_STEPS, .TF_GLM_SHARED }) |id| {
        const e = p.entry(id);
        try t.expect(e.applies("glm5_next", "cuda") and !e.applies("glm5_next", "metal") and !e.applies("qwen3_5", "cuda"));
        try t.expectEqual(p.Class.identical, e.class);
        try t.expectEqualStrings("#2370", e.evidence);
        try t.expect(p.resolve(.stable, id, null) == null);
        try t.expectEqual(id, p.registered(@tagName(id)).?);
    }
    // timing sections and waiting for an upload choose no policy
    for ([_][]const u8{ "TF_GLM_PROF", "TF_GLM_LINK_WAIT" }) |name| try t.expect(p.registered(name) == null);
}
