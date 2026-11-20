//! SM110's Qwen development build serves the measured default path only.
const std = @import("std");
const cuda = @import("cuda");
const profile = @import("core").profile;
const environment = @import("sys").environment;

pub const Modes = struct {
    slide: bool = false,
    dspine: bool = false,
    dspine_window: ?[]const u8 = null,
    kv8: bool = false,
    prompt_int8: bool = false,
    rope_yarn: bool = false,
    kv_pages: ?[]const u8 = null,
    kv_pages_all: ?[]const u8 = null,
    kv_share: ?[]const u8 = null,
    request_starts: ?[]const u8 = null,
    kv_page_test: ?[]const u8 = null,
    keep_late: ?[]const u8 = null,
    no_trees: ?[]const u8 = null,
    draft_nodes: ?[]const u8 = null,
};

pub fn current(kv8: bool, prompt_int8: bool, rope_yarn: bool) Modes {
    return .{
        .dspine_window = profile.lookup("TF_DSPINE_WINDOW"),
        .kv8 = kv8,
        .prompt_int8 = prompt_int8,
        .rope_yarn = rope_yarn,
        .kv_pages = profile.get(.TF_QWEN_KV_PAGES),
        .kv_pages_all = profile.get(.TF_QWEN_KV_PAGES_ALL),
        .kv_share = profile.get(.TF_QWEN_KV_SHARE),
        .request_starts = profile.get(.TF_QWEN_REQUEST_STARTS),
        .kv_page_test = environment.Process.get(.{}, "TF_QWEN_KV_PAGE_TEST"),
        .keep_late = environment.Process.get(.{}, "TF_QWEN_KEEP_LATE"),
        .no_trees = environment.Process.get(.{}, "TF_NO_TREES"),
        .draft_nodes = environment.Process.get(.{}, "TF_DRAFT_NODES"),
    };
}

pub fn checkDevice(ctx: *const cuda.Context, now: Modes) !void {
    if (!needsDevice(now)) return;
    try check(try ctx.capability(), now);
}

pub fn check(sm: u32, modes: Modes) !void {
    if (sm != 110) return;
    if (modes.slide) return error.QwenSm110SlideUnqualified;
    if (modes.dspine) return error.QwenSm110DspineUnqualified;
    if (count(modes.dspine_window)) return error.QwenSm110DspineWindowUnqualified;
    if (modes.kv8) return error.QwenSm110Fp8KvUnqualified;
    if (modes.prompt_int8) return error.QwenSm110Int8PromptUnqualified;
    if (modes.rope_yarn) return error.QwenSm110RopeYarnUnqualified;
    if (count(modes.kv_pages)) return error.QwenSm110KvPagesUnqualified;
    if (one(modes.kv_pages_all)) return error.QwenSm110KvPagesAllUnqualified;
    if (count(modes.kv_share)) return error.QwenSm110KvShareUnqualified;
    if (one(modes.request_starts)) return error.QwenSm110RequestStartsUnqualified;
    if (modes.kv_page_test != null) return error.QwenSm110KvPageTestUnqualified;
    if (one(modes.keep_late)) return error.QwenSm110KeepLateUnqualified;
    if (modes.no_trees != null) return error.QwenSm110NoTreesUnqualified;
    if (modes.draft_nodes != null) return error.QwenSm110DraftNodesUnqualified;
}

fn count(raw: ?[]const u8) bool {
    const v = raw orelse return false;
    return (std.fmt.parseInt(usize, v, 10) catch return true) != 0;
}

fn one(raw: ?[]const u8) bool {
    return std.mem.eql(u8, raw orelse return false, "1");
}

fn needsDevice(modes: Modes) bool {
    return modes.slide or modes.dspine or count(modes.dspine_window) or modes.kv8 or modes.prompt_int8 or modes.rope_yarn or count(modes.kv_pages) or
        one(modes.kv_pages_all) or count(modes.kv_share) or one(modes.request_starts) or
        modes.kv_page_test != null or one(modes.keep_late) or modes.no_trees != null or modes.draft_nodes != null;
}

pub fn explain(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.QwenSm110SlideUnqualified => "--slide is not qualified for Qwen on sm_110; omit it",
        error.QwenSm110DspineUnqualified => "DSpine --drafter is not qualified for Qwen on sm_110; use DFlash2",
        error.QwenSm110DspineWindowUnqualified => "TF_DSPINE_WINDOW is not qualified for Qwen on sm_110; unset it",
        error.QwenSm110Fp8KvUnqualified => "--kv-cache fp8 is not qualified for Qwen on sm_110; use bf16",
        error.QwenSm110Int8PromptUnqualified => "--prompt-precision int8 is not qualified for Qwen on sm_110; use bf16",
        error.QwenSm110RopeYarnUnqualified => "--rope-yarn is not qualified for Qwen on sm_110; omit it",
        error.QwenSm110KvPagesUnqualified => "TF_QWEN_KV_PAGES is not qualified for Qwen on sm_110; unset it or use 0",
        error.QwenSm110KvPagesAllUnqualified => "TF_QWEN_KV_PAGES_ALL=1 is not qualified for Qwen on sm_110; unset it",
        error.QwenSm110KvShareUnqualified => "TF_QWEN_KV_SHARE is not qualified for Qwen on sm_110; unset it or use 0",
        error.QwenSm110RequestStartsUnqualified => "TF_QWEN_REQUEST_STARTS=1 is not qualified for Qwen on sm_110; unset it",
        error.QwenSm110KvPageTestUnqualified => "TF_QWEN_KV_PAGE_TEST is not qualified for Qwen on sm_110; unset it",
        error.QwenSm110KeepLateUnqualified => "TF_QWEN_KEEP_LATE=1 is not qualified for Qwen on sm_110; unset it",
        error.QwenSm110NoTreesUnqualified => "TF_NO_TREES is not qualified for Qwen on sm_110; unset it",
        error.QwenSm110DraftNodesUnqualified => "TF_DRAFT_NODES is not qualified for Qwen on sm_110; unset it",
        else => null,
    };
}

test "SM110 accepts defaults, including explicit off values; other SMs retain every mode" {
    try check(110, .{});
    try check(110, .{ .kv_pages = "0", .kv_share = "0", .kv_pages_all = "0", .request_starts = "0", .keep_late = "0" });
    try check(121, .{ .kv8 = true, .prompt_int8 = true, .kv_pages = "256", .keep_late = "1" });
    try std.testing.expect(!needsDevice(.{ .kv_pages = "0", .kv_share = "0", .kv_pages_all = "0", .request_starts = "0", .keep_late = "0" }));
    try std.testing.expect(needsDevice(.{ .kv8 = true }));
}

test "SM110 refuses every unqualified Qwen serving mode with an explanation" {
    const cases = [_]struct { modes: Modes, err: anyerror }{
        .{ .modes = .{ .slide = true }, .err = error.QwenSm110SlideUnqualified },
        .{ .modes = .{ .dspine = true }, .err = error.QwenSm110DspineUnqualified },
        .{ .modes = .{ .dspine_window = "256" }, .err = error.QwenSm110DspineWindowUnqualified },
        .{ .modes = .{ .kv8 = true }, .err = error.QwenSm110Fp8KvUnqualified },
        .{ .modes = .{ .prompt_int8 = true }, .err = error.QwenSm110Int8PromptUnqualified },
        .{ .modes = .{ .rope_yarn = true }, .err = error.QwenSm110RopeYarnUnqualified },
        .{ .modes = .{ .kv_pages = "256" }, .err = error.QwenSm110KvPagesUnqualified },
        .{ .modes = .{ .kv_pages_all = "1" }, .err = error.QwenSm110KvPagesAllUnqualified },
        .{ .modes = .{ .kv_share = "32" }, .err = error.QwenSm110KvShareUnqualified },
        .{ .modes = .{ .request_starts = "1" }, .err = error.QwenSm110RequestStartsUnqualified },
        .{ .modes = .{ .kv_page_test = "256" }, .err = error.QwenSm110KvPageTestUnqualified },
        .{ .modes = .{ .keep_late = "1" }, .err = error.QwenSm110KeepLateUnqualified },
        .{ .modes = .{ .no_trees = "1" }, .err = error.QwenSm110NoTreesUnqualified },
        .{ .modes = .{ .draft_nodes = "16" }, .err = error.QwenSm110DraftNodesUnqualified },
    };
    for (cases) |case| {
        try std.testing.expectError(case.err, check(110, case.modes));
        try std.testing.expect(std.mem.indexOf(u8, explain(case.err).?, "sm_110") != null);
    }
    try std.testing.expectError(error.QwenSm110NoTreesUnqualified, check(110, .{ .no_trees = "0" }));
}

test {
    _ = @import("cuda_sm110_open_test.zig");
}
