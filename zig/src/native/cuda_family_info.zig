//! Family reasoning metadata carried into the CUDA lane host.
const std = @import("std");
const api = @import("engine_api");
const registry = @import("cuda_families.zig").registry;

/// The Info `openWith` serves: the host's fields, then what the family declares (its reasoning block, its think rules).
pub fn familyInfo(comptime F: type, host: api.Info) api.Info {
    var info = host;
    info.reasoning = reasoning(F);
    if (@hasDecl(F, "thinking")) F.thinking(&info); // a family's own think block rules
    if (@hasDecl(F, "reasoning_optional")) info.reasoning_optional = F.reasoning_optional;
    if (@hasDecl(F, "thinking_budget")) info.thinking_budget = F.thinking_budget;
    return info;
}

/// The family's Info.reasoning: false when it declares `pub const reasoning = false` (no reasoning block).
fn reasoning(comptime F: type) bool {
    return if (@hasDecl(F, "reasoning")) F.reasoning else true;
}

test "a CUDA family declares replies without reasoning with `pub const reasoning = false`; only dots.mocr does" {
    try std.testing.expect(reasoning(struct {}));
    try std.testing.expect(!reasoning(struct {
        pub const reasoning = false;
    }));
    inline for (registry) |F| try std.testing.expectEqual(!std.mem.eql(u8, F.model_type, "dots_ocr"), reasoning(F));
}

test "the Info openWith serves carries each family's reasoning declaration over the host's" {
    const Plain = struct {
        pub const reasoning = false;
    };
    try std.testing.expect(!familyInfo(Plain, .{ .reasoning = true }).reasoning);
    try std.testing.expect(familyInfo(struct {}, .{ .reasoning = false }).reasoning);
    const kept = familyInfo(Plain, .{ .lanes = 6, .context_window = 8192, .startup = "plan" });
    try std.testing.expect(kept.lanes == 6 and kept.context_window == 8192 and std.mem.eql(u8, kept.startup, "plan"));
    inline for (registry) |F| try std.testing.expectEqual(reasoning(F), familyInfo(F, .{ .reasoning = !reasoning(F) }).reasoning);
    const bare = familyInfo(@import("ricursive_cuda").native, .{}); // Ripple's think rules: <think> optional, no budget
    try std.testing.expect(bare.reasoning_optional and !bare.thinking_budget);
}

test "optional reasoning and its budget are family metadata; other host flags stay unchanged" {
    const Optional = struct {
        pub const reasoning_optional = true;
        pub const thinking_budget = false;
    };
    const host: api.Info = .{ .lanes = 3, .context_window = 4096, .reasoning_optional = false, .thinking_budget = true };
    const chosen = familyInfo(Optional, host);
    try std.testing.expect(chosen.reasoning and chosen.reasoning_optional and !chosen.thinking_budget);
    try std.testing.expect(chosen.lanes == host.lanes and chosen.context_window == host.context_window);
    const untouched = familyInfo(struct {}, host);
    try std.testing.expect(!untouched.reasoning_optional and untouched.thinking_budget);
    inline for (registry) |F| {
        const info = familyInfo(F, host);
        const optional = std.mem.eql(u8, F.model_type, "ricursive");
        try std.testing.expectEqual(optional, info.reasoning_optional);
        try std.testing.expectEqual(!optional, info.thinking_budget);
    }
}
