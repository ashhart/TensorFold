//! Family reasoning metadata carried into the CUDA lane host.
const std = @import("std");
const api = @import("engine_api");
const registry = @import("cuda_families.zig").registry;

/// The Info `openWith` serves: the host's fields, then what the family declares (its reasoning block).
pub fn familyInfo(comptime F: type, host: api.Info) api.Info {
    var info = host;
    info.reasoning = reasoning(F);
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
}
