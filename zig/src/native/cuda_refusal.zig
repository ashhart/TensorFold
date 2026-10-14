//! What the native CUDA server refuses before any device work: flags no CUDA family serves, and --drafter per family.
const std = @import("std");
const api = @import("engine_api");

/// Flags refused whatever the family; null when `o` may open.
pub fn backend(o: api.Open) ?[]const u8 {
    if (o.learn != null and !o.learn_conversations) return "--learn learns shared prompt prefixes on Macs only; CUDA keeps conversation states on disk with --learn-conversations: serve with it, or without --learn, --learn-dir or --learn-gib";
    if (o.speed_up != null) return "--speed-up links two Macs: serve without --speed-up on CUDA";
    if (o.prompt_cache_over_cap) return "--prompt-cache-over-cap lifts a Mac's 70% cap: on CUDA the prompt cache takes at most the memory the weights and streams leave, so serve without it";
    return null;
}

/// A family takes --drafter only if it says so (`takes_drafter`); null when `o` may open `F`.
pub fn family(comptime F: type, o: api.Open) ?[]const u8 {
    if (o.drafter != null and !(@hasDecl(F, "takes_drafter") and F.takes_drafter)) return "--drafter loads a separate drafter, and this CUDA family drafts with its own MTP head: serve without --drafter";
    return null;
}

test "--learn is refused whatever the family, and --slide or a learned-disk size alone is not" {
    try std.testing.expectStringStartsWith(backend(.{ .dir = "d", .model_type = "nemotron_h", .learn = "/x" }) orelse "", "--learn learns shared prompt prefixes on Macs only");
    try std.testing.expect(backend(.{ .dir = "d", .model_type = "nemotron_h", .learn = "/x", .learn_conversations = true }) == null);
    try std.testing.expect(backend(.{ .dir = "d", .model_type = "nemotron_h", .slide = true, .learn_gib = 8, .learn_min_free_gib = 2 }) == null);
}

test "flags with nothing to do on CUDA are refused, not ignored" {
    try std.testing.expectStringStartsWith(backend(.{ .dir = "d", .model_type = "nemotron_h", .speed_up = "rank0.json" }) orelse "", "--speed-up links two Macs");
    try std.testing.expectStringStartsWith(backend(.{ .dir = "d", .model_type = "nemotron_h", .prompt_cache_over_cap = true }) orelse "", "--prompt-cache-over-cap");
    try std.testing.expect(backend(.{ .dir = "d", .model_type = "nemotron_h", .prompt_cache_gib = 4 }) == null);
}

test "--drafter opens only a family that takes one" {
    const Mtp = struct {};
    const Drafted = struct {
        pub const takes_drafter = true;
    };
    try std.testing.expectStringStartsWith(family(Mtp, .{ .dir = "d", .model_type = "m", .drafter = "/x" }) orelse "", "--drafter loads a separate drafter");
    try std.testing.expect(family(Drafted, .{ .dir = "d", .model_type = "m", .drafter = "/x" }) == null);
    try std.testing.expect(family(Mtp, .{ .dir = "d", .model_type = "m" }) == null);
}

test "a family that explicitly declines a drafter refuses it" {
    const Declined = struct {
        pub const takes_drafter = false;
    };
    try std.testing.expectStringStartsWith(family(Declined, .{ .dir = "d", .model_type = "m", .drafter = "draft" }) orelse "", "--drafter loads a separate drafter");
}
