//! What the native CUDA server refuses before any device work: flags no CUDA family serves, and --drafter per family.
const std = @import("std");
const api = @import("engine_api");
const nemotron = @import("nemotron");

/// Flags refused whatever the family; null when `o` may open.
pub fn backend(o: api.Open) ?[]const u8 {
    if (o.mtp_drafts) |n| if (n > 15) return "--mtp-drafts: the CUDA engine drafts at most 15";
    if (o.mtp_confidence) |c| if (!(c >= 0 and c <= 1)) return "--mtp-confidence must be from 0 to 1";
    if (o.mtp_drafts != null or o.mtp_confidence != null) {
        if (!std.mem.eql(u8, o.model_type, nemotron.native.model_type) and
            !std.mem.eql(u8, o.model_type, nemotron.omni.model_type))
            return "--mtp-drafts: this CUDA family's drafts are not tunable; serve it without --mtp-drafts or --mtp-confidence";
    }
    if (o.learn != null and !o.learn_conversations) return "--learn learns shared prompt prefixes on Macs only; CUDA keeps conversation states on disk with --learn-conversations: serve with it, or without --learn, --learn-dir or --learn-gib";
    if (o.speed_up != null) return "--speed-up links two Macs: serve without --speed-up on CUDA";
    if (o.prompt_cache_over_cap) return "--prompt-cache-over-cap lifts a Mac's 70% cap: on CUDA the prompt cache takes at most the memory the weights and streams leave, so serve without it";
    return null;
}

/// LFM2 reads a checkpoint's pictures through its own tower without --vision: the flag is refused, not ignored.
pub fn lfm2(o: api.Open) ?[]const u8 {
    return if (o.vision != null) lfm2_vision else null;
}

const lfm2_vision = "--vision: LFM2 reads the checkpoint's images through its own tower without the flag; " ++
    "serve it without --vision";

/// A family takes --drafter only if it says so, and --learn-conversations only as Nemotron; null: `o` may open `F`.
pub fn family(comptime F: type, o: api.Open) ?[]const u8 {
    if (o.vision != null and !(@hasDecl(F, "Options") and @hasField(F.Options, "vision")))
        return "--vision: this CUDA family reads no images; serve it without --vision";
    const ranked = @hasDecl(F, "Options") and @hasField(F.Options, "rank");
    if (o.tp != 1 and !ranked) return "this CUDA family has no tensor-parallel ranks: serve without --tp";
    if (@hasDecl(F, "tp_draft_snapshots")) {
        if (!F.tp_draft_snapshots and o.tp > 1 and o.drafts and (o.prompt_cache_gib orelse 1) > 0)
            return "Flash Next tensor-rank MTP snapshots are unavailable: use --prompt-cache-gib 0 with --tp and drafting";
    }
    if (o.drafter != null and !(@hasDecl(F, "takes_drafter") and F.takes_drafter)) return "--drafter loads a separate drafter, and this CUDA family drafts with its own MTP head: serve without --drafter";
    if (o.learn_conversations and F != nemotron.native) return "--learn-conversations keeps conversation states on disk for Nemotron only on CUDA: serve this model without it";
    if (o.kv8 and !(@hasDecl(F, "Options") and @hasField(F.Options, "kv8"))) return "--kv-cache fp8: this CUDA family keeps bf16 caches; serve without it";
    return null;
}

test "a family must consume rank options before tensor parallelism is admitted" {
    const Single = struct {
        pub const Options = struct { context: usize };
    };
    const Ranked = struct {
        pub const Options = struct { context: usize, rank: usize };
    };
    try std.testing.expectStringStartsWith(family(Single, .{ .dir = "d", .model_type = "m", .tp = 2 }) orelse "", "this CUDA family has no tensor-parallel ranks");
    try std.testing.expect(family(Ranked, .{ .dir = "d", .model_type = "m", .tp = 2 }) == null);
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

test "--vision is each family's to take: Super VL reads images, LFM2 refuses the flag, the rest refuse at their open" {
    for ([_][]const u8{ "nemotron_h", "nemotron_h_omni", "glm5_next", "lfm2_vl" }) |t| {
        try std.testing.expect(backend(.{ .dir = "d", .model_type = t, .vision = .{} }) == null);
    }
    try std.testing.expectEqualStrings(lfm2_vision, lfm2(.{ .dir = "d", .model_type = "lfm2_vl", .vision = .{} }) orelse "");
    try std.testing.expect(lfm2(.{ .dir = "d", .model_type = "lfm2_vl" }) == null);
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

test "--kv-cache fp8 opens only a family whose options take it" {
    const Bf16 = struct {
        pub const Options = struct { context: usize };
    };
    const Fp8 = struct {
        pub const Options = struct { context: usize, kv8: bool = false };
    };
    try std.testing.expectStringStartsWith(family(Bf16, .{ .dir = "d", .model_type = "m", .kv8 = true }) orelse "", "--kv-cache fp8");
    try std.testing.expect(family(Fp8, .{ .dir = "d", .model_type = "m", .kv8 = true }) == null);
    try std.testing.expect(family(Bf16, .{ .dir = "d", .model_type = "m" }) == null);
}

test "a family that explicitly declines a drafter refuses it" {
    const Declined = struct {
        pub const takes_drafter = false;
    };
    try std.testing.expectStringStartsWith(family(Declined, .{ .dir = "d", .model_type = "m", .drafter = "draft" }) orelse "", "--drafter loads a separate drafter");
}

test "--learn-conversations opens Nemotron only, the one CUDA family whose states have a disk image" {
    const Other = struct {};
    try std.testing.expectStringStartsWith(family(Other, .{ .dir = "d", .model_type = "m", .learn_conversations = true }) orelse "", "--learn-conversations keeps");
    try std.testing.expect(family(nemotron.native, .{ .dir = "d", .model_type = "nemotron_h", .learn_conversations = true }) == null);
}

test "CUDA MTP controls respect window and probability bounds" {
    for ([_]u32{ 0, 8, 12, 15 }) |n| {
        try std.testing.expect(backend(.{ .dir = "d", .model_type = "nemotron_h_omni", .mtp_drafts = n }) == null);
    }
    try std.testing.expect(backend(.{ .dir = "d", .model_type = "nemotron_h_omni", .mtp_drafts = 16 }) != null);
    for ([_]f64{ 0, 0.5, 1 }) |c| {
        try std.testing.expect(backend(.{ .dir = "d", .model_type = "nemotron_h", .mtp_confidence = c }) == null);
    }
    for ([_]f64{ -0.1, 1.1, std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) }) |c| {
        try std.testing.expect(backend(.{ .dir = "d", .model_type = "nemotron_h", .mtp_confidence = c }) != null);
    }
}

test "CUDA MTP controls refuse unsupported families before device work" {
    for ([_][]const u8{ "glm5_next", "lfm2_vl" }) |model_type| {
        try std.testing.expect(backend(.{ .dir = "d", .model_type = model_type }) == null);
        try std.testing.expectStringStartsWith(backend(.{ .dir = "d", .model_type = model_type, .mtp_drafts = 0 }) orelse "", "--mtp-drafts");
        try std.testing.expectStringStartsWith(backend(.{ .dir = "d", .model_type = model_type, .mtp_confidence = 0 }) orelse "", "--mtp-drafts");
    }
}
