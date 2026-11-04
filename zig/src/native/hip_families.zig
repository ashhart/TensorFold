//! What the HIP host reads from a family's namespace and its Loaded: names, refusals and its offer.
const std = @import("std");
const api = @import("engine_api");
const Offer = @import("hip_serve").Offer;

/// Whether `F` reads `model_type`: its own, or one of its `also_model_types`.
pub fn reads(comptime F: type, model_type: []const u8) bool {
    if (std.mem.eql(u8, model_type, F.model_type)) return true;
    if (comptime @hasDecl(F, "also_model_types")) for (F.also_model_types) |alias| {
        if (std.mem.eql(u8, model_type, alias)) return true;
    };
    return false;
}

/// One capability entry per model_type a family reads, aliases after its own, with its `raw_text` when it names one.
pub fn list(comptime registry: anytype) []const api.Family {
    comptime {
        var count: usize = 0;
        for (registry) |F| count += 1 + if (@hasDecl(F, "also_model_types")) F.also_model_types.len else 0;
        var out: [count]api.Family = undefined;
        var i: usize = 0;
        for (registry) |F| {
            const raw = if (@hasDecl(F, "raw_text")) F.raw_text else true;
            out[i] = .{ .model_type = F.model_type, .formats = F.formats, .raw_text = raw };
            i += 1;
            if (@hasDecl(F, "also_model_types")) for (F.also_model_types) |alias| {
                out[i] = .{ .model_type = alias, .formats = F.formats, .raw_text = raw };
                i += 1;
            };
        }
        const final = out;
        return &final;
    }
}

/// Why `F` cannot take these flags, before any load or group join: its own `refusal(o)` first, then the shared ones.
pub fn refused(comptime F: type, o: api.Open) ?[]const u8 {
    if (comptime @hasDecl(F, "refusal")) if (F.refusal(o)) |why| return why;
    if (o.rope_yarn != null) return "--rope-yarn: the native HIP engine has no YaRN rope; serve without --rope-yarn, or on CUDA";
    if (o.slide and @hasDecl(F, "slide_refusal")) return F.slide_refusal;
    if (o.mtp_confidence != null and @hasDecl(F, "confidence_refusal")) return F.confidence_refusal;
    return null;
}

/// The family's offer, or the empty one (text, the host's defaults) when its Loaded carries none.
pub fn offerOf(loaded: anytype) Offer {
    return if (comptime @hasField(@TypeOf(loaded), "offer")) loaded.offer else .{};
}

/// Whether the family generates text: its offer says so and it answers no readouts.
pub fn generates(offer: Offer) bool {
    return offer.text_generation and offer.readouts == null;
}

/// --vision asked of a model this family serves without images: it starts text-only and logs so.
pub fn textOnly(o: api.Open, offer: Offer) bool {
    return o.vision != null and offer.vision == null;
}

const Plain = struct {
    pub const model_type = "plain";
    pub const formats: []const []const u8 = &.{"mlx"};
};
const Aliased = struct {
    pub const model_type = "first";
    pub const also_model_types: []const []const u8 = &.{"second"};
    pub const formats: []const []const u8 = &.{"bf16"};
    pub const raw_text = false;
    pub fn refusal(o: api.Open) ?[]const u8 {
        return if (o.tp > 1) "--tp: this family serves one rank" else null;
    }
};

test "a family reads its own model_type and its aliases; the list has one entry each, with raw_text" {
    try std.testing.expect(reads(Plain, "plain") and !reads(Plain, "first"));
    try std.testing.expect(reads(Aliased, "first") and reads(Aliased, "second") and !reads(Aliased, "plain"));
    const families = comptime list(.{ Plain, Aliased });
    try std.testing.expectEqual(@as(usize, 3), families.len);
    try std.testing.expectEqualStrings("plain", families[0].model_type);
    try std.testing.expect(families[0].raw_text);
    try std.testing.expectEqualStrings("second", families[2].model_type);
    try std.testing.expectEqualStrings("bf16", families[2].formats[0]);
    try std.testing.expect(!families[1].raw_text and !families[2].raw_text);
}

test "a family's own refusal comes first; --vision alone refuses nothing" {
    const many: api.Open = .{ .dir = "d", .model_type = "first", .tp = 2, .rope_yarn = 2.0 };
    try std.testing.expectEqualStrings("--tp: this family serves one rank", refused(Aliased, many).?);
    try std.testing.expect(std.mem.startsWith(u8, refused(Plain, many).?, "--rope-yarn"));
    const seen: api.Open = .{ .dir = "d", .model_type = "plain", .vision = .{} };
    try std.testing.expect(refused(Plain, seen) == null and refused(Aliased, seen) == null);
}

test "a family that answers readouts generates no text" {
    try std.testing.expect(generates(.{}) and !generates(.{ .text_generation = false }));
    try std.testing.expect(!generates(.{ .readouts = .{ .max_rows = 16384, .max_questions = 64, .max_options = 1024 } }));
}

test "a Loaded without an offer gives the empty one; --vision with no offered images is text-only" {
    const Bare = struct { streams: usize = 1 };
    const Offering = struct { offer: Offer };
    const empty = offerOf(Bare{});
    try std.testing.expect(empty.vision == null and empty.cache == null and empty.policy == null);
    const warm = offerOf(Offering{ .offer = .{ .warm_turns = true, .policy = "fixed" } });
    try std.testing.expect(warm.warm_turns);
    try std.testing.expectEqualStrings("fixed", warm.policy.?);
    const seen: api.Open = .{ .dir = "d", .model_type = "plain", .vision = .{} };
    try std.testing.expect(textOnly(seen, empty));
    try std.testing.expect(!textOnly(.{ .dir = "d", .model_type = "plain" }, empty));
}
