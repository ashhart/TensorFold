//! Explicit legacy recipes share artifact maps without claiming response calibration or changing their support.
const std = @import("std");
const pv = @import("proposal_vocab.zig");
pub const Parse = struct { vocab: u32, skip_past_vocab: bool = false, delimiters: []const u8 = " \t\r\n", tile: u32 = 1 };

/// Legacy tiling adds real token IDs only for a recipe that already did so; calibrated artifacts never call this.
pub fn ids(gpa: std.mem.Allocator, text: []const u8, recipe: Parse) ![]u32 {
    if (recipe.tile == 0) return error.InvalidTile;
    var listed: std.AutoHashMapUnmanaged(u32, void) = .empty;
    defer listed.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, recipe.delimiters);
    while (it.next()) |token| {
        const id = std.fmt.parseInt(u32, token, 10) catch return error.BadDraftVocab;
        if (id >= recipe.vocab) {
            if (recipe.skip_past_vocab) continue;
            return error.DraftIdPastVocab;
        }
        try listed.put(gpa, id, {});
    }
    if (listed.count() == 0) return error.BadDraftVocab;
    var candidate: u32 = 0;
    while (listed.count() % recipe.tile != 0) : (candidate += 1) {
        if (candidate >= recipe.vocab) return error.DraftVocabExhausted;
        if (!listed.contains(candidate)) try listed.put(gpa, candidate, {});
    }
    const out = try gpa.alloc(u32, listed.count());
    var keys = listed.keyIterator();
    var i: usize = 0;
    while (keys.next()) |key| : (i += 1) out[i] = key.*;
    std.mem.sort(u32, out, {}, std.sort.asc(u32));
    return out;
}
pub const Layout = struct { vocab: u32, hidden: u32, format: []const u8, group: u32, scale_recipe: []const u8 };

/// Trusted built-in metadata only; checkpoint admission remains the existing owner's and no calibration is invented.
pub fn artifact(gpa: std.mem.Allocator, logical: []const u32, layout: Layout) !pv.Artifact {
    if (logical.len == 0 or logical.len > layout.vocab or layout.hidden == 0) return error.InvalidIds;
    for (logical, 0..) |id, i| if (id >= layout.vocab or (i > 0 and logical[i - 1] >= id)) return error.InvalidIds;
    const arena = try gpa.create(std.heap.ArenaAllocator);
    arena.* = .init(gpa);
    errdefer {
        arena.deinit();
        gpa.destroy(arena);
    }
    const a = arena.allocator();
    const hash = try a.dupe(u8, &pv.idsHash(logical));
    const binding: pv.Binding = .{ .name = "admitted-legacy-table", .revision = "legacy", .sha256 = hash };
    const identity: pv.Identity = .{ .target = binding, .drafter = binding, .head = binding, .tokenizer = binding, .recipe = binding, .vocab = layout.vocab, .hidden = layout.hidden, .format = try a.dupe(u8, layout.format), .group = layout.group, .scale_recipe = try a.dupe(u8, layout.scale_recipe) };
    const doc: pv.Document = .{ .version = 1, .identity = identity, .ids = try a.dupe(u32, logical), .ids_sha256 = hash, .candidate_size = @intCast(logical.len), .mandatory_ids = &.{}, .coverage = &.{}, .provenance = .{ .corpus_sha256 = "", .prompts_sha256 = "", .responses_sha256 = "", .calibration_sha256 = "", .selection_sha256 = "", .heldout_sha256 = "", .settings_sha256 = "", .collector = binding, .mandatory_source = "existing admitted recipe", .policy_version = 0, .source_kind = .legacy_table } };
    const encoded = try std.json.Stringify.valueAlloc(a, doc, .{});
    var fingerprint: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(encoded, &fingerprint, .{});
    a.free(encoded);
    return .{ .owned = .{ .arena = arena, .value = doc }, .fingerprint = fingerprint };
}
