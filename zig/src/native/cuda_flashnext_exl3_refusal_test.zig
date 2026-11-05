//! Flash Next EXL3 rank admission and draft-cache refusals.
const std = @import("std");
const api = @import("engine_api");
const F = @import("flashnext").native;
const refusals = @import("cuda_refusal.zig");

test "Flash Next serves --tp for EXL3 packs only and keeps the tensor-rank draft refusal for any other pack" {
    const o: api.Open = .{ .dir = "/missing-flashnext-test-checkpoint", .model_type = F.model_type, .tp = 2, .rank = 1, .drafts = true, .prompt_cache_gib = 4 };
    try std.testing.expect(refusals.admitted(F, o, true) == null);
    try std.testing.expectStringStartsWith(refusals.admitted(F, o, false) orelse "", "Flash Next tensor-rank MTP snapshots are unavailable");
    try std.testing.expect(refusals.packRanks(F, o, true) == null);
    try std.testing.expectStringStartsWith(refusals.packRanks(F, o, false) orelse "", "--tp: on CUDA only an EXL3 pack");
    var one = o;
    one.tp = 1;
    try std.testing.expect(refusals.packRanks(F, one, false) == null);
    try std.testing.expect(refusals.admitted(F, one, false) == null);
    // a directory that holds no checkpoint is no EXL3 pack
    try std.testing.expect(!F.exl3_pack(std.testing.allocator, std.testing.io, o.dir));
}
