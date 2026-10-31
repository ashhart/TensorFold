//! The words a pair gives: the flags `qwen27_served` refuses and the policy line `/health` reports.
const std = @import("std");

pub fn refuse(drafter: bool, cache_gib: bool, learn: bool, vision: bool, why: *[]const u8) !void {
    if (drafter) {
        why.* = "--drafter does not run on a pair: rank 1 has no drafter checkpoint, so both ranks decode plain";
        return error.PairNoDrafter;
    }
    if (cache_gib) {
        why.* = "--prompt-cache-gib does not run on a pair: each Mac keeps its own halves of the slots' states";
        return error.PairNoCache;
    }
    if (learn) {
        why.* = "--learn does not run on a pair: each Mac learns its own halves of the slots' states";
        return error.PairNoLearn;
    }
    if (vision) {
        why.* = "--vision does not run on a pair: rank 1 replays rank 0's token ids, so an image prompt would run different rows on the two ranks";
        return error.PairNoVision;
    }
}

/// The line `/health` reports as policy: a pair's rank 0 answers greedy requests only, so the recorder refuses it.
pub fn healthPolicy(pair: bool) []const u8 {
    return if (pair) "speed-up mode: a two-Mac pair, rank 0 answers greedy requests only" else "";
}

test "a pair's /health names the two-Mac mode; one Mac's names no policy" {
    // the recorder refuses on the word "pair" in this line, so the line has to carry it; one Mac reports none
    try std.testing.expectEqualStrings("", healthPolicy(false));
    const pair = healthPolicy(true);
    try std.testing.expect(std.mem.indexOf(u8, pair, "pair") != null);
    try std.testing.expect(std.mem.indexOf(u8, pair, "greedy") != null);
}
