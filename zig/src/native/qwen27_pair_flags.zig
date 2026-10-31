//! The flags a pair refuses, named: `qwen27_served` reads `Options` and passes only the flags this answers.

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
