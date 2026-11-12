//! The flags a pair refuses: `qwen27_served` maps `Options` through `refuseOptions`, so only these words answer.

pub fn refuse(drafter: bool, cache_gib: bool, learn: bool, vision: bool, why: *[]const u8) !void {
    // A drafter on a pair runs on rank 0 alone: rank 1 replays the tree rounds (.tree/.commit) and never proposes.
    _ = drafter;
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

/// TF_QWEN27_LANES=0's serial draft loop sends rank 1 no rounds: a drafted pair refuses it rather than wait on rank 1.
pub fn refuseSerial(pair: bool, drafting: bool, serial: bool, why: *[]const u8) !void {
    if (!(pair and drafting and serial)) return;
    why.* = "TF_QWEN27_LANES=0 does not run on a pair with --drafter: its serial draft loop sends rank 1 no rounds, so rank 0 would wait at its first seam; unset it";
    return error.PairSerialDraft;
}

/// The flags `qwen27_served.Options` carries, mapped onto `refuse`'s booleans: `open` and `attach` both pass theirs.
pub fn refuseOptions(pair: bool, o: anytype, bound_drafter: bool, why: *[]const u8) !void {
    if (!pair) return;
    return refuse(bound_drafter or o.drafter != null, o.cache_gib != null, o.learn != null, o.vision != null or o.images != null, why);
}

/// The policy line a pair's `/health` shows; one Mac's is empty, which `/health` leaves out, as on main.
pub fn healthPolicy(pair: bool) []const u8 {
    return if (pair) "speed-up mode: a two-Mac pair, rank 0 answers greedy requests only" else "";
}

/// The host's `Info` with a pair's policy line: `attach` passes its own through, so one Mac's comes back unchanged.
pub fn withPolicy(info: anytype, pair: bool) @TypeOf(info) {
    if (!pair) return info;
    var out = info;
    out.policy = healthPolicy(true);
    return out;
}
