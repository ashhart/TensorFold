//! A shared pass's lanes: what a lane may hold, and what its rollback may keep.
const std = @import("std");
const state = @import("cuda_batch_state.zig");

const Keep = state.Keep;
const HcKeep = state.HcKeep;
const LayerKeep = state.LayerKeep;
const Lane = state.Lane;
const att_rows = state.att_rows;

/// Lanes whose rows add up to `n`, each window within its cache and the attention block, keeping a rollback at most.
pub fn checkLanes(lanes: []const Lane, n: usize) error{UnexpectedTensor}!void {
    var total: usize = 0;
    for (lanes) |l| {
        if (l.rows == 0 or l.rows > att_rows or l.pos + l.rows > l.seq.cap) return error.UnexpectedTensor;
        if (l.keep) |kp| if (!rollbackOnly(kp)) return error.UnexpectedTensor;
        total += l.rows;
    }
    if (total != n) return error.UnexpectedTensor;
}

/// A drafted window's rollback (the DeltaNet rows, history and end states alone), not a training pass's tensors.
pub fn rollbackOnly(k: *const Keep) bool {
    if (!std.meta.eql(k.mixer, HcKeep{})) return false;
    for (k.layers) |lk| {
        var rest = lk;
        rest.proj = 0;
        rest.hist = 0;
        rest.state_out = 0;
        if (!std.meta.eql(rest, LayerKeep{})) return false;
    }
    return true;
}
