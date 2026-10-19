//! Failed settle/release owners stay attached to an unavailable slot until recovery completes.
const ln = @import("lanes.zig");
pub fn retire(x: *ln.Lanes, st: *ln.St) bool {
    x.settle() catch return false;
    if (!x.release_requested[st.slot]) {
        x.target.releaseChecked(st.slot) catch return false;
        x.release_requested[st.slot] = true;
    }
    if (!x.target.available(st.slot)) return false;
    if (x.spec) |*sp| sp.forget(st.slot);
    x.pass.reset(st.slot);
    x.policy.depth.reset(st.slot);
    x.by_slot[st.slot] = null;
    x.retired[st.slot] = false;
    x.release_requested[st.slot] = false;
    x.free.appendAssumeCapacity(st.slot);
    x.drop(st);
    return true;
}
pub fn recover(x: *ln.Lanes) void {
    for (x.retired, 0..) |retired, i| if (retired) {
        _ = retire(x, x.by_slot[i].?);
    };
}
