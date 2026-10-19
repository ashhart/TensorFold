//! A family adds records only for symbols the common registry does not already own.
const std = @import("std");
fn has(comptime prior: anytype, comptime record: anytype) bool {
    for (prior) |r| if (std.mem.eql(u8, r.launch.image, record.launch.image) and std.mem.eql(u8, r.launch.symbol, record.launch.symbol)) return true;
    return false;
}
fn count(comptime prior: anytype, comptime additional: anytype) usize {
    @setEvalBranchQuota(4000000);
    var n: usize = 0;
    for (additional) |r| if (!has(prior, r)) {
        n += 1;
    };
    return n;
}
pub fn extra(comptime prior: anytype, comptime additional: anytype) [count(prior, additional)]@TypeOf(additional[0]) {
    @setEvalBranchQuota(4000000);
    var out: [count(prior, additional)]@TypeOf(additional[0]) = undefined;
    var n: usize = 0;
    for (additional) |r| if (!has(prior, r)) {
        out[n] = r;
        n += 1;
    };
    return out;
}
