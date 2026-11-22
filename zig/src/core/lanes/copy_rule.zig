//! A copy prefix's expected tokens per millisecond; no shared or model-drafter depth selection.
const depth = @import("depth.zig");
const price = @import("copy_price.zig");

pub fn rows(r: *depth.Rule, who: depth.Who, chances: []const f64, most: i64, take: bool) !i64 {
    if (most <= 0 or !r.copy_guard or r.family_costs.empty()) return most;
    const plain = (try price.plain(r, who)) orelse return most;
    var best: i64 = 1;
    var best_rate: f64 = -1;
    if (plain != 0) {
        best = 0;
        best_rate = r.k.plain_margin / plain;
    }
    var drafted: i64 = 1;
    var drafted_rate: f64 = -1;
    var expected: f64 = 1;
    var run: f64 = 1;
    var d: i64 = 1;
    while (d <= most) : (d += 1) {
        const cost = (try price.cost(r, d, who)) orelse break;
        const at: usize = @intCast(d - 1);
        run *= if (at < chances.len) chances[at] else chances[chances.len - 1];
        expected += run;
        const rate = expected / cost;
        if (rate > best_rate) {
            best = d;
            best_rate = rate;
        }
        if (rate > drafted_rate) {
            drafted = d;
            drafted_rate = rate;
        }
    }
    const st = &who.state.*.?;
    if (best > 0) {
        if (take) {
            st.copy_plain = 0;
            st.copy_wait = r.k.depth_probe_every;
        }
        return best;
    }
    const next = (st.copy_plain orelse 0) + 1;
    const wait = st.copy_wait orelse r.k.depth_probe_every;
    const probe = next >= wait;
    if (take) {
        st.copy_plain = if (probe) 0 else next;
        if (probe) st.copy_wait = @min(2 * wait, r.k.plain_wait_most);
    }
    return if (probe) drafted else 0;
}
