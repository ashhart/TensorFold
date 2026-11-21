//! Choose sibling drafts and their Markov continuations.
const Allocator = @import("std").mem.Allocator;
const ln = @import("lanes.zig");
const dspark = @import("dspark.zig");
const iface = @import("iface.zig");
const tree = @import("tree.zig");

/// The siblings after the pending token: the draft distribution's next first tokens and their Markov chains.
pub fn siblings(opts: ln.Options, a: Allocator, st: *ln.St, m: dspark.Markov, p: iface.Proposal, ask: iface.Ask) !void {
    const c = opts.shape.candidates;
    const z0 = try a.alloc(f32, c);
    dspark.markovZ(m, p.cand.?[0..c], p.base.?[0..c], ask.anchor, z0);
    const smp = ask.params.sampling();
    const got = try tree.ranked(a, p.cand.?[0..c], z0, ask.start + 1, smp, st.held[0..1], st.sib[0..opts.siblings]);
    const rows = @min(opts.sib_rows, st.n);
    for (st.sib[0..got], 0..) |sib, j| {
        st.sib_tokens[j][0] = sib.token;
        const more = try tree.continuation(a, m, p.cand.?, p.base.?, c, sib.token, rows, ask.params, st.sib_tokens[j][1..]);
        st.sib_len[j] = @intCast(1 + more);
    }
    st.n_sib = @intCast(got);
}
