//! Depth policy and shared draft prices supplied by a model.
pub const Constants = struct {
    shared_step_ms: ?f64 = null, // a shared round's head step, when it differs from the lone price

    depth_rate: f64 = 0.15, // weight of the newest round in a depth's acceptance
    learn_rows: bool = false, // shared_costs came from another machine: learn what a row costs here
    depth_probe_every: u64 = 8, // rounds between probes one depth deeper
    cost_rate: f64 = 0.2, // weight of the newest round in a depth's measured time
    plain_margin: f64 = 1.05, // a drafted depth must beat plain rounds' tokens a ms by this much
    plain_wait_most: u64 = 128, // plain rounds between probes at most
    draft_slack: i64 = 2, // drafts a stream offers past what its last shared round granted it
    grant_slack: i64 = 1, // granted depths: drafts past a grant that verified every draft the stream held
    short_slack: i64 = 0, // granted depths: drafts past a grant that left some of them unverified
    copy_first: bool = false, // a copy ahead takes the next round whole: the head drafts none for it (else one)
    floor_rates: bool = false, // under a confidence floor a greedy stream keeps its rates' depth too (else it drafts the most)
    hold_costs: []const @import("hold_price.zig").HoldCost = &.{}, // a drafter's own hold pass by blocks and depth: it prices every hold (lone, shared, granted)
    hold_context_costs: []const @import("hold_price.zig").ContextCost = &.{},
    hold_rows: u32 = 0, // rows one of those hold passes holds at most; a larger hold runs as passes in order (0: no bound)
    hold_block_rows: u32 = 0, // minimum query rows per block, independent of the returned proposal depth
};
