//! The CUDA drafter's drafts: lanes' shared walker (core/lanes/draft_tree.zig) under the engine's window.
const platform = @import("sys");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const walker = lanes.draft_tree;

pub const Candidates = walker.Candidates;
pub const Landing = walker.Landing;
pub const chain = walker.chain;
pub const lanesOf = walker.lanesOf;

/// A grown tree's nodes at most: walker.nodes under the engine's window, 0 under TF_NO_TREES.
pub fn nodes() !usize {
    return walker.nodes(engine.most_window, platform.c.getenv("TF_NO_TREES") != null);
}

/// walker.grow under the engine's window.
pub fn grow(c: Candidates, pending: u32, rows: usize, most: usize, tokens: []u32, parents: []i32, own: []f64) usize {
    return walker.grow(engine.most_window, c, pending, rows, most, tokens, parents, own);
}
