//! Nemotron as Living Weights reads it: its family (living/families.zig) and the layout its config gives.
const living = @import("living");
const Config = @import("config.zig").Config;

pub const family = living.families.nemotron_h;

/// Every MoE layer's shared expert carries the pocket; all but the last layer's learn.
pub fn layout(c: Config) living.Layout {
    var moe: living.Layers = .empty;
    for (0..c.layers) |l| if (c.kinds[l] == .moe) moe.set(l);
    return .fromExperts(c.hidden, c.shared_width, c.layers, moe);
}
