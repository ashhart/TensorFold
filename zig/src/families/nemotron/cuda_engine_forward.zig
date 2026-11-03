//! Construct the bound decode view without changing any of its arithmetic or shared-expert policy.
const Forward = @import("cuda_forward.zig").Forward;
const Dump = @import("cuda_dump.zig").Dump;
pub fn forward(e: anytype, dump: ?*Dump) Forward {
    var f = Forward.init(e.c, &e.w, &e.b, e.ops(), e.max_len, e.nch, e.sampling != null);
    f.page_elems = e.pageBytes() / 2;
    f.dump = dump;
    f.l2pf = if (e.l2pf) |*p| p else null;
    f.slot = e.slot;
    f.untaught = e.bound.untaught;
    f.tp = if (e.tp) |*g| g else null;
    f.vision = e.vision;
    if (dump == null and (e.c.latent > 0 or (e.c.experts <= 128 and e.c.slots() <= 8))) f.side = e.side; // tf_plan_routed's bounds; a latent MoE plans without it
    return f;
}
