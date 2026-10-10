//! Living Weights on the full GLM-5.3 (glm53): the host-only pieces as one module (no Metal), imported by
//! tf-glm53 as "glm53_lw" and tested by `zig build test` on any OS. The learner state machine, Sites and the shared host
//! math are the GLM-5.3-Flash learner's (families/glm/lw_*.zig), reused unchanged.
pub const sidecar = @import("families/glm53/lw_sidecar.zig");
pub const tail = @import("families/glm53/lw_tail.zig");
pub const lm = @import("families/glm/lw_math.zig");
pub const sites = @import("families/glm/lw_sites.zig");
pub const learner = @import("families/glm/lw_learner.zig");

test {
    _ = sidecar;
    _ = tail;
    _ = lm;
    _ = sites;
    _ = learner;
}
