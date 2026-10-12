//! Qwen3.8 Flash Next for the native HIP engine: the family's config and index checks, the plan of a rank's bytes.
pub const config = @import("config.zig");
pub const Config = config.Config;
pub const index = @import("index.zig");
pub const Spec = @import("spec.zig").Spec;
pub const plan = @import("hip/plan.zig");
pub const ngram = @import("hip/ngram.zig");
pub const model = @import("hip/model.zig");
pub const load = @import("hip/load.zig");
pub const cache = @import("hip/cache.zig");
pub const tables = @import("hip/tables.zig");
pub const forward = @import("hip/forward.zig");
pub const mixers = @import("hip/mixers.zig");
pub const engine = @import("hip/engine.zig");
pub const head = @import("hip/head.zig");
pub const lanes = @import("hip/lanes.zig");
pub const snapshot = @import("hip/snapshot.zig");
/// The registry entry of the native server.
pub const native = @import("hip/native.zig");

test {
    _ = config;
    _ = index;
    _ = plan;
    _ = ngram;
    _ = model;
    _ = cache;
    @import("std").testing.refAllDecls(engine.Engine);
    @import("std").testing.refAllDecls(load);
    @import("std").testing.refAllDecls(forward);
    @import("std").testing.refAllDecls(mixers);
    @import("std").testing.refAllDecls(tables.Tables);
    @import("std").testing.refAllDecls(lanes.Fx);
    @import("std").testing.refAllDecls(head);
    @import("std").testing.refAllDecls(snapshot);
    _ = native;
}
