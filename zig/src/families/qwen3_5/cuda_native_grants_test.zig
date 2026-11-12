//! Qwen CUDA's shared-round grants: a zero grant keeps absorbed history, and the MTP head keeps its step cost.
const lanes = @import("lanes");
const E = @import("cuda_engine.zig");
const Mtp = @import("cuda_mtp.zig").Mtp;
const laneFacts = @import("cuda_native.zig").laneFacts;

test "zero grant: Qwen MTP and DFlash wrappers keep their absorbed history through a long skip" {
    try lanes.shared_draft_check.check(laneFacts(16, 32), true);
}

test "shared head costs: Qwen CUDA MTP retains costed copy grants in chains and trees" {
    var e: E.Engine = undefined;
    e.g.kv_heads = 1;
    e.g.hidden = 16;
    var head: Mtp = undefined;
    head.e = &e;
    head.capacity = 16;
    head.depth = 3;
    head.step_ms = 2;
    head.n_costs = 0;
    for ([_]bool{ false, true }) |trees| {
        head.trees = trees;
        const m = try lanes.shared_head_cost_check.attached(laneFacts(16, 32), head.drafter().facts());
        try lanes.shared_head_cost_check.check(m);
    }
}
