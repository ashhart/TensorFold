//! HIP family metadata over inert host objects, without a HIP runtime or device.
const check = @import("draft_sources_check.zig").check;
const present = @import("draft_sources_check.zig").present;

test "draft sources: HIP Qwen dense and MoE checkpoint heads and explicit zero" {
    const q = @import("qwen3_5");
    var e: q.engine.Engine = undefined;
    var b: q.hip_lanes.Hip = undefined;
    b.e = &e;
    b.costs = .{};
    e.o.batch_rows = 16;
    e.o.policy.mtp.drafts = 3;
    b.head = null;
    try check(b.facts(), null, true, false);
    b.head = present(@TypeOf(b.head.?));
    try check(b.facts(), null, true, true);
    e.o.policy.mtp.drafts = 0;
    try check(b.facts(), null, true, false);
}

test "draft sources: HIP Nemotron checkpoint head and explicit zero" {
    const n = @import("nemotron");
    var e: n.engine.Engine = undefined;
    var b: n.lanes.Hip = undefined;
    b.e = &e;
    b.costs = .{};
    e.plan.rows = 16;
    e.plan.slots = 4;
    e.o.policy.mtp.drafts = 3;
    b.head = null;
    try check(b.facts(), null, true, false);
    b.head = present(@TypeOf(b.head.?));
    try check(b.facts(), null, true, true);
    e.o.policy.mtp.drafts = 0;
    try check(b.facts(), null, true, false);
}

test "draft sources: HIP Flash Next checkpoint head and explicit zero" {
    const f = @import("flashnext");
    var e: f.engine.Engine = undefined;
    var b: f.lanes.Fx = undefined;
    b.e = &e;
    b.streams = 4;
    b.drafts = 3;
    e.weights.model.mtp = null;
    try check(b.facts(), null, true, false);
    e.weights.model.mtp = present(@TypeOf(e.weights.model.mtp.?));
    try check(b.facts(), null, true, true);
    b.drafts = 0;
    try check(b.facts(), null, true, false);
}
