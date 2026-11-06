//! Assistant and DFlash metadata reach the CUDA host head-cost grants.
const lanes = @import("lanes");

pub fn check(target: lanes.Model) !void {
    var m: @import("cuda_forward.zig").Model = undefined;
    m.geo.hidden = 16;
    m.context = 512;
    var assistant: @import("cuda_assistant.zig").Drafter = undefined;
    assistant.m = &m;
    assistant.step_ms = 2;
    var dflash: @import("cuda_dflash.zig").Drafter = undefined;
    dflash.m = &m;
    dflash.c.layers = 0;
    dflash.c.kv_heads = 1;
    dflash.step_ms = 2;
    for ([_]lanes.drafter.Facts{ assistant.drafter().facts(), dflash.drafter().facts() }) |facts| {
        try lanes.shared_head_cost_check.check(try lanes.shared_head_cost_check.attached(target, facts));
    }
}
