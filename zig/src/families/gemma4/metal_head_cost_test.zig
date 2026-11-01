//! Assistant and DFlash metadata reach the Metal host head-cost grants.
const lanes = @import("lanes");

pub fn check(target: lanes.Model) !void {
    var m: @import("model.zig").Model = undefined;
    m.config.geo.hidden = 16;
    var assistant: @import("assistant.zig").Drafter = undefined;
    assistant.target = &m;
    assistant.step_ms = 2;
    var dflash: @import("dflash.zig").Drafter = undefined;
    dflash.config.layers = 0;
    dflash.options = .{ .capacity = 512 };
    dflash.step_ms = 2;
    for ([_]lanes.drafter.Facts{ assistant.drafter().facts(), dflash.drafter().facts() }) |facts| {
        try lanes.shared_head_cost_check.check(try lanes.shared_head_cost_check.attached(target, facts));
    }
}
