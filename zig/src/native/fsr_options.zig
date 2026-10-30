//! Native Flash Next residency options at the Metal admission boundary.
pub fn refuse(o: anytype, problem: *[]const u8) bool {
    if (o.fsrRefusal("metal")) |why| {
        problem.* = why;
        return true;
    }
    return false;
}

pub fn apply(o: anytype, value: anytype) @TypeOf(value) {
    var choice = value;
    choice.settings.ssd_experts = if (o.ssd_experts_gib) |g| @as(u64, @intFromFloat(g * (1 << 30))) else null;
    choice.settings.ple_on_ssd = o.ple_on_ssd;
    choice.settings.fsr_trace = o.fsr_trace;
    return choice;
}
