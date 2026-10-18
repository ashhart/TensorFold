//! Drafter weights, scratch, projection scratch and live lanes follow the served set by allocation identity.
const mtl = @import("metal");

fn buffers(d: anytype, hook: mtl.ResidencySet.Follow, joining: bool) void {
    mtl.resident.fields.apply(d, hook, joining);
    for (d.w.owned.items) |b| mtl.resident.fields.one(b, hook, joining);
    if (@hasField(@TypeOf(d.*), "attention")) mtl.resident.fields.apply(&d.attention, hook, joining);
    var it = d.lanes.valueIterator();
    while (it.next()) |l| lane(d, l.*, hook, joining);
}

pub fn set(d: anytype, hook: ?mtl.ResidencySet.Follow) void {
    if (d.follow) |f| buffers(d, f, false);
    d.projections.setResident(hook);
    d.follow = hook;
    if (hook) |f| buffers(d, f, true);
}

pub fn lane(d: anytype, l: anytype, hook: mtl.ResidencySet.Follow, joining: bool) void {
    if (@hasField(@TypeOf(l.*), "k")) {
        for (0..d.config.layers) |i| {
            mtl.resident.fields.one(l.k[i], hook, joining);
            mtl.resident.fields.one(l.v[i], hook, joining);
        }
    } else mtl.resident.fields.one(l.h, hook, joining);
}
