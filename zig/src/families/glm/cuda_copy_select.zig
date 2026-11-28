//! Native TP=2 suffix selection is independent of shared-round draft depth.
const lanes = @import("lanes");
pub const Settings = struct { on: bool = false, world: usize = 0, dsa: bool = false };
pub const name = "TF_GLM_COPY_SELECT";
pub fn read() !bool {
    return @import("sys").environment.read(bool, name, false);
}
pub fn apply(base: lanes.Model, on: bool, world: usize, dsa: bool, block: bool) lanes.Model {
    var m = base;
    m.copy_select = on and world == 2 and !dsa and !block and m.mtp and !m.root_copies and !m.context_copies;
    return m;
}
