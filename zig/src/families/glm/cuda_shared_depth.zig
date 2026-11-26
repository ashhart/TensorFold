//! Native TP2 shared-head growth is policy metadata; target verification remains the acceptance rule.
const lanes = @import("lanes");

pub const name = "TF_GLM_SHARED_MTP_DEPTH";

pub fn read() !u32 {
    return @import("sys").environment.read(u32, name, 0);
}

pub fn apply(base: lanes.Model, ceiling: u32, world: usize, dsa: bool) lanes.Model {
    var m = base;
    m.shared_draft_growth = if (world == 2 and !dsa and m.mtp and m.hidden_rows and !m.root_copies and !m.draft_streams and m.shared_drafts == 3) ceiling else 0;
    return m;
}
