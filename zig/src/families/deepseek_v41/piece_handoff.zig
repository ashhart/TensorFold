//! Prompt-piece completion shared by the target and its host ordering control.
pub fn finish(end: anytype, full: bool, save: bool) !void {
    if (full) try end.handoff();
    if (save) try end.snapshot();
}
