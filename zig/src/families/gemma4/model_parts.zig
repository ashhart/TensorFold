//! The late Model.load acquisitions, with GPU operations supplied by its private boundary.
pub fn load(m: anytype, comptime Ops: type) !void {
    errdefer Ops.releasePipelines(m);
    try Ops.pipelines(m);
    m.inv_sliding = try Ops.buffer(m, false);
    errdefer m.inv_sliding.deinit();
    m.inv_global = try Ops.buffer(m, true);
    errdefer m.inv_global.deinit();
    try Ops.inverse(m, false, m.inv_sliding);
    try Ops.inverse(m, true, m.inv_global);
}
