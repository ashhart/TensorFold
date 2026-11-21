//! Host-tier frame ownership advances before capture or replay; kernels consume the resulting device tables.
pub fn stage(selection: anytype, stream: anytype, first: usize, count: usize, position: usize, rows: usize) !void {
    if (selection.graph_staged) return error.NestedPageGraphStage;
    for (first..first + count) |layer| try selection.begin(stream, layer, position, rows);
    selection.graph_staged = true;
}

pub fn finish(selection: anytype) void {
    selection.graph_staged = false;
}
