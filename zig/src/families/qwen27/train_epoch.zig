//! The native routes around the prompt store belong to the same learned-weight epoch.
pub fn changed(back: anytype) void {
    back.standing.deinit(back.gpa);
    back.dropTree();
    back.dropRound();
}

/// Cache.clear stops and detaches borrowed disk rows; the server still owns and closes them exactly once.
pub fn takeDisk(host: anytype) @TypeOf(host.disk_rows) {
    const rows = host.disk_rows;
    host.disk_rows = null;
    return rows;
}
