//! Ownership transfers to the run only after its held-buffer registration succeeds.
pub fn allocate(r: anytype, n: usize) !@TypeOf(r.allocate(n) catch unreachable) {
    const b = try r.allocate(n);
    errdefer b.deinit();
    try r.keep(b);
    return b;
}
