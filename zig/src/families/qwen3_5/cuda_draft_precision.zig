//! Draft arithmetic follows a row's role and the 1.1.0 recipe, independently of storage and verify capacity.
pub const Role = enum { prompt, live, proposal };
pub const dflash_projection_prompt_cutoff = 16;
pub const mtp_projection_prompt_cutoff = 16;
pub const mtp_moe_prompt_cutoff = 16;
pub const native_rows = 16;

pub fn prompt(role: Role, rows: usize, cutoff: usize) bool {
    return role == .prompt and rows > cutoff;
}

pub fn affine(e: anytype, l: anytype, x: u64, xs: u64, out: u64, rows: usize, sums: bool, role: Role) !void {
    if (prompt(role, rows, dflash_projection_prompt_cutoff)) return e.affine.prompt(e.stream, x, l, out, rows);
    if (sums) try e.run(e.fns.sums, .{ rows, (l.k / 64 + 63) / 64, 1 }, .{ 64, 1, 1 }, .{ x, xs, @as(c_int, @intCast(l.k)), @as(u64, 0) });
    const tile = e.affine.verifyRows(); // verify keeps decode()'s chains a row: two m16 tiles a weight panel on GB10
    var at: usize = 0;
    while (at < rows) : (at += tile) {
        try e.affine.verify(e.stream, x + at * l.k * 2, xs + at * (l.k / 64) * 4, l, out + at * l.n * 2, @min(tile, rows - at));
    }
}

pub fn project(e: anytype, l: anytype, x: u64, xs: u64, out: u64, rows: usize, role: Role, cutoff: usize) !void {
    return e.projSums(l, x, xs, out, rows, prompt(role, rows, cutoff), 0); // native: the engine's verify tiles
}

pub fn routed(e: anytype, router: u64, experts: anytype, x: u64, out: u64, rows: usize, role: Role) !void {
    if (prompt(role, rows, mtp_moe_prompt_cutoff)) return e.routedRows(router, experts, x, out, rows, true, 0);
    var at: usize = 0;
    while (at < rows) : (at += native_rows) {
        try e.routedRows(router, experts, x + at * e.g.hidden * 2, out + at * e.g.hidden * 2, @min(native_rows, rows - at), false, 0);
    }
}
