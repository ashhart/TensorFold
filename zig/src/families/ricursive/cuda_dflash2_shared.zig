//! Ripple's DFlash2 drafter on CUDA: shared rounds of every admitted stream, timed at load for the depth rule.
const std = @import("std");
const fwd = @import("cuda_model.zig");
const be = @import("cuda_backend.zig");
const dfl = @import("cuda_dflash2.zig");

const windows = be.max_window;

/// Shared rounds of every admitted stream, w rows each (1 to 16): median of 3, priced past the lone sweep's 16 rows.
pub fn time(d: *dfl.Drafter) !void {
    const m = d.m;
    const n = @min(d.streams, m.arena.slots);
    if (n < 2) return;
    const chains = try d.gpa.alloc(fwd.Chain, n);
    defer d.gpa.free(chains);
    const ids = try d.gpa.alloc(u32, n * windows);
    defer d.gpa.free(ids);
    @memset(ids, 0);
    d.n_shared = 0;
    for (1..windows + 1) |w| {
        if (n * w > m.round_rows) break;
        for (chains, 0..) |*c, s| c.* = .{ .slot = s, .p0 = 0, .tokens = w };
        var ms: [3]f64 = undefined;
        for (0..4) |run| {
            const t0 = dfl.now(d.io);
            try m.round(chains, ids[0 .. n * w]);
            try m.rowLogits(n * w);
            try m.drawRows(n * w, d.b.ids);
            try m.s.synchronize();
            if (run > 0) ms[run - 1] = dfl.msSince(d.io, t0);
        }
        std.mem.sort(f64, &ms, {}, std.sort.asc(f64));
        std.log.info("Ripple DFlash2 on CUDA: a shared round of {d} streams x {d} rows ({d} rows) in {d:.2} ms", .{ n, w, n * w, ms[1] });
        if (n * w <= windows) continue; // the lone stream's sweep prices rounds up to its widest window
        d.shared[d.n_shared] = .{ .width = @intCast(n * w), .ms = ms[1] };
        d.n_shared += 1;
    }
    for (0..n) |s| try m.reset(s, windows);
    try m.s.synchronize();
}
