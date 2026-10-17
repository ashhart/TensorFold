//! SlotPass's ingests: a slot's taps rows into its rings, serial or overlapped, one main_proj a slot or batched.

const std = @import("std");
const cuda = @import("cuda");
const emit = @import("dspark_emit.zig");
const rows = @import("dspark_rows.zig");
const gpu = @import("dspark_gpu.zig");
const batch = @import("batch.zig");
const iface = @import("draft/iface.zig");
const rowtab = @import("rowtab.zig");
const core = @import("dspark_slots.zig");
const SlotPass = core.SlotPass;
const self = core.SlotPass.self;
const glueFn = core.SlotPass.glueFn;
const proj_rows_max = core.proj_rows_max;
const proj_check_rounds = core.proj_check_rounds;

/// A device-started round's ingests in slot order, on the side stream with TF_DSV41_DRAFT_OVERLAP, else on main.
pub fn ingestRound(x: *SlotPass, slots: []const u32, side: bool) !void {
    const r = x.f.runner;
    const b = x.bat.?;
    const st = if (side) x.side.?.stream else r.stream;
    if (side) try st.wait(x.side.?.mark);
    const prefix: ?[]const u8 = if (side) "w.ovl." else null;
    var la = std.heap.ArenaAllocator.init(x.gpa);
    defer la.deinit();
    const a = la.allocator();
    const proj = if (x.bproj != .off and slots.len > 1) try x.projRound(a, slots, st, prefix) else null;
    const pos_at = r.addressOf(if (side) "w.ovl.w.ds.pos" else "w.ds.pos") orelse return error.Unbound;
    for (slots) |sl| {
        const pd = b.pend[sl].?;
        const pos: i32 = @intCast(pd.start);
        try gpu.setPos(r.d, pos_at, pos, st);
        const sl_: emit.Slots = .{ .ring_slots = x.nslots, .slot = sl, .taps_role = "w.taps" };
        var cs = if (proj) |a0|
            try emit.emitIngestBlocks(a, x.f.cfg, x.f.widths, x.f.opts, pd.n, pd.a - a0, sl_)
        else
            try emit.emitIngestAt(a, x.f.cfg, x.f.widths, x.f.opts, pd.n, pd.a, sl_);
        if (prefix) |px| cs = try emit.renameRoles(a, cs, px);
        try r.windowOn(st, cs);
    }
    if (side) {
        const sd = &x.side.?;
        try sd.done.record(st);
        try r.stream.wait(sd.done);
        sd.rounds += 1;
        sd.slots += slots.len;
    }
}

/// TF_DSV41_DRAFT_BATCH_PROJ: main_proj + main_norm over window rows [a0, a1) on `st`; null: per-slot ingests run.
pub fn projRound(x: *SlotPass, a: std.mem.Allocator, slots: []const u32, st: cuda.Stream, prefix: ?[]const u8) !?u32 {
    const r = x.f.runner;
    const b = x.bat.?;
    var a0: u32 = std.math.maxInt(u32);
    var a1: u32 = 0;
    for (slots) |sl| {
        const pd = b.pend[sl].?;
        a0 = @min(a0, pd.a);
        a1 = @max(a1, pd.a + pd.n);
    }
    const n = a1 - a0;
    if (n > proj_rows_max) return null;
    var cs = try emit.emitIngestProj(a, x.f.cfg, x.f.widths, x.f.opts, n, a0, "w.taps");
    if (prefix) |px| cs = try emit.renameRoles(a, cs, px);
    try r.windowOn(st, cs);
    x.bstats.rounds += 1;
    x.bstats.slots += slots.len;
    if (x.bproj == .check and x.bstats.checked < proj_check_rounds) {
        if (!try x.projCheck(a, slots, st, a0, prefix)) return null;
    }
    return a0;
}

/// One check round: each slot's rows projected alone against the batched rows; any difference turns batching off.
pub fn projCheck(x: *SlotPass, a: std.mem.Allocator, slots: []const u32, st: cuda.Stream, a0: u32, prefix: ?[]const u8) !bool {
    const r = x.f.runner;
    const b = x.bat.?;
    const D: usize = x.f.cfg.hidden;
    const mx = r.addressOf(if (prefix != null) "w.ovl.w.ds.mx" else "w.ds.mx") orelse return error.Unbound;
    const chk = r.addressOf("w.chk.w.ds.mx") orelse return error.Unbound;
    const got = try a.alloc(u16, rowtab.seg_max * D);
    const want = try a.alloc(u16, rowtab.seg_max * D);
    var ok = true;
    for (slots) |sl| {
        const pd = b.pend[sl].?;
        const one = try emit.renameRoles(a, try emit.emitIngestProj(a, x.f.cfg, x.f.widths, x.f.opts, pd.n, pd.a, "w.taps"), "w.chk.");
        try r.windowOn(st, one);
        try st.synchronize();
        const bytes = 2 * @as(usize, pd.n) * D;
        try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = mx + 2 * @as(u64, pd.a - a0) * D, .len = bytes }, 0, std.mem.sliceAsBytes(got[0 .. pd.n * D]));
        try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = chk, .len = bytes }, 0, std.mem.sliceAsBytes(want[0 .. pd.n * D]));
        x.bstats.check_rows += pd.n;
        for (0..pd.n) |i| if (!std.mem.eql(u16, got[i * D ..][0..D], want[i * D ..][0..D])) {
            x.bstats.differ += 1;
            ok = false;
        };
    }
    x.bstats.checked += 1;
    if (!ok) {
        x.bproj = .off;
        std.log.err("draft batched main_proj: {d} of {d} check rows differ from their slot's projection alone: one main_proj a slot from here on", .{ x.bstats.differ, x.bstats.check_rows });
    } else if (x.bstats.checked == proj_check_rounds) {
        std.log.scoped(.dsv41).info("draft batched main_proj: {d} check rounds, {d} rows equal to their slot's projection alone, bit for bit", .{ x.bstats.checked, x.bstats.check_rows });
    }
    return ok;
}

/// An ingest of a slot's rows on the device without draining the stream first (dspark_gpu.ingestAsync's form).
pub fn ingestAsync(x: *SlotPass, slot: u32, start: u64, at: rows.TapsAt, n: u32) !void {
    const r = x.f.runner;
    const pos: i32 = @intCast(start);
    try gpu.setPos(r.d, r.addressOf("w.ds.pos") orelse return error.Unbound, pos, r.stream);
    var la = std.heap.ArenaAllocator.init(x.gpa);
    defer la.deinit();
    const sl: emit.Slots = .{ .ring_slots = x.nslots, .slot = slot, .taps_role = if (at.src == 1) batch.taps_stash_role else "w.taps" };
    const cs = try emit.emitIngestAt(la.allocator(), x.f.cfg, x.f.widths, x.f.opts, n, at.skip, sl);
    r.glue = .{ .ctx = x, .run = glueFn };
    try r.window(cs);
}

// ingest -------------------------------------------------------------------------------------------------------

pub fn ingest(p: *anyopaque, slot: u32, start: u64, taps: iface.Taps, rws: []const u32) anyerror!void {
    const x = self(p);
    if (slot >= x.nslots) return error.Slots;
    if (rws.len == 0) return;
    for (rws, 0..) |r, i| if (r != rws[0] + i) return error.NotAPrefix;
    if (taps.map != null) return error.NotAPrefix; // trees are refused with several slots (batch.zig)
    // a device-launched round already ingested this slot's window rows (afterPick): the same rows, positions
    if (x.dl) |dl| if (rws[0] == 0 and std.mem.indexOfScalar(u32, dl.slots[0..dl.k], slot) != null) if (x.bat.?.pend[slot]) |pd| if (pd.start == start and rws.len <= pd.n) return;
    const n: u32 = @intCast(rws.len);
    const r = x.f.runner;
    const at = try rows.locate(taps.address, taps.rows, 2 * @as(u64, taps.stride), rws[0], n, x.one.ingest_rows, r.addressOf("w.taps") orelse return error.Unbound, x.stash);
    try x.sendOp(&.{ rows.op_ingest, slot, @intCast(start), at.src, at.skip, n });
    try x.doIngest(slot, start, at, n);
}

pub fn doIngest(x: *SlotPass, slot: u32, start: u64, at: rows.TapsAt, n: u32) !void {
    if (at.src < 0) {
        x.valid[slot] = start + n; // the rows are not on the device: the context restarts after them
        return;
    }
    const r = x.f.runner;
    try r.stream.synchronize();
    const pos: i32 = @intCast(start);
    try gpu.setPos(r.d, r.addressOf("w.ds.pos") orelse return error.Unbound, pos, r.stream);
    var la = std.heap.ArenaAllocator.init(x.gpa);
    defer la.deinit();
    const sl: emit.Slots = .{ .ring_slots = x.nslots, .slot = slot, .taps_role = if (at.src == 1) batch.taps_stash_role else "w.taps" };
    const cs = try emit.emitIngestAt(la.allocator(), x.f.cfg, x.f.widths, x.f.opts, n, at.skip, sl);
    r.glue = .{ .ctx = x, .run = glueFn };
    try r.window(cs);
}
