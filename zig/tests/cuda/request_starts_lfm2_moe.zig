//! LFM named starts exercise production chunk taps, page snapshots and restore after slot reuse.
const std = @import("std");
const lfm = @import("lfm2_moe");
const lanes = @import("lanes");
const common = @import("request_start_checks.zig");
const check = @import("check.zig");
const cfg = lfm.config;
const mdl = lfm.model;
const kern = lfm.kernels;
const Lfm = lfm.backend.Liquid;

const Saver = struct {
    x: *Lfm,
    saved: [32]*lfm.snap.Saved = undefined,
    at: [32]u32 = undefined,
    count: usize = 0,
    err: ?anyerror = null,

    fn call(ptr: *anyopaque, s: *lanes.Stream, at: u32) void {
        const v: *Saver = @ptrCast(@alignCast(ptr));
        if (v.err != null) return;
        if (v.count == v.saved.len) {
            v.err = error.TooManySnapshots;
            return;
        }
        const saved = v.x.saveState(s, at) catch |err| {
            v.err = err;
            return;
        };
        v.saved[v.count] = saved;
        v.at[v.count] = at;
        v.count += 1;
    }
    fn drop(v: *Saver) void {
        for (v.saved[0..v.count]) |s| v.x.dropState(s);
    }
};

fn digest(gpu: check.Gpu, x: *Lfm, s: *lanes.Stream, last_row: usize) !common.Digest {
    if (x.snaps) |*p| try p.fence(x.s);
    try x.s.synchronize();
    const l = x.lanes.get(s) orelse return error.UnknownStream;
    var state = common.Sha.init(.{});
    try common.add(gpu, &state, x.m.states + l.slot * mdl.slot_elems * 2, mdl.slot_elems * 2);
    const ids = x.m.ids[l.slot * x.m.perSlot() ..];
    var at: usize = 0;
    while (at < s.prompt().len) : (at += kern.page) {
        const rows = @min(kern.page, s.prompt().len - at);
        const page = x.m.arena + @as(u64, ids[at / kern.page]) * mdl.page_elems * 2;
        for (0..2 * mdl.attention_layers) |i| {
            try common.add(gpu, &state, page + i * kern.page * mdl.kv_width * 2, rows * mdl.kv_width * 2);
        }
    }
    var logits = common.Sha.init(.{});
    try common.add(gpu, &logits, l.last, cfg.vocab * 2);
    var hidden = common.Sha.init(.{});
    try common.add(gpu, &hidden, x.m.s.h + last_row * cfg.hidden * 2, cfg.hidden * 2);
    const b = x.backend();
    const token = try b.read(try b.first(s, s.prompt().len));
    return .{ .state = state.finalResult(), .logits = logits.finalResult(), .hidden = hidden.finalResult(), .token = token };
}

fn pass(gpu: check.Gpu, x: *Lfm, ids: []const u32, starts: []const u32, saver: ?*Saver, saved: ?*lfm.snap.Saved) !common.Digest {
    var s = try lanes.Stream.init(gpu.gpa, .{ .id = "state-check", .prompt = ids, .chunks = starts, .max_new = 1, .drafts = false });
    defer s.deinit(gpu.gpa);
    if (saver) |v| s.reuse.hook = .{ .ptr = v, .at = Saver.call };
    if (saved) |v| {
        s.reuse.saved = v;
        s.reuse.at = v.at;
    }
    const b = x.backend();
    defer b.release(&s);
    var from: usize = if (saved) |v| v.at else 0;
    while (!try b.vtable.prefill_step.?(b.ptr, &s, 0)) from = s.fill_at.load(.monotonic);
    try check.expect(!s.reuse_failed and s.cached == (if (saved) |v| v.at else @as(u32, 0)), "LFM snapshot restore refused or cached count differs", .{});
    return digest(gpu, x, &s, ids.len - from - 1);
}

pub fn run(gpu: check.Gpu, model: []const u8, ids_path: []const u8, n_text: []const u8, patterns: []const u8) !void {
    const ids = try common.readIds(gpu, ids_path, n_text);
    defer gpu.gpa.free(ids);
    const other = try gpu.gpa.dupe(u32, ids);
    defer gpu.gpa.free(other);
    try common.validateIds(ids, cfg.vocab);
    common.poison(other, cfg.vocab);
    const loaded = try lfm.native.open(gpu.gpa, gpu.io, gpu.ctx, model, null, .{ .context = 16384, .drafts = false, .lanes = 2, .lanes_fixed = true });
    defer loaded.deinit(loaded.ctx);
    if (loaded.layout != null) try lfm.native.attach(loaded.ctx, 1, 2 * lfm.backend.pagesFor(ids.len, 1));
    const x = loaded.target;
    const fresh = try pass(gpu, x, ids, &.{}, null, null);
    common.show("lfm cold", fresh);
    if (std.mem.eql(u8, patterns, "-")) return;
    var it = std.mem.splitScalar(u8, patterns, ';');
    var budget: u64 = lfm.snap.page;
    while (it.next()) |pattern| {
        var buf: [32]u32 = undefined;
        const starts = try common.points(pattern, ids.len, &buf);
        var need: u64 = 0;
        for (starts) |at| need += lfm.snap.bytes(at);
        budget = @max(budget, need);
    }
    _ = try lfm.native.snap_ready(x, budget);
    std.debug.print("RESULT lfm paged={any} snapshot-budget={d}\n", .{ x.m.paged, budget });
    it = std.mem.splitScalar(u8, patterns, ';');
    while (it.next()) |pattern| {
        var buf: [32]u32 = undefined;
        const starts = try common.points(pattern, ids.len, &buf);
        try common.equal("lfm named cache-off", try pass(gpu, x, ids, starts, null, null), fresh);
        var saver: Saver = .{ .x = x };
        defer saver.drop();
        const kept = try pass(gpu, x, ids, starts, &saver, null);
        if (saver.err) |err| return err;
        try check.expect(std.mem.eql(u32, starts, saver.at[0..saver.count]), "LFM named starts {s}: saved {any}, expected {any}", .{ pattern, saver.at[0..saver.count], starts });
        try common.equal("lfm named hooked", kept, fresh);
        for (saver.saved[0..saver.count]) |saved| {
            _ = try pass(gpu, x, other, &.{}, null, null);
            std.debug.print("RESULT lfm restored-at={d}\n", .{saved.at});
            try common.equal("lfm resumed", try pass(gpu, x, ids, starts, null, saved), fresh);
        }
    }
    check.pass("lfm2-moe-request-starts: device conv, KV, hidden, logits and first token equal", .{});
}
