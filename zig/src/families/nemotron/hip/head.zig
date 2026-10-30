//! The MTP draft head on the lanes: kept rows absorbed into its cache, then greedy one-row steps chained into drafts.
const std = @import("std");
const hip = @import("hip");
const engine = @import("engine.zig");
const forward = @import("forward.zig");
const round = @import("round.zig");
const pages = @import("pages.zig");

const Tensor = hip.ops.Tensor;

/// Drafts a chain holds at most.
pub const max_depth = 3;

/// One stream's work of a head round: its model rows and the token after each, then the drafts chained from the last.
pub const Job = struct { s: *pages.Stream, hidden: u64, next: []const u32, out: []u32, got: usize = 0 };

/// Token i + 2's draft from hidden row i and token i + 1: drafts never change verified tokens, only their rounds.
pub const Head = struct {
    e: *engine.Engine,
    gpa: std.mem.Allocator,
    /// The draft rows' token ids on the host: a draft is the id of the head's best row.
    ids: []u32,
    /// Each row's best draft row of a step, on the device and read back.
    arg: hip.DeviceBuffer,
    host: hip.HostBuffer,
    /// Each job's row in the last step.
    last: []usize,
    /// Chained steps run, and drafts they gave.
    steps: u64 = 0,
    drafts: u64 = 0,

    /// The head over the engine's loaded MTP weights; null when the checkpoint has none (or it was not loaded).
    pub fn init(gpa: std.mem.Allocator, e: *engine.Engine) !?*Head {
        const mt = e.model_().mtp orelse return null;
        const rows = e.plan.rows;
        const h = try gpa.create(Head);
        errdefer gpa.destroy(h);
        const ids = try gpa.alloc(u32, mt.draft_count);
        errdefer gpa.free(ids);
        try hip.runtime.check(e.driver.api.hipMemcpyDtoH(@ptrCast(ids.ptr), @ptrFromInt(mt.draft_ids), ids.len * 4));
        var arg = try hip.DeviceBuffer.alloc(&e.driver, 4 * rows);
        errdefer arg.free();
        var host = try hip.HostBuffer.alloc(&e.driver, 4 * rows);
        errdefer host.free();
        const last = try gpa.alloc(usize, e.plan.slots);
        errdefer gpa.free(last);
        h.* = .{ .e = e, .gpa = gpa, .ids = ids, .arg = arg, .host = host, .last = last };
        return h;
    }

    pub fn deinit(h: *Head) void {
        if (h.steps > 0) std.log.info("draft head: {d} chained steps, {d} drafts", .{ h.steps, h.drafts });
        h.gpa.free(h.last);
        h.host.free();
        h.arg.free();
        h.gpa.free(h.ids);
        h.gpa.destroy(h);
    }

    /// Device bytes the head holds beside the weights, for `rows` rows a round.
    pub fn deviceBytes(rows: usize) usize {
        return 4 * rows;
    }

    /// The `n` pairs (row i of `hidden`, `next[i]`) of a prompt's rows absorbed at the stream's head positions.
    pub fn absorb(h: *Head, o: hip.ops.Ops, s: *pages.Stream, hidden: Tensor, next: []const u32) !forward.Drafted {
        const e = h.e;
        var p = o;
        p.prefill = true;
        const dev = try e.ids(p, next);
        const where: forward.Where = .{ .paged = .{ .c = e.pool.paged(e.mtp_slot, s.table()), .pos = s.mtp_len } };
        const d = try forward.draft(p, e.model_(), where, dev, hidden, next.len);
        s.mtp_len += next.len;
        return d;
    }

    /// Each row's best draft row of the last step's `rows` logits rows, read back into `host`.
    fn best(h: *Head, o: hip.ops.Ops, logits: Tensor, rows: usize) !void {
        const width = h.e.model_().mtp.?.draft_count;
        try o.argmaxRows(logits, rows, width, h.arg.base());
        try hip.raw.download(h.arg, 0, h.host.bytes[0 .. 4 * rows], o.stream);
        try h.e.stream.synchronize();
    }

    /// Every job's pairs absorbed in one head round, then each job's `out.len` greedy drafts chained a step for all.
    pub fn chains(h: *Head, o: hip.ops.Ops, jobs: []Job) !void {
        const e = h.e;
        const m = e.model_();
        const hidden = m.dims.hidden;
        const width = hidden * e.act.size();
        if (jobs.len > e.wins.len) return error.WindowTooWide;
        var total: usize = 0;
        for (jobs) |j| total += j.next.len;
        if (total == 0) return;
        // every job's model rows, contiguous: the head's hidden input in plan order
        const hid = try forward.take(o, e.act, total * hidden);
        var at: usize = 0;
        for (jobs, 0..) |*j, i| {
            e.wins[i] = .{ .s = j.s, .pos = j.s.mtp_len, .tokens = j.next };
            try e.copy(hid.ptr + at * width, j.hidden, j.next.len * width, o.stream);
            h.last[i] = at + j.next.len - 1;
            at += j.next.len;
            j.got = 0;
        }
        var l = try e.plan.send(e.wins[0..jobs.len], e.o.capacity, e.stream);
        var d = try round.draftRound(o, m, e.mtp_slot, l, hid);
        for (jobs) |j| j.s.mtp_len += j.next.len;
        var rows = total;
        var step: usize = 0;
        while (true) {
            var drafting: usize = 0;
            for (jobs) |j| drafting += @intFromBool(j.got < j.out.len);
            if (drafting == 0) return;
            try h.best(o, d.logits, rows);
            for (jobs, 0..) |*j, i| if (j.got < j.out.len) {
                j.out[j.got] = h.ids[@intCast(h.host.slice(i32)[h.last[i]])];
                j.got += 1;
                h.drafts += 1;
            };
            // the next step: each job still drafting reads its last final-normed row and its newest draft
            const next = try forward.take(o, e.act, drafting * hidden);
            var k: usize = 0;
            for (jobs, 0..) |*j, i| {
                if (j.got == j.out.len) continue;
                try e.copy(next.ptr + k * width, d.normed.ptr + h.last[i] * width, width, o.stream);
                // at a position the kept rows will overwrite
                e.wins[k] = .{ .s = j.s, .pos = j.s.mtp_len + step, .tokens = j.out[j.got - 1 ..][0..1] };
                h.last[i] = k;
                k += 1;
            }
            if (k == 0) return;
            l = try e.plan.send(e.wins[0..k], e.o.capacity, e.stream);
            d = try round.draftRound(o, m, e.mtp_slot, l, next);
            rows = k;
            step += 1;
            h.steps += 1;
        }
    }
};
