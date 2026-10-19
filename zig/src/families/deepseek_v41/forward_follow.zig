//! The forward's plan link: the leader's operation messages and a follower's loop that runs them.

const std = @import("std");
const cuda = @import("cuda");
const hooks = @import("forward_hooks.zig");
const ft = @import("forward_types.zig");
const ph = @import("phases.zig");
const fwd = @import("forward.zig");
const Forward = fwd.Forward;

pub const op_window: i64 = 1;
pub const op_greedy: i64 = 2;
pub const op_keep: i64 = 3;
pub const op_stop: i64 = 4;
pub const op_ds_ingest: i64 = 5;
pub const op_ds_propose: i64 = 6;
pub const op_prefill: i64 = 7;
/// 60: [op, ids...] the next prompt piece's ids for the coming prefill's Engram read-ahead (prefillAhead)
pub const op_ahead: i64 = 60;
pub const op_sess_save: i64 = 8;
pub const op_sess_restore: i64 = 9;
pub const op_release: i64 = 10;
pub const op_ds_reset: i64 = 11;
/// 12: [op, row0, n, T, top_k, top_p, min_p] a keyed sample (ops 12-15: the keyed sampler)
pub const op_sample: i64 = 12;
/// 13: [op, n, (row0, rows, T, top_k, top_p, min_p, seed, pos0) a segment]: TF_DSV41_SAMP_BATCH's segments
pub const op_sample_many: i64 = 13;

/// Rank 0: a sampled choice's message to the followers (before its gathers).
pub fn sendSample(f: *Forward, ints: []const i64) !void {
    if (f.leads()) try f.link.?.send(ints);
}
pub const op_tree_drop: i64 = 20;
/// 44: [op, end] the CED decoder replay (finishPrompt); 45: a prompt snapshot (ops 44-47: CED replay)
pub const op_ced_finish: i64 = 44;
pub const op_sess_prompt: i64 = 45;
/// 48: [op, need, n, spilled entry ids...] a request's admission (ops 48-51: sessions over several slots)
pub const op_sess_admit: i64 = 48;

/// 52: [op, segments, (slot, start, rows) a segment, ids...]; 53: [op, n, slots...] (ops 52-55: prefill runs)
pub const op_pf_multi: i64 = 52;
pub const op_ced_multi: i64 = 53;

/// Several slots' CED decoder replays in one run (TF_DSV41_REPLAY_RUNS), every rank; `out`: tails, taps.
pub fn replayMulti(f: *Forward, slots: []const u32, out: []ft.Replayed) !void {
    if (f.leads()) {
        var msg: [2 + 16]i64 = undefined;
        if (slots.len > 16) return error.BadWindow;
        msg[0] = op_ced_multi;
        msg[1] = @intCast(slots.len);
        for (slots, msg[2 .. 2 + slots.len]) |s, *y| y.* = s;
        try f.link.?.send(msg[0 .. 2 + slots.len]);
    }
    return (try hooks.prefill()).finishMulti(f, slots, out);
}

/// Several slots' prompt segments in one run (TF_DSV41_PIECE_RUNS); the leader sends the run first.
pub fn prefillMulti(f: *Forward, segs: []const ft.MultiSeg) !void {
    if (segs.len == 0) return;
    f.gated = false;
    if (f.leads()) {
        var msg: std.ArrayList(i64) = .empty;
        defer msg.deinit(f.gpa);
        try msg.appendSlice(f.gpa, &.{ op_pf_multi, @intCast(segs.len) });
        for (segs) |sg| try msg.appendSlice(f.gpa, &.{ sg.slot, @intCast(sg.start), @intCast(sg.ids.len) });
        for (segs) |sg| for (sg.ids) |t| try msg.append(f.gpa, t);
        try f.link.?.send(msg.items);
    }
    f.quiet = true;
    defer f.quiet = false;
    return (try hooks.prefill()).promptMulti(f, segs);
}

/// CED (TF_DSV41_PREFILL=replay): the decoder replay over the prompt's prefilled tail, every rank; full: nothing.
pub fn finishPrompt(f: *Forward) !void {
    const h = try hooks.prefill();
    if (!h.replaying(f)) return;
    if (f.leads()) try f.link.?.send(&.{ op_ced_finish, @intCast(f.slot.pos) });
    return h.finish(f);
}
/// 28: [op, sub (bind / step / release), ...] (grammar_gpu.zig; ops 28-31)
pub const op_grammar: i64 = 28;

/// 64: [op] every rank syncs its device and reads its clock; 65: [op, n] samples all-gathered (ops 64-67: calib)
pub const op_calib_mark: i64 = 64;
pub const op_calib_gather: i64 = 65;

/// A timed run's start or end on every rank: the leader sends the mark, then this rank syncs and reads its clock.
pub fn calibMark(f: *Forward) !u64 {
    if (f.leads()) try f.link.?.send(&.{op_calib_mark});
    const d = f.runner.d;
    try d.synchronize();
    return ph.nowNs();
}

/// Every rank's samples (`mine`, ns) into `all` [world][mine.len]; an unpaired follower sends -1s (refused).
pub fn calibGather(f: *Forward, mine: []const i64, all: []i64) !void {
    const w: usize = f.comm.world();
    if (all.len != w * mine.len) return error.BadWindow;
    if (f.leads()) try f.link.?.send(&.{ op_calib_gather, @intCast(mine.len) });
    if (w == 1 or mine.len == 0) return @memcpy(all[0..mine.len], mine);
    const d = f.runner.d;
    const bytes = 8 * mine.len;
    var dev = try cuda.DeviceBuffer.alloc(d, bytes * (w + 1));
    defer dev.free();
    var host = try cuda.HostBuffer.alloc(d, bytes * (w + 1));
    defer host.free();
    const h = host.slice(i64);
    @memcpy(h[0..mine.len], mine);
    const st = f.runner.stream;
    try dev.uploadAsync(0, std.mem.sliceAsBytes(h[0..mine.len]), st.handle);
    try f.comm.allGather(dev.ptr, dev.ptr + bytes, mine.len, .i64, st.handle);
    try dev.downloadAsync(bytes, std.mem.sliceAsBytes(h[mine.len .. mine.len * (w + 1)]), st.handle);
    try st.synchronize();
    @memcpy(all, h[mine.len .. mine.len * (w + 1)]);
}

/// A follower's op_calib_gather: its marks paired up (end - start) as its samples, gathered with the leader's.
pub fn followCalib(f: *Forward, n: usize) !void {
    defer f.calib_marks.clearRetainingCapacity();
    const mine = try f.gpa.alloc(i64, n);
    defer f.gpa.free(mine);
    const marks = f.calib_marks.items;
    if (marks.len == 2 * n) {
        for (mine, 0..) |*x, k| x.* = @intCast(marks[2 * k + 1] - marks[2 * k]);
    } else {
        std.log.scoped(.dsv41).err("calibration: {d} marks on this rank for {d} timed runs; sending none", .{ marks.len, n });
        @memset(mine, -1);
    }
    const all = try f.gpa.alloc(i64, n * f.comm.world());
    defer f.gpa.free(all);
    try f.calibGather(mine, all);
}

/// Drops the pending window on every rank without committing a row (a tree chain whose branch lost).
pub fn drop(f: *Forward) !void {
    if (f.slot.pending == null) return error.BadWindow;
    if (f.leads()) try f.link.?.send(&.{op_tree_drop});
    try f.transaction.rollback();
    if (f.kv) |k| try k.truncate(f.slot.pos);
    f.slot.pending = null;
}

/// Frees the slot for the next stream on every rank: position 0, no Engram tail, no pending window, pages back.
pub fn release(f: *Forward) !void {
    if (f.leads()) try f.link.?.send(&.{op_release});
    if (f.transaction.state.pending != null) try f.transaction.rollback();
    if (f.sess) |x| return x.reset(x.ptr);
    if (f.kv) |k| try k.release();
    f.slot = .{};
}

/// Rank 0: a restore to the followers, with the entry's ids when it is not in RAM ([op, id, ids...]).
pub fn sendRestore(f: *Forward, id: u32, ids: ?[]const u32) !void {
    if (!f.leads()) return;
    const extra = if (ids) |x| x.len else 0;
    const msg = try f.gpa.alloc(i64, 2 + extra);
    defer f.gpa.free(msg);
    msg[0] = op_sess_restore;
    msg[1] = id;
    if (ids) |x| for (x, msg[2..]) |t, *y| {
        y.* = t;
    };
    try f.link.?.send(msg);
}

/// Rank 0: a session operation to the followers (before running it).
pub fn sendSess(f: *Forward, op: enum { save, restore, prompt }, id: u32) !void {
    if (f.leads()) try f.link.?.send(&.{ switch (op) {
        .save => op_sess_save,
        .restore => op_sess_restore,
        .prompt => op_sess_prompt,
    }, id });
}

/// Rank 0: a request's admission to the followers (before running it): its pages and the entries to evict first.
pub fn sendSessAdmit(f: *Forward, need: u32, spills: []const u32) !void {
    if (!f.leads()) return;
    var msg: std.ArrayList(i64) = .empty;
    defer msg.deinit(f.gpa);
    try msg.appendSlice(f.gpa, &.{ op_sess_admit, need, @intCast(spills.len) });
    for (spills) |id| try msg.append(f.gpa, id);
    try f.link.?.send(msg.items);
}

pub fn dsResetOp() [1]i64 {
    return .{op_ds_reset};
}

/// Rank 0: a drafter operation to the followers (before running it).
pub fn sendDrafter(f: *Forward, ints: []const i64) !void {
    if (f.leads()) try f.link.?.send(ints);
    // the pass's anchor is the next window's row 0: its Engram rows are read while the pass runs
    if (ints.len == 3 and ints[0] == op_ds_propose) f.prefetchEngram(@intCast(ints[2]), &.{@intCast(ints[1])});
}

pub fn dsIngestOp(start: u64, row0: u32, n: u32, avail: u32) [5]i64 {
    return .{ op_ds_ingest, @intCast(start), row0, n, avail };
}

pub fn dsProposeOp(anchor: u32, start: u64) [3]i64 {
    return .{ op_ds_propose, anchor, @intCast(start) };
}

/// Rank 0: the followers' loops end (after the last operation they were sent).
pub fn stop(f: *Forward) !void {
    if (f.leads()) try f.link.?.send(&.{op_stop});
    // every rank's clean end, before rank 0 closes its sockets (tp.Session.announceStop: no fail-fast exit 70)
    if (f.leads()) if (f.on_stop) |h| h.run(h.ctx);
}

/// Ranks > 0: run the leader's operations in its order (collectives pair up) until `stop`.
pub fn follow(f: *Forward) !void {
    var buf: std.ArrayList(i64) = .empty;
    defer buf.deinit(f.gpa);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(f.gpa);
    var prompt: std.ArrayList(u32) = .empty;
    defer prompt.deinit(f.gpa);
    // greedy's rows: a window's, or a prefill's last segment (up to prefill_rows)
    var picks: std.ArrayList(u32) = .empty;
    defer picks.deinit(f.gpa);
    while (true) {
        const msg = try f.link.?.recvGrow(f.gpa, &buf);
        if (msg.len == 0) return error.BadPlan;
        const ended = followOp(f, msg, &ids, &prompt, &picks) catch |e| {
            // Runner.guard failed the leader's same window before any launch: only its request fails
            if (e != error.MissingTritonVariant) return e;
            std.log.scoped(.dsv41).warn("rank {d}: operation {d} failed ({t}), as on rank 0: the next one", .{ f.comm.rank(), msg[0], e });
            continue;
        };
        if (ended) return;
    }
}

/// One of the leader's operations (follow); true: `stop`.
fn followOp(f: *Forward, msg: []const i64, ids: *std.ArrayList(u32), prompt: *std.ArrayList(u32), picks: *std.ArrayList(u32)) !bool {
    switch (msg[0]) {
        op_window => {
            const n: usize = @intCast(msg[1]);
            ids.clearRetainingCapacity();
            for (msg[2 .. 2 + n]) |t| try ids.append(f.gpa, @intCast(t));
            try f.window(ids.items);
        },
        op_greedy => {
            try picks.resize(f.gpa, @intCast(msg[1]));
            try f.greedy(picks.items);
        },
        op_prefill => {
            const total: usize = @intCast(msg[1]);
            const at: usize = @intCast(msg[2]);
            if (at == 0) prompt.clearRetainingCapacity();
            if (at != prompt.items.len) return error.BadPlan;
            for (msg[3..]) |t| try prompt.append(f.gpa, @intCast(t));
            if (prompt.items.len > total) return error.BadPlan;
            if (prompt.items.len == total) try f.prefill(prompt.items);
        },
        op_ahead => {
            var nx: std.ArrayList(u32) = .empty;
            defer nx.deinit(f.gpa);
            for (msg[1..]) |t| try nx.append(f.gpa, @intCast(t));
            try f.prefillAhead(nx.items);
        },
        op_keep => try f.keep(@intCast(msg[1])),
        op_release => try f.release(),
        op_tree_drop => try f.drop(),
        op_calib_mark => {
            const d = f.runner.d;
            try d.synchronize();
            try f.calib_marks.append(f.gpa, ph.nowNs());
        },
        op_calib_gather => {
            if (msg.len != 2 or msg[1] < 0) return error.BadPlan;
            try f.followCalib(@intCast(msg[1]));
        },
        op_stop => return true,
        op_sess_save => {
            const x = f.sess orelse return error.BadPlan;
            try x.save(x.ptr);
        },
        op_sess_restore => {
            const x = f.sess orelse return error.BadPlan;
            if (msg.len < 2) return error.BadPlan;
            try x.restore(x.ptr, @intCast(msg[1]), msg[2..]);
        },
        op_sess_prompt => {
            const x = f.sess orelse return error.BadPlan;
            try x.prompt(x.ptr);
        },
        op_sess_admit => {
            const x = f.sess orelse return error.BadPlan;
            const run_ = x.admit orelse return error.BadPlan;
            try run_(x.ptr, msg);
        },
        op_pf_multi => {
            const ns: usize = @intCast(msg[1]);
            if (ns == 0 or ns > 16 or msg.len < 2 + 3 * ns) return error.BadPlan;
            var segs: [16]ft.MultiSeg = undefined;
            prompt.clearRetainingCapacity();
            for (msg[2 + 3 * ns ..]) |t| try prompt.append(f.gpa, @intCast(t));
            var at: usize = 0;
            for (segs[0..ns], 0..) |*sg, j| {
                const n: usize = @intCast(msg[4 + 3 * j]);
                if (at + n > prompt.items.len) return error.BadPlan;
                sg.* = .{ .slot = @intCast(msg[2 + 3 * j]), .start = @intCast(msg[3 + 3 * j]), .ids = prompt.items[at .. at + n] };
                at += n;
            }
            if (at != prompt.items.len) return error.BadPlan;
            try f.prefillMulti(segs[0..ns]);
        },
        op_ced_multi => {
            const ns: usize = @intCast(msg[1]);
            if (ns < 2 or ns > 16 or msg.len != 2 + ns) return error.BadPlan;
            var slots: [16]u32 = undefined;
            for (msg[2..], slots[0..ns]) |x, *y| y.* = @intCast(x);
            var out: [16]ft.Replayed = undefined;
            try f.replayMulti(slots[0..ns], &out);
        },
        op_ced_finish => {
            // the leader's slot ends where this one does, and both run replay (the same TF_DSV41_PREFILL)
            const h = try hooks.prefill();
            if (!h.replaying(f) or msg.len < 2 or msg[1] != f.slot.pos) return error.BadPlan;
            try h.finish(f);
        },
        op_ds_ingest => {
            const d = f.drafter orelse return error.BadPlan;
            try d.ingest(d.ptr, @intCast(msg[1]), @intCast(msg[2]), @intCast(msg[3]), @intCast(msg[4]));
        },
        op_ds_reset => {
            const d = f.drafter orelse return error.BadPlan;
            d.reset(d.ptr);
        },
        op_ds_propose => {
            const d = f.drafter orelse return error.BadPlan;
            f.prefetchEngram(@intCast(msg[2]), &.{@intCast(msg[1])});
            try d.propose(d.ptr, @intCast(msg[1]), @intCast(msg[2]));
        },
        op_sample, op_sample_many => {
            const x = f.sampler orelse return error.BadPlan;
            try x.follow(x.ptr, msg);
        },
        op_grammar => {
            const x = f.grammar orelse return error.BadPlan;
            try x.follow(x.ptr, msg);
        },
        else => {
            const x = f.ext orelse return error.BadPlan;
            try x.run(x.ptr, msg);
        },
    }
    return false;
}
