//! GLM-5.3 on a pair: a stream a slot; prompt chunks, verify windows and MTP drafts, run on both Macs in step.
const std = @import("std");
const mtl = @import("metal");
const frags = @import("../../core/frags.zig");
const pr = @import("../../core/pair.zig");
const cfg = @import("config.zig");
const kn = @import("kernels.zig");
const gpu = @import("gpu.zig");
const md = @import("model.zig");
const snap = @import("snap.zig");
const pw = @import("prompt.zig");
const df2 = @import("dflash2.zig");
pub const tr = @import("tree.zig");
const slot_mod = @import("slot.zig");
const Ref = kn.Ref;

const R = gpu.max_rows;
pub const max_depth = slot_mod.max_depth;
const passes = 2 + max_depth + 2;
pub const Win = slot_mod.Win;
pub const Draft = slot_mod.Draft;
pub const Slot = slot_mod.Slot;

/// One GPU pass's host-written tables: token ids, rows, picks and gather indices.
const Pass = struct { ids: mtl.Buffer, rows: mtl.Buffer, picks: mtl.Buffer, idx: mtl.Buffer };

pub const Engine = struct {
    resident: ?mtl.resident.Resident = null,
    gpa: std.mem.Allocator,
    io: std.Io,
    device: mtl.Device,
    queue: mtl.Queue,
    pair: *pr.Pair,
    m: md.Model,
    dflash2: ?*df2.Drafter = null,
    s: gpu.Scratch,
    slots: []Slot,
    pass: [passes]Pass,
    /// Normed hidden rows for the MTP head: the last window's [0, R), a chunk's [R, 2R), each slot's last row from 2R.
    xs: mtl.Buffer,
    accept_meta: mtl.Buffer,
    compact_disp: mtl.Buffer,
    gate_meta: mtl.Buffer,
    gate_disp: mtl.Buffer,
    draft_rows: mtl.Buffer,
    draft_x: mtl.Buffer,
    words: std.ArrayList(u32) = .empty,
    /// A prompt window's buffers.
    q: gpu.Prompt,
    /// Kept prompt states by id (the same ids on both Macs).
    snaps: std.AutoHashMapUnmanaged(u32, *snap.Snap) = .empty,
    context: u32,
    use_tree: bool = true,
    dispatch: mtl.DispatchType = .concurrent,
    win_dispatch: mtl.DispatchType = .serial,
    load_seconds: f64 = 0,
    /// The GPU's time on the last command buffer (ms), for speed work.
    gpu_ms: f64 = 0,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, settings: ?[]const u8, layers: ?u32, mtp: bool, context: u32, streams: u32, limit: u64) !*Engine {
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        const t0 = std.Io.Timestamp.now(io, .awake);
        e.gpa = gpa;
        e.io = io;
        e.device = try mtl.Device.init();
        const want_m4 = @import("sys").environment.read(bool, "TF_METAL4", true) catch true;
        e.queue = if (want_m4) (e.device.queue4() catch try e.device.queue()) else try e.device.queue();
        if (e.queue.isMetal4()) std.log.info("glm_dsa: Metal 4 command queue", .{});
        try frags.check(e.device, e.queue, gpa); // the fragment layout nax.h gives the prompt MMA kernels
        const slot_bytes = gpu.prompt_rows * 6144 * 4;
        e.pair = if (settings) |sp| try pr.Pair.init(gpa, io, e.device, sp, slot_bytes) else try pr.Pair.initSingle(gpa, e.device, slot_bytes);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(gpa, "{s}/config.json", .{dir}), gpa, .limited(1 << 20));
        var c = try cfg.parse(gpa, text);
        try cfg.splitSettings(&c, e.pair.cfg, layers);
        e.context = context;
        const use_conc = @import("sys").environment.read(bool, "TF_GLM_CONCURRENT", true) catch true;
        e.use_tree = !(@import("sys").environment.read(bool, "TF_NO_TREES", false) catch false);
        e.dispatch = if (use_conc) .concurrent else .serial;
        e.win_dispatch = e.dispatch;
        const effective_limit: u64 = if (e.pair.cfg.load_limit_gb) |gb| @intFromFloat(gb * (1024.0 * 1024.0 * 1024.0)) else limit;
        e.m = try md.Model.load(gpa, io, e.device, dir, c, mtp, context, streams, e.pair, effective_limit);
        e.dflash2 = if (mtp and e.m.c.pp == 1 and e.m.vocab.lm_bf16_off != 0) try df2.Drafter.tryLoad(gpa, io, e.device, &e.m.c, dir, streams) else null;
        e.s = try gpu.Scratch.init(e.device, &e.m.c, context);
        e.q = try gpu.Prompt.init(e.device, &e.m.c, context, e.m.is_fp8);
        e.slots = try gpa.alloc(Slot, streams);
        for (e.slots) |*s| s.* = .{};
        for (&e.pass) |*p| p.* = .{ .ids = try buf(e.device, R * 4), .rows = try buf(e.device, R * @sizeOf(kn.Row)), .picks = try buf(e.device, 2 * R * 4), .idx = try buf(e.device, R * 4) };
        @memcpy(u32s(e.pass[7].idx, 2), &[_]u32{ 0, 0 });
        @memcpy(u32s(e.pass[8].idx, 2), &[_]u32{ 0, 2 });
        @memcpy(u32s(e.pass[9].idx, 2), &[_]u32{ 2, 0 });
        @memcpy(u32s(e.pass[10].idx, 2), &[_]u32{ 0, 8 });
        e.xs = try buf(e.device, (2 * R + @as(usize, streams)) * e.m.c.hidden * 4);
        e.accept_meta = try buf(e.device, 64);
        e.compact_disp = try buf(e.device, 64);
        e.gate_meta = try buf(e.device, 64);
        e.gate_disp = try buf(e.device, 16 * 16);
        e.draft_rows = try buf(e.device, 10 * @sizeOf(kn.Row));
        e.draft_x = try buf(e.device, e.m.c.hidden * 4);
        e.words = .empty;
        e.snaps = .empty;
        e.resident = null;
        try e.pair.ready();
        e.load_seconds = @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e9;
        return e;
    }

    fn buf(device: mtl.Device, n: usize) !mtl.Buffer {
        return device.buffer(n, mtl.ResourceOptions.shared);
    }
    fn u32s(b: mtl.Buffer, n: usize) []u32 {
        return b.slice(u32, n);
    }
    fn f32s(b: mtl.Buffer, n: usize) []f32 {
        return b.slice(f32, n);
    }

    pub fn deinit(e: *Engine) void {
        snap.deinit(e);
        @import("residency.zig").drop(e);
        e.pair.deinit();
        for (e.slots) |*s| s.prompt.deinit(e.gpa);
        if (e.dflash2) |d| d.deinit();
        e.m.deinit();
        e.s.deinit();
        e.q.deinit();
        for (e.pass) |p| inline for (.{ p.ids, p.rows, p.picks, p.idx }) |b| b.deinit();
        inline for (.{ e.xs, e.accept_meta, e.compact_disp, e.gate_meta, e.gate_disp, e.draft_rows, e.draft_x }) |b| b.deinit();
        e.words.deinit(e.gpa);
        e.queue.deinit();
        e.device.deinit();
        e.gpa.destroy(e);
    }

    pub fn rank(e: *const Engine) u32 {
        return e.pair.rank;
    }
    pub fn hasMtp(e: *const Engine) bool {
        return e.m.mtp != null;
    }
    pub fn hasDFlash2(e: *const Engine) bool {
        return e.dflash2 != null;
    }
    pub fn hasDraft(e: *const Engine) bool {
        return e.m.mtp != null or e.dflash2 != null;
    }

    /// A scratch whose rows table is pass `i`'s.
    pub fn scratch(e: *const Engine, i: usize) gpu.Scratch {
        var s = e.s;
        s.rows = e.pass[i].rows;
        return s;
    }

    pub fn scratchDraftAt(e: *const Engine, off_rows: u32) gpu.Scratch {
        var s = e.s;
        s.rows = e.draft_rows;
        s.rows_off = @as(usize, off_rows) * @sizeOf(kn.Row);
        return s;
    }

    pub fn settleSlot(e: *Engine, i: u32, kept: u32) void {
        settle(&e.slots[i], kept);
    }

    pub fn setRow(e: *Engine, i: usize, r: usize, slot: u32, pos: u32) void {
        e.pass[i].rows.slice(kn.Row, R)[r] = .{ .slot = slot, .pos = pos, .list = @intCast(r * e.m.c.i_topk) };
    }

    pub fn finish(e: *Engine, cb: mtl.CommandBuffer) !void {
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("glm_dsa: GPU command failed: {s}", .{msg});
            return error.GpuFailure;
        }
        if (e.pair.gaveUp() != 0 or e.pair.failed.load(.acquire)) return error.PairLinkFailed;
        e.gpu_ms = (cb.gpuEnd() - cb.gpuStart()) * 1000;
    }

    /// A fresh stream in slot `i` with its prompt.
    pub fn begin(e: *Engine, i: u32, prompt: []const u32) !void {
        if (i >= e.slots.len or prompt.len == 0 or prompt.len + max_depth + R + 1 > e.context) return error.ContextFull;
        const s = &e.slots[i];
        s.used = true;
        s.len = 0;
        s.open = 0;
        s.held_n = 0;
        s.held_probs_valid = false;
        s.fused_valid = false;
        s.fused_b1 = false;
        s.cal0_scale = 1.0;
        s.cal_scale = 1.0;
        s.fused_depth = 0;
        s.tree = .{};
        if (e.dflash2) |d| d.resetSlot(i);
        s.prompt.clearRetainingCapacity();
        try s.prompt.appendSlice(e.gpa, prompt);
    }

    /// Prompt rows [at, at + n) of slot i (a prompt window) into its caches and the head's; the first token at the end.
    pub fn chunk(e: *Engine, i: u32, at: u32, n: u32, every: bool) !?u32 {
        if (i >= e.slots.len) return error.ChunkOutOfStep;
        const s = &e.slots[i];
        if (!s.used or s.open > 0 or at != s.len or n == 0 or n > gpu.prompt_rows or at + n > s.prompt.items.len) return error.ChunkOutOfStep;
        const hid = e.m.c.hidden;
        const last = at + n == s.prompt.items.len;
        const toks = s.prompt.items;
        const q = &e.q;
        @memcpy(u32s(q.ids, n), toks[at..][0..n]);
        for (q.rows.slice(kn.Row, n), 0..) |*row, r| row.* = .{ .slot = i, .pos = at + @as(u32, @intCast(r)), .list = @intCast(r * e.m.c.i_topk) };
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const s0 = e.scratch(0);
        const profile_prompt = @import("sys").environment.read(bool, "TF_GLM_PROMPT_PROFILE", false) catch false;
        if (profile_prompt) {
            try pw.backboneProfiled(&e.m, e.queue, &s0, q, n, at);
            const cb = e.queue.commandBuffer();
            const enc = cb.compute(.serial);
            var k: u32 = if (every) 0 else n - 1;
            while (k < n and (last or every)) : (k += R) {
                const rows = @min(R, n - k);
                try e.m.pick(enc, .{ .buf = q.x, .off = @as(usize, k) * hid * 4 }, rows, .{ .buf = q.picks, .off = @as(usize, k) * 4 });
            }
            enc.end();
            try e.finish(cb);
            s.len = at + n;
            return if (last) u32s(q.picks, n)[n - 1] else null;
        }
        var cb = e.queue.commandBuffer();
        var enc = cb.compute(e.dispatch);
        const tap_d = if (e.dflash2) |d| (if (at + n > toks.len -| (d.c.window + df2.block_size)) d else null) else null;
        const ane_wait = try pw.backboneTap(&e.m, &s0, q, e.queue, &cb, &enc, e.dispatch, n, at, tap_d);
        var k: u32 = if (every) 0 else n - 1;
        while (k < n and (last or every)) : (k += R) {
            const rows = @min(R, n - k);
            try e.m.pick(enc, .{ .buf = q.x, .off = @as(usize, k) * hid * 4 }, rows, .{ .buf = q.picks, .off = @as(usize, k) * 4 });
        }
        if (e.hasMtp()) {
            // runs inside CB2 right after pick() while 128-core ANE fc computes in parallel!
            const carry: u32 = @intFromBool(at > 0);
            const mr = carry + n - 1;
            for (0..mr) |r| {
                const pos: u32 = at - carry + @as(u32, @intCast(r));
                u32s(q.ids_mtp, gpu.prompt_rows)[r] = toks[pos + 1];
                q.rows_mtp.slice(kn.Row, gpu.prompt_rows)[r] = .{ .slot = i, .pos = pos, .list = @intCast(r * e.m.c.i_topk) };
            }
            u32s(e.pass[1].idx, 1)[0] = 2 * R + i;
            if (carry == 1) kn.gather(&e.m.K.k, enc, .{ .buf = e.xs }, .{ .buf = e.pass[1].idx }, hid, hid, hid, .{ .buf = e.m.head.prev }, 1);
            if (n > 1) kn.gather(&e.m.K.k, enc, .{ .buf = q.x }, .{ .buf = q.idx }, hid, hid, hid, .{ .buf = e.m.head.prev, .off = @as(usize, carry) * hid * 4 }, n - 1);
            enc.barrier();
            u32s(e.pass[3].idx, 1)[0] = n - 1;
            kn.gather(&e.m.K.k, enc, .{ .buf = q.x }, .{ .buf = e.pass[3].idx }, hid, hid, hid, .{ .buf = e.xs, .off = (2 * R + @as(usize, i)) * hid * 4 }, 1);
            enc.barrier();
            if (mr > 0) try pw.draft(&e.m, &s0, q, enc, mr, at - carry);
        }
        if (tap_d) |d| d.absorb(e.pair, e.queue, &cb, &enc, e.dispatch, i, at, 0, n, ane_wait, false);
        enc.end();
        try e.finish(cb);
        s.len = at + n;
        return if (last) u32s(q.picks, n)[n - 1] else null;
    }

    /// One forward over every window's rows: each row's next token into `out` (rows in window order).
    pub fn window(e: *Engine, wins: []const Win, out: []u32) !void {
        var rows: u32 = 0;
        var top: u32 = 0;
        for (wins) |w| {
            if (w.slot >= e.slots.len) return error.WindowOutOfStep;
            const s = &e.slots[w.slot];
            if (s.open > 0) settle(s, s.open);
            if (!s.used or w.tokens.len == 0 or s.len + w.tokens.len + max_depth + 1 > e.context) return error.WindowOutOfStep;
            if (rows + w.tokens.len > R) return error.WindowOutOfStep;
            s.win_row = rows;
            s.open = @intCast(w.tokens.len);
            s.kept = 0;
            s.held_n = 0;
            s.held_probs_valid = false;
            s.fused_valid = false;
            s.fused_depth = 0;
            s.tree = .{};
            for (w.tokens, 0..) |t, j| {
                u32s(e.pass[0].ids, R)[rows] = t;
                e.setRow(0, rows, w.slot, s.len + @as(u32, @intCast(j)));
                top = @max(top, s.len + @as(u32, @intCast(j)));
                rows += 1;
            }
        }
        if (out.len < rows) return error.WindowOutOfStep;
        const hid = e.m.c.hidden;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const s0 = e.scratch(0);
        const profile_win = @import("sys").environment.read(bool, "TF_GLM_PROMPT_PROFILE", false) catch false;
        if (profile_win) {
            try pw.windowProfiled(&e.m, e.queue, &s0, rows, top, .{ .buf = e.pass[0].ids });
            const cb = e.queue.commandBuffer();
            const enc = cb.compute(.serial);
            try e.m.pick(enc, .{ .buf = s0.x }, rows, .{ .buf = e.pass[0].picks });
            for (u32s(e.pass[2].idx, rows), 0..) |*x, r| x.* = @intCast(r);
            kn.gather(&e.m.K.k, enc, .{ .buf = s0.x }, .{ .buf = e.pass[2].idx }, hid, hid, hid, .{ .buf = e.xs }, rows);
            enc.end();
            try e.finish(cb);
            @memcpy(out[0..rows], u32s(e.pass[0].picks, rows));
            return;
        }
        var rname_buf: [16]u8 = undefined;
        const rname = std.fmt.bufPrint(&rname_buf, "win{d}", .{rows}) catch "win";
        e.queue.roundBegin(rname);
        defer e.queue.roundEnd();
        const cb = e.queue.commandBuffer();
        const enc = cb.compute(e.win_dispatch);
        const tap_w = if (e.dflash2) |d| (if (wins.len == 1 and d.slots[wins[0].slot].use_mtp) null else d) else null;
        try e.m.backboneTap(&s0, enc, rows, top, .{ .buf = e.pass[0].ids }, tap_w);
        if (e.hasMtp()) {
            for (u32s(e.pass[2].idx, rows), 0..) |*x, r| x.* = @intCast(r);
            kn.gather(&e.m.K.k, enc, .{ .buf = s0.x }, .{ .buf = e.pass[2].idx }, hid, hid, hid, .{ .buf = e.xs }, rows);
        }
        try e.m.pick(enc, .{ .buf = s0.x }, rows, .{ .buf = e.pass[0].picks });
        enc.end();
        try e.finish(cb);
        @memcpy(out[0..rows], u32s(e.pass[0].picks, rows));
    }

    pub fn gateMtp(e: *Engine, enc: mtl.ComputeEncoder, step: u32, pos: u32, more: bool, picks_buf: mtl.Buffer) void {
        e.gateMtpThresh(enc, step, pos, more, picks_buf, e.m.mtp_gate_thresh);
    }

    pub fn gateMtpThresh(e: *Engine, enc: mtl.ComputeEncoder, step: u32, pos: u32, more: bool, picks_buf: mtl.Buffer, thresh: f32) void {
        var a = e.m.mtpGateArgs(step, pos, more);
        a.thresh = thresh;
        kn.fast.mtpGate(&e.m.K.k.fast_p, enc, .{ .buf = picks_buf }, .{ .buf = e.gate_meta }, .{ .buf = e.gate_disp }, a);
        enc.barrier();
    }

    pub fn runDraftStep(e: *Engine, enc: mtl.ComputeEncoder, d: u32, nq: u32, pos: u32, prev_idx: ?mtl.Buffer, h_src: mtl.Buffer, h_idx: mtl.Buffer, sd: *const gpu.Scratch, use_gate: bool, more: bool, tree_args: kn.TreeArgs) !bool {
        const p = &e.pass[d];
        const hid = e.m.c.hidden;
        const gbuf: ?mtl.Buffer = if (use_gate) e.gate_disp else null;
        const gmeta: ?mtl.Buffer = if (use_gate) e.gate_meta else null;
        if (prev_idx) |pi| kn.gatherGated(&e.m.K.k, enc, .{ .buf = e.pass[d - 1].picks }, .{ .buf = pi }, 1, 1, 1, .{ .buf = p.ids }, nq, gbuf);
        kn.gatherGated(&e.m.K.k, enc, .{ .buf = h_src }, .{ .buf = h_idx }, hid, hid, hid, .{ .buf = e.m.head.prev }, nq, gbuf);
        enc.barrier();
        try e.m.draftGated(sd, enc, nq, pos, .{ .buf = p.ids }, .{ .buf = e.m.head.prev }, gbuf, gmeta, null, tree_args);
        const ok = try e.m.pickDraftGated(enc, .{ .buf = sd.x }, nq, .{ .buf = p.picks }, gbuf, gmeta);
        if (use_gate) e.gateMtp(enc, d, pos + 1, more, p.picks);
        return ok;
    }

    /// Single-CB fused verify + GPU greedy accept + DFlash2 or MTP draft chain (0 CB2!).
    pub fn windowDraft(e: *Engine, slot: u32, tokens: []const u32, pred_depth: u32, out: []u32) !void {
        if (e.dflash2) |d| if (!d.slots[slot].use_mtp) return d.windowFused(e, slot, tokens, pred_depth, out);
        if (!e.hasMtp() or slot >= e.slots.len) return error.WindowOutOfStep;
        const rows: u32 = @intCast(tokens.len);
        if (rows == 0 or rows > 8 or pred_depth > max_depth or out.len < rows) return error.WindowOutOfStep;
        const s = &e.slots[slot];
        if (s.open > 0) settle(s, s.open);
        if (!s.used or s.len + rows + pred_depth + 3 > e.context) return error.WindowOutOfStep;
        const base_pos = s.len;
        const top = base_pos + rows - 1;
        s.win_row = 0;
        s.fused_start = base_pos;
        s.open = 0;
        s.kept = 0;
        s.held_n = 0;
        s.held_probs_valid = false;
        s.fused_valid = false;
        s.fused_b1 = false;
        s.fused_depth = 0;
        s.tree = .{};
        for (tokens, 0..) |t, j| {
            u32s(e.pass[0].ids, R)[j] = t;
            e.setRow(0, j, slot, base_pos + @as(u32, @intCast(j)));
            e.setRow(1, j, slot, base_pos + @as(u32, @intCast(j)));
        }
        const main_n = @min(pred_depth, 6);
        const hid = e.m.c.hidden;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        var rname_buf: [16]u8 = undefined;
        const rname = std.fmt.bufPrint(&rname_buf, "wd{d}_{d}", .{ rows, main_n }) catch "wd";
        e.queue.roundBegin(rname);
        defer e.queue.roundEnd();
        const cb = e.queue.commandBuffer();
        const enc = cb.compute(e.win_dispatch);
        const s0 = e.scratch(0);
        try e.m.backbone(&s0, enc, rows, top, .{ .buf = e.pass[0].ids });
        kn.gather(&e.m.K.k, enc, .{ .buf = s0.x }, .{ .buf = e.q.idx }, hid, hid, hid, .{ .buf = e.xs }, rows);
        try e.m.pick(enc, .{ .buf = s0.x }, rows, .{ .buf = e.pass[0].picks });
        const has_q4 = e.m.vocab.lm_q4_w != null and e.m.vocab.lm_q4_s != null and (hid % 1024 == 0);
        const batch_r2 = has_q4 and e.use_tree and main_n >= 3 and (top + 1) < 2048;
        kn.fast.mtpAccept(&e.m.K.k.fast_p, enc, .{ .buf = e.pass[0].ids }, .{ .buf = e.pass[0].picks }, .{
            .rows = rows,
            .slot = slot,
            .base_pos = base_pos,
            .depth = main_n,
            .i_topk = e.m.c.i_topk,
            .pad0 = @intFromBool(batch_r2),
        }, .{ .buf = e.accept_meta }, .{ .buf = e.pass[1].ids }, .{ .buf = e.pass[2].idx }, .{ .buf = e.draft_rows });
        enc.barrier();
        const s1 = e.scratch(1);
        try e.m.draftGated(&s1, enc, rows, top, .{ .buf = e.pass[1].ids }, .{ .buf = e.xs }, null, null, e.accept_meta, .{});
        var all_prob = false;
        var has_b1 = false;
        if (main_n > 0) {
            kn.gather(&e.m.K.k, enc, .{ .buf = s1.x }, .{ .buf = e.pass[2].idx }, hid, hid, hid, .{ .buf = e.draft_x }, 1);
            enc.barrier();
            all_prob = try e.m.pickDraft(enc, .{ .buf = e.draft_x }, 1, .{ .buf = e.pass[1].picks });
            const res = try tr.runBatchedMtp(e, enc, main_n, top + 1, true, all_prob, batch_r2);
            all_prob = all_prob and res.all_prob;
            has_b1 = res.has_b1;
        }
        enc.end();
        try e.finish(cb);
        const kept = u32s(e.accept_meta, 1)[0];
        @memcpy(out[0..rows], u32s(e.pass[0].picks, rows));
        @memcpy(s.fused_picks[0..rows], out[0..rows]);
        s.fused_path = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
        s.len = base_pos + kept;
        s.kept = kept;
        s.open = 0;
        s.fused_depth = main_n;
        s.fused_valid = true;
        s.fused_b1 = has_b1;
        tr.finishMtpPicks(e, s, main_n, all_prob, has_b1);
    }

    pub fn windowTreeDraft(e: *Engine, slot: u32, tokens: []const u32, parents: []const i32, pred_depth: u32, out: []u32) !void {
        return tr.windowTreeDraft(e, slot, tokens, parents, pred_depth, out);
    }

    /// Extend a still-valid fused windowDraft chain from `s.fused_depth` to `want` with GPU gating.
    pub fn extendDraft(e: *Engine, slot: u32, want: u32) !void {
        return tr.extendDraft(e, slot, want);
    }

    fn settle(s: *Slot, kept: u32) void {
        s.len += kept;
        s.kept = kept;
        s.open = 0;
    }

    /// The cache length the lane core counts: a window no keep settled yet counts whole.
    pub fn cacheLen(e: *const Engine, i: u32) u32 {
        return e.slots[i].len + e.slots[i].open;
    }

    /// The first `kept` rows of slot i's last window stay in its caches (a draft may have settled them already).
    pub fn keep(e: *Engine, i: u32, kept: u32) !void {
        if (i >= e.slots.len) return error.KeepOutOfStep;
        const s = &e.slots[i];
        if (!s.used or kept == 0) return error.KeepOutOfStep;
        if (s.fused_valid) {
            if (kept > s.kept) return error.KeepOutOfStep;
            if (kept < s.kept) {
                s.len = s.fused_start + kept;
                s.kept = kept;
                s.fused_valid = false;
                s.held_n = 0;
                s.held_probs_valid = false;
                s.tree = .{};
                if (e.dflash2) |d| d.slots[i].end = s.len;
            }
            return;
        }
        if (s.open > 0) {
            if (kept > s.open) return error.KeepOutOfStep;
            settle(s, kept);
        } else if (kept != s.kept) return error.KeepOutOfStep;
        if (e.dflash2) |d| if (!d.slots[i].use_mtp and d.slots[i].end < s.len) {
            e.gpu_ms = try d.absorbAndDraft(&e.m, e.queue, e.win_dispatch, i, d.slots[i].end, s.win_row, s.len - d.slots[i].end, 0, 0, 0, e.use_tree, &s.held, &s.held_probs);
        };
    }

    /// Each request's rows into the MTP head's caches, then its chain of drafts (held, read with `held`).
    pub fn draft(e: *Engine, reqs: []const Draft) !void {
        if (!e.hasDraft()) return error.NoMtpHead;
        if (reqs.len == 0 or reqs.len > R) return error.DraftOutOfStep;
        if (e.dflash2) |d| if (reqs.len != 1 or !d.slots[reqs[0].slot].use_mtp) {
            for (reqs) |q| try @import("dflash2_hybrid.zig").draftSlot(d, e, q);
            return;
        };
        var rows: u32 = 0;
        var depth: u32 = 0;
        var ends: [R]u32 = undefined;
        var top: u32 = 0;
        var compact_slot: ?u32 = null;
        var compact_base: u32 = 0;
        var compact_path: []const u32 = &.{};
        for (reqs, 0..) |q, k| {
            if (q.slot >= e.slots.len) return error.DraftOutOfStep;
            const s = &e.slots[q.slot];
            const n: u32 = @intCast(q.follow.len);
            if (!s.used or q.depth > max_depth or n == 0 or rows + n > R) return error.DraftOutOfStep;
            s.fused_valid = false;
            s.fused_depth = 0;
            s.tree = .{};
            const base_before = s.len;
            if (q.prompt and (n != 1 or s.open > 0 or s.len != s.prompt.items.len)) return error.DraftOutOfStep;
            if (!q.prompt and s.open > 0) {
                if (q.path.len == n) {
                    if (!tr.validPath(q.path, s.open)) return error.DraftOutOfStep;
                    if (tr.needsCompact(q.path)) {
                        compact_slot = q.slot;
                        compact_base = base_before;
                        compact_path = q.path;
                    }
                } else if (n > s.open) return error.DraftOutOfStep;
                settle(s, n);
            } else if (!q.prompt and n != s.kept) return error.DraftOutOfStep;
            for (q.follow, 0..) |t, j| {
                const pos = s.len - n + @as(u32, @intCast(j));
                const src_row: u32 = if (q.path.len == n) q.path[j] else @intCast(j);
                u32s(e.pass[1].idx, R)[rows] = if (q.prompt) 2 * R + q.slot else s.win_row + src_row;
                u32s(e.pass[1].ids, R)[rows] = t;
                e.setRow(1, rows, q.slot, pos);
                top = @max(top, pos);
                rows += 1;
            }
            ends[k] = rows - 1;
            depth = @max(depth, q.depth);
            if (s.len + max_depth + 1 > e.context) return error.ContextFull;
        }
        const hid = e.m.c.hidden;
        const nq: u32 = @intCast(reqs.len);
        const main_n = if (nq == 1) @min(depth, 6) else depth;
        if (nq == 1) {
            u32s(e.pass[2].idx, 1)[0] = ends[0];
            tr.prepRows(e, reqs[0].slot, e.slots[reqs[0].slot].len, main_n);
        }
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        e.queue.roundBegin("draft");
        defer e.queue.roundEnd();
        const cb = e.queue.commandBuffer();
        const enc = cb.compute(e.win_dispatch);
        if (compact_slot) |cs| {
            tr.compactKv(&e.m, enc, cs, compact_base, e.context, compact_path);
            enc.barrier();
        }
        kn.gather(&e.m.K.k, enc, .{ .buf = e.xs }, .{ .buf = e.pass[1].idx }, hid, hid, hid, .{ .buf = e.m.head.prev }, rows);
        enc.barrier();
        const s1 = e.scratch(1);
        try e.m.draft(&s1, enc, rows, top, .{ .buf = e.pass[1].ids }, .{ .buf = e.m.head.prev });
        var all_prob = (nq == 1 and main_n > 0);
        var has_b1 = false;
        if (main_n > 0) {
            if (nq == 1) {
                kn.gather(&e.m.K.k, enc, .{ .buf = s1.x }, .{ .buf = e.pass[2].idx }, hid, hid, hid, .{ .buf = e.draft_x }, 1);
                enc.barrier();
                all_prob = (try e.m.pickDraft(enc, .{ .buf = e.draft_x }, 1, .{ .buf = e.pass[1].picks })) and all_prob;
                const res = try tr.runBatchedMtp(e, enc, main_n, top + 1, false, all_prob, all_prob and e.use_tree and main_n >= 2 and (top + 1) < 2048);
                all_prob = all_prob and res.all_prob;
                has_b1 = res.has_b1;
                e.slots[reqs[0].slot].fused_b1 = has_b1;
            } else {
                _ = try e.m.pickDraft(enc, .{ .buf = s1.x }, rows, .{ .buf = e.pass[1].picks });
                var d: u32 = 2;
                while (d <= main_n) : (d += 1) {
                    const p = &e.pass[d];
                    for (reqs, 0..) |q, k| {
                        u32s(p.idx, R)[k] = if (d == 2) ends[k] else @intCast(k);
                        e.setRow(d, k, q.slot, e.slots[q.slot].len - 1 + d - 1);
                    }
                    const sd = e.scratch(d);
                    _ = try e.runDraftStep(enc, d, nq, top + d - 1, p.idx, e.s.x, p.idx, &sd, false, false, .{});
                }
            }
        }
        enc.end();
        try e.finish(cb);
        if (nq == 1) {
            tr.finishMtpPicks(e, &e.slots[reqs[0].slot], main_n, all_prob, has_b1);
        } else for (reqs, 0..) |q, k| {
            const s = &e.slots[q.slot];
            s.held_n = q.depth;
            s.held_probs_valid = false;
            for (0..q.depth) |j| s.held[j] = u32s(e.pass[j + 1].picks, 2 * R)[if (j == 0) ends[k] else k];
        }
    }
    pub fn picks(e: *const Engine, n: usize) []const u32 {
        return u32s(e.q.picks, n);
    }
    pub fn held(e: *const Engine, i: u32) []const u32 {
        return e.slots[i].held[0..e.slots[i].held_n];
    }
    pub fn release(e: *Engine, i: u32) void {
        if (i < e.slots.len) e.slots[i].used = false;
    }
};
