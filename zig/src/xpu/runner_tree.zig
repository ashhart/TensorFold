//! Single-slot tree/window verify, commit, masked argmax, and host logit readout for the Qwen3.8-27B XPU Runner.
const std = @import("std");
const ze = @import("ze");
const q = @import("qwen27.zig");
const dev = @import("dev.zig");
const fwd = @import("forward.zig");

const Runner = q.Runner;
const MAX_SLOTS = q.MAX_SLOTS;
const VOCAB = q.VOCAB;
const MASK_WORDS = q.MASK_WORDS;
const D = q.D;
const alloc = q.alloc;
const copy = dev.copy;
const off = dev.off;
const forwardRunner = fwd.forwardRunner;
const kvCap = fwd.kvCap;

pub fn select(r: *Runner, slot: u32) !void {
    if (slot >= MAX_SLOTS) return error.NoSuchSlot;
    if (slot == r.slot) return;
    try r.ensure(slot);
    r.store();
    r.seqs[r.slot] = .{ .pos = r.pos, .last = r.last };
    for (0..r.tp) |rk| {
        for (&r.modelAt(rk).layers) |*l| {
            if (l.attn) {
                l.kc = l.slot_kc[slot];
                l.vc = l.slot_vc[slot];
            } else {
                l.rec = l.slot_rec[slot];
                l.rec_save = l.slot_save[slot];
                l.conv = l.slot_conv[slot];
                l.conv_cur = l.slot_cur[slot];
            }
        }
    }
    r.pos = r.seqs[slot].pos;
    r.last = r.seqs[slot].last;
    r.slot = slot;
    r.block_rows = 0;
    r.tree_on = false;
}

pub fn store(r: *Runner) void {
    for (0..r.tp) |rk| {
        for (&r.modelAt(rk).layers) |*l| {
            if (l.attn) {
                l.slot_kc[r.slot] = l.kc;
                l.slot_vc[r.slot] = l.vc;
            } else {
                l.slot_rec[r.slot] = l.rec;
                l.slot_save[r.slot] = l.rec_save;
                l.slot_conv[r.slot] = l.conv;
                l.slot_cur[r.slot] = l.conv_cur;
            }
        }
    }
    r.seqs[r.slot] = .{ .pos = r.pos, .last = r.last };
}

pub fn reset(r: *Runner) !void {
    const sh = q.tpShape();
    const rec_b = @as(usize, sh.gv) * 128 * 128 * 4;
    const conv_b = 3 * @as(usize, sh.nconv) * 2;
    for (0..r.tp) |rk| {
        const d = r.devAt(rk);
        for (&r.modelAt(rk).layers) |*l| if (!l.attn) {
            try d.fill(l.rec.?, @as(u32, 0), rec_b);
            for (l.conv) |cv| try d.fill(cv.?, @as(u16, 0), conv_b);
            l.conv_cur = 0;
        };
    }
    r.pos = 0;
}

pub fn step(r: *Runner, token: u32) !void {
    if (r.pos + 1 > r.context) return error.ContextFull;
    const id: i32 = @bitCast(token);
    r.last = @intCast((try forwardRunner(r, &.{id}, r.pos, .{ .taps = r.taps_on, .cand = r.candMode() })).?);
    r.pos += 1;
    if (r.taps_on) if (r.on_taps) |cb| try cb.f(cb.ctx, r.acts.taps, 1);
}

pub fn verify(r: *Runner, ids: []const u32) ![]const u32 {
    if (ids.len == 0 or ids.len > 16) return error.BadBlock;
    if (r.pos + ids.len > r.context) return error.ContextFull;
    _ = try forwardRunner(r, @ptrCast(ids), r.pos, .{ .want = .rows, .stash = true, .taps = r.taps_on, .rows_out = &r.tok_host, .cand = r.candMode() });
    for (r.block[0..ids.len], r.tok_host[0..ids.len]) |*o, t| o.* = @intCast(t);
    r.block_rows = @intCast(ids.len);
    return r.block[0..ids.len];
}

pub fn commit(r: *Runner, keep: u32) !void {
    if (keep == 0 or keep > r.block_rows or r.tree_on) return error.BadCommit;
    const sh = q.tpShape();
    const nul: ?*anyopaque = null;
    for (0..r.tp) |rk| {
        const d = r.devAt(rk);
        const a = r.actsAt(rk);
        for (&r.modelAt(rk).layers) |*l| if (!l.attn) {
            if (keep == r.block_rows) {
                std.mem.swap(?*anyopaque, &l.rec, &l.rec_save);
                continue;
            }
            const st = l.stash.?;
            try d.launch(d.k.gdn_step, .{ .x = 16, .y = sh.gv }, .{ off(st, sh.st_q), off(st, sh.st_k), off(st, sh.st_v), off(st, sh.st_g), off(st, sh.st_b), l.rec.?, l.rec.?, a.y, @as(i32, @intCast(keep)), nul, nul, nul, nul });
            try d.launch(d.k.conv_commit, .{ .x = sh.nconv / 256 }, .{ l.conv[1 - l.conv_cur].?, off(st, sh.st_p), l.conv[l.conv_cur].?, @as(i32, @intCast(keep)), nul });
        };
    }
    r.pos += keep;
    r.last = r.block[keep - 1];
    r.block_rows = 0;
}

pub fn verifyTree(r: *Runner, ids: []const u32, parents: []const i32) ![]const u32 {
    if (ids.len == 0 or ids.len > 16 or parents.len != ids.len) return error.BadBlock;
    var depth_max: u32 = 0;
    for (0..ids.len) |t| {
        const row = r.tree_host[t * 17 ..][0..17];
        if (t == 0) {
            row[0] = 0;
            row[1] = 0;
            continue;
        }
        const p: usize = @intCast(parents[t]);
        if (p >= t) return error.BadBlock;
        const pr = r.tree_host[p * 17 ..][0..17];
        row[0] = pr[0] + 1;
        @memcpy(row[1 .. 1 + @as(usize, @intCast(pr[0])) + 1], pr[1 .. 1 + @as(usize, @intCast(pr[0])) + 1]);
        row[1 + @as(usize, @intCast(row[0]))] = @intCast(t);
        depth_max = @max(depth_max, @as(u32, @intCast(row[0])));
    }
    for (1..ids.len) |t| {
        const p: usize = @intCast(parents[t]);
        if (p + 1 != t) {
            r.tree_host[t * 17] |= 512;
            r.tree_host[p * 17] |= 256;
        }
    }
    if (r.pos + depth_max + 1 > r.context or r.pos + ids.len > r.context) return error.ContextFull;
    for (0..r.tp) |rk| {
        const d = r.devAt(rk);
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, r.actsAt(rk).tree, &r.tree_host, ids.len * 17 * 4, null, 0, null));
    }
    _ = try forwardRunner(r, @ptrCast(ids), r.pos, .{ .want = .rows, .stash = true, .taps = r.taps_on, .tree = true, .rows_out = &r.tok_host, .cand = r.candMode() });
    for (r.block[0..ids.len], r.tok_host[0..ids.len]) |*o, t| o.* = @intCast(t);
    r.block_rows = @intCast(ids.len);
    r.tree_on = true;
    return r.block[0..ids.len];
}

pub fn commitPath(r: *Runner, path: []const u32) !void {
    if (!r.tree_on or path.len == 0 or path.len > r.block_rows or path[0] != 0) return error.BadCommit;
    const sh = q.tpShape();
    var straight = true;
    for (path, 0..) |row, dd| {
        r.path_host[dd] = @intCast(row);
        straight = straight and row == dd;
    }
    const keep: i32 = @intCast(path.len);
    const nul: ?*anyopaque = null;
    for (0..r.tp) |rk| {
        const d = r.devAt(rk);
        const a = r.actsAt(rk);
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, a.path, &r.path_host, path.len * 4, null, 0, null));
        for (&r.modelAt(rk).layers) |*l| {
            if (l.attn) {
                if (!straight) try d.launch(d.k.kv_move, .{ .x = sh.akv }, .{ l.kc.?, l.vc.?, @as(i32, @intCast(r.pos)), a.path, keep, kvCap() });
                continue;
            }
            const st = l.stash.?;
            try d.launch(d.k.gdn_step, .{ .x = 16, .y = sh.gv }, .{ off(st, sh.st_q), off(st, sh.st_k), off(st, sh.st_v), off(st, sh.st_g), off(st, sh.st_b), l.rec.?, l.rec.?, a.y, keep, a.path, nul, nul, nul });
            try d.launch(d.k.conv_commit, .{ .x = sh.nconv / 256 }, .{ l.conv[1 - l.conv_cur].?, off(st, sh.st_p), l.conv[l.conv_cur].?, keep, a.path });
        }
    }
    if (r.taps_on and !straight) for (path, 0..) |row, dd| if (row != dd)
        try copy(&r.d, off(r.acts.taps, dd * 5 * D * 2), off(r.acts.taps, row * 5 * D * 2), 5 * D * 2);
    r.pos += @intCast(path.len);
    r.last = r.block[path[path.len - 1]];
    r.block_rows = 0;
    r.tree_on = false;
    try r.syncAll();
}

pub fn argmaxMasked(r: *Runner, rows: []const u32, bits: []const u32, words: usize, out: []u32) !void {
    const n = rows.len;
    if (n == 0 or n > 16 or words > MASK_WORDS or bits.len != n * words or out.len < n) return error.BadMasks;
    const d = &r.d;
    var rh: [16]i32 = undefined;
    for (rows, 0..) |x, j| rh[j] = @intCast(x);
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, r.acts.mrows, &rh, n * 4, null, 0, null));
    errdefer d.settle();
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, r.acts.mbits, bits.ptr, bits.len * 4, null, 0, null));
    try d.launch(d.k.argmax_part_masked, .{ .x = 256, .y = @intCast(n) }, .{ r.acts.logits, @as(i32, VOCAB), r.acts.mbits, @as(i32, @intCast(words)), r.acts.mrows, r.acts.pv, r.acts.pi });
    try d.launch(d.k.argmax, .{ .x = @intCast(n) }, .{ r.acts.pv, r.acts.pi, r.acts.mtok });
    var toks: [16]i32 = undefined;
    try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, &toks, r.acts.mtok, n * 4, null, 0, null));
    try d.sync();
    for (out[0..n], toks[0..n]) |*o, t| o.* = @intCast(t);
}

pub fn rowLogits(r: *Runner, row: u32) ![]const f32 {
    const raw = try alloc.alloc(f16, VOCAB);
    defer alloc.free(raw);
    try ze.check(r.d.g.api.zeCommandListAppendMemoryCopy(r.d.list, raw.ptr, off(r.acts.logits, row * VOCAB * 2), VOCAB * 2, null, 0, null));
    try r.d.sync();
    for (r.host_logits, raw) |*o, v| o.* = v;
    return r.host_logits;
}

pub fn logits(r: *Runner) ![]const f32 {
    const raw = try alloc.alloc(f16, VOCAB);
    defer alloc.free(raw);
    try ze.check(r.d.g.api.zeCommandListAppendMemoryCopy(r.d.list, raw.ptr, r.acts.logits, VOCAB * 2, null, 0, null));
    try r.d.sync();
    for (r.host_logits, raw) |*o, v| o.* = v;
    return r.host_logits;
}
