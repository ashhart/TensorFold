//! Shared verify rounds across slots (TP={1,2,4}): one forward over several windows, then each keeps accepted rows.
const std = @import("std");
const ze = @import("ze");
const q = @import("qwen27.zig");
const dev = @import("dev.zig");
const fwd = @import("forward.zig");
const LAYERS = q.LAYERS;
const Runner = q.Runner;
const forwardRunner = fwd.forwardRunner;
const kvCap = fwd.kvCap;
const off = dev.off;

pub const Win = struct { slot: u32, ids: []const u32, parents: ?[]const i32 = null };

pub fn verifyMulti(r: *Runner, wins: []const Win) !void {
    if (wins.len == 0 or wins.len > 16) return error.BadBlock;
    r.store();
    var T: usize = 0;
    var max_pos: u32 = 0;
    for (wins, 0..) |w, i| {
        for (wins[0..i]) |o| if (o.slot == w.slot) return error.SlotTwice;
        if (w.ids.len == 0 or T + w.ids.len > 16) return error.BadBlock;
        const p = r.seqs[w.slot].pos;
        if (p + w.ids.len > r.context) return error.ContextFull;
        r.win_host[2 * i] = @intCast(T);
        r.win_host[2 * i + 1] = @intCast(w.ids.len);
        for (w.ids, 0..) |id, j| {
            r.pos_host[T + j] = @intCast(p + j);
            r.ids_host[T + j] = id;
        }
        max_pos = @max(max_pos, p + @as(u32, @intCast(w.ids.len)) - 1);
        T += w.ids.len;
    }
    var tree = false;
    for (wins) |w| if (w.parents) |pa| {
        if (pa.len != w.ids.len) return error.BadBlock;
        for (pa, 0..) |x, j| if (x != @as(i32, @intCast(j)) - 1) {
            tree = true;
        };
    };
    if (tree) {
        max_pos = 0;
        for (wins, 0..) |w, i| {
            const first: usize = @intCast(r.win_host[2 * i]);
            const p = r.seqs[w.slot].pos;
            for (0..w.ids.len) |j| {
                const t = first + j;
                const row = r.tree_host[t * 17 ..][0..17];
                const pj: i32 = if (w.parents) |pa| pa[j] else @as(i32, @intCast(j)) - 1;
                if (j == 0) {
                    if (pj != -1) return error.BadBlock;
                    row[0] = 0;
                    row[1] = @intCast(t);
                } else {
                    if (pj < 0 or pj >= j) return error.BadBlock;
                    const pr = r.tree_host[(first + @as(usize, @intCast(pj))) * 17 ..][0..17];
                    const dp: usize = @intCast(pr[0] & 255);
                    row[0] = @intCast(dp + 1);
                    @memcpy(row[1 .. dp + 2], pr[1 .. dp + 2]);
                    row[dp + 2] = @intCast(t);
                    if (pj + 1 != j) {
                        row[0] |= 512;
                        pr[0] |= 256;
                    }
                }
                r.pos_host[t] = @intCast(p + @as(u32, @intCast(row[0] & 255)));
                max_pos = @max(max_pos, p + @as(u32, @intCast(row[0] & 255)));
            }
        }
    }

    for (0..r.tp) |rk| {
        const d = r.devAt(rk);
        const m = r.modelAt(rk);
        const a = r.actsAt(rk);
        const mt_buf = &r.mt_hosts[rk];
        if (tree) {
            try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, a.tree, &r.tree_host, T * 17 * 4, null, 0, null));
        }
        for (&m.layers, 0..) |*l, i| for (wins, 0..) |w, wi| {
            const first: usize = @intCast(r.win_host[2 * wi]);
            for (0..w.ids.len) |j| {
                const e = mt_buf[(i * 16 + first + j) * 8 ..][0..8];
                e.* = @splat(0);
                e[6] = first;
                e[7] = w.ids.len;
                if (l.attn) {
                    e[0] = @intFromPtr(l.slot_kc[w.slot].?);
                    e[1] = @intFromPtr(l.slot_vc[w.slot].?);
                } else {
                    const cur = l.slot_cur[w.slot];
                    e[2] = @intFromPtr(l.slot_conv[w.slot][cur].?);
                    e[3] = @intFromPtr(l.slot_conv[w.slot][1 - cur].?);
                    e[4] = @intFromPtr(l.slot_rec[w.slot].?);
                    e[5] = @intFromPtr(l.slot_save[w.slot].?);
                }
            }
        };
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, a.mt, mt_buf, LAYERS * 16 * 8 * 8, null, 0, null));
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, a.mpos, &r.pos_host, T * 4, null, 0, null));
    }

    _ = try forwardRunner(r, @ptrCast(r.ids_host[0..T]), 0, .{ .want = .rows, .stash = true, .taps = r.taps_on, .tree = tree, .multi_max_pos = max_pos, .rows_out = &r.tok_host, .cand = r.candMode() });
    for (r.block[0..T], r.tok_host[0..T]) |*o, t| o.* = @intCast(t);
    for (wins, 0..) |w, i| r.multi_slots[i] = w.slot;
    r.multi_count = @intCast(wins.len);
    r.multi_tree = tree;
}

pub fn keepMulti(r: *Runner, keep: []const u32) !void {
    var rows: [16]u32 = undefined;
    for (&rows, 0..) |*x, i| x.* = @intCast(i);
    var paths: [16][]const u32 = undefined;
    if (keep.len > 16) return error.BadCommit;
    for (keep, 0..) |k, i| paths[i] = rows[0..@min(k, 16)];
    return r.keepPaths(paths[0..keep.len]);
}

pub fn keepPaths(r: *Runner, paths: []const []const u32) !void {
    if (paths.len != r.multi_count) return error.BadCommit;
    const sh = q.tpShape();
    const nul: ?*anyopaque = null;
    for (paths, 0..) |path, i| {
        const first: u32 = @intCast(r.win_host[2 * i]);
        const width: u32 = @intCast(r.win_host[2 * i + 1]);
        if (path.len == 0 or path.len > width or path[0] != 0) return error.BadCommit;
        for (path, 0..) |row, dd| {
            if (row >= width or (dd > 0 and row <= path[dd - 1])) return error.BadCommit;
            r.mpath_host[(i * 2) * 16 + dd] = @intCast(first + row);
            r.mpath_host[(i * 2 + 1) * 16 + dd] = @intCast(row);
        }
    }

    for (0..r.tp) |rk| {
        const d = r.devAt(rk);
        const m = r.modelAt(rk);
        const a = r.actsAt(rk);
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, a.mpath, &r.mpath_host, paths.len * 2 * 16 * 4, null, 0, null));
        for (paths, 0..) |path, i| {
            const sl = r.multi_slots[i];
            const first: usize = @intCast(r.win_host[2 * i]);
            const width: u32 = @intCast(r.win_host[2 * i + 1]);
            const k: u32 = @intCast(path.len);
            var straight = true;
            for (path, 0..) |row, dd| straight = straight and row == dd;
            const brows = off(a.mpath, (i * 2) * 16 * 4);
            const wrows = off(a.mpath, (i * 2 + 1) * 16 * 4);
            for (&m.layers) |*l| {
                if (l.attn) {
                    if (!straight) try d.launch(d.k.kv_move, .{ .x = sh.akv }, .{ l.slot_kc[sl].?, l.slot_vc[sl].?, @as(i32, @intCast(r.seqs[sl].pos)), wrows, @as(i32, @intCast(k)), kvCap() });
                    continue;
                }
                const cur = l.slot_cur[sl];
                if (k == width and straight and !r.multi_tree) {
                    std.mem.swap(?*anyopaque, &l.slot_rec[sl], &l.slot_save[sl]);
                } else if (straight and !r.multi_tree) {
                    const st = l.stash.?;
                    try d.launch(d.k.gdn_step, .{ .x = 16, .y = sh.gv }, .{
                        off(st, sh.st_q + first * @as(usize, sh.gk) * 128 * 2),
                        off(st, sh.st_k + first * @as(usize, sh.gk) * 128 * 2),
                        off(st, sh.st_v + first * @as(usize, sh.gv) * 128 * 2),
                        off(st, sh.st_g + first * @as(usize, sh.gv) * 4),
                        off(st, sh.st_b + first * @as(usize, sh.gv) * 4),
                        l.slot_rec[sl].?,
                        l.slot_rec[sl].?,
                        a.y,
                        @as(i32, @intCast(k)),
                        nul,
                        nul,
                        nul,
                        nul,
                    });
                    try d.launch(d.k.conv_commit, .{ .x = sh.nconv / 256 }, .{ l.slot_conv[sl][cur].?, off(st, sh.st_p + first * @as(usize, sh.gdn_in) * 2), l.slot_conv[sl][1 - cur].?, @as(i32, @intCast(k)), nul });
                } else {
                    const st = l.stash.?;
                    try d.launch(d.k.gdn_step, .{ .x = 16, .y = sh.gv }, .{ off(st, sh.st_q), off(st, sh.st_k), off(st, sh.st_v), off(st, sh.st_g), off(st, sh.st_b), l.slot_rec[sl].?, l.slot_rec[sl].?, a.y, @as(i32, @intCast(k)), brows, nul, nul, nul });
                    try d.launch(d.k.conv_commit, .{ .x = sh.nconv / 256 }, .{ l.slot_conv[sl][cur].?, off(st, sh.st_p), l.slot_conv[sl][1 - cur].?, @as(i32, @intCast(k)), brows });
                }
                l.slot_cur[sl] = 1 - cur;
            }
        }
        for (&m.layers) |*l| if (!l.attn) {
            l.rec = l.slot_rec[r.slot];
            l.rec_save = l.slot_save[r.slot];
            l.conv_cur = l.slot_cur[r.slot];
        };
    }

    for (paths, 0..) |path, i| {
        const sl = r.multi_slots[i];
        const first: u32 = @intCast(r.win_host[2 * i]);
        const k: u32 = @intCast(path.len);
        r.seqs[sl].pos += k;
        r.seqs[sl].last = r.block[first + path[k - 1]];
    }
    r.multi_count = 0;
    r.multi_tree = false;
    r.pos = r.seqs[r.slot].pos;
    r.last = r.seqs[r.slot].last;
}

pub fn stepMulti(r: *Runner, slots: []const u32, tokens: []const u32) !void {
    var wins: [16]Win = undefined;
    if (slots.len > 16 or tokens.len != slots.len) return error.BadBlock;
    for (slots, tokens, 0..) |sl, *t, i| wins[i] = .{ .slot = sl, .ids = t[0..1] };
    try r.verifyMulti(wins[0..slots.len]);
    const ones: [16]u32 = @splat(1);
    try r.keepMulti(ones[0..slots.len]);
}
