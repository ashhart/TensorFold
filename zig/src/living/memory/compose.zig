//! Living-memory bank routing and composition: conflict search, gate routing and prefix + unit state assembly.
const std = @import("std");
const quant = @import("quant.zig");
const gate = @import("gate.zig");
const types = @import("types.zig");
const Bank = @import("bank.zig").Bank;
const RawState = types.RawState;
const SnapHandle = types.SnapHandle;

/// Vector-only entity conflict search: returns up to `max_out` active small-fact parent IDs overlapping `gate_ids`.
pub fn findConflictParents(self: *Bank, gate_ids: []const u32, max_out: usize) ![]gate.ConflictMatch {
    if (!self.hasUnits() or gate_ids.len == 0) return try self.gpa.alloc(gate.ConflictMatch, 0);
    const arch = self.backend.arch;
    const q_slice = if (gate_ids.len > 128) gate_ids[0..128] else gate_ids;
    const n = q_slice.len;
    const hl = try self.gpa.alloc(f32, n * arch.hidden);
    defer self.gpa.free(hl);
    const hs = try self.gpa.alloc(f32, n * arch.hidden);
    defer self.gpa.free(hs);
    try self.backend.extractHidden(self.backend.ctx, q_slice, arch.layer_lex, arch.layer_sem, hl, hs);

    var tl = try self.w_lex.?.transform(self.gpa, hl, n);
    defer tl.deinit(self.gpa);
    var ts = try self.w_sem.?.transform(self.gpa, hs, n);
    defer ts.deinit(self.gpa);

    const views = try self.gpa.alloc(gate.UnitView, self.units.items.len);
    defer self.gpa.free(views);
    for (self.units.items, views, 0..) |*u, *v, idx| {
        v.* = .{ .index = idx, .parent_id = u.parent_id, .n_tokens = u.n_tokens, .gate = &u.gate };
    }
    return gate.findConflictParents(self.gpa, views, &tl, &ts, self.w_lex.?.norm_floor, self.w_sem.?.norm_floor, max_out);
}

/// Runs the gate (ClusterIndex, token guard, coverage, 2-hop bridges, policies) on `query_ids`; null on miss.
pub fn routeAndCompose(self: *Bank, query_ids: []const u32) !?SnapHandle {
    if (!self.hasUnits() or query_ids.len == 0) return null;
    const arch = self.backend.arch;

    // Clamp very long queries to the last 256 tokens for fast O(1) gating
    const q_slice = if (query_ids.len > 256) query_ids[query_ids.len - 256 ..] else query_ids;
    const n = q_slice.len;
    const hl = try self.gpa.alloc(f32, n * arch.hidden);
    defer self.gpa.free(hl);
    const hs = try self.gpa.alloc(f32, n * arch.hidden);
    defer self.gpa.free(hs);
    try self.backend.extractHidden(self.backend.ctx, q_slice, arch.layer_lex, arch.layer_sem, hl, hs);

    var tl = try self.w_lex.?.transform(self.gpa, hl, n);
    defer tl.deinit(self.gpa);
    var ts = try self.w_sem.?.transform(self.gpa, hs, n);
    defer ts.deinit(self.gpa);

    const views = try self.gpa.alloc(gate.UnitView, self.units.items.len);
    defer self.gpa.free(views);
    for (self.units.items, views, 0..) |*u, *v, idx| {
        v.* = .{ .index = idx, .parent_id = u.parent_id, .n_tokens = u.n_tokens, .gate = &u.gate };
    }

    const cidx: ?*const gate.ClusterIndex = if (self.cluster_index.isReady()) &self.cluster_index else null;
    const active = try gate.routeUnitsIndexed(
        self.gpa,
        views,
        self.bridges,
        cidx,
        &tl,
        &ts,
        self.w_lex.?.norm_floor,
        self.w_sem.?.norm_floor,
        self.route_cfg,
    );
    defer self.gpa.free(active);
    if (self.trace) {
        std.debug.print("lw6 route: q_len={d} active=[", .{n});
        for (active, 0..) |idx, k| {
            if (k > 0) std.debug.print(",", .{});
            const u = &self.units.items[idx];
            const tag: []const u8 = if (u.gate.is_standing) ":S" else if (u.gate.is_digest) ":D" else "";
            std.debug.print("u{d}(p{d}{s})", .{ u.unit_id, u.parent_id, tag });
        }
        std.debug.print("]", .{});
        for (views) |v| {
            if (!v.gate.isActive() or v.gate.is_standing or v.gate.is_digest) continue;
            const sc = gate.scoreUnitDual(self.gpa, v.gate, &tl, &ts, self.w_lex.?.norm_floor, self.w_sem.?.norm_floor) catch continue;
            const t_sem = @max(self.route_cfg.sem_floor, v.gate.neg_sem_max + self.route_cfg.sem_margin);
            if (sc.lex_gate >= 0.30 or sc.sem_score >= t_sem - 0.05) {
                std.debug.print(" | u{d}(p{d}):lg={d:.2},ns={d},ss={d:.2}/{d:.2}", .{ self.units.items[v.index].unit_id, v.parent_id, sc.lex_gate, sc.n_strong_lex, sc.sem_score, t_sem });
            }
        }
        std.debug.print("\n", .{});
    }
    if (active.len == 0) return null;

    var composed = try self.composeRawState(active);
    defer composed.deinit(self.gpa);
    return try self.backend.buildSnap(self.backend.ctx, &composed, self.suffix_ids);
}

/// Assembles prefix + active-unit KV (K shifted to contiguous offsets) and SVD recurrent state (prefix if rank 0).
pub fn composeRawState(self: *const Bank, active_indices: []const usize) !RawState {
    const arch = self.backend.arch;
    const pref = &self.prefix_state.?;

    var total_len: usize = pref.len;
    for (active_indices) |idx| total_len += self.units.items[idx].n_tokens;

    var out = try RawState.init(self.gpa, arch, @intCast(total_len));
    errdefer out.deinit(self.gpa);

    // 1. Attention KV: the prefix at [0..p_len), then each active unit's own K and V, K shifted to `cur`
    var tmp_k: std.ArrayList(u16) = .empty;
    defer tmp_k.deinit(self.gpa);
    var tmp_v: std.ArrayList(u16) = .empty;
    defer tmp_v.deinit(self.gpa);

    for (0..arch.num_attn) |a| {
        for (0..arch.kv_heads) |h| {
            const src_k = pref.kv_k[a][h * pref.len * arch.head_dim ..][0 .. pref.len * arch.head_dim];
            const src_v = pref.kv_v[a][h * pref.len * arch.head_dim ..][0 .. pref.len * arch.head_dim];
            const dst_k = out.kv_k[a][h * total_len * arch.head_dim ..][0 .. pref.len * arch.head_dim];
            const dst_v = out.kv_v[a][h * total_len * arch.head_dim ..][0 .. pref.len * arch.head_dim];
            @memcpy(dst_k, src_k);
            @memcpy(dst_v, src_v);
        }

        var cur: usize = pref.len;
        for (active_indices) |idx| {
            const u = &self.units.items[idx];
            const u_len: usize = u.n_tokens;
            const rows = arch.kv_heads * u_len;
            const elems = rows * arch.head_dim;
            try tmp_k.resize(self.gpa, elems);
            try tmp_v.resize(self.gpa, elems);

            @memcpy(tmp_k.items, u.kv[a].k);
            @memcpy(tmp_v.items, u.kv[a].v);
            quant.ropeShiftKvBf16(tmp_k.items, arch.kv_heads, u_len, arch.head_dim, cur - pref.len, arch.rotary_dim, arch.rope_theta);

            for (0..arch.kv_heads) |h| {
                const sk = tmp_k.items[h * u_len * arch.head_dim ..][0 .. u_len * arch.head_dim];
                const sv = tmp_v.items[h * u_len * arch.head_dim ..][0 .. u_len * arch.head_dim];
                const dk = out.kv_k[a][(h * total_len + cur) * arch.head_dim ..][0 .. u_len * arch.head_dim];
                const dv = out.kv_v[a][(h * total_len + cur) * arch.head_dim ..][0 .. u_len * arch.head_dim];
                @memcpy(dk, sk);
                @memcpy(dv, sv);
            }
            cur += u_len;
        }
    }

    // 2. Pure-attention transformers (`num_rec == 0`) finish immediately after KV assembly!
    if (arch.num_rec == 0) return out;

    // 3. Parent-balanced SVD recurrent state blending (plus sleep-time consolidated `skill_rec` steering!)
    var has_standing_active = false;
    var max_active_rank: usize = 0;
    for (active_indices) |idx| {
        const u = &self.units.items[idx];
        if (u.gate.is_standing) has_standing_active = true;
        if (u.rec.len > 0 and u.rec[0].rank > max_active_rank) max_active_rank = u.rec[0].rank;
    }
    const use_skill_rec = has_standing_active and self.skill_rec.len == arch.num_rec and self.skill_rec[0].rank > 0;
    if (use_skill_rec and self.skill_rec[0].rank > max_active_rank) {
        max_active_rank = self.skill_rec[0].rank;
    }

    if (max_active_rank == 0) {
        for (0..arch.num_rec) |r| {
            @memcpy(out.rec_state[r], pref.rec_state[r]);
            @memcpy(out.rec_conv[r], pref.rec_conv[r]);
        }
        return out;
    }

    var unique_parents: [16]u32 = undefined;
    var parent_counts: [16]usize = @splat(0);
    var n_parents: usize = 0;
    for (active_indices) |idx| {
        const u = &self.units.items[idx];
        if (use_skill_rec and u.gate.is_standing) continue; // represented jointly by `skill_rec`
        const pid = u.parent_id;
        var found_p: ?usize = null;
        for (0..n_parents) |pi| if (unique_parents[pi] == pid) {
            found_p = pi;
            break;
        };
        if (found_p) |pi| {
            parent_counts[pi] += 1;
        } else if (n_parents < unique_parents.len) {
            unique_parents[n_parents] = pid;
            parent_counts[n_parents] = 1;
            n_parents += 1;
        }
    }
    const total_groups: usize = @max(1, n_parents + @intFromBool(use_skill_rec));

    const weights = try self.gpa.alloc(f32, active_indices.len);
    defer self.gpa.free(weights);
    for (active_indices, weights) |idx, *w| {
        const u = &self.units.items[idx];
        if (use_skill_rec and u.gate.is_standing) {
            w.* = 0.0;
            continue;
        }
        const pid = u.parent_id;
        var pc: usize = 1;
        for (0..n_parents) |pi| if (unique_parents[pi] == pid) {
            pc = parent_counts[pi];
            break;
        };
        w.* = 1.0 / @as(f32, @floatFromInt(@max(1, total_groups * pc)));
    }
    const skill_w: f32 = if (use_skill_rec) 1.0 / @as(f32, @floatFromInt(total_groups)) else 0.0;

    const head_mat = arch.rec_rows * arch.rec_cols;

    for (0..arch.num_rec) |r| {
        @memcpy(out.rec_state[r], pref.rec_state[r]);
        if (use_skill_rec) {
            const rl = &self.skill_rec[r];
            const rk: usize = rl.rank;
            for (0..arch.rec_heads) |h| {
                const fu = rl.u[h * arch.rec_rows * rk ..][0 .. arch.rec_rows * rk];
                const fv = rl.v[h * rk * arch.rec_cols ..][0 .. rk * arch.rec_cols];
                const dst = out.rec_state[r][h * head_mat ..][0..head_mat];
                quant.lowRankAdd(fu, fv, arch.rec_rows, rk, arch.rec_cols, skill_w, dst);
            }
        }
        for (active_indices, weights) |idx, w| {
            if (w == 0.0) continue;
            const u = &self.units.items[idx];
            if (u.rec.len <= r) continue;
            const rl = &u.rec[r];
            const rk: usize = rl.rank;
            if (rk == 0) continue;
            for (0..arch.rec_heads) |h| {
                const fu = rl.u[h * arch.rec_rows * rk ..][0 .. arch.rec_rows * rk];
                const fv = rl.v[h * rk * arch.rec_cols ..][0 .. rk * arch.rec_cols];
                const dst = out.rec_state[r][h * head_mat ..][0..head_mat];
                quant.lowRankAdd(fu, fv, arch.rec_rows, rk, arch.rec_cols, w, dst);
            }
        }
        const use_unit_conv = active_indices.len == 1 and
            self.units.items[active_indices[0]].rec.len > r and
            self.units.items[active_indices[0]].rec[r].conv.len == arch.conv_elems;
        const conv_src = if (use_unit_conv)
            self.units.items[active_indices[0]].rec[r].conv
        else
            pref.rec_conv[r];
        @memcpy(out.rec_conv[r], conv_src);
    }
    return out;
}
