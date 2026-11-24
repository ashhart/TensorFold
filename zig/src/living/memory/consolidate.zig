//! Living-memory bank sleep-time consolidation: prunes superseded units and compiles standing policies' skill_rec.
const std = @import("std");
const quant = @import("quant.zig");
const types = @import("types.zig");
const Bank = @import("bank.zig").Bank;
const RecLayerData = types.RecLayerData;
const ConsolidateReport = types.ConsolidateReport;

/// Sleep-time consolidation: can prune tombstones, builds policy delta `skill_rec`, rebuilds indexes, compacts.
pub fn consolidate(self: *Bank, prune_superseded: bool, compact_policies: bool) !ConsolidateReport {
    var rep = ConsolidateReport{ .compacted_deltas = self.pending_deltas };

    if (prune_superseded) {
        var i: usize = 0;
        while (i < self.units.items.len) {
            if (self.units.items[i].gate.superseded_by != 0) {
                self.units.items[i].deinit(self.gpa);
                _ = self.units.orderedRemove(i);
                rep.pruned_superseded += 1;
            } else {
                i += 1;
            }
        }
    }

    // Sleep-time spectral recurrent steering consolidation across active standing policies
    self.freeSkillRec();
    const arch = self.backend.arch;
    if (compact_policies and arch.num_rec > 0) {
        var pol_indices: std.ArrayList(usize) = .empty;
        defer pol_indices.deinit(self.gpa);
        for (self.units.items, 0..) |*u, idx| {
            if (u.gate.isActive() and u.gate.is_standing and u.rec.len == arch.num_rec and u.rec[0].rank > 0) {
                try pol_indices.append(self.gpa, idx);
            }
        }
        if (pol_indices.items.len >= 1) {
            const target_rk = @min(@as(usize, 8), @min(arch.rec_rows, arch.rec_cols));
            try buildConsolidatedSkillRec(self, pol_indices.items, target_rk);
            rep.consolidated_policies = @intCast(pol_indices.items.len);
        }
    }

    try self.rebuildBridges();
    rep.bytes = try self.saveFull();
    rep.units = self.activeUnitCount();
    rep.clusters = @intCast(self.cluster_index.n_clusters);
    return rep;
}

fn buildConsolidatedSkillRec(self: *Bank, pol_indices: []const usize, rank: usize) !void {
    const arch = self.backend.arch;
    const head_mat = arch.rec_rows * arch.rec_cols;
    const n_pol_f: f32 = @floatFromInt(pol_indices.len);
    // Energy-preserving 1/sqrt(N) scaling preserves individual policy steering strength in Mamba-2/GDN
    const w_each: f32 = 1.0 / @sqrt(n_pol_f);

    const acc = try self.gpa.alloc(f32, head_mat);
    defer self.gpa.free(acc);

    const rec = try self.gpa.alloc(RecLayerData, arch.num_rec);
    var made: usize = 0;
    errdefer {
        for (rec[0..made]) |*r| r.deinit(self.gpa);
        self.gpa.free(rec);
    }
    var scratch_arena = std.heap.ArenaAllocator.init(self.gpa);
    defer scratch_arena.deinit();

    for (0..arch.num_rec) |r| {
        const fu = try self.gpa.alloc(f32, arch.rec_heads * arch.rec_rows * rank);
        errdefer self.gpa.free(fu);
        const fv = try self.gpa.alloc(f32, arch.rec_heads * rank * arch.rec_cols);
        errdefer self.gpa.free(fv);
        const cv = try self.gpa.alloc(u16, 0);
        errdefer self.gpa.free(cv);

        for (0..arch.rec_heads) |h| {
            _ = scratch_arena.reset(.retain_capacity);
            @memset(acc, 0.0);
            for (pol_indices) |idx| {
                const rl = &self.units.items[idx].rec[r];
                const rk: usize = rl.rank;
                if (rk == 0) continue;
                quant.lowRankAdd(
                    rl.u[h * arch.rec_rows * rk ..][0 .. arch.rec_rows * rk],
                    rl.v[h * rk * arch.rec_cols ..][0 .. rk * arch.rec_cols],
                    arch.rec_rows,
                    rk,
                    arch.rec_cols,
                    w_each,
                    acc,
                );
            }
            _ = try quant.randomizedSvdBalanced(
                scratch_arena.allocator(),
                acc,
                arch.rec_rows,
                arch.rec_cols,
                rank,
                fu[h * arch.rec_rows * rank ..][0 .. arch.rec_rows * rank],
                fv[h * rank * arch.rec_cols ..][0 .. rank * arch.rec_cols],
            );
        }
        rec[r] = .{ .rank = @intCast(rank), .u = fu, .v = fv, .conv = cv };
        made += 1;
    }
    self.skill_rec = rec;
    self.skill_policy_count = @intCast(pol_indices.len);
}
