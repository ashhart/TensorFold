//! Living-memory bank sleep-time consolidation: prunes superseded units and compiles standing policies' skill_rec.
const std = @import("std");
const addScaled = @import("compress.zig").addScaled;
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
            if (u.gate.isActive() and u.gate.is_standing and u.rec.len == arch.num_rec) {
                try pol_indices.append(self.gpa, idx);
            }
        }
        if (pol_indices.items.len >= 1) {
            try buildConsolidatedSkillRec(self, pol_indices.items);
            rep.consolidated_policies = @intCast(pol_indices.items.len);
        }
    }

    try self.rebuildBridges();
    rep.bytes = try self.saveFull();
    rep.units = self.activeUnitCount();
    rep.clusters = @intCast(self.cluster_index.n_clusters);
    self.version += 1; // pruned history, compacted policies or a new skill steering: a new published version
    return rep;
}

/// The standing policies' recurrent changes summed whole at 1/sqrt(N) each: the steering composes in their place.
fn buildConsolidatedSkillRec(self: *Bank, pol_indices: []const usize) !void {
    const arch = self.backend.arch;
    const n_pol_f: f32 = @floatFromInt(pol_indices.len);
    const w_each: f32 = 1.0 / @sqrt(n_pol_f);
    const rec = try self.gpa.alloc(RecLayerData, arch.num_rec);
    var made: usize = 0;
    errdefer {
        for (rec[0..made]) |*r| r.deinit(self.gpa);
        self.gpa.free(rec);
    }
    for (0..arch.num_rec) |r| {
        const acc = try self.gpa.alloc(f32, arch.rec_heads * arch.rec_rows * arch.rec_cols);
        errdefer self.gpa.free(acc);
        @memset(acc, 0.0);
        for (pol_indices) |idx| addScaled(self.units.items[idx].rec[r].delta, w_each, acc);
        rec[r] = .{ .delta = acc, .conv = try self.gpa.alloc(u16, 0) };
        made += 1;
    }
    self.skill_rec = rec;
    self.skill_policy_count = @intCast(pol_indices.len);
}
