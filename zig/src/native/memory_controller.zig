//! Shared native wrapper turning a `living.memory.Bank` into `api.MemoryController`.
const std = @import("std");
const api = @import("engine_api");
const living = @import("living");
const mem = living.memory;
const Allocator = std.mem.Allocator;

pub const Holder = struct {
    gpa: Allocator,
    bank: *mem.Bank,
    /// Optional family-owned adapter struct freed on `deinit`.
    adapter_ctx: ?*anyopaque = null,
    adapter_free: ?*const fn (*anyopaque, Allocator) void = null,

    pub fn init(
        gpa: Allocator,
        io: std.Io,
        dir: []const u8,
        backend: mem.Backend,
        adapter_ctx: ?*anyopaque,
        adapter_free: ?*const fn (*anyopaque, Allocator) void,
    ) !*Holder {
        const h = try gpa.create(Holder);
        errdefer gpa.destroy(h);
        const bank = try mem.Bank.init(gpa, io, dir, backend);
        errdefer bank.deinit();
        bank.trace = @import("sys").environment.read(bool, "TF_SLIDE_TRACE", false) catch false;
        h.* = .{
            .gpa = gpa,
            .bank = bank,
            .adapter_ctx = adapter_ctx,
            .adapter_free = adapter_free,
        };
        if (try bank.loadIfPresent()) {
            std.log.info(
                "living weights (v6): loaded {d} active/{d} total folded units ({d:.1} MiB, {d} bridges, {d} clusters, {d} deltas) from {s}/{s}",
                .{
                    bank.activeUnitCount(),
                    bank.units.items.len,
                    @as(f64, @floatFromInt(bank.last_bytes)) / (1024.0 * 1024.0),
                    bank.bridges.len,
                    bank.cluster_index.n_clusters,
                    bank.pending_deltas,
                    dir,
                    mem.store.file_name,
                },
            );
        }
        return h;
    }

    pub fn deinit(self: *Holder) void {
        self.bank.deinit();
        if (self.adapter_free) |f| if (self.adapter_ctx) |c| f(c, self.gpa);
        self.gpa.destroy(self);
    }

    pub fn controller(self: *Holder) api.MemoryController {
        return .{
            .ctx = self,
            .hasUnits = hasUnitsFn,
            .routeAndCompose = routeAndComposeFn,
            .dropSnap = dropSnapFn,
            .exec = execFn,
        };
    }

    fn of(ctx: *anyopaque) *Holder {
        return @ptrCast(@alignCast(ctx));
    }

    fn hasUnitsFn(ctx: *anyopaque) bool {
        return of(ctx).bank.hasUnits();
    }

    fn routeAndComposeFn(ctx: *anyopaque, query_ids: []const u32) anyerror!?api.MemorySnapHandle {
        const b = of(ctx).bank;
        const h = (try b.routeAndCompose(query_ids)) orelse return null;
        return .{ .saved = h.saved, .at = h.at };
    }

    fn dropSnapFn(ctx: *anyopaque, snap: *anyopaque) void {
        const b = of(ctx).bank;
        b.backend.dropSnap(b.backend.ctx, snap);
    }

    fn execFn(ctx: *anyopaque, op: *const api.MemoryOp) api.MemoryResult {
        const b = of(ctx).bank;
        switch (op.*) {
            .init => |x| {
                b.ensureCalibrated(x.cal_ids, x.prefix_ids, x.suffix_ids) catch |e| {
                    return .{ .err = @errorName(e) };
                };
                return .{
                    .units = @intCast(b.units.items.len),
                    .affected = b.maxParentId() + 1,
                    .bytes = b.last_bytes,
                };
            },
            .teach_chunk => |x| {
                _ = b.teachChunk(.{
                    .parent_id = x.parent_id,
                    .body_ids = x.body_ids,
                    .gate_ids = x.gate_ids,
                    .title_ids = x.title_ids,
                    .cue_ids = x.cue_ids,
                    .is_small_fact = x.is_small_fact,
                    .is_standing = x.is_standing,
                    .is_digest = x.is_digest,
                }) catch |e| {
                    return .{ .err = @errorName(e) };
                };
                return .{
                    .units = @intCast(b.units.items.len),
                    .affected = 1,
                    .tokens = @intCast(x.body_ids.len),
                    .bytes = b.last_bytes,
                };
            },
            .find_conflicts => |x| {
                const matches = b.findConflictParents(x.gate_ids, 4) catch |e| {
                    return .{ .err = @errorName(e) };
                };
                defer b.gpa.free(matches);
                var out: api.MemoryResult = .{
                    .units = b.activeUnitCount(),
                    .bytes = b.last_bytes,
                    .deltas = b.pending_deltas,
                    .clusters = @intCast(b.cluster_index.n_clusters),
                };
                for (matches, 0..) |m, i| {
                    out.conflict_ids[i] = m.parent_id;
                    out.conflict_count += 1;
                }
                return out;
            },
            .supersede_parent => |x| {
                const changed = b.supersedeParent(x.old_parent_id, x.new_parent_id);
                const nbytes = if (changed > 0)
                    (b.persistTombstone(.supersede_parent, x.old_parent_id, x.new_parent_id) catch |e| {
                        return .{ .err = @errorName(e) };
                    })
                else
                    b.last_bytes;
                return .{
                    .units = b.activeUnitCount(),
                    .affected = changed,
                    .bytes = nbytes,
                    .deltas = b.pending_deltas,
                    .clusters = @intCast(b.cluster_index.n_clusters),
                };
            },
            .consolidate => |x| {
                const rep = b.consolidate(x.prune_superseded, x.compact_policies) catch |e| {
                    return .{ .err = @errorName(e) };
                };
                return .{
                    .units = rep.units,
                    .affected = rep.consolidated_policies,
                    .pruned = rep.pruned_superseded,
                    .clusters = rep.clusters,
                    .deltas = rep.compacted_deltas,
                    .bytes = rep.bytes,
                };
            },
            .save => {
                const nbytes = b.save() catch |e| {
                    return .{ .err = @errorName(e) };
                };
                return .{
                    .units = b.activeUnitCount(),
                    .bytes = nbytes,
                    .deltas = b.pending_deltas,
                    .clusters = @intCast(b.cluster_index.n_clusters),
                };
            },
            .forget_parent => |pid| {
                // durable first: on a failed write the bank is unchanged and the error is the result
                const removed = b.forgetDurable(pid) catch |e| {
                    return .{ .err = @errorName(e) };
                };
                return .{
                    .units = b.activeUnitCount(),
                    .affected = removed,
                    .bytes = b.last_bytes,
                    .deltas = b.pending_deltas,
                    .clusters = @intCast(b.cluster_index.n_clusters),
                };
            },
            .clear_all => {
                const removed = b.clearDurable() catch |e| { // the files go first: on failure the units stay
                    return .{ .err = @errorName(e) };
                };
                return .{ .units = 0, .affected = removed, .bytes = 0 };
            },
            .stats => {
                return .{
                    .units = b.activeUnitCount(),
                    .bytes = b.last_bytes,
                    .deltas = b.pending_deltas,
                    .clusters = @intCast(b.cluster_index.n_clusters),
                };
            },
        }
    }
};

/// Legacy pocket slide opt-in (`TF_SLIDE_LEGACY`, checked at startup with --slide); folded memory is the default.
pub fn useLegacyPocket() bool {
    return @import("sys").environment.read(bool, "TF_SLIDE_LEGACY", false) catch false;
}
