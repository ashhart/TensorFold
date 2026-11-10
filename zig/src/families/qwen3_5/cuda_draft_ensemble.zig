//! DFlash2 and native MTP share one target's features and offer one merged tree for its exact verification.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const Draft = @import("cuda_draft.zig").Draft;
const Mtp = @import("cuda_mtp.zig").Mtp;
const coop = @import("cuda_draft_cooperative.zig");
const dr = lanes.drafter;
const Features = lanes.backend.Features;
const max_streams = @import("cuda_draft_frame.zig").max_streams;
const most = engine.most_window;

pub fn viewBytes(hidden: usize, df_taps: usize) !usize {
    if (hidden == 0 or df_taps == 0 or df_taps > 16) return error.BadEnsembleFeatures;
    const row = try std.math.mul(usize, hidden, 2);
    return std.math.mul(usize, engine.tap_rows, try std.math.mul(usize, row, df_taps + 1));
}

pub const CopyPlan = struct {
    copies: [4]cuda.abi.Memcpy2D = undefined,
    count: usize = 0,
    df: Features,
    mtp: Features,
};

/// Preserve absolute ring positions while compacting each drafter's columns, including a wrapped verify frame.
pub fn copyPlan(hidden: usize, df_taps: usize, union_taps: usize, mtp_index: usize, feature: Features, arena: cuda.DeviceBuffer) !CopyPlan {
    const bytes = try viewBytes(hidden, df_taps);
    const row_bytes = try std.math.mul(usize, hidden, 2);
    const df_pitch = try std.math.mul(usize, df_taps, row_bytes);
    const source_pitch = try std.math.mul(usize, union_taps, row_bytes);
    if (union_taps < df_taps or union_taps > df_taps + 1 or mtp_index >= union_taps or
        feature.space != .device or feature.dtype != .bf16 or feature.ready != .none or
        feature.rows == 0 or feature.row_bytes != source_pitch or arena.len < bytes) return error.BadEnsembleFeatures;
    const source_end = try std.math.add(u64, feature.buffer, try std.math.mul(usize, engine.tap_rows, source_pitch));
    const dest_end = try std.math.add(u64, arena.ptr, bytes);
    if (feature.buffer < dest_end and arena.ptr < source_end) return error.EnsembleFeaturesAlias;
    var plan: CopyPlan = .{ .df = feature, .mtp = feature };
    plan.df.buffer = arena.ptr;
    plan.df.row_bytes = @intCast(df_pitch);
    plan.mtp.buffer = arena.ptr + engine.tap_rows * df_pitch;
    plan.mtp.row_bytes = @intCast(row_bytes);
    var remaining: usize = @min(feature.rows, engine.tap_rows);
    const end = try std.math.add(u64, feature.offset, feature.rows);
    var start = end - remaining;
    while (remaining > 0) {
        const slot = start % engine.tap_rows;
        const height = @min(remaining, engine.tap_rows - slot);
        plan.copies[plan.count] = .{ .src = feature.buffer + slot * source_pitch, .src_pitch = source_pitch, .dst = plan.df.buffer + slot * df_pitch, .dst_pitch = df_pitch, .width = df_pitch, .height = height };
        plan.copies[plan.count + 1] = .{ .src = feature.buffer + slot * source_pitch + mtp_index * row_bytes, .src_pitch = source_pitch, .dst = plan.mtp.buffer + slot * row_bytes, .dst_pitch = row_bytes, .width = row_bytes, .height = height };
        plan.count += 2;
        remaining -= height;
        start += height;
    }
    return plan;
}

pub const Ensemble = struct {
    gpa: std.mem.Allocator,
    e: *engine.Engine,
    df: *Draft,
    mtp: *Mtp,
    taps_storage: [17]u32 = undefined,
    n_taps: usize,
    mtp_index: usize,
    weight: f64,
    graft: bool = false,
    prefix: bool = false,
    query_nodes: usize = 15,
    graft_ms: f64 = 1.0,
    prior: [most]f64 = @splat(0.7),
    bytes: usize,
    views: std.AutoHashMapUnmanaged(*lanes.Stream, cuda.DeviceBuffer) = .empty,
    pool: std.ArrayList(cuda.DeviceBuffer) = .empty,

    pub fn init(gpa: std.mem.Allocator, df: *Draft, mtp: *Mtp) !*Ensemble {
        if (df.c.kind != .dflash2 or !@import("cuda_geometry.zig").dense27(df.e.g) or df.e != mtp.e or
            df.e.head != .a4 or df.tree_nodes == 0 or mtp.trees or !mtp.batch) return error.UnsupportedEnsemble;
        const x = try gpa.create(Ensemble);
        errdefer gpa.destroy(x);
        const df_taps = df.taps();
        if (df_taps.len == 0 or df_taps.len > 16) return error.UnsupportedEnsemble;
        const mt = mtp.taps()[0];
        const index = std.mem.indexOfScalar(u32, df_taps, mt) orelse df_taps.len;
        x.* = .{ .gpa = gpa, .e = df.e, .df = df, .mtp = mtp, .n_taps = df_taps.len + @intFromBool(index == df_taps.len), .mtp_index = index, .weight = 0.3, .bytes = try viewBytes(df.e.g.hidden, df_taps.len) };
        @memcpy(x.taps_storage[0..df_taps.len], df_taps);
        if (index == df_taps.len) x.taps_storage[index] = mt;
        if (try @import("sys").environment.given("TF_DRAFT_MTP_WEIGHT")) |raw| {
            x.weight = try std.fmt.parseFloat(f64, raw);
            if (!std.math.isFinite(x.weight) or x.weight < 0 or x.weight > 1) return error.BadEnsembleWeight;
        }
        x.graft = try @import("cuda_draft_graft.zig").enabled(try @import("sys").environment.given("TF_DRAFT_MTP_GRAFT"));
        if (x.graft and (df.tree_nodes > 15 or df.cooperative_lanes != 1)) return error.UnsupportedEnsemble;
        x.prefix = try @import("cuda_draft_graft.zig").enabled(try @import("sys").environment.given("TF_DRAFT_MTP_PREFIX"));
        if (x.prefix and !x.graft) return error.UnsupportedEnsemble;
        if (x.graft) mtp.head_enabled = false;
        x.query_nodes = df.tree_nodes;
        if (try @import("sys").environment.given("TF_DRAFT_MTP_QUERIES")) |raw| {
            x.query_nodes = try std.fmt.parseInt(usize, raw, 10);
            if (!x.graft or x.query_nodes < df.tree_nodes or x.query_nodes >= most) return error.UnsupportedEnsemble;
        }
        if (x.graft) df.auxiliary_nodes = x.query_nodes;
        const prior = df.drafter().facts().prior;
        if (prior.len > 0) for (&x.prior, 0..) |*p, i| {
            p.* = prior[@min(i, prior.len - 1)];
        };
        std.log.info("DFlash2 + MTP proposals: {d} auxiliary tokens, weight {d:.2}", .{ mtp.depth, x.weight });
        return x;
    }

    pub fn taps(x: *const Ensemble) []const u32 {
        return x.taps_storage[0..x.n_taps];
    }

    pub fn deinit(x: *Ensemble) void {
        cuda.completion.drain(x.e.d, x.e.stream, "ensemble work did not drain before its feature views were freed");
        var it = x.views.valueIterator();
        while (it.next()) |arena| arena.free();
        x.views.deinit(x.gpa);
        for (x.pool.items) |*arena| arena.free();
        x.pool.deinit(x.gpa);
        x.df.deinit();
        x.mtp.deinit();
        x.gpa.destroy(x);
    }

    fn makeView(x: *Ensemble) !cuda.DeviceBuffer {
        var arena = try cuda.DeviceBuffer.alloc(x.e.d, x.bytes);
        errdefer arena.free();
        try arena.fill8(0, null);
        errdefer cuda.completion.drain(x.e.d, x.e.stream, "an ensemble feature warmup failed to drain");
        const h = x.e.g.hidden;
        const grid: [3]usize = .{ 1, (h / 64 + 63) / 64, 1 };
        for ([_]u64{ arena.ptr, arena.ptr + engine.tap_rows * x.df.c.n_taps * h * 2 }) |ptr|
            try x.e.run(x.e.fns.sums, grid, .{ 64, 1, 1 }, .{ ptr, x.df.xs, @as(c_int, @intCast(h)), @as(u64, 0) });
        try x.e.stream.synchronize();
        return arena;
    }

    pub fn reserve(x: *Ensemble, streams: u32) !void {
        if (streams > max_streams) return error.BadEnsembleStreams;
        try x.df.reserve(streams);
        try x.mtp.reserve(streams);
        try x.pool.ensureTotalCapacity(x.gpa, streams);
        while (x.pool.items.len < streams) x.pool.appendAssumeCapacity(try x.makeView());
        if (x.graft) {
            x.graft_ms = try @import("cuda_draft_graft.zig").warm(x);
            std.log.info("DFlash2 batched MTP tree correction: {d:.2} ms", .{x.graft_ms});
        }
    }

    pub fn drafter(x: *Ensemble) dr.Drafter {
        return .{ .kind = "DFlash2 + MTP", .ptr = x, .vtable = &.{ .taps = tapsFn, .facts = factsFn, .open = openFn, .absorb = absorbFn, .hold = holdFn, .held = heldFn, .release = releaseFn, .tree = treeFn, .chances = chancesFn } };
    }

    fn cast(p: *anyopaque) *Ensemble {
        return @ptrCast(@alignCast(p));
    }
    fn tapsFn(p: *anyopaque) []const u32 {
        return cast(p).taps();
    }
    fn factsFn(p: *anyopaque) dr.Facts {
        const x = cast(p);
        var f = x.df.drafter().facts();
        const m = x.mtp.drafter().facts();
        f.stream_bytes += m.stream_bytes + x.bytes;
        if (x.graft) {
            const df_ms = f.step_ms * @as(f64, @floatFromInt(f.depth));
            f.depth = @intCast(@min(x.df.c.block, x.df.tree_nodes));
            f.prior = x.prior[0..f.depth];
            f.step_ms = (df_ms + x.graft_ms) / @as(f64, @floatFromInt(f.depth));
        } else f.step_ms += m.step_ms * @as(f64, @floatFromInt(m.depth)) / @as(f64, @floatFromInt(f.depth));
        return f;
    }

    fn openFn(p: *anyopaque, stream: *lanes.Stream) !void {
        const x = cast(p);
        if (x.views.contains(stream)) return error.StreamOpen;
        try x.df.drafter().open(stream);
        errdefer x.df.drafter().release(stream);
        try x.mtp.drafter().open(stream);
        errdefer x.mtp.drafter().release(stream);
        var arena = x.pool.pop() orelse try x.makeView();
        errdefer arena.free();
        try x.views.put(x.gpa, stream, arena);
    }

    fn absorbFn(p: *anyopaque, items: []const dr.Absorb) !void {
        const x = cast(p);
        if (items.len > max_streams) return error.BadEnsembleStreams;
        var df_items: [max_streams]dr.Absorb = undefined;
        var mtp_items: [max_streams]dr.Absorb = undefined;
        errdefer cuda.completion.drain(x.e.d, x.e.stream, "ensemble feature copies did not drain after a failed absorb");
        for (items, 0..) |item, i| {
            const arena = x.views.get(item.stream) orelse return error.UnknownStream;
            var plan = try copyPlan(x.e.g.hidden, x.df.c.n_taps, x.n_taps, x.mtp_index, item.features, arena);
            for (plan.copies[0..plan.count]) |*copy|
                try x.e.d.check(x.e.d.api.cuMemcpy2DAsync_v2(copy, x.e.stream.handle), "cuMemcpy2DAsync");
            df_items[i] = item;
            df_items[i].features = plan.df;
            mtp_items[i] = item;
            mtp_items[i].features = plan.mtp;
        }
        try x.df.drafter().absorb(df_items[0..items.len]);
        try x.mtp.drafter().absorb(mtp_items[0..items.len]);
    }

    fn holdFn(p: *anyopaque, items: []const dr.Hold) !void {
        const x = cast(p);
        if (items.len > max_streams) return error.BadEnsembleStreams;
        errdefer {
            cuda.completion.drain(x.e.d, x.e.stream, "a failed ensemble hold did not drain");
            for (items) |item| {
                if (x.df.streams.get(item.stream)) |ctx| {
                    ctx.grown = 0;
                    ctx.held_n = 0;
                }
                if (x.mtp.streams.get(item.stream)) |ctx| ctx.held_n = 0;
            }
        }
        for (items) |item| if (item.lanes != null) return error.EnsembleNeedsGrownTree;
        try x.df.drafter().hold(items);
        if (x.graft) return @import("cuda_draft_graft.zig").forward(x, items);
        var holds: [max_streams]dr.Hold = undefined;
        var n: usize = 0;
        for (items) |item| if (item.depth > 0) {
            holds[n] = item;
            holds[n].depth = @min(item.depth, x.mtp.depth);
            n += 1;
        };
        if (n > 0) try x.mtp.drafter().hold(holds[0..n]);
        for (items) |item| {
            const df = x.df.streams.get(item.stream) orelse return error.UnknownStream;
            const mtp = x.mtp.streams.get(item.stream) orelse return error.UnknownStream;
            const count = if (item.depth == 0) 0 else @min(mtp.held_n, x.mtp.depth);
            var parents: [most]i32 = undefined;
            var own: [most]f64 = undefined;
            for (parents[0..count], own[0..count], 0..) |*parent, *chance, i| {
                parent.* = @as(i32, @intCast(i)) - 1;
                chance.* = @log(@as(f64, 0.7));
            }
            const merged = try coop.merge(&.{
                .{ .tokens = df.held[0..df.grown], .parents = df.parents[0..df.grown], .own = df.own[0..df.grown], .weight = 1 - x.weight },
                .{ .tokens = mtp.held[0..count], .parents = parents[0..count], .own = own[0..count], .weight = x.weight },
            }, x.df.tree_nodes, 0.5);
            @memcpy(df.held[0..merged.count], merged.tokens[0..merged.count]);
            @memcpy(df.parents[0..merged.count], merged.parents[0..merged.count]);
            @memcpy(df.own[0..merged.count], merged.own[0..merged.count]);
            df.held_n = merged.count;
            df.grown = merged.count;
        }
    }

    fn heldFn(p: *anyopaque, streams: []const *lanes.Stream, out: []const []u32) !void {
        return cast(p).df.drafter().held(streams, out);
    }
    fn treeFn(p: *anyopaque, stream: *lanes.Stream, gpa: std.mem.Allocator) !?lanes.stream.Held {
        const d = cast(p).df.drafter();
        return d.vtable.tree.?(d.ptr, stream, gpa);
    }
    fn chancesFn(p: *anyopaque, stream: *lanes.Stream, out: []f64) !bool {
        const d = cast(p).df.drafter();
        return d.vtable.chances.?(d.ptr, stream, out);
    }
    fn releaseFn(p: *anyopaque, stream: *lanes.Stream) void {
        const x = cast(p);
        x.df.drafter().release(stream);
        x.mtp.drafter().release(stream);
        if (x.views.fetchRemove(stream)) |entry| {
            var arena = entry.value;
            x.pool.append(x.gpa, arena) catch {
                cuda.completion.drain(x.e.d, x.e.stream, "an ensemble view could not return to its pool");
                arena.free();
            };
        }
    }
};

test "ensemble features preserve both compact row layouts across a ring wrap and reject aliasing" {
    const h = 5120;
    const rb = h * 2;
    const arena: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0x300000000, .len = try viewBytes(h, 5) };
    var f: Features = .{ .buffer = 0x100000000, .offset = engine.tap_rows - 1, .rows = 3, .row_bytes = 6 * rb, .space = .device, .dtype = .bf16 };
    const plan = try copyPlan(h, 5, 6, 5, f, arena);
    try std.testing.expectEqual(@as(usize, 4), plan.count);
    try std.testing.expectEqual(@as(usize, 5 * rb), plan.copies[0].width);
    try std.testing.expectEqual(@as(usize, rb), plan.copies[1].width);
    try std.testing.expectEqual(@as(usize, 1), plan.copies[0].height);
    try std.testing.expectEqual(@as(usize, 2), plan.copies[2].height);
    try std.testing.expectEqual(f.buffer, plan.copies[2].src);
    try std.testing.expectEqual(f.buffer + 5 * rb, plan.copies[3].src);
    try std.testing.expectEqual(arena.ptr, plan.copies[2].dst);
    try std.testing.expectEqual(arena.ptr + engine.tap_rows * 5 * rb, plan.copies[3].dst);
    try std.testing.expectEqual(f.offset, plan.df.offset);
    try std.testing.expectEqual(@as(u32, 5 * rb), plan.df.row_bytes);
    try std.testing.expectEqual(@as(u32, rb), plan.mtp.row_bytes);
    f.row_bytes = 5 * rb;
    try std.testing.expectError(error.BadEnsembleFeatures, copyPlan(h, 5, 6, 5, f, arena));
    f.row_bytes = 6 * rb;
    f.buffer = arena.ptr;
    try std.testing.expectError(error.EnsembleFeaturesAlias, copyPlan(h, 5, 6, 5, f, arena));
}

test {
    _ = @import("cuda_draft_graft.zig");
}
