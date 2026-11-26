//! CUDA DSpark adapter: committed target taps into per-stream speculative heads.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const dspark = @import("dspark.zig");
const engine = @import("loop_engine.zig");

const dr = lanes.drafter;
const Stream = lanes.Stream;
const prior = [_]f64{ 0.55, 0.45, 0.35 };

const Entry = struct {
    context: dspark.Context,
    held: [dspark.depth]u32 = undefined,
    count: usize = 0,
};

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    s: cuda.Stream,
    kernels: *engine.Kernels,
    context: u32,
    lane: cuda.qmmf.Lane,
    weights: dspark.Weights,
    head: dspark.Head,
    entries: std.AutoHashMapUnmanaged(*Stream, *Entry) = .empty,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, target: *engine.Model, kernels: *engine.Kernels, s: cuda.Stream, target_dir: []const u8, dir: []const u8) !*Drafter {
        const out = try gpa.create(Drafter);
        errdefer gpa.destroy(out);
        const mtp = target.w.c.mtp orelse return error.DSparkNotConfigured;
        if (mtp.layers != 3 or mtp.q_heads != 24 or mtp.kv_heads != 8 or mtp.head_dim != 128 or
            mtp.inter != dspark.inter or mtp.window != 1024 or mtp.block != dspark.rows or
            mtp.rank != dspark.rank or mtp.rope_theta != 1e6 or mtp.taps_n != dspark.taps.len or
            !std.mem.eql(u32, mtp.taps[0..mtp.taps_n], &dspark.taps) or
            !mtp.local[0] or mtp.local[1] or !mtp.local[2])
            return error.UnsupportedDSparkConfig;
        const d = target.d;
        var weights = try dspark.load(gpa, io, d, target_dir, dir, target.w.c.text.vocab);
        errdefer weights.deinit(gpa);
        var lane = try cuda.qmmf.Lane.load(d, (try d.currentCapability()) / 10);
        errdefer lane.unload();
        const head = try dspark.Head.init(gpa, d, &weights, &kernels.draft.?, &kernels.glue, .{ .w = target.w, .lane = &lane, .unembed = weights.unembed }, target.context);
        out.* = .{
            .gpa = gpa,
            .kernels = kernels,
            .d = d,
            .s = s,
            .context = @intCast(target.context),
            .lane = lane,
            .weights = weights,
            .head = head,
        };
        // A Head points at the owners, not the temporary values from load.
        out.head.w = &out.weights;
        out.head.t.lane = &out.lane;
        return out;
    }

    pub fn deinit(out: *Drafter) void {
        out.s.synchronize() catch {};
        var it = out.entries.valueIterator();
        while (it.next()) |entry| {
            entry.*.context.deinit();
            out.gpa.destroy(entry.*);
        }
        out.entries.deinit(out.gpa);
        out.head.deinit(out.gpa);
        out.lane.unload();
        out.weights.deinit(out.gpa);
        out.gpa.destroy(out);
    }

    pub fn reserve(_: *Drafter, streams: u32) !void {
        if (streams > engine.max_slots) return error.Inkling2TooManyStreams;
    }

    pub fn drafter(out: *Drafter) dr.Drafter {
        return .{ .kind = "DSpark", .ptr = out, .vtable = &.{
            .taps = tapsFn,
            .facts = factsFn,
            .open = openFn,
            .absorb = absorbFn,
            .hold = holdFn,
            .held = heldFn,
            .release = releaseFn,
        } };
    }

    fn cast(ptr: *anyopaque) *Drafter {
        return @ptrCast(@alignCast(ptr));
    }

    fn tapsFn(_: *anyopaque) []const u32 {
        return &dspark.taps;
    }

    fn factsFn(ptr: *anyopaque) dr.Facts {
        const out = cast(ptr);
        return .{
            .depth = dspark.depth,
            .step_ms = 0,
            .prior = &prior,
            .plain_guard = false,
            .fixed_block = true,
            .batched = true,
            .stream_bytes = 2 * (@as(usize, 2 * dspark.local_ring) + out.context) * dspark.kv * 2 + dspark.hidden * 2,
        };
    }

    fn openFn(ptr: *anyopaque, stream: *Stream) anyerror!void {
        const out = cast(ptr);
        if (out.entries.contains(stream)) return error.DSparkStreamAlreadyOpen;
        if (out.entries.count() >= engine.max_slots) return error.Inkling2TooManyStreams;
        const entry = try out.gpa.create(Entry);
        errdefer out.gpa.destroy(entry);
        entry.* = .{ .context = try dspark.Context.init(out.d, out.context) };
        errdefer entry.context.deinit();
        try out.entries.put(out.gpa, stream, entry);
    }

    fn releaseFn(ptr: *anyopaque, stream: *Stream) void {
        const out = cast(ptr);
        if (out.entries.fetchRemove(stream)) |owned| {
            out.s.synchronize() catch {};
            owned.value.context.deinit();
            out.gpa.destroy(owned.value);
        }
    }

    fn absorbFn(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const out = cast(ptr);
        for (items) |item| {
            const entry = out.entries.get(item.stream) orelse return error.DraftStreamUnknown;
            const f = item.features;
            if (item.rows != null or f.ready != .none or f.space != .device or f.dtype != .bf16 or
                f.row_bytes != dspark.taps.len * dspark.hidden * 2 or f.rows == 0 or f.rows != item.follow.len)
                return error.DraftFeaturesUnsupported;
            try out.head.absorb(out.s, &entry.context, f.buffer + f.offset, item.start, f.rows);
            entry.count = 0;
        }
    }

    fn holdFn(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        const out = cast(ptr);
        if (items.len == 0 or items.len > engine.max_slots) return error.DSparkHoldGeometry;
        var contexts: [engine.max_slots]*dspark.Context = undefined;
        var pending: [engine.max_slots]u32 = undefined;
        var max_depth: usize = 0;
        for (items, 0..) |item, i| {
            const entry = out.entries.get(item.stream) orelse return error.DraftStreamUnknown;
            if (item.lanes != null or item.depth == 0 or item.depth > dspark.depth or
                item.position != entry.context.end + 1) return error.DSparkHoldGeometry;
            contexts[i] = &entry.context;
            pending[i] = item.pending;
            max_depth = @max(max_depth, item.depth);
        }
        var picks: [engine.max_slots * dspark.rows]u32 = undefined;
        try out.head.draftMany(out.s, contexts[0..items.len], pending[0..items.len], max_depth, picks[0 .. items.len * dspark.rows]);
        for (items, 0..) |item, i| {
            const entry = out.entries.get(item.stream).?;
            @memcpy(entry.held[0..item.depth], picks[i * dspark.rows + 1 ..][0..item.depth]);
            entry.count = item.depth;
        }
    }

    fn heldFn(ptr: *anyopaque, streams: []const *Stream, destinations: []const []u32) anyerror!void {
        const out = cast(ptr);
        if (streams.len != destinations.len) return error.DraftHeldCount;
        for (streams, destinations) |stream, dest| {
            const entry = out.entries.get(stream) orelse return error.DraftStreamUnknown;
            if (dest.len > entry.count) return error.DraftHeldFewer;
            @memcpy(dest, entry.held[0..dest.len]);
        }
    }
};
