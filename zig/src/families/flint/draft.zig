//! Flint's published DFlash2 head uses the shared graph, Metal runtime and lane verifier with per-stream contexts.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const q = @import("../qwen27/qwen27.zig");
const rt = q.dflash.runtime_model;
const cfg = q.dflash.config;
const dr = lanes.drafter;
const Model = @import("model.zig").Model;
const Lane = struct { context: *rt.Context, held: [7]u32 = undefined, count: usize = 0 };

pub const Drafter = struct {
    a: std.mem.Allocator,
    model: *rt.Model,
    glue: @import("../qwen27/glue_gpu.zig").Kernels,
    kernels: @import("../qwen27/quant_gpu.zig").Kernels,
    dims: mtl.Buffer,
    lanes: std.AutoHashMapUnmanaged(*lanes.Stream, Lane) = .empty,
    follow: ?mtl.ResidencySet.Follow = null,
    step_ms: f64 = 0,

    pub fn init(a: std.mem.Allocator, io: std.Io, target: *Model, dir: []const u8, bits: u8) !*Drafter {
        if (!target.device.tensorUnits()) return error.FlintDFlash2RequiresM5;
        if (bits != 0 and bits != 4) return error.UnsupportedDraftBits;
        const d = try a.create(Drafter);
        errdefer a.destroy(d);
        d.* = .{ .a = a, .model = undefined, .glue = try @import("../qwen27/glue_gpu.zig").Kernels.init(target.device), .kernels = .{ .allocator = a, .device = target.device, .raw_tiles = false }, .dims = undefined };
        errdefer d.glue.deinit();
        errdefer d.kernels.deinit();
        d.dims = try target.device.buffer(4096, mtl.ResourceOptions.shared);
        errdefer d.dims.deinit();
        const w = target.weights.head;
        const bindings: q.dflash.runtime_backend.Target = .{ .device = target.device, .queue = target.queue, .checkpoint = &target.checkpoint, .config = .{ .hidden = target.config.hidden, .vocab = target.config.vocab, .layers = target.config.layers }, .weights = .{ .embed = .{ .weight = try target.checkpoint.get("model.embed_tokens.weight"), .scales = try target.checkpoint.get("model.embed_tokens.scales"), .biases = try target.checkpoint.get("model.embed_tokens.biases"), .bits = target.weights.embed.bits, .group = @intCast(target.weights.embed.group) }, .head = .{ .words = .{ .buffer = w.w, .offset = w.w_off }, .pairs = .{ .buffer = w.scales, .offset = w.s_off }, .n = @intCast(w.n), .k = @intCast(w.k), .tile = 32, .slices = 1, .raw = w, .bits = w.bits, .group = @intCast(w.group) } }, .glue = &d.glue, .kernels = &d.kernels, .dims = .{ .buffer = d.dims } };
        d.model = try rt.Model.loadBound(a, io, bindings, dir, if (bits == 4) .prepared_q4_reference else .bf16_reference);
        errdefer d.model.deinit();
        if (!std.mem.eql(u32, d.model.graph.config.tapIds(), cfg.flint_tap_ids[0..5])) return error.UnsupportedFlintTaps;
        try d.model.setTreeBlock(8);
        try d.warm();
        return d;
    }

    pub fn deinit(d: *Drafter) void {
        d.setResident(null);
        var it = d.lanes.valueIterator();
        while (it.next()) |l| l.context.deinit(d.a);
        d.lanes.deinit(d.a);
        d.model.deinit();
        d.dims.deinit();
        d.kernels.deinit();
        d.glue.deinit();
        d.a.destroy(d);
    }

    pub fn buffers(d: *Drafter, list: *std.ArrayList(mtl.Buffer)) !void {
        var it = d.model.checkpoint.tensors.valueIterator();
        while (it.next()) |t| try list.append(d.a, t.buffer);
        try list.appendSlice(d.a, &.{ d.dims, d.model.pending });
        try @import("../qwen27/residency.zig").appendDraft(&d.model.backend, d.a, list);
    }

    pub fn setResident(d: *Drafter, hook: ?mtl.ResidencySet.Follow) void {
        if (d.follow) |old| old.leave(old.ctx, d.model.pending);
        if (hook) |h| h.join(h.ctx, d.model.pending);
        d.follow = hook;
        d.model.backend.setResident(hook);
        d.kernels.setResident(hook);
        var it = d.lanes.valueIterator();
        while (it.next()) |l| l.context.setResident(hook);
    }

    pub fn fixedBytes(d: *Drafter) !u64 {
        var list: std.ArrayList(mtl.Buffer) = .empty;
        defer list.deinit(d.a);
        try d.buffers(&list);
        var bytes: u64 = 0;
        for (list.items, 0..) |b, i| {
            var found = false;
            for (list.items[0..i]) |old| if (old.id == b.id) {
                found = true;
            };
            if (!found) bytes += b.length();
        }
        return bytes;
    }

    pub fn streamBytes(d: *const Drafter) usize {
        const c = d.model.graph.config;
        return @as(usize, c.window) * c.tapWidth() * 2 + @as(usize, c.layers) * (c.window + 16) * c.kvWidth() * 4;
    }

    pub fn drafter(d: *Drafter) dr.Drafter {
        return .{ .kind = "DFlash2", .ptr = d, .vtable = &.{ .taps = taps, .facts = facts, .open = open, .absorb = absorb, .hold = hold, .held = held, .release = release } };
    }
    fn self(ptr: *anyopaque) *Drafter {
        return @ptrCast(@alignCast(ptr));
    }
    fn taps(_: *anyopaque) []const u32 {
        return cfg.flint_tap_ids[0..5];
    }
    fn facts(ptr: *anyopaque) dr.Facts {
        const d = self(ptr);
        return .{ .depth = 7, .step_ms = d.step_ms, .batched = true, .stream_bytes = d.streamBytes() };
    }
    fn open(ptr: *anyopaque, s: *lanes.Stream) !void {
        const d = self(ptr);
        if (d.lanes.contains(s)) return error.DraftStreamOpen;
        const context = try rt.Context.init(d.a, d.model, true);
        errdefer context.deinit(d.a);
        context.setResident(d.follow);
        try d.lanes.put(d.a, s, .{ .context = context });
    }
    fn absorb(ptr: *anyopaque, items: []const dr.Absorb) !void {
        const d = self(ptr);
        const c = d.model.graph.config;
        for (items) |item| {
            const l = d.lanes.getPtr(item.stream) orelse return error.DraftStreamUnknown;
            const x = l.context;
            const f = item.features;
            if (f.space != .host or f.dtype != .bf16 or f.ready != .none or f.row_bytes != c.tapWidth() * 2 or item.rows != null) return error.BadFlintFeatures;
            if (x.committed_end == 0 and item.start > 0) {
                x.committed_end = item.start;
                x.context = .{ .cache = 1, .begin = item.start, .end = item.start };
                x.session.context = x.context;
            }
            if (x.committed_end != item.start) return error.DraftContextGap;
            const bytes: [*]const u8 = @ptrFromInt(f.buffer + f.offset);
            try x.ring().absorbBytes(bytes[0 .. @as(usize, f.rows) * f.row_bytes], f.rows);
            l.count = 0;
        }
    }
    fn hold(ptr: *anyopaque, items: []const dr.Hold) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const d = self(ptr);
        var batch: [8]rt.Model.Proposal = undefined;
        var trees: [8]q.dflash.selector.Tree = undefined;
        var first: usize = 0;
        while (first < items.len) {
            const n = @min(items.len - first, batch.len);
            for (items[first..][0..n], batch[0..n]) |item, *p| {
                const l = d.lanes.getPtr(item.stream) orelse return error.DraftStreamUnknown;
                if (item.depth == 0 or item.depth > 7 or item.position != l.context.committed_end + 1 or item.lanes != null) return error.BadDraftPosition;
                l.count = 0;
                p.* = .{ .context = l.context, .pending = item.pending, .nodes = item.depth, .chain = true };
            }
            if (n == 1) {
                try d.model.swap(batch[0].context.?);
                const result = d.model.proposeOptions(batch[0].pending, rt.Model.selection(batch[0]));
                try d.model.swap(batch[0].context.?);
                trees[0] = try result;
            } else try d.model.proposeBatch(batch[0..n], trees[0..n]);
            defer for (trees[0..n]) |*tree| tree.deinit();
            for (items[first..][0..n], trees[0..n]) |item, tree| {
                const l = d.lanes.getPtr(item.stream).?;
                if (tree.tokens.len != item.depth) return error.ShortDraftChain;
                for (tree.parents, 0..) |p, i| if (p != @as(i32, @intCast(i)) - 1) return error.NotDraftChain;
                @memcpy(l.held[0..item.depth], tree.tokens);
                l.count = item.depth;
            }
            first += n;
        }
    }
    fn held(ptr: *anyopaque, streams: []const *lanes.Stream, out: []const []u32) !void {
        const d = self(ptr);
        if (streams.len != out.len) return error.BadDraftStreams;
        for (streams, out) |s, ids| {
            const l = d.lanes.getPtr(s) orelse return error.DraftStreamUnknown;
            if (ids.len > l.count) return error.ShortDraftChain;
            @memcpy(ids, l.held[0..ids.len]);
        }
    }
    fn release(ptr: *anyopaque, s: *lanes.Stream) void {
        const d = self(ptr);
        if (d.lanes.fetchRemove(s)) |l| l.value.context.deinit(d.a);
    }
    fn warm(d: *Drafter) !void {
        for (0..3) |i| {
            try d.model.reset();
            @memset(d.model.pending.contents()[0 .. d.model.graph.config.tapWidth() * 2], 0);
            d.model.committed_end = 1;
            const t0 = @import("forward.zig").seconds();
            var tree = try d.model.proposeOptions(1000, .{ .nodes = 7, .children = 1 });
            tree.deinit();
            if (i == 2) d.step_ms = (@import("forward.zig").seconds() - t0) * 1000 / 7;
        }
        try d.model.reset();
    }
};
