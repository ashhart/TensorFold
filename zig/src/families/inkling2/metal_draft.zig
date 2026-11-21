//! Inkling 2 Mini's bundled DSpark draft head as a `lanes.drafter.Drafter` on Metal: logical-layer taps in, up to 7 drafts out.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const config = @import("config.zig");
const core_mod = @import("metal_draft_core.zig");
const gm = @import("metal_vision_gemm.zig");

const dr = lanes.drafter;
const Core = core_mod.Core;
const Lane = core_mod.Lane;
const Ref = gm.Ref;

pub const Target = core_mod.Target;

pub const Options = struct {
    /// Streams the adapter keeps lanes for at once (`open` refuses more).
    streams: u32 = 1,
    /// Positions a stream's full-attention layer holds: the longest context (prompt + generated) a stream reaches.
    capacity: u32,
    /// Drafts a round at most; 0: the block's own (7).
    depth: u32 = 0,
};

/// Device bytes one opened stream's caches take at `capacity` positions: what host admission counts before allocating.
pub fn laneBytes(cfg: *const config.Config, capacity: u32) u64 {
    const m = cfg.mtp orelse return 0;
    return Core.laneBytes(m, cfg.text.hidden, capacity);
}

pub const Adapter = struct {
    gpa: std.mem.Allocator,
    core: *Core,
    opts: Options,
    depth: u32,
    step_ms: f64 = 0,
    lanes: std.AutoHashMapUnmanaged(*lanes.Stream, *Lane) = .empty,
    follow: ?mtl.ResidencySet.Follow = null,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, device: mtl.Device, queue: mtl.Queue, cfg: *const config.Config, target: Target, opts: Options) !*Adapter {
        _ = io;
        if (opts.streams == 0) return error.UnsupportedDraftHead;
        const m = cfg.mtp orelse return error.NoDraftHead;
        const a = try gpa.create(Adapter);
        errdefer gpa.destroy(a);
        a.* = .{ .gpa = gpa, .core = try Core.init(gpa, dir, device, queue, cfg, target, opts.capacity), .opts = opts, .depth = @min(m.block - 1, if (opts.depth == 0) m.block - 1 else opts.depth) };
        errdefer a.core.deinit();
        try a.warm();
        return a;
    }

    pub fn deinit(a: *Adapter) void {
        a.setResident(null);
        var it = a.lanes.valueIterator();
        while (it.next()) |l| a.core.dropLane(l.*);
        a.lanes.deinit(a.gpa);
        a.core.deinit();
        a.gpa.destroy(a);
    }

    /// Device bytes held besides the streams' lanes (weights, tables, scratch); each open stream adds `laneBytes()`.
    pub fn bytes(a: *const Adapter) u64 {
        return a.core.bytes();
    }

    /// One stream's caches (also `Facts.stream_bytes`).
    pub fn laneBytes(a: *const Adapter) u64 {
        return Core.laneBytes(a.core.m, a.core.hidden, a.opts.capacity);
    }

    pub fn drafter(a: *Adapter) dr.Drafter {
        return .{ .kind = "DSpark", .ptr = a, .vtable = &.{ .taps = taps, .facts = facts, .open = open, .absorb = absorb, .hold = hold, .held = held, .release = release } };
    }

    /// Makes the residency set follow the adapter's and its streams' buffers (null: stop following, leaving them).
    pub fn setResident(a: *Adapter, hook: ?mtl.ResidencySet.Follow) void {
        const k = a.core;
        if (a.follow) |f| {
            for (k.s) |b| f.leave(f.ctx, b);
            for ([_]mtl.Buffer{ k.alpha, k.inv, k.fold_w, k.fold_b }) |b| f.leave(f.ctx, b);
            for (k.w.ckpt.shards.items) |s| f.leave(f.ctx, s.buffer);
            if (k.staging) |b| f.leave(f.ctx, b);
            var it = a.lanes.valueIterator();
            while (it.next()) |l| a.track(f, l.*, false);
        }
        a.follow = hook;
        if (hook) |f| {
            for (k.s) |b| f.join(f.ctx, b);
            for ([_]mtl.Buffer{ k.alpha, k.inv, k.fold_w, k.fold_b }) |b| f.join(f.ctx, b);
            for (k.w.ckpt.shards.items) |s| f.join(f.ctx, s.buffer);
            if (k.staging) |b| f.join(f.ctx, b);
            var it = a.lanes.valueIterator();
            while (it.next()) |l| a.track(f, l.*, true);
        }
    }

    fn track(a: *const Adapter, f: mtl.ResidencySet.Follow, l: *const Lane, join: bool) void {
        const call = if (join) f.join else f.leave;
        for (0..a.core.m.layers) |i| {
            call(f.ctx, l.k[i]);
            call(f.ctx, l.v[i]);
        }
        call(f.ctx, l.f);
        call(f.ctx, l.seq);
    }

    fn self(ptr: *anyopaque) *Adapter {
        return @ptrCast(@alignCast(ptr));
    }

    fn taps(ptr: *anyopaque) []const u32 {
        const m = &self(ptr).core.m;
        return m.taps[0..m.taps_n];
    }

    fn facts(ptr: *anyopaque) dr.Facts {
        const a = self(ptr);
        return .{ .fixed_block = true, .depth = a.depth, .step_ms = a.step_ms, .plain_guard = true, .stream_bytes = @intCast(a.laneBytes()) };
    }

    fn open(ptr: *anyopaque, s: *lanes.Stream) !void {
        const a = self(ptr);
        if (a.lanes.contains(s)) return error.DraftStreamOpen;
        if (a.lanes.count() >= a.opts.streams) return error.DraftStreamsFull;
        try a.lanes.ensureUnusedCapacity(a.gpa, 1);
        const l = try a.core.newLane(a.opts.capacity);
        if (a.follow) |f| a.track(f, l, true);
        a.lanes.putAssumeCapacity(s, l);
    }

    fn release(ptr: *anyopaque, s: *lanes.Stream) void {
        const a = self(ptr);
        const l = (a.lanes.fetchRemove(s) orelse return).value;
        if (a.follow) |f| a.track(f, l, false);
        a.core.dropLane(l);
    }

    const Pass = struct {
        a: *Adapter,
        cb: ?mtl.CommandBuffer = null,
        e: mtl.ComputeEncoder = undefined,

        fn encoder(p: *Pass) mtl.ComputeEncoder {
            if (p.cb == null) {
                p.cb = p.a.core.queue.commandBuffer();
                p.e = p.cb.?.compute(.serial);
            }
            return p.e;
        }

        fn flush(p: *Pass) !void {
            const cb = p.cb orelse return;
            p.cb = null;
            p.e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| return Core.failed(msg);
        }
    };

    fn absorb(ptr: *anyopaque, items: []const dr.Absorb) !void {
        const a = self(ptr);
        const k = a.core;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        var pass: Pass = .{ .a = a };
        errdefer pass.flush() catch {};
        const row_bytes: u32 = k.m.taps_n * k.hidden * 2;
        for (items) |item| {
            const l = a.lanes.get(item.stream) orelse return error.DraftStreamUnknown;
            const f = item.features;
            if (f.dtype != .bf16 or f.ready != .none or f.row_bytes != row_bytes or item.rows != null or f.rows == 0) return error.BadDraftFeatures;
            if (item.start + f.rows > k.capacity) return error.DraftContextFull;
            if (item.start > l.end or l.end == 0) {
                // rows before `start` never reached this head (a restored prefix): the context begins at `start`
                l.from = item.start;
                l.lfrom = item.start;
            } else if (item.start < l.end and l.end - item.start > core_mod.Core.ring(k.m) - k.m.window) {
                l.lfrom = @max(l.lfrom, l.end - Core.ring(k.m)); // a rollback past the rings' slack clobbered the oldest rows
            }
            if (item.start < l.from) l.from = item.start;
            l.end = item.start;
            var done: u32 = 0;
            while (done < f.rows) {
                const n = @min(f.rows - done, core_mod.absorb_rows);
                const base = f.offset + @as(u64, done) * row_bytes;
                var src: Ref = undefined;
                switch (f.space) {
                    .device => src = .{ .buf = .{ .id = @ptrFromInt(f.buffer) }, .off = @intCast(base) },
                    .host => {
                        try pass.flush(); // the staging buffer is read by the passes before it
                        const stage = try a.staging();
                        const source: [*]const u8 = @ptrFromInt(f.buffer + base);
                        @memcpy(stage.slice(u8, @as(usize, n) * row_bytes), source[0 .. @as(usize, n) * row_bytes]);
                        src = .{ .buf = stage };
                    },
                }
                try k.absorb(pass.encoder(), l, src, l.end, n);
                l.end += n;
                done += n;
            }
            l.held_len = 0;
        }
        try pass.flush();
    }

    fn staging(a: *Adapter) !mtl.Buffer {
        const k = a.core;
        if (k.staging == null) {
            k.staging = try k.device.buffer(@as(usize, core_mod.absorb_rows) * k.m.taps_n * k.hidden * 2, mtl.ResourceOptions.shared);
            if (a.follow) |f| f.join(f.ctx, k.staging.?);
        }
        return k.staging.?;
    }

    fn hold(ptr: *anyopaque, items: []const dr.Hold) !void {
        const a = self(ptr);
        const k = a.core;
        for (items) |item| {
            const l = a.lanes.get(item.stream) orelse return error.DraftStreamUnknown;
            if (item.depth == 0 or item.depth > a.depth or item.lanes != null) return error.DraftDepth;
            if (item.position == 0 or l.end != item.position - 1 or item.pending >= k.vocab) return error.DraftContextGap;
        }
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        var pass: Pass = .{ .a = a };
        errdefer pass.flush() catch {};
        for (items) |item| {
            const l = a.lanes.get(item.stream).?;
            l.held_len = 0;
            l.seq.slice(u32, k.m.block)[0] = item.pending;
            try k.block(pass.encoder(), l, l.end);
        }
        try pass.flush();
        for (items) |item| a.lanes.get(item.stream).?.held_len = item.depth;
    }

    /// One block at the lane's anchor with `pending` as its anchor token; seq[1 ..] its drafts.
    fn run(a: *Adapter, l: *const Lane, pending: u32) !void {
        const k = a.core;
        l.seq.slice(u32, k.m.block)[0] = pending;
        var pass: Pass = .{ .a = a };
        errdefer pass.flush() catch {};
        try k.block(pass.encoder(), l, l.end);
        try pass.flush();
    }

    fn held(ptr: *anyopaque, streams: []const *lanes.Stream, out: []const []u32) !void {
        const a = self(ptr);
        if (streams.len != out.len) return error.BadDraftStreams;
        for (streams, out) |s, ids| {
            const l = a.lanes.get(s) orelse return error.DraftStreamUnknown;
            if (ids.len > l.held_len) return error.ShortDraftChain;
            @memcpy(ids, l.seq.slice(u32, a.core.m.block)[1 .. 1 + ids.len]);
        }
    }

    /// Times a block on a throwaway lane (no context) for `Facts.step_ms`, per draft.
    fn warm(a: *Adapter) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const l = try a.core.newLane(core_mod.absorb_rows);
        defer a.core.dropLane(l);
        for (0..3) |i| {
            const t0 = seconds();
            try a.run(l, 2);
            if (i == 2) a.step_ms = (seconds() - t0) * 1000 / @as(f64, @floatFromInt(a.core.m.block - 1));
        }
    }

    fn seconds() f64 {
        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &ts);
        return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) * 1e-9;
    }
};
