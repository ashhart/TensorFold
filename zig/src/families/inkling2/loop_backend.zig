//! Inkling 2 Mini lane execution and snapshot adapter.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("loop_engine.zig");
const ops = @import("ops.zig");

const Model = engine.Model;
const Kernels = engine.Kernels;
const prompt_rows = engine.prompt_rows;
const window_rows = engine.window_rows;
const max_streams = engine.max_segments;
const max_batch_rows = engine.max_batch_rows;
const token_ring = 1024;

const Lane = struct {
    slot: u32 = 0, // engine cache slot 0..max_streams-1
    len: usize = 0,
    fill: ?usize = null,
    base: usize = 0,
    rows: std.ArrayList(u32) = .empty,
    last: u64, // per-lane fp32 logits row
    rule: u64, // per-lane sampling rule
    last_b: cuda.DeviceBuffer,
    rule_b: cuda.DeviceBuffer,
    tap_buf: u64 = 0,
    tap_start: u64 = 0,
    tap_rows: usize = 0,
    sampled: bool = false,
};

pub const Backend = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    m: *Model,
    k: *const Kernels,
    s: cuda.Stream,
    lanes: std.AutoHashMapUnmanaged(*lanes.Stream, Lane) = .empty,
    slots: [max_streams]bool = @splat(false), // engine cache slots in use
    tokens: cuda.DeviceBuffer,
    logits16: cuda.DeviceBuffer,
    meta: cuda.DeviceBuffer,
    next: u64 = 0,
    prepared: bool = false,
    taps_buf: ?cuda.DeviceBuffer = null,
    verify_taps: [max_streams]?cuda.DeviceBuffer = @splat(null),
    batch_taps: ?cuda.DeviceBuffer = null, // a batched verify's taps, all its rows, copied out to each lane's verify_taps
    sink: ?be.FeatureSink = null,

    const be = lanes.backend;

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, m: *Model, k: *const Kernels, s: cuda.Stream) !Backend {
        const v = m.w.c.text.unpadded_vocab;
        // the ring, then one verify round's draws: every window's rows, one after another
        var tokens = try cuda.DeviceBuffer.alloc(d, (token_ring + max_streams * window_rows) * 4);
        errdefer tokens.free();
        var logits16 = try cuda.DeviceBuffer.alloc(d, window_rows * v * 2);
        errdefer logits16.free();
        var meta = try cuda.DeviceBuffer.alloc(d, 16);
        errdefer meta.free();
        return .{ .gpa = gpa, .d = d, .m = m, .k = k, .s = s, .tokens = tokens, .logits16 = logits16, .meta = meta };
    }

    pub fn deinit(b: *Backend) void {
        for (&b.verify_taps) |*vt| if (vt.*) |*buf| buf.free();
        if (b.batch_taps) |*bt| bt.free();
        if (b.taps_buf) |*tb| tb.free();
        var it = b.lanes.valueIterator();
        while (it.next()) |l| {
            l.rows.deinit(b.gpa);
            l.last_b.free();
            l.rule_b.free();
        }
        b.lanes.deinit(b.gpa);
        b.meta.free();
        b.logits16.free();
        b.tokens.free();
        b.* = undefined;
    }

    pub fn backend(b: *Backend) be.Backend {
        return .{ .ptr = b, .vtable = &vtable };
    }

    pub fn facts(b: *const Backend) lanes.Model {
        return .{ .exact_width = window_rows, .gpu_tokens = false, .mtp = false, .speculate = false, .hidden_rows = true, .batch_rows = prompt_rows, .max_streams = @intCast(@min(b.m.segments, max_streams)), .streams_exact = true };
    }

    pub fn admits(b: *const Backend, prompt: usize, max_new: usize) bool {
        if (prompt > b.m.context or max_new > b.m.context - prompt) return false;
        var used: usize = 0;
        for (b.slots[0..@min(b.m.segments, max_streams)]) |held| used += @intFromBool(held);
        return used < @min(b.m.segments, max_streams);
    }

    fn self(ptr: *anyopaque) *Backend {
        return @ptrCast(@alignCast(ptr));
    }

    fn upload(b: *Backend, dst: u64, bytes: []const u8) !void {
        try cuda.copy_source.enqueue(b.d, dst, bytes, b.s);
    }

    fn laneOf(b: *Backend, s: *const lanes.Stream) !*Lane {
        const l = b.lanes.getPtr(@constCast(s)) orelse return error.UnknownStream;
        b.m.slot = l.slot;
        return l;
    }

    fn resumeSaved(b: *Backend, s: *lanes.Stream) usize {
        const saved = s.reuse.saved orelse return 0;
        const k: *const Model.Saved = @ptrCast(@alignCast(saved));
        if (k.at != s.reuse.at or k.at == 0 or k.at >= s.prompt().len) {
            s.reuse_failed = true;
            return 0;
        }
        b.m.restore(b.s, k) catch {
            s.reuse_failed = true;
            return 0;
        };
        s.cached = k.at;
        return k.at;
    }

    fn opening(b: *Backend, s: *lanes.Stream) !*Lane {
        const prompt = s.prompt();
        if (prompt.len == 0) return error.EmptyPrompt;
        if (prompt.len > b.m.context or s.max_new > b.m.context - prompt.len) return error.PastContext;
        if (b.lanes.contains(s)) return b.laneOf(s);
        const cap = @min(b.m.segments, max_streams);
        var slot: u32 = max_streams;
        for (b.slots[0..cap], 0..) |held, i| if (!held) {
            slot = @intCast(i);
            break;
        };
        if (@as(usize, slot) >= cap) return error.NoFreeSlot;
        const v = b.m.w.c.text.unpadded_vocab;
        var last_b = try cuda.DeviceBuffer.alloc(b.d, v * 4);
        errdefer last_b.free();
        var rule_b = try cuda.DeviceBuffer.alloc(b.d, @sizeOf(cuda.sample.Rule));
        errdefer rule_b.free();
        b.slots[slot] = true;
        errdefer b.slots[slot] = false;
        b.m.slot = slot;
        try b.m.resetStream(b.s, slot);
        var ln = Lane{ .slot = slot, .last = last_b.ptr, .rule = rule_b.ptr, .last_b = last_b, .rule_b = rule_b };
        ln.sampled = false;
        if (cuda.sample.ruleFor(s.sampling)) |r| {
            try b.upload(ln.rule, std.mem.asBytes(&r));
            ln.sampled = true;
        }
        ln.len = 0;
        const at = b.resumeSaved(s);
        ln.len = at;
        ln.fill = at;
        s.fill_at.store(@intCast(at), .monotonic);
        try b.lanes.put(b.gpa, s, ln);
        return b.laneOf(s);
    }

    fn chunk(b: *Backend, s: *lanes.Stream, l: *Lane) !bool {
        const prompt = s.prompt();
        const at = l.fill.?;
        if (s.isCancelled()) return error.Cancelled;
        var end = @min(prompt.len, (at / prompt_rows + 1) * prompt_rows);
        for (s.reuse.marks) |mark| {
            if (mark > at and mark < end) end = mark;
        }
        const done = end == prompt.len;
        const taps_ptr: u64 = if (b.prepared) b.taps_buf.?.ptr else 0;
        try b.m.forward(b.k, b.s, prompt[at..end], at, end - at - 1, @intFromBool(done), false, taps_ptr);
        l.tap_buf = taps_ptr;
        l.tap_start = at;
        l.tap_rows = end - at;
        if (b.sink) |sink| if (b.prepared) {
            const emit_rows = (end - at) - @intFromBool(done);
            if (emit_rows > 0) {
                const row_bytes = 4 * b.m.w.c.text.hidden * 2;
                try sink.take(sink.ptr, s, .{
                    .buffer = taps_ptr,
                    .offset = 0,
                    .rows = @intCast(emit_rows),
                    .row_bytes = @intCast(row_bytes),
                    .space = .device,
                    .dtype = .bf16,
                }, at);
            }
        };
        if (done) try b.d.check(b.d.api.cuMemcpyDtoDAsync_v2(l.last, b.m.s.logits, b.m.w.c.text.unpadded_vocab * 4, b.s.handle), "cuMemcpyDtoDAsync");
        l.len = end;
        l.fill = if (done) null else end;
        s.fill_at.store(@intCast(end), .monotonic);
        if (s.reuse.hook) |hook| if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(end)) != null)
            hook.at(hook.ptr, s, @intCast(end));
        return done;
    }

    // Global draw-handle ring shared by all streams: at most max_streams in-flight queued rows land here
    // between reads, far below the 1024-handle wrap horizon (verify rounds draw past the ring, at token_ring).
    fn handle(b: *Backend) usize {
        const h = b.next;
        b.next += 1;
        return @intCast(h % token_ring);
    }

    fn drawRows(b: *Backend, l: *const Lane, logits: u64, rows: usize, position0: u64, at: usize) !void {
        const v = b.m.w.c.text.unpadded_vocab;
        if (!l.sampled) return ops.top1(&b.k.glue, b.s, logits, rows, v, v, 0, b.m.s.t + at * 4, b.tokens.ptr + at * 4);
        try ops.tobf(&b.k.glue, b.s, logits, rows, v, v, b.logits16.ptr, v, 0, v);
        const metadata = [4]i32{ @intCast(position0 - 1), 0, 0, 0 };
        try b.upload(b.meta.ptr, std.mem.asBytes(&metadata));
        try b.k.draws.draw(b.s, b.logits16.ptr, v, l.rule, b.meta.ptr, 0, b.tokens.ptr + at * 4, rows, null, null);
    }

    fn read(ptr: *anyopaque, h: u64) anyerror!u32 {
        const b = self(ptr);
        if (h >= b.next or b.next - h > token_ring) return error.NoSuchToken;
        var out: [1]u32 = undefined;
        try b.s.synchronize();
        try b.tokens.download(@intCast((h % token_ring) * 4), std.mem.sliceAsBytes(&out));
        return out[0];
    }

    fn validateTaps(taps: []const u32) !void {
        if (!std.mem.eql(u32, taps, &.{ 5, 23, 41, 48 })) return error.UnsupportedInkling2Taps;
    }

    fn prepareFeatures(ptr: *anyopaque, taps: []const u32) anyerror!void {
        const b = self(ptr);
        try validateTaps(taps);
        if (b.prepared) return;
        const bytes = prompt_rows * 4 * b.m.w.c.text.hidden * 2;
        b.taps_buf = try cuda.DeviceBuffer.alloc(b.d, bytes);
        const vbytes = window_rows * 4 * b.m.w.c.text.hidden * 2;
        for (0..@min(b.m.segments, max_streams)) |slot| {
            b.verify_taps[slot] = try cuda.DeviceBuffer.alloc(b.d, vbytes);
        }
        if (b.m.segments > 1) b.batch_taps = try cuda.DeviceBuffer.alloc(b.d, max_batch_rows * 4 * b.m.w.c.text.hidden * 2);
        b.prepared = true;
    }

    fn featuresFn(ptr: *anyopaque, s: *lanes.Stream, taps: []const u32, start: u64, count: u32) anyerror!be.Features {
        const b = self(ptr);
        try validateTaps(taps);
        if (!b.prepared) return error.UnpreparedInkling2Taps;
        const ln = try b.laneOf(s);
        if (ln.tap_buf == 0 or start < ln.tap_start or start + count > ln.tap_start + ln.tap_rows) return error.Inkling2TapsNotHeld;
        const row_bytes = 4 * b.m.w.c.text.hidden * 2;
        return .{
            .buffer = ln.tap_buf,
            .offset = (start - ln.tap_start) * row_bytes,
            .rows = count,
            .row_bytes = @intCast(row_bytes),
            .space = .device,
            .dtype = .bf16,
        };
    }

    fn prefillFeatures(ptr: *anyopaque, s: *lanes.Stream, taps: []const u32, sink: be.FeatureSink) anyerror!void {
        const b = self(ptr);
        try validateTaps(taps);
        if (!b.prepared) return error.UnpreparedInkling2Taps;
        b.sink = sink;
        defer b.sink = null;
        return prefill(ptr, s);
    }

    const vtable: be.Backend.VTable = .{
        .prefill = prefill,
        .prefill_step = prefillStep,
        .prepare_features = prepareFeatures,
        .features = featuresFn,
        .prefill_features = prefillFeatures,
        .first = first,
        .first_masked = firstMasked,
        .queue = queue,
        .read = read,
        .verify = verify,
        .keep = keep,
        .draft = draft,
        .release = release,
        .reports_fill = true,
    };

    fn prefill(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const b = self(ptr);
        const l = try b.opening(s);
        errdefer release(ptr, s);
        while (!try b.chunk(s, l)) {}
    }

    fn prefillStep(ptr: *anyopaque, s: *lanes.Stream, _: u32) anyerror!bool {
        const b = self(ptr);
        const going = if (b.lanes.getPtr(s)) |l| l.fill != null else false;
        const l = if (going) try b.laneOf(s) else try b.opening(s);
        errdefer release(ptr, s);
        return b.chunk(s, l);
    }

    fn first(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const b = self(ptr);
        const l = try b.laneOf(s);
        if (position != l.len or l.fill != null) return error.PositionMismatch;
        const h = b.next;
        try b.drawRows(l, l.last, 1, position, b.handle());
        return h;
    }

    fn firstMasked(_: *anyopaque, _: *lanes.Stream, _: u64, _: []const u32) anyerror!u64 {
        return error.StructuresUnsupported;
    }

    fn feed(b: *Backend, value: be.Feed) !u32 {
        return switch (value) {
            .value => |v| v,
            .handle => |h| read(b, h),
        };
    }

    fn queue(ptr: *anyopaque, s: *lanes.Stream, value: be.Feed, position: u64) anyerror!u64 {
        const b = self(ptr);
        const l = try b.laneOf(s);
        const token = try b.feed(value);
        if (position != l.len + 1) return error.PositionMismatch;
        const taps_ptr: u64 = if (b.prepared) b.verify_taps[l.slot].?.ptr else 0;
        try b.m.forward(b.k, b.s, &.{token}, l.len, 0, 1, false, taps_ptr);
        l.tap_buf = taps_ptr;
        l.tap_start = l.len;
        l.tap_rows = 1;
        l.len += 1;
        const h = b.next;
        try b.drawRows(l, b.m.s.logits, 1, position, b.handle());
        return h;
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const b = self(ptr);
        if (windows.len == 0 or windows.len > max_streams or out.len != windows.len) return error.BadVerifyWindows;
        // Validate every window first so a late failure cannot leave lanes half-advanced.
        for (windows, 0..) |w, i| {
            if (w.held != 0) return error.NoHeldDrafts;
            if (w.parents != null) return error.ChainsOnly;
            if (w.masks.len != 0) return error.StructuresUnsupported;
            const l = try b.laneOf(w.stream);
            const n = w.rows();
            if (n == 0 or n > window_rows) return error.WindowTooWide;
            for (w.positions, 0..) |position, j| if (position != l.len + j + 1) return error.PositionMismatch;
            for (windows[0..i]) |prev| if (prev.stream == w.stream) return error.BadVerifyWindows;
        }
        for (windows) |w| {
            const l = b.lanes.getPtr(w.stream).?;
            l.base = l.len;
            l.rows.clearRetainingCapacity();
            try l.rows.append(b.gpa, w.pending);
            try l.rows.appendSlice(b.gpa, w.tokens);
        }
        // Windows in order, as few forwards as max_batch_rows allows (16 eight-row windows take one): each forward
        // runs its windows' rows on their own slots (the weights read once for them), each window's draws land at
        // token_ring + its first row of the round, then one sync and one download for the round. A lone window is
        // forward() itself, graphed into its lane's own taps.
        const lone = windows.len == 1;
        const row_bytes = 4 * b.m.w.c.text.hidden * 2;
        const v = b.m.w.c.text.unpadded_vocab;
        var rows: usize = 0;
        var w0: usize = 0;
        while (w0 < windows.len) {
            var segs: [max_streams]Model.BatchedSeg = undefined;
            var w1 = w0;
            var group_rows: usize = 0;
            while (w1 < windows.len and group_rows + windows[w1].rows() <= max_batch_rows) : (w1 += 1) {
                const l = b.lanes.getPtr(windows[w1].stream).?;
                segs[w1 - w0] = .{ .tokens = l.rows.items, .p0 = l.base, .slot = l.slot };
                group_rows += windows[w1].rows();
            }
            const taps_ptr: u64 = if (!b.prepared) 0 else if (lone) b.verify_taps[segs[0].slot].?.ptr else b.batch_taps.?.ptr;
            try b.m.forwardBatched(b.k, b.s, segs[0 .. w1 - w0], group_rows, true, taps_ptr);
            var row0: usize = 0;
            for (windows[w0..w1]) |w| {
                const l = b.lanes.getPtr(w.stream).?;
                const n = w.rows();
                l.tap_buf = 0;
                if (b.prepared) {
                    l.tap_buf = b.verify_taps[l.slot].?.ptr;
                    if (!lone) try b.d.check(b.d.api.cuMemcpyDtoDAsync_v2(l.tap_buf, taps_ptr + row0 * row_bytes, n * row_bytes, b.s.handle), "cuMemcpyDtoDAsync");
                }
                l.tap_start = l.base;
                l.tap_rows = n;
                try b.drawRows(l, b.m.s.logits + row0 * v * 4, n, w.positions[0], token_ring + rows + row0);
                row0 += n;
            }
            rows += group_rows;
            w0 = w1;
        }
        var got: [max_streams * window_rows]u32 = undefined;
        try b.s.synchronize();
        try b.tokens.download(token_ring * 4, std.mem.sliceAsBytes(got[0..rows]));
        var row0: usize = 0;
        for (windows, out) |w, *o| {
            const l = b.lanes.getPtr(w.stream).?;
            const n = w.rows();
            @memcpy(o.sampled[0..n], got[row0 .. row0 + n]);
            @memcpy(o.drafts[0 .. n - 1], l.rows.items[1..n]);
            l.len = l.base + n;
            row0 += n;
        }
    }

    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const b = self(ptr);
        if (windows.len == 0 or windows.len > max_streams or paths.len != windows.len) return error.BadVerifyWindows;
        for (windows, paths) |w, path| {
            const l = try b.laneOf(w.stream);
            if (path.len == 0 or path.len > l.rows.items.len) return error.EmptyPath;
            for (path, 0..) |row, i| if (row != i) return error.ChainsOnly;
        }
        for (windows, paths) |w, path| {
            const l = try b.laneOf(w.stream);
            try b.m.rewind(b.k, b.s, path.len);
            l.len = l.base + path.len;
        }
    }

    fn draft(_: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        for (requests) |r| if (r.depth != 0 or r.lanes != null) return error.NoDraftHead;
    }

    pub fn saveState(b: *Backend, owner: ?*anyopaque, at: u32) !*Model.Saved {
        const s: *const lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoOwner));
        const l = try b.laneOf(s);
        if ((l.fill orelse l.len) != at) return error.NotAtMark;
        return b.m.save(b.s, at);
    }

    pub fn restoreState(b: *Backend, owner: ?*anyopaque, saved: *const Model.Saved) !void {
        const s: *const lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoOwner));
        const l = try b.laneOf(s);
        try b.m.restore(b.s, saved);
        l.len = saved.at;
    }

    pub fn dropState(b: *Backend, saved: *Model.Saved) void {
        b.m.drop(b.s, saved);
    }

    fn release(ptr: *anyopaque, s: *lanes.Stream) void {
        const b = self(ptr);
        const entry = b.lanes.fetchRemove(s) orelse return;
        var ln = entry.value;
        ln.rows.deinit(b.gpa);
        ln.last_b.free();
        ln.rule_b.free();
        if (ln.slot < max_streams) b.slots[ln.slot] = false;
    }
};
