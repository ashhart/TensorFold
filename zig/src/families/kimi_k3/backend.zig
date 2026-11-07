//! K3's resident graph in the existing lane core: shared windows, exact recurrent rollback and independent slots.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const be = lanes.backend;
const Graph = @import("graph.zig").Graph;
const State = @import("state.zig").State;
const round = @import("round.zig");
const Kernels = @import("kernels.zig").Kernels;
const Index = @import("index.zig");
const gpu_rows: usize = 256;
pub const window: u32 = 16;
pub const Slot = struct { stream: ?*lanes.Stream = null, first: usize = 0, verified: usize = 0, start: u32 = 0 };

pub const Metal = struct {
    a: std.mem.Allocator,
    graph: *Graph,
    device: mtl.Device,
    queue: mtl.Queue,
    k: *const Kernels,
    state: State,
    scratch: round.Scratch,
    index: ?Index.Work = null,
    slots: []Slot,
    last: mtl.Buffer,
    resident: mtl.ResidencySet,
    values: []f32,
    answers: [4096]struct { handle: u64 = 0, token: u32 = 0 } = @splat(.{}),
    next: u64 = 0,
    healthy: std.atomic.Value(bool) = .init(true),

    pub fn init(a: std.mem.Allocator, graph: *Graph, device: mtl.Device, queue: mtl.Queue, k: *const Kernels, streams: u32, capacity: u32, source_buffers: []const mtl.Buffer) !*Metal {
        if (streams == 0 or streams > 8 or capacity == 0) return error.InvalidKimiCapacity;
        const m = try a.create(Metal);
        errdefer a.destroy(m);
        const c = graph.c;
        const layers = try a.alloc(u32, c.layers);
        defer a.free(layers);
        for (layers, 0..) |*li, i| li.* = @intCast(i);
        var state = try State.init(a, device, c, .{ .slots = streams, .log_rows = window, .max_ctx = capacity }, layers, c.kda_heads);
        errdefer state.deinit();
        var scratch = try round.Scratch.init(device, &c, gpu_rows);
        errdefer scratch.deinit();
        var index: ?Index.Work = if (c.dsa != null) try Index.Work.init(device, gpu_rows, capacity) else null;
        errdefer if (index) |*ix| ix.deinit();
        const slots = try a.alloc(Slot, streams);
        errdefer a.free(slots);
        @memset(slots, .{});
        const last = try device.buffer(@as(usize, streams) * c.vocab * 2, mtl.ResourceOptions.shared);
        errdefer last.deinit();
        const values = try a.alloc(f32, c.vocab);
        errdefer a.free(values);
        var buffers: std.ArrayList(mtl.Buffer) = .empty;
        defer buffers.deinit(a);
        try buffers.appendSlice(a, source_buffers);
        try graph.buffers(&buffers, a);
        try state.buffers(&buffers, a);
        if (index) |*ix| try ix.buffers(&buffers, a);
        inline for (@typeInfo(round.Scratch).@"struct".field_names, @typeInfo(round.Scratch).@"struct".field_types) |f, T|
            if (T == mtl.Buffer) try buffers.append(a, @field(scratch, f));
        try buffers.append(a, last);
        const resident = try device.residencySet(buffers.items.len);
        errdefer resident.deinit();
        for (buffers.items) |b| resident.add(b);
        resident.commit();
        queue.addResidencySet(resident);
        m.* = .{ .a = a, .graph = graph, .device = device, .queue = queue, .k = k, .state = state, .scratch = scratch, .index = index, .slots = slots, .last = last, .values = values, .resident = resident };
        return m;
    }

    pub fn deinit(m: *Metal) void {
        m.queue.removeResidencySet(m.resident);
        m.resident.deinit();
        if (m.index) |*ix| ix.deinit();
        m.last.deinit();
        m.state.deinit();
        m.scratch.deinit();
        m.a.free(m.values);
        m.a.free(m.slots);
        m.a.destroy(m);
    }

    pub fn facts(m: *const Metal) lanes.Model {
        return .{ .exact_width = window, .hidden_rows = true, .batch_rows = gpu_rows, .max_streams = @intCast(m.slots.len), .streams_exact = true, .copy_drafts = true };
    }

    pub fn backend(m: *Metal) be.Backend {
        return .{ .ptr = m, .chunk_invariant = true, .vtable = &.{ .prefill = prefill, .prefill_step = step, .reports_fill = true, .first = first, .first_masked = firstMasked, .queue = queueOne, .read = read, .verify = verify, .keep = keep, .draft = draft, .held = held, .release = release } };
    }

    fn self(ptr: *anyopaque) *Metal {
        return @ptrCast(@alignCast(ptr));
    }
    pub fn find(m: *Metal, s: *lanes.Stream) !u32 {
        for (m.slots, 0..) |slot, i| if (slot.stream == s) return @intCast(i);
        return error.UnknownKimiStream;
    }
    pub fn logits(m: *Metal, slot: u32) []u16 {
        const v = m.graph.c.vocab;
        return m.last.slice(u16, m.slots.len * v)[@as(usize, slot) * v ..][0..v];
    }
    pub fn acquire(m: *Metal, s: *lanes.Stream) !u32 {
        if (!m.healthy.load(.acquire)) return error.KimiEngineFailed;
        const cap = m.state.limits.max_ctx;
        if (s.prompt_len == 0 or s.prompt_len > cap or s.max_new > cap - s.prompt_len) return error.KimiContextLimit;
        if (s.images.len > 0 or s.embedding != null) return error.KimiTextOnly;
        if (s.logprobs != null or s.prompt_logprobs != null) return error.KimiLogprobsUnsupported;
        for (s.prompt()) |id| if (id >= m.graph.c.vocab) return error.KimiTokenOutOfRange;
        for (m.slots, 0..) |*slot, i| if (slot.stream == null) {
            m.state.reset(@intCast(i));
            slot.* = .{ .stream = s };
            errdefer slot.* = .{};
            if (s.reuse.saved) |ptr| {
                const snap: *const @import("snapshot.zig").Snap = @ptrCast(@alignCast(ptr));
                try @import("snapshot.zig").restore(m, @intCast(i), snap, s.prompt(), @intCast(s.reuse.at));
            }
            s.cached = m.state.streams[i].pos;
            s.fill_at.store(m.state.streams[i].pos, .release);
            return @intCast(i);
        };
        return error.KimiNoStreamSlot;
    }

    pub fn run(m: *Metal, segments: []const round.Segment, ids: []const u32) !void {
        if (!m.healthy.load(.acquire)) return error.KimiEngineFailed;
        for (ids) |id| if (id >= m.graph.c.vocab) return error.KimiTokenOutOfRange;
        try m.scratch.load(&m.state, segments, ids);
        if (m.index) |*ix| try ix.load(m.scratch.mrows.slice(@import("kernels.zig").MlaRow, m.scratch.rows));
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = m.queue.commandBuffer();
        const encoder = cb.compute(.serial);
        var local = @import("experts.zig").Local.all(m.k, &m.graph.c);
        const stage: @import("forward.zig").Stage = .{ .first = 0, .last = m.graph.c.layers, .layers = m.graph.layers, .head = &m.graph.head, .embed = m.graph.head.embed };
        stage.encode(.{ .k = m.k, .c = &m.graph.c, .sc = &m.scratch, .st = &m.state, .experts = local.experts(), .index = if (m.index) |*ix| ix else null }, encoder, null) catch |err| {
            encoder.end();
            return err;
        };
        encoder.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) {
            m.healthy.store(false, .release);
            return error.KimiGpuFailed;
        }
    }

    fn storeLast(m: *Metal, slot: u32, row: usize) void {
        const v = m.graph.c.vocab;
        @memcpy(m.logits(slot), m.scratch.logits.slice(u16, (row + 1) * v)[row * v ..]);
    }
    fn prefill(ptr: *anyopaque, s: *lanes.Stream) !void {
        while (!try step(ptr, s, 0)) {}
    }
    fn step(ptr: *anyopaque, s: *lanes.Stream, max_rows: u32) !bool {
        const m = self(ptr);
        const slot = m.find(s) catch try m.acquire(s);
        const pos = m.state.streams[slot].pos;
        if (s.isCancelled()) {
            s.keepInterruptedPrefill(pos);
            return error.Cancelled;
        }
        if (pos == s.prompt_len) return true;
        var end = @min(s.prompt_len, pos + @min(gpu_rows, if (max_rows == 0) gpu_rows else max_rows));
        for (s.chunks) |cut| if (cut > pos) {
            end = @min(end, cut);
            break;
        };
        for (s.reuse.marks) |cut| if (cut > pos) {
            end = @min(end, cut);
            break;
        };
        const rows: u32 = @intCast(end - pos);
        try m.run(&.{.{ .slot = slot, .rows = rows, .commit = true }}, s.prompt()[pos..end]);
        lanes.prefill_widths.completed(m.a, s, rows);
        m.storeLast(slot, rows - 1);
        m.state.advance(slot, rows, true, rows);
        s.fill_at.store(@intCast(end), .release);
        if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(end)) != null)
            if (s.reuse.hook) |hook| hook.at(hook.ptr, s, @intCast(end));
        return end == s.prompt_len;
    }

    fn draw(m: *Metal, s: *lanes.Stream, logits_: []const u16, pos: u64, mask: []const u32) !u32 {
        for (logits_, m.values, 0..) |bits, *out, i| {
            out.* = @bitCast(@as(u32, bits) << 16);
            if (mask.len > 0 and !lanes.grammar.allows(mask, i)) out.* = -std.math.inf(f32);
        }
        if (s.sampling) |sample| if (sample.temperature > 0)
            return lanes.gpu_full.sample(m.a, m.values, sample, @intCast(pos), null);
        return greedy(m.values);
    }
    fn answer(m: *Metal, token: u32) !u64 {
        m.next = try std.math.add(u64, m.next, 1);
        m.answers[m.next % m.answers.len] = .{ .handle = m.next, .token = token };
        return m.next;
    }
    fn first(ptr: *anyopaque, s: *lanes.Stream, pos: u64) !u64 {
        return firstMasked(ptr, s, pos, &.{});
    }
    fn firstMasked(ptr: *anyopaque, s: *lanes.Stream, pos: u64, mask: []const u32) !u64 {
        const m = self(ptr);
        const slot = try m.find(s);
        if (pos != m.state.streams[slot].pos) return error.KimiPosition;
        return m.answer(try m.draw(s, m.logits(slot), pos, mask));
    }
    fn queueOne(ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, pos: u64) !u64 {
        const m = self(ptr);
        const slot = try m.find(s);
        if (pos != @as(u64, m.state.streams[slot].pos) + 1) return error.KimiPosition;
        const token = switch (feed) {
            .value => |id| id,
            .handle => |h| try read(ptr, h),
        };
        try m.run(&.{.{ .slot = slot, .rows = 1, .commit = true }}, &.{token});
        m.storeLast(slot, 0);
        m.state.advance(slot, 1, true, 1);
        return m.answer(try m.draw(s, m.logits(slot), pos, &.{}));
    }
    fn read(ptr: *anyopaque, handle: u64) !u32 {
        const m = self(ptr);
        const value = m.answers[handle % m.answers.len];
        if (handle == 0 or value.handle != handle) return error.KimiStaleDraw;
        return value.token;
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const m = self(ptr);
        if (windows.len != out.len or windows.len > m.slots.len) return error.KimiWindow;
        var segments: [8]round.Segment = undefined;
        var ids: [gpu_rows]u32 = undefined;
        var total: usize = 0;
        for (windows, out, 0..) |w, o, i| {
            const n = w.rows();
            if (n > window or total + n > gpu_rows or w.parents != null or w.held != 0 or o.sampled.len != n or o.drafts.len != n - 1 or w.positions.len != n) return error.KimiWindow;
            const slot = try m.find(w.stream);
            for (segments[0..i]) |s| if (s.slot == slot) return error.KimiWindow;
            for (w.positions, 0..) |p, row| if (p != @as(u64, m.state.streams[slot].pos) + row + 1) return error.KimiPosition;
            segments[i] = .{ .slot = slot, .rows = @intCast(n), .commit = false };
            ids[total] = w.pending;
            @memcpy(ids[total + 1 ..][0..w.tokens.len], w.tokens);
            m.slots[slot].start = m.state.streams[slot].pos;
            m.slots[slot].first = total;
            m.slots[slot].verified = n;
            total += n;
        }
        try m.run(segments[0..windows.len], ids[0..total]);
        const vocab = m.graph.c.vocab;
        const all = m.scratch.logits.slice(u16, total * vocab);
        const words = (vocab + 31) / 32;
        for (windows, out, segments[0..windows.len]) |w, o, segment| {
            @memcpy(o.drafts, w.tokens);
            for (o.sampled, 0..) |*sampled, r| {
                const row = m.slots[segment.slot].first + r;
                const mask = if (w.masks.len >= (r + 1) * words) w.masks[r * words ..][0..words] else &.{};
                sampled.* = try m.draw(w.stream, all[row * vocab ..][0..vocab], w.positions[r], mask);
            }
            // The lane core omits keep when the whole window lands, so publish that full window now.
            m.storeLast(segment.slot, m.slots[segment.slot].first + o.sampled.len - 1);
            windowPosition(&m.state, segment.slot, m.slots[segment.slot], @intCast(o.sampled.len));
        }
    }
    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
        const m = self(ptr);
        if (windows.len != paths.len) return error.KimiWindow;
        for (windows, paths) |w, path| {
            const slot = try m.find(w.stream);
            if (path.len == 0 or path.len > w.rows() or m.slots[slot].verified != w.rows()) return error.KimiWindow;
            for (path, 0..) |p, i| if (p != i) return error.KimiTreeUnsupported;
        }
        for (windows, paths) |w, path| {
            const slot = try m.find(w.stream);
            m.storeLast(slot, m.slots[slot].first + path.len - 1);
            windowPosition(&m.state, slot, m.slots[slot], @intCast(path.len));
            m.slots[slot].verified = 0;
        }
    }
    fn draft(_: *anyopaque, requests: []const be.DraftRequest) !void {
        for (requests) |r| if (r.depth != 0) return error.KimiNoDraftHead;
    }
    fn held(_: *anyopaque, _: *lanes.Stream, out: []u32) !void {
        if (out.len != 0) return error.KimiNoDraftHead;
    }
    fn release(ptr: *anyopaque, s: *lanes.Stream) void {
        const m = self(ptr);
        const slot = m.find(s) catch return;
        m.state.reset(slot);
        m.slots[slot] = .{};
    }
};

// Match k3_argmax: ignore NaNs, start at -infinity, and keep the lower id on a tie.
fn greedy(values: []const f32) u32 {
    var best: f32 = -std.math.inf(f32);
    var id: u32 = 0;
    for (values, 0..) |v, i| if (v > best) {
        best = v;
        id = @intCast(i);
    };
    return id;
}

test "Kimi host greedy follows the GPU's NaN, infinity and lower-id tie rule" {
    const nan = std.math.nan(f32);
    const inf = std.math.inf(f32);
    try std.testing.expectEqual(@as(u32, 1), greedy(&.{ nan, 4, 4, -inf }));
    try std.testing.expectEqual(@as(u32, 2), greedy(&.{ -inf, nan, inf, inf }));
    try std.testing.expectEqual(@as(u32, 0), greedy(&.{ -inf, nan, -inf }));
    try std.testing.expectEqual(@as(u32, 0), greedy(&.{ -0.0, 0.0 }));
}

/// Each verify provisionally keeps every row; a later partial keep rolls back from the same saved start.
fn windowPosition(state: *State, slot: u32, record: Slot, kept: u32) void {
    state.streams[slot].pos = record.start;
    state.advance(slot, @intCast(record.verified), false, kept);
}

test "Kimi full verification publishes its position before an optional partial keep" {
    var positions = [_]@import("state.zig").Stream{.{ .pos = 17 }};
    var state: State = undefined;
    state.streams = &positions;
    var record: Slot = .{ .start = 17, .verified = 6 };
    windowPosition(&state, 0, record, 6);
    try std.testing.expectEqual(@as(u32, 23), positions[0].pos);
    try std.testing.expectEqual(@as(u32, 6), positions[0].kept);
    windowPosition(&state, 0, record, 2);
    try std.testing.expectEqual(@as(u32, 19), positions[0].pos);
    try std.testing.expectEqual(@as(u32, 2), positions[0].kept);
    record = .{ .start = positions[0].pos, .verified = 1 };
    windowPosition(&state, 0, record, 1);
    try std.testing.expectEqual(@as(u32, 20), positions[0].pos);
}
