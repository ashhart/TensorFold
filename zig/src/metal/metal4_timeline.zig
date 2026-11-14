//! TF_METAL4_TIMELINE: a GPU timestamp between every two stages of a named round, logged by kernel and by stage.
const std = @import("std");
const objc = @import("objc.zig");
const types = @import("types.zig");

const Id = objc.Id;
const heap = std.heap.c_allocator;
/// Stamps a round at most: the most a timestamp heap holds.
const cap = 4096;
const warm = 8;
/// Kernels a stage is named by; more are counted, not named.
const kept = 3;
const shown_kernels = 25;
const shown_stages = 15;

/// What ran before stamp i: the stage it closes (`n` dispatches), or an encoder's start when `n` is 0.
const Mark = struct { enc: u32 = 0, n: u32 = 0, pipes: [kept]?Id = @splat(null), more: bool = false };

const Sum = struct { s: f64 = 0, stages: u64 = 0 };

/// One name's rounds, summed: the whole span, inside encoders and between them, by kernel and by stage index.
const Rounds = struct {
    name: []u8,
    seen: u32 = 0,
    done: u32 = 0,
    span: f64 = 0,
    inside: f64 = 0,
    between: f64 = 0,
    encoders: u32 = 0,
    dispatches: u64 = 0,
    kernels: std.StringArrayHashMapUnmanaged(Sum) = .empty,
    each: std.ArrayList(f64) = .empty,
    keys: std.ArrayList([]const u8) = .empty,
    encs: std.ArrayList(u32) = .empty,
    fits: bool = true,
};

pub const Timeline = struct {
    counters: Id,
    freq: f64,
    want: u32,
    on: bool = false,
    cur: ?*Rounds = null,
    count: usize = 0,
    enc: u32 = 0,
    pipe: ?Id = null,
    open: Mark = .{},
    marks: []Mark,
    rounds: std.ArrayList(*Rounds) = .empty,
    labels: std.AutoHashMapUnmanaged(usize, []const u8) = .empty,

    pub fn init(device: Id, want: u32) ?*Timeline {
        const d = objc.msg(Id, objc.msg(Id, objc.class("MTL4CounterHeapDescriptor"), "alloc", .{}), "init", .{});
        defer objc.release(d);
        objc.msg(void, d, "setType:", .{@as(isize, 1)});
        objc.msg(void, d, "setCount:", .{@as(usize, cap)});
        var err: ?Id = null;
        const counters = objc.msg(?Id, device, "newCounterHeapWithDescriptor:error:", .{ d, &err }) orelse {
            std.log.warn("metal 4 timeline: no timestamp heap ({s})", .{objc.errorText(err)});
            return null;
        };
        const t = heap.create(Timeline) catch return null;
        t.* = .{
            .counters = counters,
            .freq = @floatFromInt(objc.msg(u64, device, "queryTimestampFrequency", .{})),
            .want = want,
            .marks = heap.alloc(Mark, cap) catch return null,
        };
        return t;
    }

    pub fn deinit(t: *Timeline) void {
        for (t.rounds.items) |r| {
            for (r.kernels.keys()) |k| heap.free(k);
            r.kernels.deinit(heap);
            r.each.deinit(heap);
            for (r.keys.items) |k| heap.free(k);
            r.keys.deinit(heap);
            r.encs.deinit(heap);
            heap.free(r.name);
            heap.destroy(r);
        }
        t.rounds.deinit(heap);
        var it = t.labels.valueIterator();
        while (it.next()) |v| heap.free(v.*);
        t.labels.deinit(heap);
        heap.free(t.marks);
        objc.release(t.counters);
        heap.destroy(t);
    }

    pub fn begin(t: *Timeline, name: []const u8) void {
        t.on = false;
        const r = t.find(name) orelse return;
        r.seen += 1;
        if (r.seen <= warm or r.done >= t.want) return;
        t.on = true;
        t.cur = r;
        t.count = 0;
        t.enc = 0;
        t.open = .{};
    }

    fn find(t: *Timeline, name: []const u8) ?*Rounds {
        for (t.rounds.items) |r| if (std.mem.eql(u8, r.name, name)) return r;
        const r = heap.create(Rounds) catch return null;
        r.* = .{ .name = heap.dupe(u8, name) catch return null };
        t.rounds.append(heap, r) catch return null;
        return r;
    }

    fn stamp(t: *Timeline, e: Id, m: Mark) void {
        if (t.count == cap) {
            t.on = false;
            return;
        }
        objc.msg(void, e, "writeTimestampWithGranularity:intoHeap:atIndex:", .{ @as(isize, 1), t.counters, t.count });
        t.marks[t.count] = m;
        t.count += 1;
    }

    pub fn started(t: *Timeline, e: Id) void {
        if (!t.on) return;
        t.enc += 1;
        t.open = .{ .enc = t.enc };
        t.stamp(e, t.open);
    }

    pub fn piped(t: *Timeline, id: Id) void {
        if (t.on) t.pipe = id;
    }

    pub fn dispatched(t: *Timeline) void {
        if (!t.on) return;
        t.open.n += 1;
        for (&t.open.pipes) |*p| {
            if (p.* == t.pipe) return;
            if (p.* == null) {
                p.* = t.pipe;
                return;
            }
        }
        t.open.more = true;
    }

    pub fn stage(t: *Timeline, e: Id) void {
        if (!t.on) return;
        t.stamp(e, t.open);
        t.open = .{ .enc = t.enc };
    }

    pub fn ended(t: *Timeline, e: Id) void {
        t.stage(e);
    }

    /// After the round's last wait: its stamps read back and added to its name's sums.
    pub fn end(t: *Timeline) void {
        if (!t.on) return;
        t.on = false;
        const r = t.cur orelse return;
        if (t.count < 2) return;
        const data = objc.msg(?Id, t.counters, "resolveCounterRange:", .{types.Range{ .location = 0, .length = t.count }}) orelse return;
        const w: [*]align(1) const u64 = @ptrCast(objc.msg([*]const u8, data, "bytes", .{}));
        const first = r.done == 0;
        r.done += 1;
        r.span += t.secs(w[0], w[t.count - 1]);
        r.encoders = t.enc;
        var j: usize = 0;
        for (1..t.count) |i| {
            const d = t.secs(w[i - 1], w[i]);
            const m = t.marks[i];
            if (m.n == 0) {
                r.between += d;
                continue;
            }
            r.inside += d;
            r.dispatches += m.n;
            const key = t.keyOf(m) catch continue;
            const g = r.kernels.getOrPut(heap, key) catch continue;
            if (g.found_existing) heap.free(key) else g.value_ptr.* = .{};
            g.value_ptr.s += d;
            g.value_ptr.stages += 1;
            if (first) {
                r.each.append(heap, d) catch {};
                r.keys.append(heap, heap.dupe(u8, g.key_ptr.*) catch "") catch {};
                r.encs.append(heap, m.enc) catch {};
            } else if (j < r.each.items.len) r.each.items[j] += d else r.fits = false;
            j += 1;
        }
        if (j != r.each.items.len) r.fits = false;
        if (r.done == t.want) t.report(r);
    }

    fn secs(t: *const Timeline, a: u64, b: u64) f64 {
        return if (b > a) @as(f64, @floatFromInt(b - a)) / t.freq else 0;
    }

    /// The stage's kernels by label, joined.
    fn keyOf(t: *Timeline, m: Mark) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(heap);
        for (m.pipes) |p| if (p) |id| {
            if (out.items.len > 0) try out.appendSlice(heap, " + ");
            try out.appendSlice(heap, t.label(id));
        };
        if (m.more) try out.appendSlice(heap, " + more");
        return out.toOwnedSlice(heap);
    }

    fn label(t: *Timeline, id: Id) []const u8 {
        const g = t.labels.getOrPut(heap, @intFromPtr(id)) catch return "?";
        if (g.found_existing) return g.value_ptr.*;
        const text = if (objc.msg(?Id, id, "label", .{})) |s| std.mem.span(objc.utf8(s)) else "unnamed kernel";
        g.value_ptr.* = heap.dupe(u8, text) catch "?";
        return g.value_ptr.*;
    }

    fn report(t: *Timeline, r: *Rounds) void {
        _ = t;
        const n: f64 = @floatFromInt(r.done);
        const ms = 1e3 / n;
        var stages: u64 = 0;
        for (r.kernels.values()) |v| stages += v.stages;
        std.log.info("metal 4 timeline: {s}, {d} rounds after {d} warm: {d} encoders, {d} stages and {d} dispatches a round", .{ r.name, r.done, warm, r.encoders, stages / r.done, r.dispatches / r.done });
        std.log.info("metal 4 timeline: {s}: {d:.3} ms a round from the first stamp to the last, {d:.3} in stages, {d:.3} between encoders", .{ r.name, r.span * ms, r.inside * ms, r.between * ms });
        const order = heap.alloc(usize, r.kernels.count()) catch return;
        defer heap.free(order);
        for (order, 0..) |*o, i| o.* = i;
        const vals = r.kernels.values();
        std.mem.sort(usize, order, vals, struct {
            fn more(v: []const Sum, a: usize, b: usize) bool {
                return v[a].s > v[b].s;
            }
        }.more);
        std.log.info("metal 4 timeline: {s} by kernel: ms a round, stages a round, us a stage", .{r.name});
        for (order[0..@min(order.len, shown_kernels)]) |i| {
            const v = vals[i];
            const per = @as(f64, @floatFromInt(v.stages)) / n;
            std.log.info("  {d:>8.3} {d:>6.1} {d:>8.1}  {s}", .{ v.s * ms, per, v.s * 1e6 / @as(f64, @floatFromInt(v.stages)), r.kernels.keys()[i] });
        }
        if (!r.fits) return std.log.info("metal 4 timeline: {s}: rounds differ in stages, so no stage-by-stage list", .{r.name});
        const idx = heap.alloc(usize, r.each.items.len) catch return;
        defer heap.free(idx);
        for (idx, 0..) |*o, i| o.* = i;
        std.mem.sort(usize, idx, r.each.items, struct {
            fn more(v: []const f64, a: usize, b: usize) bool {
                return v[a] > v[b];
            }
        }.more);
        std.log.info("metal 4 timeline: {s}: the longest single stages, us a round", .{r.name});
        for (idx[0..@min(idx.len, shown_stages)]) |i| {
            std.log.info("  stage {d:>4} of encoder {d}: {d:>8.1}  {s}", .{ i, r.encs.items[i], r.each.items[i] * 1e6 / n, r.keys.items[i] });
        }
    }
};
