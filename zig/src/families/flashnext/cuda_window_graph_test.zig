//! The window graphs' cache on a driver that counts calls: two-way slots, LRU updates in place, exact key replays.
const std = @import("std");
const t = std.testing;
const cuda = @import("cuda");
const abi = cuda.abi;
const wg = @import("cuda_window_graph.zig");
const parts = @import("cuda_batch_state.zig");
const wt = @import("cuda_prompt_wt.zig");

const max_fake_execs = 512;

const Fake = struct {
    const State = struct { capturing: bool = false, records: usize = 0, instantiates: usize = 0, updates: usize = 0, launches: usize = 0, eager: usize = 0, destroyed: usize = 0, verdict: abi.ExecUpdateResult = .success, recorded_tag: u64 = 0, last_tag: u64 = 0, last_exec: usize = 0, exec_tag: [max_fake_execs]u64 = @splat(0), live: [max_fake_execs]bool = @splat(false) };
    var st: State = .{};
    fn reset() cuda.Driver {
        st = .{};
        var d: cuda.Driver = undefined;
        d.api.cuStreamBeginCapture_v2 = begin;
        d.api.cuStreamEndCapture = end;
        d.api.cuStreamIsCapturing = status;
        d.api.cuGraphInstantiateWithFlags = instantiate;
        d.api.cuGraphExecUpdate_v2 = update;
        d.api.cuGraphDestroy = graphDestroy;
        d.api.cuGraphUpload = upload;
        d.api.cuGraphLaunch = launch;
        d.api.cuGraphExecDestroy = execDestroy;
        return d;
    }
    fn begin(_: abi.Stream, _: abi.CaptureMode) callconv(.c) abi.Result {
        st.capturing = true;
        return 0;
    }
    fn end(_: abi.Stream, out: *abi.Graph) callconv(.c) abi.Result {
        st.capturing = false;
        st.records += 1;
        out.* = @ptrFromInt(1);
        return 0;
    }
    fn status(_: abi.Stream, s: *abi.CaptureStatus) callconv(.c) abi.Result {
        s.* = if (st.capturing) .active else .none;
        return 0;
    }
    fn instantiate(e: *abi.GraphExec, _: abi.Graph, _: u64) callconv(.c) abi.Result {
        st.instantiates += 1;
        std.debug.assert(st.instantiates < max_fake_execs);
        st.exec_tag[st.instantiates] = st.recorded_tag;
        st.live[st.instantiates] = true;
        e.* = @ptrFromInt(st.instantiates);
        return 0;
    }
    fn update(exec: abi.GraphExec, _: abi.Graph, info: *abi.ExecUpdateResultInfo) callconv(.c) abi.Result {
        st.updates += 1;
        info.result = st.verdict;
        if (st.verdict != .success) return 910; // CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE
        const id = @intFromPtr(exec);
        std.debug.assert(st.live[id]);
        st.exec_tag[id] = st.recorded_tag;
        return 0;
    }
    fn graphDestroy(_: abi.Graph) callconv(.c) abi.Result {
        return 0;
    }
    fn upload(exec: abi.GraphExec, _: abi.Stream) callconv(.c) abi.Result {
        std.debug.assert(st.live[@intFromPtr(exec)]);
        return 0;
    }
    fn launch(exec: abi.GraphExec, _: abi.Stream) callconv(.c) abi.Result {
        const id = @intFromPtr(exec);
        std.debug.assert(st.live[id]);
        st.launches += 1;
        st.last_exec = id;
        st.last_tag = st.exec_tag[id];
        return 0;
    }
    fn execDestroy(exec: abi.GraphExec) callconv(.c) abi.Result {
        const id = @intFromPtr(exec);
        std.debug.assert(st.live[id]);
        st.live[id] = false;
        st.destroyed += 1;
        return 0;
    }
};

/// A pass whose tag records which buffers/sizes were baked into a graph, or into `last_tag` when run eagerly.
const Pass = struct {
    tag: u64 = 0,
    pub fn run(p: Pass) !void {
        if (Fake.st.capturing) Fake.st.recorded_tag = p.tag else {
            Fake.st.eager += 1;
            Fake.st.last_tag = p.tag;
        }
    }
};

const Rig = struct {
    d: cuda.Driver = undefined,
    prev: abi.Context = null,
    fn init(r: *Rig) cuda.Stream {
        r.d = Fake.reset();
        r.prev = cuda.copy_source.current();
        cuda.copy_source.setCurrent(@ptrFromInt(0x3254));
        return .{ .d = &r.d, .handle = @ptrFromInt(1) };
    }
    fn deinit(r: Rig) void {
        cuda.copy_source.setCurrent(r.prev);
    }
};

fn key(moving: u64, rows: u32, shape: u64) wg.Key {
    return .{ .rows = rows, .shape = shape, .moving = moving };
}

fn runTag(c: anytype, stream: cuda.Stream, k: anytype, tag: u64) !void {
    try c.run(t.allocator, stream, k, Pass{ .tag = tag });
    try t.expectEqual(tag, Fake.st.last_tag);
}

/// After one more run: these counts of recordings, instantiations, updates, eager runs and graph launches.
fn expectAfter(c: *wg.Cache(wg.Key), stream: cuda.Stream, k: wg.Key, counts: [5]usize) !void {
    try runTag(c, stream, k, k.moving);
    try t.expectEqual(counts, [5]usize{ Fake.st.records, Fake.st.instantiates, Fake.st.updates, Fake.st.eager, Fake.st.launches });
}

test "window graphs: a slot holds two keys, updates the older in place on a third, and replaces on new launches" {
    var rig: Rig = .{};
    const stream = rig.init();
    defer rig.deinit();
    var c: wg.Cache(wg.Key) = .{};
    defer c.clear(t.allocator);

    try expectAfter(&c, stream, key(1, 3, 7), .{ 0, 0, 0, 1, 0 }); // a new slot runs eagerly once
    try expectAfter(&c, stream, key(1, 3, 7), .{ 1, 1, 0, 1, 1 }); // way 0 captured
    try expectAfter(&c, stream, key(1, 3, 7), .{ 1, 1, 0, 1, 2 }); // way 0 replayed
    try expectAfter(&c, stream, key(2, 3, 7), .{ 2, 2, 0, 1, 3 }); // way 1 captured alongside way 0
    try expectAfter(&c, stream, key(1, 3, 7), .{ 2, 2, 0, 1, 4 }); // way 0 still resident: replayed
    try expectAfter(&c, stream, key(2, 3, 7), .{ 2, 2, 0, 1, 5 }); // way 1 still resident: replayed
    try expectAfter(&c, stream, key(3, 3, 7), .{ 3, 2, 1, 1, 6 }); // third key updates LRU way 0 in place
    try expectAfter(&c, stream, key(3, 3, 7), .{ 3, 2, 1, 1, 7 });
    Fake.st.verdict = .topology_changed;
    try expectAfter(&c, stream, key(4, 3, 7), .{ 4, 3, 2, 1, 8 }); // other launches: a new executable, the old freed
    try t.expectEqual(@as(usize, 1), Fake.st.destroyed);
    Fake.st.verdict = .success;
    try expectAfter(&c, stream, key(4, 3, 7), .{ 4, 3, 2, 1, 9 });
    try expectAfter(&c, stream, key(4, 4, 7), .{ 4, 3, 2, 2, 9 }); // another row count is another slot
    try expectAfter(&c, stream, key(4, 3, 8), .{ 4, 3, 2, 3, 9 }); // and another shape
    try t.expectEqual([3]u64{ 2, 2, 5 }, [3]u64{ c.captures, c.updates, c.replays });
    try t.expectEqual(@as(usize, 2), c.held);
    c.clear(t.allocator);
    try t.expectEqual(@as(usize, 0), c.held);
    try t.expectEqual(@as(usize, 3), Fake.st.destroyed);
}

const Buf = struct { ptr: u64 };
const Pg = @import("cuda_glue.zig").Pg;
const linear_n = wt.linear_n;
const attn_n = wt.attn_n;

const Seq = struct {
    gen: u64,
    cap: usize = 262144,
    conv: [linear_n]Buf = @splat(.{ .ptr = 0 }),
    state: [linear_n]Buf = @splat(.{ .ptr = 0 }),
    kc: [attn_n]Buf = @splat(.{ .ptr = 0 }),
    vc: [attn_n]Buf = @splat(.{ .ptr = 0 }),
    ikc: [attn_n]Buf = @splat(.{ .ptr = 0 }),
    pooled: [attn_n]Buf = @splat(.{ .ptr = 0 }),
    tail: Buf = .{ .ptr = 0 },
    pg: Pg = .{},
    fn at(gen: u64, base: u64) Seq {
        var q: Seq = .{ .gen = gen };
        var a = base;
        inline for (.{ "conv", "state", "kc", "vc", "ikc", "pooled" }) |f| for (&@field(q, f)) |*b| {
            b.ptr = a;
            a += 0x10000;
        };
        q.tail.ptr = a;
        return q;
    }
};

const FakeDraft = struct { gen: u64 = 11, cap: usize = 262144, len: usize = 9000, kc: Buf = .{ .ptr = 0x5000_0000 }, streams: Buf = .{ .ptr = 0x5100_0000 }, out: Buf = .{ .ptr = 0x5200_0000 }, dlogits: Buf = .{ .ptr = 0x5300_0000 }, saved: Buf = .{ .ptr = 0x5400_0000 }, keep: parts.Keep = .{} };

fn scratchOf(h: u64, cap: usize) parts.Scratch {
    var s: parts.Scratch = undefined;
    s.h = h;
    s.cap = cap;
    return s;
}

fn windowKey(seq: Seq, pos0: usize, rows: usize, hooks: parts.Hooks) wg.Key {
    const s = scratchOf(0x3000, 64);
    return .init(seq, pos0, rows, &s, 0x5000, 0x4000, hooks);
}

fn aimKeep(keep: *parts.Keep, spare: *const [linear_n]Buf) void {
    var li: usize = 0;
    for (&keep.layers, 0..) |*lk, layer| {
        if (!wt.linearLayer(layer)) continue;
        lk.state_out = spare[li].ptr;
        lk.proj = 0x9000_0000 + li * 0x1000;
        lk.hist = 0x9100_0000 + li * 0x1000;
        li += 1;
    }
}

/// Stream recordings (capture + in-place update) a one-graph-per-slot cache makes over `keys`.
fn oneWayRecords(comptime K: type, keys: []const K) !usize {
    var seen: std.AutoHashMapUnmanaged(K, void) = .empty;
    defer seen.deinit(t.allocator);
    var map: std.AutoHashMapUnmanaged(K, K) = .empty;
    defer map.deinit(t.allocator);
    var n: usize = 0;
    for (keys) |k| {
        const slot = wg.slotOf(k);
        if (map.getPtr(slot)) |held| {
            if (!std.meta.eql(held.*, k)) {
                held.* = k;
                n += 1;
            }
        } else if (seen.contains(slot)) {
            try map.put(t.allocator, slot, k);
            n += 1;
        } else try seen.put(t.allocator, slot, {});
    }
    return n;
}

test "window graphs: ping-pong of settled states and head inputs hits after the first two captures" {
    var rig: Rig = .{};
    const stream = rig.init();
    defer rig.deinit();

    var seq: Seq = .at(7, 0x1000_0000);
    var spare: [linear_n]Buf = undefined;
    for (&spare, 0..) |*x, i| x.ptr = 0x3000_0000 + i * 0x10000;
    var keep: parts.Keep = .{};
    aimKeep(&keep, &spare);

    var win_cache: wg.Cache(wg.Key) = .{};
    defer win_cache.clear(t.allocator);
    var win_keys: [19]wg.Key = undefined;
    var steady_before: usize = 0;
    for (&win_keys, 0..) |*wk, round| {
        const hooks: parts.Hooks = .{ .logits = .all, .logits_out = 0x6000, .streams = 0x8000, .keep = &keep, .host_logits = false };
        wk.* = windowKey(seq, 9000 + round * 4, 4, hooks);
        try runTag(&win_cache, stream, wk.*, seq.state[0].ptr ^ keep.layers[0].state_out);
        if (round == 2) steady_before = Fake.st.records;
        for (&seq.state, &spare) |*a, *b| std.mem.swap(Buf, a, b);
        aimKeep(&keep, &spare);
    }
    // A single-way slot records on every swap; the two-way slot holds both orientations and replays.
    try t.expectEqual(@as(usize, 18), try oneWayRecords(wg.Key, &win_keys));
    try t.expectEqual(@as(usize, 2), steady_before);
    try t.expectEqual(steady_before, Fake.st.records);
    try t.expectEqual([3]u64{ 2, 0, 16 }, [3]u64{ win_cache.captures, win_cache.updates, win_cache.replays });

    var hd: FakeDraft = .{ .len = 9000 };
    const s = scratchOf(0x3000, 64);
    var head_cache: wg.Cache(wg.HeadKey) = .{};
    defer head_cache.clear(t.allocator);
    var head_keys: [19]wg.HeadKey = undefined;
    const base_records = Fake.st.records;
    for (&head_keys, 0..) |*hk, step| {
        const in_ptr = if (step % 2 == 0) hd.streams.ptr else hd.out.ptr;
        hk.* = .init(&hd, 1, true, true, in_ptr, &s, 0x5000);
        try runTag(&head_cache, stream, hk.*, in_ptr);
        hd.len += 1;
    }
    try t.expectEqual(@as(usize, 18), try oneWayRecords(wg.HeadKey, &head_keys));
    try t.expectEqual(@as(usize, 2), Fake.st.records - base_records);
    try t.expectEqual([3]u64{ 2, 0, 16 }, [3]u64{ head_cache.captures, head_cache.updates, head_cache.replays });
}

test "window graphs: a third distinct key on a slot evicts the least recently used way" {
    var rig: Rig = .{};
    const stream = rig.init();
    defer rig.deinit();
    var c: wg.Cache(wg.Key) = .{};
    defer c.clear(t.allocator);

    // Slot 7: populate kA (way 0) and kB (way 1), then replay kA so kB is least recently used.
    const kA, const kB, const kC = .{ key(10, 3, 7), key(20, 3, 7), key(30, 3, 7) };
    try runTag(&c, stream, kA, 10);
    try runTag(&c, stream, kA, 10);
    const exec_a = Fake.st.last_exec;
    try runTag(&c, stream, kB, 20);
    const exec_b = Fake.st.last_exec;
    try t.expect(exec_a != exec_b);
    try runTag(&c, stream, kA, 10); // replay kA -> kB is now LRU
    try t.expectEqual(exec_a, Fake.st.last_exec);
    try runTag(&c, stream, kC, 30); // must evict kB (exec_b), keeping kA
    try t.expectEqual(@as(u64, 1), c.updates);
    try t.expectEqual(exec_b, Fake.st.last_exec);
    try runTag(&c, stream, kA, 10); // kA was MRU: hits with no update
    try t.expectEqual([2]u64{ 1, 2 }, [2]u64{ c.updates, c.replays });
    try t.expectEqual(exec_a, Fake.st.last_exec);
    try runTag(&c, stream, kB, 20); // kB was evicted: misses and evicts kC (now LRU)
    try t.expectEqual([2]u64{ 2, 2 }, [2]u64{ c.updates, c.replays });
    try t.expectEqual(exec_b, Fake.st.last_exec);

    // Slot 9: populate kD (way 0) then kE (way 1) without replaying kD, so kD is LRU and kE is MRU.
    const kD, const kE, const kF = .{ key(40, 3, 9), key(50, 3, 9), key(60, 3, 9) };
    try runTag(&c, stream, kD, 40);
    try runTag(&c, stream, kD, 40);
    const exec_d = Fake.st.last_exec;
    try runTag(&c, stream, kE, 50);
    const exec_e = Fake.st.last_exec;
    try runTag(&c, stream, kF, 60); // must evict kD (exec_d), keeping kE
    try t.expectEqual(@as(u64, 3), c.updates);
    try t.expectEqual(exec_d, Fake.st.last_exec);
    try runTag(&c, stream, kE, 50); // kE was MRU: hits
    try t.expectEqual([2]u64{ 3, 3 }, [2]u64{ c.updates, c.replays });
    try t.expectEqual(exec_e, Fake.st.last_exec);
}

fn expectNoReplay(c: anytype, stream: cuda.Stream, k: anytype, tag: u64) !void {
    const before = c.replays;
    try runTag(c, stream, k, tag);
    try t.expectEqual(before, c.replays);
}

fn withField(base: anytype, comptime f: []const u8, v: anytype) @TypeOf(base) {
    var out = base;
    @field(out, f) = v;
    return out;
}

fn withBuf(seq: Seq, comptime f: []const u8, i: usize) Seq {
    var out = seq;
    if (comptime std.mem.eql(u8, f, "tail")) out.tail.ptr += 0x100 else @field(out, f)[i].ptr += 0x100;
    return out;
}

test "window graphs: replay only happens on an exact match of pointers, sizes and shapes" {
    var rig: Rig = .{};
    const stream = rig.init();
    defer rig.deinit();

    const seq_a: Seq = .at(7, 0x1000_0000);
    var seq_b = seq_a;
    var spare: [linear_n]Buf = undefined;
    for (&spare, 0..) |*x, i| x.ptr = 0x3000_0000 + i * 0x10000;
    for (&seq_b.state, &spare) |*x, *y| std.mem.swap(Buf, x, y);

    var keep_a: parts.Keep = .{};
    aimKeep(&keep_a, &spare);
    var keep_b: parts.Keep = .{};
    aimKeep(&keep_b, &seq_a.state);
    var m_keep = keep_a;
    m_keep.layers[0].state_out += 0x100;

    const s64, const s128, const s_moved = .{ scratchOf(0x3000, 64), scratchOf(0x3000, 128), scratchOf(0x3100, 64) };
    const hk_a: parts.Hooks = .{ .logits = .all, .logits_out = 0x6000, .streams = 0x8000, .keep = &keep_a, .host_logits = false };
    const hk_b: parts.Hooks = .{ .logits = .all, .logits_out = 0x6000, .streams = 0x8000, .keep = &keep_b, .host_logits = false };
    const k_a = wg.Key.init(seq_a, 9000, 3, &s64, 0x5000, 0x4000, hk_a);
    const k_b = wg.Key.init(seq_b, 9000, 3, &s64, 0x5000, 0x4000, hk_b);

    var c: wg.Cache(wg.Key) = .{};
    defer c.clear(t.allocator);
    try runTag(&c, stream, k_a, 1);
    try runTag(&c, stream, k_a, 1);
    try runTag(&c, stream, k_b, 2);
    try runTag(&c, stream, k_a, 1);
    try runTag(&c, stream, k_b, 2);
    try t.expectEqual(@as(u64, 2), c.replays);

    const m_state = withBuf(seq_a, "state", linear_n - 1);
    const pg_alt: Pg = .{ .pt = 0x7000_0000, .ps = 4, .ptok = 256 };
    const mismatches = [_]wg.Key{
        .init(m_state, 9000, 3, &s64, 0x5000, 0x4000, hk_a),
        .init(withBuf(seq_a, "conv", 0), 9000, 3, &s64, 0x5000, 0x4000, hk_a),
        .init(withBuf(seq_a, "kc", attn_n - 1), 9000, 3, &s64, 0x5000, 0x4000, hk_a),
        .init(withBuf(seq_a, "tail", 0), 9000, 3, &s64, 0x5000, 0x4000, hk_a),
        .init(withField(seq_a, "pg", pg_alt), 9000, 3, &s64, 0x5000, 0x4000, hk_a),
        .init(seq_a, 9000, 3, &s_moved, 0x5000, 0x4000, hk_a),
        .init(seq_a, 9000, 3, &s64, 0x5100, 0x4000, hk_a),
        .init(seq_a, 9000, 3, &s64, 0x5000, 0x4100, hk_a),
        .init(seq_a, 9000, 3, &s64, 0x5000, 0x4000, withField(hk_a, "logits_out", @as(u64, 0x6100))),
        .init(seq_a, 9000, 3, &s64, 0x5000, 0x4000, withField(hk_a, "streams", @as(u64, 0x8100))),
        .init(seq_a, 9000, 3, &s64, 0x5000, 0x4000, withField(hk_a, "keep", @as(?*parts.Keep, &m_keep))),
        .init(withField(seq_a, "cap", @as(usize, 131072)), 9000, 3, &s64, 0x5000, 0x4000, hk_a),
        .init(seq_a, 9000, 3, &s128, 0x5000, 0x4000, hk_a),
        .init(withField(seq_a, "gen", @as(u64, 8)), 9000, 3, &s64, 0x5000, 0x4000, hk_a),
        .init(seq_a, 9000, 4, &s64, 0x5000, 0x4000, hk_a),
        .init(seq_a, 9300, 3, &s64, 0x5000, 0x4000, hk_a),
        .init(seq_a, 100, 3, &s64, 0x5000, 0x4000, hk_a),
        .init(seq_a, 9000, 3, &s64, 0x5000, 0x4000, withField(hk_a, "keep", @as(?*parts.Keep, null))),
    };
    for (mismatches, 0..) |mk, idx| {
        try runTag(&c, stream, k_a, 1);
        try runTag(&c, stream, k_b, 2);
        try expectNoReplay(&c, stream, mk, 100 + idx);
    }

    var hd: FakeDraft = .{};
    var hc: wg.Cache(wg.HeadKey) = .{};
    defer hc.clear(t.allocator);
    const h_a: wg.HeadKey = .init(&hd, 1, true, true, hd.streams.ptr, &s64, 0x5000);
    const h_b: wg.HeadKey = .init(&hd, 1, true, true, hd.out.ptr, &s64, 0x5000);
    try runTag(&hc, stream, h_a, 1);
    try runTag(&hc, stream, h_a, 1);
    try runTag(&hc, stream, h_b, 2);
    try runTag(&hc, stream, h_a, 1);
    try t.expectEqual(@as(u64, 1), hc.replays);
    const head_mismatches = [_]wg.HeadKey{
        .init(&hd, 1, true, true, hd.streams.ptr + 0x100, &s64, 0x5000),
        .init(&withField(hd, "out", Buf{ .ptr = hd.out.ptr + 0x100 }), 1, true, true, hd.streams.ptr, &s64, 0x5000),
        .init(&withField(hd, "dlogits", Buf{ .ptr = hd.dlogits.ptr + 0x100 }), 1, true, true, hd.streams.ptr, &s64, 0x5000),
        .init(&withField(hd, "cap", @as(usize, 131072)), 1, true, true, hd.streams.ptr, &s64, 0x5000),
        .init(&hd, 2, true, true, hd.streams.ptr, &s64, 0x5000),
        .init(&hd, 1, false, true, hd.streams.ptr, &s64, 0x5000),
        .init(&hd, 1, true, false, hd.streams.ptr, &s64, 0x5000),
    };
    for (head_mismatches, 0..) |mk, idx| {
        try runTag(&hc, stream, h_a, 1);
        try runTag(&hc, stream, h_b, 2);
        try expectNoReplay(&hc, stream, mk, 200 + idx);
    }

    var hmc: wg.Cache(wg.HeadManyKey) = .{};
    defer hmc.clear(t.allocator);
    const ds = [_]*const FakeDraft{&hd};
    const nx1, const nx2 = .{ [_][]const u32{&.{1}}, [_][]const u32{&.{ 1, 2 }} };
    const hm_a: wg.HeadManyKey = .init(&ds, &nx1, &.{hd.streams.ptr}, 1, true, &s64, 0x5000);
    const hm_b: wg.HeadManyKey = .init(&ds, &nx1, &.{hd.out.ptr}, 1, true, &s64, 0x5000);
    try runTag(&hmc, stream, hm_a, 1);
    try runTag(&hmc, stream, hm_a, 1);
    try runTag(&hmc, stream, hm_b, 2);
    try expectNoReplay(&hmc, stream, wg.HeadManyKey.init(&ds, &nx1, &.{hd.streams.ptr + 0x100}, 1, true, &s64, 0x5000), 250);
    try expectNoReplay(&hmc, stream, wg.HeadManyKey.init(&ds, &nx2, &.{hd.streams.ptr}, 2, true, &s64, 0x5000), 251);

    var sc: wg.Cache(wg.SharedKey) = .{};
    defer sc.clear(t.allocator);
    var lb_a: [2 * linear_n + 4 * attn_n]cuda.DeviceBuffer = undefined;
    var lb_b: [2 * linear_n + 4 * attn_n]cuda.DeviceBuffer = undefined;
    var lb_m: [2 * linear_n + 4 * attn_n]cuda.DeviceBuffer = undefined;
    const sh_a = wg.SharedKey.init(&.{laneOf(&rig.d, &seq_a, &lb_a)}, &s64, 0x5000, 0x4000, 0x6000, false);
    const sh_b = wg.SharedKey.init(&.{laneOf(&rig.d, &seq_b, &lb_b)}, &s64, 0x5000, 0x4000, 0x6000, false);
    try runTag(&sc, stream, sh_a, 1);
    try runTag(&sc, stream, sh_a, 1);
    try runTag(&sc, stream, sh_b, 2);
    try expectNoReplay(&sc, stream, wg.SharedKey.init(&.{laneOf(&rig.d, &m_state, &lb_m)}, &s64, 0x5000, 0x4000, 0x6000, false), 260);
    try expectNoReplay(&sc, stream, wg.SharedKey.init(&.{laneOf(&rig.d, &seq_a, &lb_a)}, &s64, 0x5000, 0x4000, 0x6000, true), 261);

    var rc: wg.Cache(wg.RollbackKey) = .{};
    defer rc.clear(t.allocator);
    const r_a: wg.RollbackKey = .init(seq_a, &hd, 2, &s64);
    const r_b: wg.RollbackKey = .init(seq_b, &hd, 2, &s64);
    try runTag(&rc, stream, r_a, 1);
    try runTag(&rc, stream, r_a, 1);
    try runTag(&rc, stream, r_b, 2);
    try expectNoReplay(&rc, stream, wg.RollbackKey.init(seq_a, &hd, 3, &s64), 301);
    try expectNoReplay(&rc, stream, wg.RollbackKey.init(m_state, &hd, 2, &s64), 302);
}

test "window graphs: held executables never exceed 128 across two-way slots and clear frees all" {
    var rig: Rig = .{};
    const stream = rig.init();
    defer rig.deinit();
    var c: wg.Cache(wg.Key) = .{};
    defer c.clear(t.allocator);

    for (0..64) |slot_i| {
        const sh: u64 = 100 + slot_i;
        try runTag(&c, stream, key(1, 3, sh), 1);
        try runTag(&c, stream, key(1, 3, sh), 1);
        try runTag(&c, stream, key(2, 3, sh), 2);
        try t.expect(c.held <= 128);
    }
    try t.expectEqual([2]usize{ 128, 0 }, [2]usize{ c.held, Fake.st.destroyed });
    try runTag(&c, stream, key(1, 3, 999), 9); // 129th past 64x2 clears and runs eagerly
    try t.expectEqual([2]usize{ 0, 128 }, [2]usize{ c.held, Fake.st.destroyed });

    for (0..128) |slot_i| {
        const sh: u64 = 200 + slot_i;
        try runTag(&c, stream, key(1, 3, sh), 1);
        try runTag(&c, stream, key(1, 3, sh), 1);
        try t.expect(c.held <= 128);
    }
    try t.expectEqual(@as(usize, 128), c.held);
    try runTag(&c, stream, key(2, 3, 200), 2); // second way at 128 held clears instead of exceeding 128
    try t.expectEqual(@as(usize, 0), c.held);
    try t.expectEqual(Fake.st.instantiates, Fake.st.destroyed);
    c.clear(t.allocator);
}

test "window graph keys: a length, a request or a lane moves in place; the score grid and outputs are slots" {
    const lane0: Seq = .at(1, 0x1000_0000);
    const lane1: Seq = .at(2, 0x2000_0000);
    const hooks: parts.Hooks = .{ .logits = .all, .logits_out = 0x6000, .host_logits = false };
    const base = windowKey(lane0, 9000, 3, hooks);
    const same = [_]wg.Key{
        windowKey(lane0, 9000 + 256, 3, hooks), // 256 tokens on: another score grid size
        windowKey(lane1, 9000, 3, hooks), // the next request, on another lane's buffers
        windowKey(lane0, 9000, 3, .{ .logits = .all, .logits_out = 0x7000, .host_logits = false }),
    };
    for (same) |k| {
        try t.expect(!std.meta.eql(k, base));
        try t.expectEqual(wg.slotOf(base), wg.slotOf(k));
    }
    try t.expectEqual(base, windowKey(lane0, 9000, 3, hooks));
    const other = [_]wg.Key{
        windowKey(lane0, 100, 3, hooks), // no score grid yet: other kernels
        windowKey(lane0, 9000, 4, hooks),
        windowKey(lane0, 9000, 3, .{ .logits = .all, .logits_out = 0x6000, .streams = 0x8000, .host_logits = false }),
    };
    for (other) |k| try t.expect(!std.meta.eql(wg.slotOf(base), wg.slotOf(k)));
    var kept: parts.Keep = .{};
    kept.layers[0].proj = 0x9000;
    const with_keep = windowKey(lane0, 9000, 3, .{ .logits = .all, .logits_out = 0x6000, .keep = &kept, .host_logits = false });
    try t.expect(!std.meta.eql(wg.slotOf(base), wg.slotOf(with_keep)));
    kept.layers[0].proj = 0xa000;
    try t.expectEqual(wg.slotOf(with_keep), wg.slotOf(windowKey(lane0, 9000, 3, .{ .logits = .all, .logits_out = 0x6000, .keep = &kept, .host_logits = false })));
}

const hooks0: parts.Hooks = .{ .logits = .all, .logits_out = 0x6000, .host_logits = false };

/// The key's value of `seq` after `f` moved one of its buffers: another key, the same slot.
fn expectMoves(seq: Seq, comptime f: []const u8, i: usize) !void {
    const base = windowKey(seq, 9000, 3, hooks0);
    var moved = seq;
    if (comptime std.mem.eql(u8, f, "tail")) moved.tail.ptr += 0x100 else @field(moved, f)[i].ptr += 0x100;
    const k = windowKey(moved, 9000, 3, hooks0);
    try t.expect(!std.meta.eql(k, base));
    try t.expectEqual(wg.slotOf(base), wg.slotOf(k));
}

test "window graph keys: any layer's state, conv window or cache at a new address is a new key on the same slot" {
    const q: Seq = .at(7, 0x1000_0000);
    for (0..linear_n) |i| {
        try expectMoves(q, "state", i);
        try expectMoves(q, "conv", i);
    }
    for (0..attn_n) |i| {
        inline for (.{ "kc", "vc", "ikc", "pooled" }) |f| try expectMoves(q, f, i);
    }
    try expectMoves(q, "tail", 0);
    var paged = q;
    paged.pg = .{ .pt = 0x7000_0000, .ps = 4, .ptok = 256 };
    try t.expect(!std.meta.eql(windowKey(q, 9000, 3, hooks0), windowKey(paged, 9000, 3, hooks0)));
}

test "window graph keys: equal tables give equal keys; a kept window's swapped states another, swapped back the first" {
    const a: Seq = .at(7, 0x1000_0000);
    var b: Seq = .at(7, 0x1000_0000);
    try t.expectEqual(windowKey(a, 9000, 3, hooks0), windowKey(b, 9000, 3, hooks0));
    // the states and a draft's spares trade places (Draft.settle), all layers at once
    var spare: [linear_n]Buf = undefined;
    for (&spare, 0..) |*x, i| x.ptr = 0x3000_0000 + i * 0x10000;
    for (&b.state, &spare) |*x, *y| std.mem.swap(Buf, x, y);
    const swapped = windowKey(b, 9000, 3, hooks0);
    try t.expect(!std.meta.eql(windowKey(a, 9000, 3, hooks0), swapped));
    for (&b.state, &spare) |*x, *y| std.mem.swap(Buf, x, y);
    try t.expectEqual(windowKey(a, 9000, 3, hooks0), windowKey(b, 9000, 3, hooks0));
}

var history: [2]i64 = .{ 0, 0 };

/// A shared pass's lane over device buffers at `seq`'s addresses (a fake driver: nothing is called).
fn laneOf(d: *const cuda.Driver, seq: *const Seq, bufs: *[2 * linear_n + 4 * attn_n]cuda.DeviceBuffer) parts.Lane {
    var n: usize = 0;
    inline for (.{ "conv", "state", "kc", "vc", "ikc", "pooled" }) |f| {
        for (@field(seq, f)) |b| {
            bufs[n] = .{ .d = d, .ptr = b.ptr, .len = 0, .borrowed = true };
            n += 1;
        }
    }
    const tail: cuda.DeviceBuffer = .{ .d = d, .ptr = seq.tail.ptr, .len = 0, .borrowed = true };
    const l, const s, const kv = .{ linear_n, 2 * linear_n, 2 * linear_n + attn_n };
    return .{ .seq = .{ .gen = seq.gen, .conv = bufs[0..l], .state = bufs[l..s], .kc = bufs[s..kv], .vc = bufs[kv..][0..attn_n], .ikc = bufs[kv + attn_n ..][0..attn_n], .pooled = bufs[kv + 2 * attn_n ..][0..attn_n], .tail = tail, .history = &history, .cap = seq.cap, .scores = 0, .pg = seq.pg }, .pos = 9000, .rows = 3 };
}

fn sharedKey(lanes: []const parts.Lane) wg.SharedKey {
    const s = scratchOf(0x3000, 64);
    return .init(lanes, &s, 0x5000, 0x4000, 0x6000, false);
}

test "shared window keys: any lane's layer at a new address is a new key on the same slot; equal tables equal keys" {
    var d: cuda.Driver = undefined;
    const q0: Seq = .at(1, 0x1000_0000);
    const q1: Seq = .at(2, 0x2000_0000);
    var b0: [2 * linear_n + 4 * attn_n]cuda.DeviceBuffer = undefined;
    var b1: [2 * linear_n + 4 * attn_n]cuda.DeviceBuffer = undefined;
    var lanes = [_]parts.Lane{ laneOf(&d, &q0, &b0), laneOf(&d, &q1, &b1) };
    const base = sharedKey(&lanes);
    var c0: [2 * linear_n + 4 * attn_n]cuda.DeviceBuffer = undefined;
    var c1: [2 * linear_n + 4 * attn_n]cuda.DeviceBuffer = undefined;
    try t.expectEqual(base, sharedKey(&[_]parts.Lane{ laneOf(&d, &q0, &c0), laneOf(&d, &q1, &c1) }));
    for ([_]*[2 * linear_n + 4 * attn_n]cuda.DeviceBuffer{ &b0, &b1 }) |bufs| {
        for (bufs) |*b| {
            b.ptr += 0x100;
            const k = sharedKey(&lanes);
            b.ptr -= 0x100;
            try t.expect(!std.meta.eql(k, base));
            try t.expectEqual(wg.slotOf(base), wg.slotOf(k));
        }
    }
    try t.expectEqual(base, sharedKey(&lanes));
}
