//! flashnext-exl3-graph eq and fork: device-plan and graph steps against the host cut, bit for bit.

const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("../../check.zig");
const Gpu = check.Gpu;
const prompt = flash.prompt;
const X = flash.exl3;
const dplan = X.dplan;
const graph = X.graph;
const vocab = prompt.vocabulary;
const harness = @import("flashnext_exl3_graph_engine.zig");
const Eng = harness.Eng;
const ms = harness.ms;
const ints = harness.ints;
const cap_default = harness.cap_default;

const modes = [_]graph.Mode{ .off, .dplan, .graph };

const Run = struct {
    ids: []u32,
    /// Per decode step, the hash of the step's whole bf16 logits row; then per window row the same.
    round_h: []u64,
    window_h: []u64,
    ms_token: f64 = 0,
    ms_window_row: f64 = 0,
    hit: f64 = 0,
    copies: f64 = 0,
};

const win_sizes = [_]usize{ 3, 1, 4, 2, 8, 16 };

fn runMode(g: *Eng, ids: []const u32, steps: usize, r: *Run) !void {
    const a = g.gpu.gpa;
    // decode rounds
    try dplan.toHost(&g.e.exl3_dplan);
    try g.reset();
    try g.forward(ids, .{ .logits = .last });
    var tok = prompt.argmax(g.seq.logits);
    r.ids[0] = tok;
    _ = try g.swap(1);
    const before = try g.stats();
    const t0 = check.now(g.gpu.io);
    for (0..steps) |i| {
        tok = try g.round(tok);
        r.ids[i + 1] = tok;
        r.round_h[i] = std.hash.Wyhash.hash(0, g.seq.logits);
    }
    const t1 = check.now(g.gpu.io);
    const after = try g.stats();
    r.ms_token = ms(t0, t1) / @as(f64, @floatFromInt(steps));
    const hits = after.hit_slots - before.hit_slots;
    const all = hits + after.miss_slots - before.miss_slots;
    r.hit = 100 * @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(@max(all, 1)));
    r.copies = @as(f64, @floatFromInt(after.misses - before.misses)) / @as(f64, @floatFromInt(steps));
    // the same continuation teacher-forced in windows of 3, 1, 4, 2 rows (every row's logits)
    try dplan.toHost(&g.e.exl3_dplan);
    try g.reset();
    try g.forward(ids, .{ .logits = .last });
    var at: usize = 0;
    var k: usize = 0;
    const host = try a.alloc(u16, 16 * vocab);
    defer a.free(host);
    const tw = check.now(g.gpu.io);
    while (at < steps) : (k += 1) {
        const n = @min(win_sizes[k % win_sizes.len], steps - at);
        try g.forward(r.ids[at..][0..n], .{ .logits = .all, .logits_out = g.out.ptr });
        try g.out.download(0, std.mem.sliceAsBytes(host[0 .. n * vocab]));
        for (0..n) |j| r.window_h[at + j] = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(host[j * vocab ..][0..vocab]));
        at += n;
    }
    r.ms_window_row = ms(tw, check.now(g.gpu.io)) / @as(f64, @floatFromInt(steps));
}

/// Two sequences in one decode round (two rows, each its own caches):
fn pairRun(g: *Eng, b: *prompt.Seq, ids_a: []const u32, ids_b: []const u32, steps: usize, out: []u64) !f64 {
    try dplan.toHost(&g.e.exl3_dplan);
    try g.reset();
    try g.forward(ids_a, .{ .logits = .last });
    for ([_][]cuda.DeviceBuffer{ b.conv, b.state, b.kc, b.vc, b.ikc, b.pooled }) |bs| for (bs) |x| try x.fill8(0, g.stream.handle);
    try b.tail.fill8(0, g.stream.handle);
    try g.stream.synchronize();
    b.fed = 0;
    b.history = .{ g.e.eos, g.e.eos };
    try g.e.forward(flash.Tri, g.tri(), b, ids_b, .{ .logits = .last });
    if (g.ranks) |r| try r.collective.finish(g.stream) else try g.stream.synchronize();
    var t = [2]u32{ prompt.argmax(g.seq.logits), prompt.argmax(b.logits) };
    const q = [_]*prompt.Seq{ &g.seq, b };
    const t0 = check.now(g.gpu.io);
    for (0..steps) |i| {
        try g.e.round(flash.Tri, g.tri(), &q, &t, g.out.ptr);
        if (g.ranks) |r| try r.collective.finish(g.stream);
        out[i] = std.hash.Wyhash.hash(0, g.seq.logits) ^ std.hash.Wyhash.hash(7, b.logits);
        t = .{ prompt.argmax(g.seq.logits), prompt.argmax(b.logits) };
    }
    return ms(t0, check.now(g.gpu.io)) / @as(f64, @floatFromInt(steps));
}

pub fn eqMode(gpu: Gpu, rank: usize, world: usize, port: u16, dir: []const u8, golden_path: []const u8, steps: usize, which: []const u8) !void {
    const a = gpu.gpa;
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, golden_path, a, .limited(1 << 26));
    defer a.free(text);
    var parsed = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    defer parsed.deinit();
    const g = try Eng.init(gpu, rank, world, port, dir);
    defer g.deinit();
    var all_ok = true;
    for (parsed.value.object.get("prompts").?.array.items) |p| {
        const ids = try ints(a, p.object.get("ids").?);
        defer a.free(ids);
        var runs: [modes.len]Run = undefined;
        for (&runs) |*r| r.* = .{ .ids = try a.alloc(u32, steps + 1), .round_h = try a.alloc(u64, steps), .window_h = try a.alloc(u64, steps) };
        defer for (&runs) |*r| {
            a.free(r.ids);
            a.free(r.round_h);
            a.free(r.window_h);
        };
        // warm the pools once (first-touch copies are not any mode's cost)
        graph.force = .off;
        try runMode(g, ids, steps, &runs[0]);
        for (modes, 0..) |m, mi| {
            if (std.mem.indexOf(u8, which, @tagName(m)) == null and m != .off) continue;
            graph.force = m;
            const g0 = graph.stats;
            try runMode(g, ids, steps, &runs[mi]);
            const g1 = graph.stats;
            const h = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(runs[mi].round_h)) ^ std.hash.Wyhash.hash(1, std.mem.sliceAsBytes(runs[mi].window_h));
            const other = try g.swap(h);
            if (world == 2) try check.expect(other == h, "rank {d}: the two ranks' logits differ ({s})", .{ rank, @tagName(m) });
            const rs = runs[mi];
            var same_r: usize = 0;
            var same_w: usize = 0;
            for (rs.round_h, runs[0].round_h) |x, y| same_r += @intFromBool(x == y);
            for (rs.window_h, runs[0].window_h) |x, y| same_w += @intFromBool(x == y);
            const ok = same_r == steps and same_w == steps and std.mem.eql(u32, rs.ids, runs[0].ids);
            if (mi != 0) all_ok = all_ok and ok;
            if (rank == 0) std.debug.print("RESULT tp={d} {s} {s}: decode {d:.2} ms/token (hit {d:.1}%, {d:.1} copies a token), windows {d:.2} ms/row; logits bit-equal to off: rounds {d}/{d}, window rows {d}/{d}; greedy ids {s}; graph captures {d} (updates {d}) replays {d}, eager-dplan steps {d}, declined {d}, capture {d:.1} ms\n", .{ world, p.object.get("name").?.string, @tagName(m), rs.ms_token, rs.hit, rs.copies, rs.ms_window_row, same_r, steps, same_w, steps, if (std.mem.eql(u32, rs.ids, runs[0].ids)) "equal" else "DIFFER", g1.captures - g0.captures, g1.updates - g0.updates, g1.replays - g0.replays, g1.eager_dplan - g0.eager_dplan, g1.declined - g0.declined, @as(f64, @floatFromInt(g1.capture_ns - g0.capture_ns)) / 1e6 });
        }
        if (std.mem.indexOf(u8, which, "inject") != null) {
            // captures fail on rank 1 and instantiations on rank 0: both ranks run those rounds eagerly, bits unchanged
            graph.clear();
            graph.fallback.clear();
            graph.force = .graph;
            graph.fallback.inject_capture_failures = if (rank == 1 or world == 1) 3 else 0;
            graph.fallback.inject_instantiate_failures = if (rank == 0) 2 else 0;
            var injected: Run = .{ .ids = try a.alloc(u32, steps + 1), .round_h = try a.alloc(u64, steps), .window_h = try a.alloc(u64, steps) };
            defer a.free(injected.ids);
            defer a.free(injected.round_h);
            defer a.free(injected.window_h);
            const before = graph.stats;
            try runMode(g, ids, steps, &injected);
            graph.fallback.inject_capture_failures = 0;
            graph.fallback.inject_instantiate_failures = 0;
            const h = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(injected.round_h)) ^ std.hash.Wyhash.hash(1, std.mem.sliceAsBytes(injected.window_h));
            if (world == 2) try check.expect(try g.swap(h) == h, "rank {d}: the two ranks' logits differ (injected capture failures)", .{rank});
            var same_r: usize = 0;
            var same_w: usize = 0;
            for (injected.round_h, runs[0].round_h) |x, y| same_r += @intFromBool(x == y);
            for (injected.window_h, runs[0].window_h) |x, y| same_w += @intFromBool(x == y);
            const failures = graph.stats.capture_failures - before.capture_failures;
            const ok = same_r == steps and same_w == steps and std.mem.eql(u32, injected.ids, runs[0].ids) and failures > 0;
            all_ok = all_ok and ok;
            if (rank == 0) std.debug.print("RESULT tp={d} injected failures: {d} captures/instantiations failed on this rank, {d} captures and {d} replays after; logits bit-equal to off: rounds {d}/{d}, window rows {d}/{d}\n", .{ world, failures, graph.stats.captures - before.captures, graph.stats.replays - before.replays, same_r, steps, same_w, steps });
        }
        graph.force = null;
    }
    // two streams in one round
    {
        const ps = parsed.value.object.get("prompts").?.array.items;
        const ia = try ints(a, ps[0].object.get("ids").?);
        defer a.free(ia);
        const ib = try ints(a, ps[ps.len - 1].object.get("ids").?);
        defer a.free(ib);
        var b = try g.e.newSeq(cap_default);
        defer b.deinit(a);
        const pair_steps = @min(steps, 32);
        var ref: [64]u64 = undefined;
        var got: [64]u64 = undefined;
        graph.force = .off;
        _ = try pairRun(g, &b, ia, ib, pair_steps, &ref);
        for (modes) |m| {
            if (std.mem.indexOf(u8, which, @tagName(m)) == null and m != .off) continue;
            graph.force = m;
            const t = try pairRun(g, &b, ia, ib, pair_steps, &got);
            var same: usize = 0;
            for (got[0..pair_steps], ref[0..pair_steps]) |x, y| same += @intFromBool(x == y);
            if (m != .off) all_ok = all_ok and same == pair_steps;
            const h = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(got[0..pair_steps]));
            if (world == 2) try check.expect(try g.swap(h) == h, "rank {d}: the two ranks' two-stream logits differ", .{rank});
            if (rank == 0) std.debug.print("RESULT tp={d} two streams {s}: {d:.2} ms a round of 2 rows; both rows' logits bit-equal to off in {d}/{d} rounds\n", .{ world, @tagName(m), t, same, pair_steps });
        }
        graph.force = null;
    }
    if (dplan.current(&g.e.exl3_dplan)) |dp| if (rank == 0) std.debug.print("RESULT dplan owner switches: {d} pushes, {d} pulls; {d} experts read in place (rounds with more misses than free lines)\n", .{ dp.pushes, dp.pulls, dp.inplace });
    try check.expect(all_ok, "rank {d}: a device-plan or graph step's logits differ from the host cut's", .{rank});
    check.pass("rank {d}/{d}: decode rounds and verify windows on the device plan and in graphs are bit-identical to the host cut ({d} steps a prompt)", .{ rank, world, steps });
}

/// A scheduling change's bits: every golden prompt's decode rounds and windows (runMode) under `which` (off, dplan).
pub fn forkMode(gpu: Gpu, rank: usize, world: usize, port: u16, dir: []const u8, golden_path: []const u8, steps: usize, which: []const u8) !void {
    const a = gpu.gpa;
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, golden_path, a, .limited(1 << 26));
    defer a.free(text);
    var parsed = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    defer parsed.deinit();
    const g = try Eng.init(gpu, rank, world, port, dir);
    defer g.deinit();
    graph.force = if (std.mem.eql(u8, which, "graph")) .graph else if (std.mem.eql(u8, which, "dplan")) .dplan else .off;
    defer graph.force = null;
    defer X.fork.force = null;
    var ok = true;
    for (parsed.value.object.get("prompts").?.array.items) |p| {
        const ids = try ints(a, p.object.get("ids").?);
        defer a.free(ids);
        var runs: [3]Run = undefined;
        for (&runs) |*r| r.* = .{ .ids = try a.alloc(u32, steps + 1), .round_h = try a.alloc(u64, steps), .window_h = try a.alloc(u64, steps) };
        defer for (&runs) |*r| {
            a.free(r.ids);
            a.free(r.round_h);
            a.free(r.window_h);
        };
        for ([_]bool{ false, false, true }, 0..) |on, i| {
            X.fork.force = on;
            try runMode(g, ids, steps, &runs[i]);
        }
        const same = std.mem.eql(u64, runs[1].round_h, runs[2].round_h) and std.mem.eql(u64, runs[1].window_h, runs[2].window_h) and std.mem.eql(u32, runs[1].ids, runs[2].ids);
        ok = ok and same;
        if (world == 2) {
            const h = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(runs[2].round_h));
            try check.expect(try g.swap(h) == h, "rank {d}: the ranks differ", .{rank});
        }
        if (rank == 0) std.debug.print("RESULT tp={d} {s} {s}: decode {d:.2} -> {d:.2} ms/token, windows {d:.2} -> {d:.2} ms/row (serial -> parallel parts); logits {s}\n", .{ world, which, p.object.get("name").?.string, runs[1].ms_token, runs[2].ms_token, runs[1].ms_window_row, runs[2].ms_window_row, if (same) "bit-equal" else "DIFFER" });
    }
    try check.expect(ok, "rank {d}: parallel parts changed the logits", .{rank});
    check.pass("rank {d}/{d}: parallel projection parts give the same bits ({s})", .{ rank, world, which });
}
