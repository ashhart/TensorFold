//! Kolibri-1 on the real checkpoint: one-row steps, the lane rounds, shared rounds, prompt cuts and kept states.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const kolibri1 = @import("kolibri1");
const check = @import("check.zig");
const layer_tests = @import("kolibri_layer_tests.zig");
const Gpu = check.Gpu;

const fwd = kolibri1.forward;
const Rig = layer_tests.Rig;
const readIds = layer_tests.readIds;
/// Rows `logits` writes: 2,048, as the quality runs read them.
const quality_rows = 2048;

fn hash(tokens: []const u32) u64 {
    return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(tokens));
}

fn argmax(x: []const f32) u32 {
    var best: usize = 0;
    for (x, 0..) |v, i| if (v > x[best]) {
        best = i;
    };
    return @intCast(best);
}

const Case = struct { prompt: []const u32, sampling: ?lanes.Sampling = null, drafts: bool };
const Run = struct { out: [][]u32, rate: f64, rounds: u64, accepted: u64 };

/// Cases admitted together into one lane engine on the backend, run to the end: tokens, rate, rounds, drafts landed.
fn lanesRun(gpu: Gpu, kb: *kolibri1.backend.Kolibri, cases: []const Case, count: u32) !Run {
    const gpa = gpu.gpa;
    var cfg = try lanes.Config.init(gpa, kb.facts(), kolibri1.backend.max_window, kolibri1.backend.max_window - 1);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(gpa, &cfg, kb.backend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(lanes.Stream, cases.len);
    defer gpa.free(streams);
    const props = try gpa.alloc(lanes.SuffixLookup, cases.len);
    defer gpa.free(props);
    for (cases, streams, props) |c, *s, *p| {
        p.* = try lanes.SuffixLookup.init(gpa, .{ .min_match = 4 });
        s.* = try lanes.Stream.init(gpa, .{ .id = "k", .prompt = c.prompt, .max_new = count, .sampling = c.sampling, .drafts = c.drafts, .proposer = p.proposer() });
    }
    defer for (streams, props) |*s, *p| {
        s.deinit(gpa);
        p.deinit();
    };
    for (streams) |*s| try engine.addStream(s);
    const t0 = check.now(gpu.io);
    while (engine.activeCount() > 0) try engine.step();
    const secs = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e9;
    var run: Run = .{ .out = try gpa.alloc([]u32, cases.len), .rate = 0, .rounds = 0, .accepted = 0 };
    var n: usize = 0;
    for (run.out, streams) |*o, *s| {
        o.* = try gpa.dupe(u32, s.emitted());
        n += o.len;
        run.rounds += s.rounds;
        run.accepted += s.accepted;
    }
    run.rate = @as(f64, @floatFromInt(n)) / secs;
    return run;
}

fn freeRun(gpa: std.mem.Allocator, r: Run) void {
    for (r.out) |o| gpa.free(o);
    gpa.free(r.out);
}

/// Greedy tokens by one eager decode row at a time (the plain reference every other path must equal).
fn eager(r: *Rig, prompt: []const u32, count: usize) ![]u32 {
    const out = try r.gpu.gpa.alloc(u32, count);
    const first = try r.prompt(0, prompt);
    out[0] = argmax(first);
    r.gpu.gpa.free(first);
    for (1..count) |i| {
        const row = try r.decode(0, prompt.len + i - 1, out[i - 1]);
        defer r.gpu.gpa.free(row);
        out[i] = argmax(row);
    }
    return out;
}

/// Each prompt (JSON id lists) decoded greedy for `count` tokens: ids and the top-2 margin of each step, as JSON lines.
pub fn greedy(gpu: Gpu, model: []const u8, prompts: []const u8, count_text: []const u8, out_path: []const u8) !void {
    const gpa = gpu.gpa;
    const count = try std.fmt.parseInt(usize, count_text, 10);
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, prompts, gpa, .limited(1 << 26));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice([][]u32, gpa, text, .{});
    defer parsed.deinit();
    var rig: Rig = undefined;
    try rig.init(gpu, model, null, 8192, 1, fwd.prompt_rows);
    defer rig.deinit();
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    for (parsed.value) |p| {
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(gpa);
        try out.writer.writeAll("{\"steps\": [");
        var row = try rig.prompt(0, p);
        for (0..count) |i| {
            const best = argmax(row);
            var second: f32 = -std.math.inf(f32);
            for (row, 0..) |v, j| if (j != best and v > second) {
                second = v;
            };
            try out.writer.print("{s}[{d}, {d}]", .{ if (i == 0) "" else ", ", best, row[best] - second });
            try ids.append(gpa, best);
            gpa.free(row);
            row = if (i + 1 < count) try rig.decode(0, p.len + i, best) else &.{};
        }
        try out.writer.print("], \"hash\": \"{x}\"}}\n", .{hash(ids.items)});
    }
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = out_path, .data = out.written() });
    std.debug.print("RESULT kolibri1 greedy: {d} prompts, {d} tokens each, to {s}\n", .{ parsed.value.len, count, out_path });
}

/// The full model: eager rows == captured steps == lane rounds, drafted == plain, shared rounds == solo, cuts, states.
pub fn full(gpu: Gpu, model: []const u8, ids_file: []const u8, count_text: []const u8) !void {
    const gpa = gpu.gpa;
    const count = try std.fmt.parseInt(u32, count_text, 10);
    const ids = try readIds(gpu, ids_file);
    defer gpa.free(ids);
    var rig: Rig = undefined;
    try rig.init(gpu, model, null, 20480, 4, fwd.prompt_rows);
    defer rig.deinit();
    var kb = try kolibri1.backend.Kolibri.init(gpa, gpu.d, &rig.m, &rig.k, rig.s);
    defer kb.deinit();
    var bad: usize = 0;
    // a short prompt, and a long one whose second half repeats its first (copy drafts land there)
    const short = ids[0..200];
    const long = try gpa.alloc(u32, 2 * 1500);
    defer gpa.free(long);
    @memcpy(long[0..1500], ids[0..1500]);
    @memcpy(long[1500..], ids[0..1500]);
    for ([_][]const u32{ short, long }, [_][]const u8{ "short", "long" }) |p, name| {
        const plain = try eager(&rig, p, count);
        defer gpa.free(plain);
        const lone = try lanesRun(gpu, &kb, &.{.{ .prompt = p, .drafts = false }}, count);
        defer freeRun(gpa, lone);
        const drafted = try lanesRun(gpu, &kb, &.{.{ .prompt = p, .drafts = true }}, count);
        defer freeRun(gpa, drafted);
        const ok = std.mem.eql(u32, plain, lone.out[0]) and std.mem.eql(u32, plain, drafted.out[0]);
        std.debug.print("RESULT kolibri1 {s}: eager rows {x}, steps {x} ({d:.2} tok/s), drafted {x} ({d:.2} tok/s, {d} rounds, {d} drafts landed)\n", .{ name, hash(plain), hash(lone.out[0]), lone.rate, hash(drafted.out[0]), drafted.rate, drafted.rounds, drafted.accepted });
        if (!ok) bad += 1;
    }
    // sampled: drafted == plain at a seed; then four streams in shared rounds against each alone
    const sampled: lanes.Sampling = .{ .seed = 7, .temperature = 0.8, .top_k = 20, .top_p = 0.95 };
    const sp = try lanesRun(gpu, &kb, &.{.{ .prompt = long, .sampling = sampled, .drafts = false }}, count);
    defer freeRun(gpa, sp);
    const sd = try lanesRun(gpu, &kb, &.{.{ .prompt = long, .sampling = sampled, .drafts = true }}, count);
    defer freeRun(gpa, sd);
    std.debug.print("RESULT kolibri1 sampled: plain {x}, drafted {x}\n", .{ hash(sp.out[0]), hash(sd.out[0]) });
    if (!std.mem.eql(u32, sp.out[0], sd.out[0])) bad += 1;
    const four = [_]Case{ .{ .prompt = long, .drafts = true }, .{ .prompt = short, .sampling = sampled, .drafts = true }, .{ .prompt = ids[300..700], .drafts = true }, .{ .prompt = long[0 .. long.len - 7], .sampling = sampled, .drafts = true } };
    const together = try lanesRun(gpu, &kb, &four, count);
    defer freeRun(gpa, together);
    for (four, 0..) |c, i| {
        const alone = try lanesRun(gpu, &kb, &.{c}, count);
        defer freeRun(gpa, alone);
        const eq = std.mem.eql(u32, together.out[i], alone.out[0]);
        std.debug.print("RESULT kolibri1 shared rounds: stream {d} {x} together, {x} alone\n", .{ i, hash(together.out[i]), hash(alone.out[0]) });
        if (!eq) bad += 1;
    }
    std.debug.print("RESULT kolibri1 shared rounds: {d:.2} tok/s for four streams\n", .{together.rate});
    bad += try cuts(&rig, ids);
    bad += try states(&rig, &kb, ids, count);
    try check.expect(bad == 0, "kolibri1 full: {d} paths differ from the plain rows", .{bad});
    check.pass("kolibri1 full: steps, drafted, sampled, shared rounds, prompt cuts and kept states equal the plain rows ({d} tokens each)", .{count});
}

/// 5,000 and 20,000 rows' last logits on the pass grid against 1,000-, 1,536- and 2,048-row chunks: same bytes?
fn cuts(r: *Rig, ids: []const u32) !usize {
    const gpa = r.gpu.gpa;
    if (ids.len < 20000) return error.TooFewIds;
    var bad: usize = 0;
    for ([_]usize{ 5000, 20000 }) |len| {
        const p = ids[0..len];
        const grid = try r.prompt(0, p);
        defer gpa.free(grid);
        for ([_]usize{ 1000, 1536, 2048 }) |chunk| {
            var at: usize = 0;
            while (at < p.len) : (at += chunk) {
                const end = @min(p.len, at + chunk);
                try r.m.forward(r.ops(), &.{.{ .slot = 1, .p0 = at, .tokens = p[at..end] }}, true, null, r.logits.ptr);
            }
            const got = try r.read(1);
            defer gpa.free(got);
            const eq = std.mem.eql(u8, std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(grid));
            var most: f32 = 0;
            for (got, grid) |g, w| most = @max(most, @abs(g - w));
            std.debug.print("RESULT kolibri1 cuts: {d} rows in {d}-row chunks against {d}-row passes: {s} (largest difference {e:.3})\n", .{ len, chunk, fwd.prompt_rows, if (eq) "same bytes" else "differ", most });
            if (!eq) bad += 1;
        }
    }
    return bad;
}

/// States kept at 2,048, 3,001 and 4,096 by one stream, resumed by another: greedy replies against fresh.
fn states(r: *Rig, kb: *kolibri1.backend.Kolibri, ids: []const u32, count: u32) !usize {
    const gpa = r.gpu.gpa;
    const p = ids[0..5000];
    const fresh = try lanesRun(r.gpu, kb, &.{.{ .prompt = p, .drafts = true }}, count);
    defer freeRun(gpa, fresh);
    var bad: usize = 0;
    for ([_]u32{ 2048, 3001, 4096 }) |at| {
        // keep the state through the backend's own path: a pass marked at `at`, its hook saving it
        const Keep = struct {
            kb: *kolibri1.backend.Kolibri,
            saved: ?*kolibri1.snap.Saved = null,
            fn hook(ptr: *anyopaque, s: *lanes.Stream, mark: u32) void {
                const k: *@This() = @ptrCast(@alignCast(ptr));
                k.saved = k.kb.saveState(s, mark) catch null;
            }
        };
        var keep: Keep = .{ .kb = kb };
        var marks = [_]u32{at};
        var s = try lanes.Stream.init(gpa, .{ .id = "keep", .prompt = p[0 .. at + 100], .max_new = 1, .drafts = false, .reuse = .{ .marks = &marks, .hook = .{ .ptr = &keep, .at = Keep.hook } } });
        defer s.deinit(gpa);
        const be = kb.backend();
        try be.prefill(&s);
        be.release(&s);
        const saved = keep.saved orelse return error.NothingKept;
        defer kb.dropState(saved);
        // a second stream resumes from it, prefilling the rest of a longer prompt
        var lookup = try lanes.SuffixLookup.init(gpa, .{ .min_match = 4 });
        defer lookup.deinit();
        var t = try lanes.Stream.init(gpa, .{ .id = "resume", .prompt = p, .max_new = count, .drafts = true, .proposer = lookup.proposer(), .reuse = .{ .saved = saved, .at = at } });
        defer t.deinit(gpa);
        var cfg = try lanes.Config.init(gpa, kb.facts(), kolibri1.backend.max_window, kolibri1.backend.max_window - 1);
        defer cfg.deinit(gpa);
        var clock: lanes.fake.FixedClock = .{};
        var engine = lanes.Engine.init(gpa, &cfg, be, clock.clock());
        defer engine.deinit();
        try engine.addStream(&t);
        while (engine.activeCount() > 0) try engine.step();
        const eq = std.mem.eql(u32, t.emitted(), fresh.out[0]) and t.cached == at and !t.reuse_failed;
        std.debug.print("RESULT kolibri1 kept state at {d}: {d} cached, reply {x}, fresh {x}\n", .{ at, t.cached, hash(t.emitted()), hash(fresh.out[0]) });
        if (!eq) bad += 1;
    }
    return bad;
}

/// Teacher-forced fp32 logits of the first `quality_rows` rows to `out_path`; with `prompts`, each one's last row too.
pub fn logits(gpu: Gpu, model: []const u8, ids_file: []const u8, out_path: []const u8, prompts: ?[]const u8) !void {
    const gpa = gpu.gpa;
    const ids = try readIds(gpu, ids_file);
    defer gpa.free(ids);
    const n = @min(ids.len, quality_rows);
    var rig: Rig = undefined;
    try rig.init(gpu, model, null, 4096, 1, fwd.prompt_rows);
    defer rig.deinit();
    const rows = try gpa.alloc(usize, n);
    defer gpa.free(rows);
    for (rows, 0..) |*x, i| x.* = i;
    try rig.m.forward(rig.ops(), &.{.{ .slot = 0, .p0 = 0, .tokens = ids[0..n] }}, true, rows, rig.logits.ptr);
    const host = try rig.read(n);
    defer gpa.free(host);
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = out_path, .data = std.mem.sliceAsBytes(host) });
    std.debug.print("RESULT kolibri1 logits: {d} rows x {d} to {s}\n", .{ n, rig.w.config.vocab, out_path });
    const file = prompts orelse return;
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, file, gpa, .limited(1 << 26));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice([][]u32, gpa, text, .{});
    defer parsed.deinit();
    var last: std.ArrayList(f32) = .empty;
    defer last.deinit(gpa);
    for (parsed.value) |p| {
        const row = try rig.prompt(0, p);
        defer gpa.free(row);
        try last.appendSlice(gpa, row);
    }
    const path = try std.fmt.allocPrint(gpa, "{s}.last", .{out_path});
    defer gpa.free(path);
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = path, .data = std.mem.sliceAsBytes(last.items) });
    std.debug.print("RESULT kolibri1 logits: the last rows of {d} prompts to {s}\n", .{ parsed.value.len, path });
}
