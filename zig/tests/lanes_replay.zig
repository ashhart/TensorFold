//! Replays tools/zig/record_lanes.py model traces: the same draws and times must give Python's log, line for line.
const std = @import("std");
const lanes = @import("lanes");
const fx = @import("fixture.zig");
const be = lanes.backend;

const gpa = std.testing.allocator;

/// The recorded inputs, served in the order the loop asks for them.
const Replay = struct {
    firsts: []u32,
    queued: []u32,
    rounds: []Round,
    costs: []f64,
    overheads: []f64,
    at: struct { first: usize = 0, queue: usize = 0, round: usize = 0, cost: usize = 0, overhead: usize = 0 } = .{},
    drawn: std.ArrayList(u32) = .empty,
    a: std.mem.Allocator,

    const Round = struct { sampled: []u32, window: []u32 };

    fn init(a: std.mem.Allocator, events: []const fx.Line) !Replay {
        var firsts: std.ArrayList(u32) = .empty;
        var queued: std.ArrayList(u32) = .empty;
        var rounds: std.ArrayList(Round) = .empty;
        var costs: std.ArrayList(f64) = .empty;
        var overheads: std.ArrayList(f64) = .empty;
        for (events) |l| {
            const kind = fx.str(fx.get(l.value, "ev"));
            if (std.mem.eql(u8, kind, "first")) {
                try firsts.append(a, @intCast(fx.int(fx.get(l.value, "drawn"))));
            } else if (std.mem.eql(u8, kind, "queue")) {
                try queued.append(a, @intCast(fx.int(fx.get(l.value, "token"))));
            } else if (std.mem.eql(u8, kind, "round")) {
                try rounds.append(a, .{ .sampled = try fx.u32s(a, fx.get(l.value, "sampled")), .window = try fx.u32s(a, fx.get(l.value, "window")) });
            } else if (std.mem.eql(u8, kind, "cost")) {
                try costs.append(a, fx.float(fx.get(l.value, "ms")));
            } else if (std.mem.eql(u8, kind, "overhead")) {
                try overheads.append(a, fx.float(fx.get(l.value, "ms")));
            }
        }
        return .{ .firsts = firsts.items, .queued = queued.items, .rounds = rounds.items, .costs = costs.items, .overheads = overheads.items, .a = a };
    }

    fn self(ptr: *anyopaque) *Replay {
        return @ptrCast(@alignCast(ptr));
    }

    fn take(r: *Replay, list: []u32, at: *usize) !u64 {
        if (at.* >= list.len) return error.TraceExhausted;
        try r.drawn.append(r.a, list[at.*]);
        at.* += 1;
        return r.drawn.items.len - 1;
    }

    fn backend(r: *Replay) be.Backend {
        return .{ .ptr = r, .vtable = &.{ .prefill = prefill, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release } };
    }

    fn prefill(_: *anyopaque, _: *lanes.Stream) anyerror!void {}
    fn keep(_: *anyopaque, _: []const be.Window, _: []const []const u32) anyerror!void {}
    fn draft(_: *anyopaque, _: []const be.DraftRequest) anyerror!void {}
    fn release(_: *anyopaque, _: *lanes.Stream) void {}

    fn first(ptr: *anyopaque, _: *lanes.Stream, _: u64) anyerror!u64 {
        const r = self(ptr);
        return r.take(r.firsts, &r.at.first);
    }

    fn queue(ptr: *anyopaque, _: *lanes.Stream, _: be.Feed, _: u64) anyerror!u64 {
        const r = self(ptr);
        return r.take(r.queued, &r.at.queue);
    }

    fn read(ptr: *anyopaque, handle: u64) anyerror!u32 {
        return self(ptr).drawn.items[handle];
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const r = self(ptr);
        for (windows, out) |w, o| {
            if (r.at.round >= r.rounds.len) return error.TraceExhausted;
            const got = r.rounds[r.at.round];
            r.at.round += 1;
            if (got.sampled.len != w.rows()) return error.WindowWidthDiffers;
            @memcpy(o.sampled, got.sampled);
            @memcpy(o.drafts, got.window[1..]);
        }
    }

    fn clock(r: *Replay) be.Clock {
        return .{ .ptr = r, .vtable = &.{ .start = start, .elapsed_ms = elapsed } };
    }

    fn start(_: *anyopaque) void {}

    fn elapsed(ptr: *anyopaque, mark: be.Mark) f64 {
        const r = self(ptr);
        const list, const at = switch (mark) {
            .cost => .{ r.costs, &r.at.cost },
            .overhead => .{ r.overheads, &r.at.overhead },
        };
        if (at.* >= list.len) return std.math.nan(f64);
        at.* += 1;
        return list[at.* - 1];
    }
};

fn model(a: std.mem.Allocator, m: fx.Value) !lanes.Model {
    const costs = struct {
        fn of(al: std.mem.Allocator, v: fx.Value) ![]lanes.config.Cost {
            const ps = try fx.pairs(al, v);
            const out = try al.alloc(lanes.config.Cost, ps.len);
            for (out, ps) |*o, p| o.* = .{ .width = @intCast(p.key), .ms = p.value };
            return out;
        }
    }.of;
    return .{
        .exact_width = @intCast(fx.int(fx.get(m, "exact_width"))),
        .first_copy_rows = @intCast(fx.int(fx.get(m, "first_copy_rows"))),
        .gpu_tokens = fx.boolean(fx.get(m, "gpu_tokens")),
        .mtp = fx.boolean(fx.get(m, "mtp")),
        .speculate = fx.boolean(fx.get(m, "speculate")),
        .speculate_early = fx.boolean(fx.get(m, "speculate_early")),
        .draft_prior = try fx.floats(a, fx.get(m, "draft_prior")),
        .plain_guard = fx.boolean(fx.get(m, "plain_guard")),
        .drafts = @intCast(fx.int(fx.get(m, "drafts"))),
        .window_costs = try costs(a, fx.get(m, "window_costs")),
        .mtp_step_ms = fx.float(fx.get(m, "mtp_step_ms")),
        .streams_exact = fx.boolean(fx.get(m, "streams_exact")),
        .hidden_rows = fx.boolean(fx.get(m, "hidden_rows")),
        .batch_rows = @intCast(fx.int(fx.get(m, "batch_rows"))),
        .max_streams = @intCast(fx.int(fx.get(m, "max_streams"))),
        .shared_costs = try costs(a, fx.get(m, "shared_costs")),
        .draft_probabilities = fx.boolean(fx.get(m, "draft_probabilities")),
        .draft_streams = fx.boolean(fx.get(m, "draft_streams")),
    };
}

/// Python's derived constants must be what the Zig setup derives from the same attributes.
fn checkDerived(a: std.mem.Allocator, cfg: *const lanes.Config, d: fx.Value, k: fx.Value) !void {
    const t = std.testing;
    try t.expectEqual(fx.int(fx.get(d, "family_width")), cfg.family_width);
    try t.expectEqual(fx.int(fx.get(d, "max_copy")), cfg.max_copy);
    try t.expectEqual(fx.int(fx.get(d, "first_copy")), cfg.first_copy);
    try t.expectEqual(fx.int(fx.get(d, "base_width")), cfg.base_width);
    try t.expectEqual(fx.boolean(fx.get(d, "family_mtp")), cfg.family_mtp);
    try t.expectEqual(fx.boolean(fx.get(d, "speculate_early")), cfg.speculate_early);
    try t.expectEqual(fx.boolean(fx.get(d, "pipelined")), cfg.pipelined);
    try t.expectEqual(fx.boolean(fx.get(d, "plain_guard")), cfg.plain_guard);
    try t.expectEqual(fx.int(fx.get(d, "most_drafts")), cfg.most_drafts);
    try t.expectEqual(fx.boolean(fx.get(d, "family_streams")), cfg.family_streams);
    try t.expectEqual(fx.int(fx.get(d, "batch_rows")), cfg.batch_rows);
    try t.expectEqual(fx.int(fx.get(d, "batch_streams")), cfg.batch_streams);
    try t.expectEqual(fx.boolean(fx.get(d, "node_probabilities")), cfg.node_probabilities);
    try t.expectEqualSlices(f64, try fx.floats(a, fx.get(d, "depth_prior")), cfg.depth_prior);
    try t.expectEqual(fx.float(fx.get(d, "mtp_step_ms")), cfg.mtp_step_ms);
    try fx.expectTable(a, fx.get(d, "family_costs"), cfg.family_costs);
    try fx.expectTable(a, fx.get(d, "shared_costs"), cfg.shared_costs);
    try t.expectEqual(fx.int(fx.get(k, "enter_match")), cfg.enter_match);
    try t.expectEqual(fx.float(fx.get(k, "copy_rate")), cfg.copy_rate);
    try t.expectEqual(fx.float(fx.get(k, "depth_rate")), cfg.k.depth_rate);
    try t.expectEqual(fx.int(fx.get(k, "depth_probe_every")), @as(i64, @intCast(cfg.k.depth_probe_every)));
    try t.expectEqual(fx.float(fx.get(k, "cost_rate")), cfg.k.cost_rate);
    try t.expectEqual(fx.float(fx.get(k, "plain_margin")), cfg.k.plain_margin);
    try t.expectEqual(fx.int(fx.get(k, "plain_wait_most")), @as(i64, @intCast(cfg.k.plain_wait_most)));
    try t.expectEqual(fx.int(fx.get(k, "draft_slack")), cfg.k.draft_slack);
    try t.expectEqualSlices(f64, try fx.floats(a, fx.get(k, "default_prior")), &lanes.config.default_prior);
}

/// Replay one trace; the count of matching lines (every line must match).
fn replay(name: []const u8) !usize {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const text = try fx.read(a, name);
    const all = try fx.lines(a, text);
    const head = all[0].value;
    const events = all[1..];
    const settings = fx.get(head, "engine");
    var cfg = try lanes.Config.init(gpa, try model(a, fx.get(head, "model")), @intCast(fx.int(fx.get(settings, "max_rows"))), @intCast(fx.int(fx.get(settings, "max_draft"))));
    defer cfg.deinit(gpa);
    try checkDerived(a, &cfg, fx.get(head, "derived"), fx.get(head, "constants"));

    const specs = fx.get(head, "streams").array.items;
    const streams = try a.alloc(lanes.Stream, specs.len);
    const proposers = try a.alloc(lanes.SuffixLookup, specs.len);
    var made: usize = 0;
    defer for (streams[0..made], proposers[0..made]) |*s, *p| {
        s.deinit(gpa);
        p.deinit();
    };
    for (specs, streams, proposers) |spec, *s, *p| {
        const sp = fx.get(spec, "sampling");
        const pp = fx.get(spec, "proposer");
        p.* = try lanes.SuffixLookup.init(gpa, if (fx.isNull(pp)) .{} else .{
            .ngram = @intCast(fx.int(fx.get(pp, "ngram"))),
            .min_match = fx.int(fx.get(pp, "min_match")),
            .max_extension = fx.int(fx.get(pp, "max_extension")),
            .silence_rounds = fx.int(fx.get(pp, "silence_rounds")),
            .window = @intCast(fx.int(fx.get(pp, "window"))),
            .confident_match = fx.int(fx.get(pp, "confident_match")),
        });
        s.* = try lanes.Stream.init(gpa, .{
            .id = fx.str(fx.get(spec, "id")),
            .prompt = try fx.u32s(a, fx.get(spec, "prompt")),
            .max_new = @intCast(fx.int(fx.get(spec, "max_new"))),
            .eos = try fx.u32s(a, fx.get(spec, "eos")),
            .drafts = fx.boolean(fx.get(spec, "drafts")),
            .proposer = if (fx.isNull(pp)) null else p.proposer(),
            .sampling = if (fx.isNull(sp)) null else fx.sampling(sp),
        });
        made += 1;
    }

    var trace = try Replay.init(a, events);
    var log = lanes.events.Log.init(gpa);
    defer log.deinit();
    var engine = lanes.Engine.init(gpa, &cfg, trace.backend(), trace.clock());
    defer engine.deinit();
    engine.log = &log;
    for (events) |l| {
        const kind = fx.str(fx.get(l.value, "ev"));
        if (std.mem.eql(u8, kind, "add")) {
            const id = fx.str(fx.get(l.value, "stream"));
            for (streams[0..made]) |*s| {
                if (std.mem.eql(u8, s.id, id)) try engine.addStream(s);
            }
        } else if (std.mem.eql(u8, kind, "step")) try engine.step();
    }
    for (events, 0..) |l, i| {
        if (i >= log.events.items.len) {
            std.debug.print("{s}: Zig stopped after {d} lines; Python's next:\n  {s}\n", .{ name, i, l.text });
            return error.TraceDiffers;
        }
        const mine = try log.line(gpa, i);
        defer gpa.free(mine);
        if (!std.mem.eql(u8, mine, l.text)) {
            std.debug.print("{s}: line {d} differs\n  python: {s}\n  zig:    {s}\n", .{ name, i + 1, l.text, mine });
            return error.TraceDiffers;
        }
    }
    try std.testing.expectEqual(events.len, log.events.items.len);
    return events.len;
}

test "greedy one stream" {
    std.debug.print("greedy: {d} lines equal\n", .{try replay("greedy.jsonl")});
}

test "sampled one stream (temperature 0.7, top-p 0.95)" {
    std.debug.print("sampled: {d} lines equal\n", .{try replay("sampled.jsonl")});
}

test "two concurrent streams" {
    std.debug.print("concurrent: {d} lines equal\n", .{try replay("concurrent.jsonl")});
}

test "drafts off" {
    std.debug.print("drafts-off: {d} lines equal\n", .{try replay("drafts-off.jsonl")});
}

/// The host reference of the GPU sampler against the kernel's own picks on recorded logits.
fn gpuRows(stem: []const u8) !void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const meta_name = try std.fmt.allocPrint(a, "{s}.json", .{stem});
    const bin_name = try std.fmt.allocPrint(a, "{s}.bin", .{stem});
    const meta = try std.json.parseFromSliceLeaky(fx.Value, a, try fx.read(a, meta_name), .{});
    const raw = try fx.read(a, bin_name);
    const vocab: usize = @intCast(fx.int(fx.get(meta, "vocab")));
    const rows = fx.get(meta, "rows").array.items;
    var differ: usize = 0;
    for (rows, 0..) |row, i| {
        const bytes = raw[i * vocab * 4 ..][0 .. vocab * 4];
        const logits = try a.alloc(f32, vocab);
        @memcpy(std.mem.sliceAsBytes(logits), bytes);
        const sp = fx.get(row, "sampling");
        const s: ?lanes.Sampling = if (fx.isNull(sp)) null else fx.sampling(sp);
        const pick = try lanes.gpu_rule.sample(a, logits, s, @intCast(fx.int(fx.get(row, "position"))), null);
        if (pick != fx.int(fx.get(row, "token"))) differ += 1;
    }
    std.debug.print("{s}: host rule == GPU kernel on {d} of {d} rows\n", .{ stem, rows.len - differ, rows.len });
    try std.testing.expectEqual(@as(usize, 0), differ);
}

test "GPU sampler rule on recorded rows" {
    try gpuRows("gpu_rows_sampled");
    try gpuRows("gpu_rows_greedy");
}
