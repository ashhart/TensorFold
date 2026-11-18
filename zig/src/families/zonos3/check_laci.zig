//! zonos3-generate --laci-replay: a LACI session driven by a recorded trace, against the reference's own results.
const std = @import("std");
const z3 = @import("zonos3");

const laci = z3.laci;
const W = z3.config.frame_width;
const Value = std.json.Value;
const cfg = z3.config;
const D = cfg.dim;

fn nowNs(io: std.Io) i128 {
    return @intCast(std.Io.Clock.awake.now(io).toNanoseconds());
}

fn read(a: std.mem.Allocator, io: std.Io, path: []const u8) !Value {
    return std.json.parseFromSliceLeaky(Value, a, try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 26)), .{});
}

fn usize_(v: ?Value, default: usize) usize {
    return if (v) |x| @intCast(x.integer) else default;
}

fn float(v: Value) f64 {
    return if (v == .float) v.float else @floatFromInt(v.integer);
}

const Out = std.ArrayList(u8);

fn optional(a: std.mem.Allocator, out: *Out, n: ?usize) !void {
    if (n) |x| try out.print(a, "{d}", .{x}) else try out.appendSlice(a, "null");
}

/// The masked heads' windows as the reference lists them: [layer, head, lo, hi] a head, or null without a mask.
fn mask(a: std.mem.Allocator, out: *Out, heads: laci.Heads, center: ?usize) !void {
    const c = center orelse return out.appendSlice(a, "null");
    var buf: [16]laci.Window = undefined;
    try out.append(a, '[');
    for (heads.windows(c, &buf), 0..) |w, i| try out.print(a, "{s}[{d}, {d}, {d}, {d}]", .{ if (i == 0) "" else ", ", w.layer, w.head, w.lo, w.hi });
    try out.append(a, ']');
}

/// Column 0 of each row, comma separated: what the reference's results list.
fn tags(a: std.mem.Allocator, out: *Out, rows: []const [W]i32, first: *bool) !void {
    for (rows) |r| {
        try out.print(a, "{s}{d}", .{ if (first.*) "" else ", ", r[0] });
        first.* = false;
    }
}

fn row(tag: usize, codes: ?Value, step: usize) [W]i32 {
    var r: [W]i32 = @splat(0);
    r[0] = @intCast(tag);
    if (codes) |list| for (list.array.items[step].array.items, 0..) |code, k| {
        r[k] = @intCast(code.integer);
    };
    return r;
}

/// Every value of `mine` equals the reference's at the same place (numbers to 1e-9); `path`: the first that is not.
fn same(mine: Value, theirs: Value, path: *std.ArrayList(u8), a: std.mem.Allocator) !bool {
    switch (mine) {
        .object => |o| {
            if (theirs != .object) return false;
            for (o.keys(), o.values()) |key, v| {
                const mark = path.items.len;
                try path.print(a, ".{s}", .{key});
                if (!try same(v, theirs.object.get(key) orelse return false, path, a)) return false;
                path.shrinkRetainingCapacity(mark);
            }
            return true;
        },
        .array => |list| {
            if (theirs != .array or theirs.array.items.len != list.items.len) return false;
            for (list.items, theirs.array.items, 0..) |v, w, i| {
                const mark = path.items.len;
                try path.print(a, "[{d}]", .{i});
                if (!try same(v, w, path, a)) return false;
                path.shrinkRetainingCapacity(mark);
            }
            return true;
        },
        .integer, .float => return (theirs == .integer or theirs == .float) and @abs(float(mine) - float(theirs)) <= 1e-9 * @max(1.0, @abs(float(theirs))),
        .null => return theirs == .null,
        .string => |s| return theirs == .string and std.mem.eql(u8, s, theirs.string),
        .bool => |b| return theirs == .bool and theirs.bool == b,
        else => return false,
    }
}

fn numbers(a: std.mem.Allocator, out: *Out, comptime T: type, xs: []const T) !void {
    for (xs, 0..) |x, i| try out.print(a, "{s}{e}", .{ if (i == 0) "" else ", ", @as(f64, x) });
}

/// test_ggml_laci_replay._follower: ranges, deltas and the gate's settings of a trace.
fn followerOf(gpa: std.mem.Allocator, a: std.mem.Allocator, j: std.json.ObjectMap) !z3.span_follow.Follower {
    const list = j.get("ranges").?.array.items;
    const ranges = try a.alloc([2]u32, list.len);
    for (ranges, list) |*r, v| r.* = .{ @intCast(v.array.items[0].integer), @intCast(v.array.items[1].integer) };
    var c: z3.span_follow.Config = .{};
    if (j.get("cfg")) |v| {
        if (v.object.get("lead")) |x| c.lead = float(x);
        if (v.object.get("lag")) |x| c.lag = float(x);
        if (v.object.get("smooth")) |x| c.smooth = @intCast(x.integer);
        if (v.object.get("ramp")) |x| c.ramp = @intCast(x.integer);
    }
    return z3.span_follow.Follower.init(gpa, ranges, c);
}

/// test_ggml_laci_replay.run_span_follow: a follower's scripted readings, decisions and rollbacks, each result kept.
fn replayFollow(gpa: std.mem.Allocator, a: std.mem.Allocator, t: std.json.ObjectMap, name: []const u8, expected: Value) !u8 {
    var f = try followerOf(gpa, a, t);
    defer f.deinit();
    const origin = usize_(t.get("origin"), 0);
    const rows = t.get("deltas").?.array.items;
    const width = rows[0].array.items.len;
    const deltas = try a.alloc(f32, rows.len * width);
    for (rows, 0..) |r, k| for (r.array.items, 0..) |v, i| {
        deltas[k * width + i] = @floatCast(float(v));
    };
    const w = try a.alloc(f32, rows.len);
    const sum = try a.alloc(f32, width);
    var out: Out = .empty;
    try out.appendSlice(a, "{\"results\": [");
    for (t.get("ops").?.array.items, 0..) |op, i| {
        const args = op.array.items;
        const what = args[0].string;
        if (i != 0) try out.appendSlice(a, ", ");
        if (std.mem.eql(u8, what, "observe") or std.mem.eql(u8, what, "on_step")) {
            const at: usize = @intCast(args[1].integer);
            try f.observe(if (what[1] == 'n') origin + at else at, @intCast(args[2].integer));
            try out.print(a, "{e}", .{f.best});
        } else if (std.mem.eql(u8, what, "weights")) {
            try f.decide(@intCast(args[1].integer), f.scratch);
            try out.append(a, '[');
            try numbers(a, &out, f64, f.scratch);
            try out.append(a, ']');
        } else if (std.mem.eql(u8, what, "delta")) {
            // span_follow.delta: the weights in fp32 times each span's row, summed in span order; null when none is on
            try f.weights(@intCast(args[1].integer), w);
            var any = false;
            for (w, 0..) |x, k| {
                any = any or x > 0;
                for (sum, deltas[k * width ..][0..width]) |*o, d| o.* = if (k == 0) x * d else o.* + x * d;
            }
            if (any) {
                try out.append(a, '[');
                try numbers(a, &out, f32, sum);
                try out.append(a, ']');
            } else try out.appendSlice(a, "null");
        } else if (std.mem.eql(u8, what, "truncate")) {
            f.truncate(@intCast(args[1].integer));
            try out.print(a, "{e}", .{f.best});
        } else if (std.mem.eql(u8, what, "position")) {
            try out.print(a, "{e}", .{f.best});
        } else return error.UnknownFollowOp;
    }
    try out.appendSlice(a, "]}");
    const mine = try std.json.parseFromSliceLeaky(Value, a, out.items, .{});
    var path: std.ArrayList(u8) = .empty;
    const pass = try same(mine, expected, &path, a);
    std.debug.print("RESULT zonos3 laci-replay {s} {s} ops={d}{s}{s}\n", .{ if (pass) "PASS" else "FAIL", name, t.get("ops").?.array.items.len, if (pass) "" else " first difference at ", if (pass) "" else path.items });
    return if (pass) 0 else 1;
}

/// test_ggml_laci_replay.run_laci over TRACE, compared with EXPECTED; `registry`: registries/short_text.json.
pub fn replay(gpa: std.mem.Allocator, io: std.Io, trace_path: []const u8, expected_path: []const u8, registry: []const u8) !u8 {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const t = (try read(a, io, trace_path)).object;
    const name = std.fs.path.stem(std.fs.path.stem(trace_path));
    if (t.get("mode")) |m| if (std.mem.eql(u8, m.string, "span_follow")) return replayFollow(gpa, a, t, name, try read(a, io, expected_path));
    for ([_][]const u8{ "duration_target", "rates", "rate_deltas", "cfg", "ops" }) |unserved| if (t.get(unserved) != null) {
        std.debug.print("RESULT zonos3 laci-replay SKIP {s}: the trace uses {s}, which is not ported\n", .{ name, unserved });
        return 0;
    };
    var heads = try laci.Heads.parse(a, try std.json.Stringify.valueAlloc(a, t.get("heads_json").?, .{}));
    if (t.get("frames_per_token")) |v| if (v != .null) {
        heads.frames_per_token = float(v);
    };
    const n_prefill = usize_(t.get("n_prefill"), 0);
    var short: ?z3.short_text.Cut = null;
    var short_config: z3.short_text.Config = undefined;
    if (t.get("short_text")) |v| if (v != .null) {
        if (registry.len == 0) return error.ShortTextTraceNeedsTheRegistry;
        short_config = try z3.short_text.Config.parse(a, try std.Io.Dir.cwd().readFileAlloc(io, registry, a, .limited(1 << 22)));
        short = z3.short_text.Cut.init(gpa, &short_config, @intCast(v.integer));
    };
    const prefill = try a.alloc([W]i32, n_prefill);
    @memset(prefill, @splat(-1));
    const hold = if (t.get("hold")) |v| v.bool else false;
    var s = try laci.Session.init(gpa, heads, .{}, prefill, if (t.get("seed")) |v| @intCast(v.integer) else 0, usize_(t.get("max_tokens"), 4000), hold, short);
    defer s.deinit();
    // a follower the session feeds: its weights are read before each step, as the reference's harness reads them
    var follower: ?z3.span_follow.Follower = if (t.get("follower")) |v| try followerOf(gpa, a, v.object) else null;
    defer if (follower) |*f| f.deinit();
    if (follower) |*f| s.follower = f;
    const lone = [1]Value{.{ .object = t }};
    const attempts: []const Value = if (t.get("attempts")) |v| v.array.items else &lone;
    const C = usize_(t.get("n_codebooks"), 10);
    if (C != z3.config.codebooks) return error.TraceOfAnotherCodec;
    var out: Out = .empty;
    try out.appendSlice(a, "{\"attempts\": [");
    const max_attempts = usize_(t.get("max_attempts"), 64);
    var k: usize = 0;
    while (true) : (k += 1) {
        if (k >= max_attempts) return error.MoreThanMaxAttempts;
        const at = attempts[@min(k, attempts.len - 1)].object;
        const tag = k + 1;
        const eos_at: ?usize = if (at.get("eos_at")) |v| (if (v == .null) null else @intCast(v.integer)) else null;
        const eos_lag = usize_(at.get("eos_lag"), 0);
        const codes = at.get("codes");
        var probe: laci.Probe = .{ .text_len = usize_(t.get("text_len"), 0), .center = s.probeCenter() };
        try out.appendSlice(a, if (k == 0) "{\"mask\": " else ", {\"mask\": ");
        try mask(a, &out, heads, probe.center);
        var changes: Out = .empty;
        var drained: Out = .empty;
        var first = true;
        try s.append(row(tag, codes, 0), null);
        try tags(a, &drained, s.drain(false), &first);
        var countdown: ?usize = null;
        var steps_run: usize = 0;
        var stop_step: ?usize = null;
        var boost_step: ?usize = null;
        var fweights: Out = .empty;
        for (at.get("steps").?.array.items, 1..) |pair, step| {
            if (follower) |*f| {
                try f.decide(s.attempt_start + step - s.n_prefill, f.scratch);
                try fweights.appendSlice(a, if (step == 1) "[" else ", [");
                try numbers(a, &fweights, f64, f.scratch);
                try fweights.append(a, ']');
            }
            const before = probe.center;
            try s.onStep(&probe, step, @intCast(pair.array.items[0].integer), float(pair.array.items[1]));
            steps_run = step;
            if (probe.center != before) {
                try changes.print(a, "{s}[{d}, ", .{ if (changes.items.len == 0) "" else ", ", step });
                try mask(a, &changes, heads, probe.center);
                try changes.append(a, ']');
            }
            var eos: ?usize = null;
            if (eos_at) |e| if (step >= e) {
                eos = e - eos_lag;
                if (countdown == null) countdown = C + 1;
            };
            try s.append(row(tag * 100000 + step, codes, step), eos);
            if (boost_step == null and s.stopBoost() > 0) boost_step = step;
            if (countdown) |*n| n.* -= 1;
            if (probe.stop or (countdown != null and countdown.? == 0)) {
                if (probe.stop) stop_step = step;
                break;
            }
            try tags(a, &drained, s.drain(false), &first);
        }
        try out.print(a, ", \"mask_changes\": [{s}], \"steps_run\": {d}, \"stop_step\": ", .{ changes.items, steps_run });
        try optional(a, &out, stop_step);
        try out.print(a, ", \"drained\": [{s}], \"boost_step\": ", .{drained.items});
        try optional(a, &out, boost_step);
        if (follower != null) try out.print(a, ", \"follower_weights\": [{s}]", .{fweights.items});
        const cont = try s.finishAttempt();
        if (cont) |c| {
            try out.print(a, ", \"continuation\": {{\"n_rows\": {d}, \"history_rows\": {d}, \"seed\": ", .{ c.rows, c.history_rows });
            try optional(a, &out, if (c.seed) |x| @intCast(x) else null);
            try out.print(a, ", \"max_tokens\": {d}, \"probe_mask\": ", .{c.max_tokens});
            try mask(a, &out, heads, s.probeCenter());
            try out.appendSlice(a, "}}");
        } else try out.appendSlice(a, ", \"continuation\": null}");
        if (cont == null) break;
    }
    var first = true;
    try out.appendSlice(a, "], \"final_drain\": [");
    try tags(a, &out, s.drain(true), &first);
    try out.appendSlice(a, "], \"eos_frame\": ");
    try optional(a, &out, s.eosFrame());
    const end = s.end.?;
    try out.print(a, ", \"end\": [\"{s}\", ", .{@tagName(end.kind)});
    try optional(a, &out, end.row);
    try out.print(a, "], \"record\": {{\"attempts\": {d}, \"capped\": {}, \"end\": \"{s}\", \"text_len\": {d}, \"n_bins\": {d}, \"final_rows\": {d}, \"generated_total\": {d}, \"final_eos_row\": ", .{ s.attempts, s.attempts >= s.c.max_retries, @tagName(end.kind), s.text_len, if (s.detector) |d| d.nb else 0, s.finalRows(), s.generated_total });
    try optional(a, &out, end.row);
    if (s.detector) |d| try out.print(a, ", \"pace_tok\": {e}", .{d.paceTok()});
    try out.appendSlice(a, ", \"failures\": [");
    for (s.failures.items, 0..) |f, i| {
        try out.print(a, "{s}{{\"kind\": \"{s}\", \"detect_row\": {d}, \"onset_row\": {d}, \"rollback_row\": {d}, \"clamped\": {}, \"bin\": {d}, \"frontier\": {d}, \"ignored\": {}, \"retry_seed\": ", .{ if (i == 0) "" else ", ", @tagName(f.failure.kind), f.failure.detect_row, f.failure.onset_row, f.rollback_row, f.clamped, f.failure.bin, f.failure.frontier, f.ignored });
        try optional(a, &out, if (f.retry_seed) |x| @intCast(x) else null);
        try out.append(a, '}');
    }
    first = true;
    try out.appendSlice(a, "]}, \"stream\": [");
    try tags(a, &out, s.stream.items[n_prefill..], &first);
    try out.appendSlice(a, "]}");
    const mine = try std.json.parseFromSliceLeaky(Value, a, out.items, .{});
    var path: std.ArrayList(u8) = .empty;
    const pass = try same(mine, try read(a, io, expected_path), &path, a);
    std.debug.print("RESULT zonos3 laci-replay {s} {s} attempts={d} failures={d} end={s} final_rows={d}{s}{s}\n", .{ if (pass) "PASS" else "FAIL", name, s.attempts, s.failures.items.len, @tagName(s.end.?.kind), s.finalRows(), if (pass) "" else " first difference at ", path.items });
    return if (pass) 0 else 1;
}

/// A steered dump's steer.npy [frames + 1][sources][dim] and steer.json (zonos3_reference.py rows --steer-bundle).
pub const Dumped = struct {
    deltas: []f32 = &.{},
    sources: []u32 = &.{},
    active: []bool = &.{},
    steer: []z3.Steer = &.{},
    twin: bool = false,
    count: usize = 0,

    pub fn deinit(d: *Dumped, gpa: std.mem.Allocator) void {
        gpa.free(d.deltas);
        gpa.free(d.sources);
        gpa.free(d.active);
        gpa.free(d.steer);
    }

    pub fn load(d: *Dumped, gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !bool {
        const path = try std.fs.path.join(gpa, &.{ dir, "steer.npy" });
        defer gpa.free(path);
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 28)) catch |e| switch (e) {
            error.FileNotFound => return false,
            else => return e,
        };
        defer gpa.free(bytes);
        const arr = try z3.npy.parse(bytes);
        if (arr.rank != 3 or arr.shape[2] != D) return error.BadSteerDump;
        d.deltas = try arr.floats(gpa);
        const side = try std.fs.path.join(gpa, &.{ dir, "steer.json" });
        defer gpa.free(side);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, side, gpa, .limited(1 << 22));
        defer gpa.free(text);
        var tree = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        defer tree.deinit();
        const o = tree.value.object;
        const sources = o.get("sources").?.array.items;
        const active = o.get("active").?.array.items;
        if (sources.len != arr.shape[1] or active.len != arr.shape[0]) return error.BadSteerDump;
        d.sources = try gpa.alloc(u32, sources.len);
        for (d.sources, sources) |*s, v| s.* = @intCast(v.integer);
        d.active = try gpa.alloc(bool, active.len);
        for (d.active, active) |*a, v| a.* = v.bool;
        d.steer = try gpa.alloc(z3.Steer, sources.len);
        d.twin = o.get("twin_steered").?.bool;
        return true;
    }

    pub fn at(ctx: *anyopaque, frame: usize) anyerror![]const z3.Steer {
        const d: *Dumped = @ptrCast(@alignCast(ctx));
        if (frame >= d.active.len or !d.active[frame]) return &.{};
        d.count += 1;
        for (d.steer, d.sources, 0..) |*s, source, j| s.* = .{ .source = source, .delta = d.deltas[(frame * d.sources.len + j) * D ..][0..D] };
        return d.steer;
    }
};

/// GEN_DIR's request again under LACI, greedy: every attempt's readings, rows and record against laci.json.
pub const Hook = struct { steer: z3.generation.Run.Steering, follower: *z3.span_follow.Follower };

pub fn engine(gpa: std.mem.Allocator, io: std.Io, e: *z3.Engine, dir: []const u8, prompt: []const [W]i32, replaced: []const z3.Replace, heads_path: []const u8, registry: []const u8, max_frames: usize, hook: ?Hook) !u8 {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cwd = std.Io.Dir.cwd();
    const want = (try std.json.parseFromSliceLeaky(std.json.Value, a, try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "laci.json" }), a, .limited(1 << 26)), .{})).object;
    const cond = (try std.json.parseFromSliceLeaky(std.json.Value, a, try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "input_ids.cond.json" }), a, .limited(1 << 22)), .{})).object;
    const heads = try z3.laci.Heads.parse(a, try cwd.readFileAlloc(io, heads_path, a, .limited(1 << 20)));
    var dumped: Dumped = .{};
    defer dumped.deinit(gpa);
    const steered = try dumped.load(gpa, io, dir);
    // a short text's dump names the filler's first token: the take is cut where the reading reaches it
    var short_config: z3.short_text.Config = undefined;
    var short: ?z3.short_text.Cut = null;
    if (want.get("short_text")) |v| if (v != .null) {
        if (registry.len == 0) return error.ShortTextDumpNeedsTheRegistry;
        short_config = try z3.short_text.Config.parse(a, try cwd.readFileAlloc(io, registry, a, .limited(1 << 22)));
        short = z3.short_text.Cut.init(gpa, &short_config, @intCast(v.integer));
    };
    const setup: z3.laci_run.Setup = .{ .short = short, .heads = heads, .text_start = @intCast(want.get("text_start").?.integer), .text_len = @intCast(want.get("text_len").?.integer), .prefill = @intCast(cond.get("prefill_frames").?.integer), .hold = true };
    const t0 = nowNs(io);
    var run = try z3.laci_run.Run(z3.Replace).begin(gpa, e, prompt, replaced, setup, .{ .temperature = 0, .seed = @intCast(want.get("seed").?.integer) }, max_frames);
    defer run.deinit();
    if (steered) run.steer = .{ .ctx = &dumped, .at = Dumped.at, .twin = false };
    if (hook) |h| {
        run.steer = h.steer;
        run.session.follower = h.follower;
    }
    const attempts = want.get("attempts").?.array.items;
    var out: [cfg.head_rows]f32 = undefined;
    var k: usize = 0; // the attempt, and the step and row in it
    var step: usize = 0;
    var at_row: usize = 0;
    var positions_differ: usize = 0;
    var masks_differ: usize = 0;
    var rows_differ: usize = 0;
    var worst: f64 = 0;
    var steps: usize = 0;
    var extra = false;
    while (true) {
        const got = try run.step(e);
        const theirs = if (k < attempts.len) attempts[k].object else null;
        if (theirs) |at| {
            const their_rows = at.get("rows").?.array.items;
            if (at_row >= their_rows.len) rows_differ += 1 else for (their_rows[at_row].array.items, got.row) |x, y| {
                if (x.integer != y) {
                    rows_differ += 1;
                    break;
                }
            }
            if (got.reading) |r| {
                const their_steps = at.get("steps").?.array.items;
                if (step >= their_steps.len) positions_differ += 1 else {
                    const s = their_steps[step].array.items;
                    positions_differ += @intFromBool(s[0].integer != r.pos);
                    worst = @max(worst, @abs(r.m - (if (s[1] == .float) s[1].float else @as(f64, @floatFromInt(s[1].integer)))));
                    // the mask the reading leaves: the reference's list of windows, or none
                    masks_differ += @intFromBool((s[2] == .null) != (got.center == null));
                    if (s[2] != .null and got.center != null) {
                        var buf: [16]z3.laci.Window = undefined;
                        for (heads.windows(got.center.?, &buf), s[2].array.items) |w, v| masks_differ += @intFromBool(v.array.items[2].integer != w.lo or v.array.items[3].integer != w.hi);
                    }
                }
                step += 1;
                steps += 1;
            }
        } else extra = true;
        at_row += 1;
        if (got.final) break;
        if (got.turn) |turn| {
            try e.runLanes(&.{.{ .lane = e.active, .row = turn.row, .out = &out, .steer = turn.steer, .probe = turn.probe }});
            @memcpy(@as(*[cfg.head_rows]f32, @ptrCast(run.run.logits)), &out);
        } else {
            k += 1;
            step = 0;
            at_row = 0;
        }
    }
    const seconds = @as(f64, @floatFromInt(nowNs(io) - t0)) / 1e9;
    const s = &run.session;
    const record = want.get("record").?.object;
    const their_failures = record.get("failures").?.array.items;
    var record_same = record.get("attempts").?.integer == s.attempts and std.mem.eql(u8, record.get("end").?.string, @tagName(s.end.?.kind)) and record.get("final_rows").?.integer == s.finalRows() and record.get("generated_total").?.integer == s.generated_total and their_failures.len == s.failures.items.len;
    if (record_same) for (their_failures, s.failures.items) |v, f| {
        const o = v.object;
        record_same = record_same and std.mem.eql(u8, o.get("kind").?.string, @tagName(f.failure.kind)) and o.get("detect_row").?.integer == f.failure.detect_row and o.get("onset_row").?.integer == f.failure.onset_row and o.get("rollback_row").?.integer == f.rollback_row and o.get("bin").?.integer == f.failure.bin and o.get("frontier").?.integer == f.failure.frontier and o.get("ignored").?.bool == f.ignored;
    };
    const pass = !extra and k + 1 == attempts.len and positions_differ == 0 and masks_differ == 0 and rows_differ == 0 and record_same and worst < 1e-2;
    std.debug.print("RESULT zonos3 laci {s} attempts={d} reference_attempts={d} failures={d} end={s} final_rows={d} steps={d} positions_differ={d} masks_differ={d} rows_differ={d} worst_centroid_diff={e:.2} record_equal={} steered={} seconds={d:.3}\n", .{ if (pass) "PASS" else "FAIL", k + 1, attempts.len, s.failures.items.len, @tagName(s.end.?.kind), s.finalRows(), steps, positions_differ, masks_differ, rows_differ, worst, record_same, steered, seconds });
    return if (pass) 0 else 1;
}
