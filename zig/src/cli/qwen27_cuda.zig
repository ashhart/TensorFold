//! tf-qwen27-cuda: the 27B's Zig CUDA engine against the Python engine's capture (prompts.json, results.json).

const std = @import("std");
const cuda = @import("cuda");
const qwen = @import("qwen27");

const usage =
    \\usage: tf-qwen27-cuda run MODEL --kernels DIR --prompts prompts.json [--expect results.json]
    \\         [--max-tokens N] [--context N] [--device N]
    \\  Greedy, drafts off: each prompt's tokens and their sha against the capture's.
    \\       tf-qwen27-cuda windows MODEL --kernels DIR --prompts windows.json
    \\  capture_windows.py's chains and shared rounds: each row's logits against the capture's sha256.
    \\       tf-qwen27-cuda draft MODEL --kernels DIR --drafter DIR --prompts prompts.json [--expect draft.json]
    \\  draft_decode with DFlash2, greedy: tokens (equal to serial) and rounds against capture_draft.py's.
    \\       tf-qwen27-cuda rounds MODEL --kernels DIR --prompts windows.json [--max-tokens ROUNDS]
    \\  Greedy shared rounds of one row a stream over every prompt, timed.
    \\       tf-qwen27-cuda multi MODEL --kernels DIR --drafter DIR --prompts multi.json
    \\  capture_multi.py's drafted shared rounds: each round's windows and every reply against the capture's.
    \\
;

const Options = struct {
    model: []const u8,
    kernels: []const u8 = "",
    prompts: []const u8 = "",
    expect: ?[]const u8 = null,
    max_tokens: usize = 128,
    context: usize = 16384,
    device: c_int = 0,
    dump: ?[]const u8 = null, // windows: a case's per-norm rows into this folder
    cases: ?[]const u8 = null, // windows: only these cases, comma separated
    drafter: ?[]const u8 = null, // draft: the DFlash2 checkpoint
};

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    const windows = args.len >= 2 and std.mem.eql(u8, args[1], "windows");
    const rounds = args.len >= 2 and std.mem.eql(u8, args[1], "rounds");
    const drafting = args.len >= 2 and std.mem.eql(u8, args[1], "draft");
    const multi = args.len >= 2 and std.mem.eql(u8, args[1], "multi");
    if (args.len < 3 or !(std.mem.eql(u8, args[1], "run") or windows or rounds or drafting or multi)) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    var o: Options = .{ .model = args[2] };
    var i: usize = 3;
    while (i + 1 < args.len) : (i += 2) {
        const k = args[i];
        const v = args[i + 1];
        if (std.mem.eql(u8, k, "--kernels")) o.kernels = v else if (std.mem.eql(u8, k, "--prompts")) o.prompts = v else if (std.mem.eql(u8, k, "--expect")) o.expect = v else if (std.mem.eql(u8, k, "--max-tokens")) o.max_tokens = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--context")) o.context = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--device")) o.device = try std.fmt.parseInt(c_int, v, 10) else if (std.mem.eql(u8, k, "--dump")) o.dump = v else if (std.mem.eql(u8, k, "--cases")) o.cases = v else if (std.mem.eql(u8, k, "--drafter")) o.drafter = v else {
            std.debug.print("unknown option {s}\n{s}", .{ k, usage });
            return 2;
        }
    }
    if (o.kernels.len == 0 or o.prompts.len == 0) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const prompts = try readJson(arena, init.io, o.prompts);
    const expect = if (o.expect) |p| try readJson(arena, init.io, p) else null;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, o.device);
    defer ctx.deinit();
    const clock = std.Io.Clock.awake;
    var t0 = clock.now(init.io).toNanoseconds();
    const e = try qwen.Engine.init(gpa, init.io, &ctx, o.model, o.kernels, .{ .context = o.context });
    defer e.deinit();
    std.debug.print("loaded in {d:.1}s, weights {d:.2} GiB\n", .{ @as(f64, @floatFromInt(clock.now(init.io).toNanoseconds() - t0)) / 1e9, @as(f64, @floatFromInt(e.w.bytes())) / (1 << 30) });
    if (windows) return checkWindows(gpa, arena, init.io, e, prompts, o.dump, o.cases);
    if (rounds) return benchRounds(gpa, arena, init.io, e, prompts, o.max_tokens);
    if (drafting) return runDrafted(gpa, arena, init.io, e, o, prompts, expect);
    if (multi) return runMulti(gpa, arena, init.io, e, o, prompts);
    var seq = try e.sequence(o.context);
    defer seq.deinit();
    var mismatches: usize = 0;
    var it = prompts.object.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        const ids = try tokenList(arena, entry.value_ptr.*);
        var out: std.ArrayList(u32) = .empty;
        defer out.deinit(gpa);
        t0 = clock.now(init.io).toNanoseconds();
        try e.generate(&seq, ids, o.max_tokens, &out);
        const seconds = @as(f64, @floatFromInt(clock.now(init.io).toNanoseconds() - t0)) / 1e9;
        const sha = try sha12(arena, out.items);
        var verdict: []const u8 = "";
        if (expect) |x| blk: {
            const want_v = (x.object.get("results") orelse break :blk).object.get(name) orelse break :blk;
            const want = try tokenList(arena, want_v.object.get("tokens").?);
            const same = std.mem.eql(u32, want, out.items);
            if (!same) mismatches += 1;
            var first: usize = 0;
            while (first < @min(want.len, out.items.len) and want[first] == out.items[first]) first += 1;
            verdict = if (same) " MATCH" else try std.fmt.allocPrint(arena, " DIFFER at token {d} (python sha {s})", .{ first, try sha12(arena, want) });
        }
        std.debug.print("{s}: prompt {d}, {d} tokens sha {s} in {d:.2}s{s}\n", .{ name, ids.len, out.items.len, sha, seconds, verdict });
    }
    return if (mismatches == 0) 0 else 1;
}

fn readJson(a: std.mem.Allocator, io: std.Io, path: []const u8) !std.json.Value {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 30));
    return (try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}));
}

fn tokenList(a: std.mem.Allocator, v: std.json.Value) ![]u32 {
    if (v != .array) return error.BadPromptFile;
    const out = try a.alloc(u32, v.array.items.len);
    for (v.array.items, out) |x, *t| t.* = @intCast(x.integer);
    return out;
}

/// The capture's sha12: sha256 of json.dumps(tokens), "[1, 2, 3]", its first 12 hex digits.
fn sha12(a: std.mem.Allocator, tokens: []const u32) ![]const u8 {
    var text: std.ArrayList(u8) = .empty;
    try text.append(a, '[');
    for (tokens, 0..) |t, i| {
        if (i > 0) try text.appendSlice(a, ", ");
        try text.print(a, "{d}", .{t});
    }
    try text.append(a, ']');
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(text.items, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    return a.dupe(u8, hex[0..12]);
}

/// Each case's streams prefilled afresh, its round run, every row's logits hashed; shared cases commit and go on.
fn checkWindows(gpa: std.mem.Allocator, a: std.mem.Allocator, io: std.Io, e: *qwen.Engine, doc: std.json.Value, dump: ?[]const u8, only: ?[]const u8) !u8 {
    const root = doc.object;
    const prompts = root.get("prompts").?.array.items;
    const cases = root.get("cases").?.array.items;
    var bad: usize = 0;
    const row_bytes = qwen.config.vocab * 2;
    const host = try gpa.alloc(u8, row_bytes);
    defer gpa.free(host);
    for (cases, 0..) |case, ci| {
        if (only) |list| {
            var it = std.mem.splitScalar(u8, list, ',');
            const listed = while (it.next()) |x| {
                if (std.fmt.parseInt(usize, x, 10) catch null == ci) break true;
            } else false;
            if (!listed) continue;
        }
        const streams = case.object.get("streams").?.array.items;
        std.debug.print("case {d}: {s}, {d} streams\n", .{ ci, case.object.get("kind").?.string, streams.len });
        var seqs: [qwen.state.max_streams]qwen.state.Seq = undefined;
        var parts: [qwen.state.max_streams]qwen.Part = undefined;
        var made: usize = 0;
        defer for (seqs[0..made]) |*sq| sq.deinit();
        for (streams, 0..) |sv, k| {
            const sid: usize = @intCast(sv.object.get("stream").?.integer);
            const ids = try tokenList(a, prompts[sid]);
            seqs[k] = try e.sequence(4096);
            made += 1;
            try e.prefill(&seqs[k], ids);
            parts[k] = .{ .seq = &seqs[k], .tokens = try tokenList(a, sv.object.get("tokens").?) };
            if (sv.object.get("parents")) |pv| {
                const ps = try a.alloc(i32, pv.array.items.len);
                for (pv.array.items, ps) |x, *y| y.* = @intCast(x.integer);
                parts[k].parents = ps;
            }
        }
        const n = streams.len;
        var f = e.forward();
        if (dump) |dir| f.dump = .{ .io = io, .dir = dir };
        try f.round(parts[0..n]);
        f.dump = null;
        bad += try compareRows(&f, host, case.object.get("rows").?.array.items, ci, "rows");
        if (case.object.get("paths")) |paths_v| {
            var paths: [qwen.state.max_streams][]const u32 = undefined;
            var next: [qwen.state.max_streams]u32 = undefined;
            for (paths_v.array.items, 0..) |pv, k| paths[k] = try tokenList(a, pv);
            try f.commit(parts[0..n], paths[0..n]);
            for (case.object.get("next").?.array.items, 0..) |nv, k| {
                next[k] = @intCast(nv.integer);
                parts[k].tokens = next[k .. k + 1];
                parts[k].parents = null;
            }
            try f.round(parts[0..n]);
            bad += try compareRows(&f, host, case.object.get("after").?.array.items, ci, "after");
        }
        if (case.object.get("keep")) |keep_v| {
            var kept: [qwen.state.max_streams]usize = undefined;
            var next: [qwen.state.max_streams]u32 = undefined;
            const prefix = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
            var paths: [qwen.state.max_streams][]const u32 = undefined;
            for (keep_v.array.items, 0..) |kv, k| {
                kept[k] = @intCast(kv.integer);
                paths[k] = prefix[0..kept[k]];
            }
            try f.commit(parts[0..n], paths[0..n]);
            const reply_rows = doc.object.get("replies").?.array.items;
            for (streams, 0..) |sv, k| {
                const sid: usize = @intCast(sv.object.get("stream").?.integer);
                next[k] = @intCast(reply_rows[sid].array.items[kept[k]].integer);
                parts[k].tokens = next[k .. k + 1];
            }
            try f.round(parts[0..n]);
            bad += try compareRows(&f, host, case.object.get("after").?.array.items, ci, "after");
        }
    }
    std.debug.print("{d} cases, {d} rows differ\n", .{ cases.len, bad });
    return if (bad == 0) 0 else 1;
}

fn compareRows(f: *qwen.Forward, host: []u8, want: []const std.json.Value, case: usize, what: []const u8) !usize {
    var bad: usize = 0;
    for (want, 0..) |rv, r| {
        try f.ops.download(host, f.s.logits + r * host.len);
        try f.ops.s.synchronize();
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(host, &digest, .{});
        const hex = std.fmt.bytesToHex(digest, .lower);
        if (!std.mem.eql(u8, &hex, rv.object.get("sha").?.string)) {
            std.debug.print("case {d} {s} row {d}: logits differ\n", .{ case, what, r });
            bad += 1;
        }
    }
    return bad;
}

/// Every prompt prefilled, then greedy shared rounds of one row a stream; ms a round and tokens a second.
fn benchRounds(gpa: std.mem.Allocator, a: std.mem.Allocator, io: std.Io, e: *qwen.Engine, doc: std.json.Value, count: usize) !u8 {
    {
        // the served engine's cost curve: the forward's ms by a round's rows
        var lanes_ = try qwen.Lanes.init(gpa, e, qwen.state.max_streams);
        defer lanes_.deinit();
        try lanes_.calibrate(io, qwen.state.max_streams);
        for (lanes_.costs[0..lanes_.cost_count]) |cst| std.debug.print("{d} rows: {d:.2} ms\n", .{ cst.width, cst.ms });
    }
    const prompts = doc.object.get("prompts").?.array.items;
    const n = @min(prompts.len, qwen.state.max_streams);
    var seqs: [qwen.state.max_streams]qwen.state.Seq = undefined;
    var parts: [qwen.state.max_streams]qwen.Part = undefined;
    var next: [qwen.state.max_streams]u32 = undefined;
    var ones: [qwen.state.max_streams][]const u32 = @splat(&.{0});
    var made: usize = 0;
    defer for (seqs[0..made]) |*sq| sq.deinit();
    var f = e.forward();
    for (prompts[0..n], 0..) |pv, k| {
        seqs[k] = try e.sequence(4096);
        made += 1;
        try e.prefill(&seqs[k], try tokenList(a, pv));
        try f.head(&seqs[k]);
        var pick: [1]u32 = undefined;
        try f.picks(1, &pick);
        next[k] = pick[0];
    }
    const clock = std.Io.Clock.awake;
    const t0 = clock.now(io).toNanoseconds();
    for (0..count) |_| {
        for (0..n) |k| parts[k] = .{ .seq = &seqs[k], .tokens = next[k .. k + 1] };
        try f.round(parts[0..n]);
        try f.picks(n, next[0..n]);
        try f.commit(parts[0..n], ones[0..n]);
    }
    try f.ops.s.synchronize();
    const ms = @as(f64, @floatFromInt(clock.now(io).toNanoseconds() - t0)) / 1e6;
    std.debug.print("{d} streams, {d} rounds: {d:.2} ms a round, {d:.1} tok/s\n", .{ n, count, ms / @as(f64, @floatFromInt(count)), @as(f64, @floatFromInt(n * count)) / ms * 1e3 });
    return 0;
}

/// Each prompt drafted with DFlash2 as capture_draft.py runs it: tokens, rounds and accepted drafts.
fn runDrafted(gpa: std.mem.Allocator, a: std.mem.Allocator, io: std.Io, e: *qwen.Engine, o: Options, prompts: std.json.Value, expect: ?std.json.Value) !u8 {
    const clock = std.Io.Clock.awake;
    var t0 = clock.now(io).toNanoseconds();
    const f0 = e.forward();
    const d = try qwen.draft.DFlash2.init(gpa, io, f0.ops, f0.t, &e.w, o.drafter orelse return error.NoDrafter, o.model);
    defer d.deinit();
    std.debug.print("drafter loaded in {d:.1}s\n", .{@as(f64, @floatFromInt(clock.now(io).toNanoseconds() - t0)) / 1e9});
    var ctx = try qwen.draft.Context.init(f0.ops.k.d);
    defer ctx.deinit();
    var seq = try e.sequence(o.context);
    defer seq.deinit();
    var bad: usize = 0;
    var it = prompts.object.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        const ids = try tokenList(a, entry.value_ptr.*);
        var out: std.ArrayList(u32) = .empty;
        defer out.deinit(gpa);
        var stats: qwen.lone.Stats = .{};
        t0 = clock.now(io).toNanoseconds();
        try qwen.lone.prefill(e, &seq, &ctx, d, ids);
        var f = e.forward();
        try f.head(&seq);
        var pick: [1]u32 = undefined;
        try f.picks(1, &pick);
        const t1 = clock.now(io).toNanoseconds();
        try qwen.lone.decode(e, &seq, &ctx, d, ids, pick[0], o.max_tokens, 12, &out, &stats);
        const secs = @as(f64, @floatFromInt(clock.now(io).toNanoseconds() - t1)) / 1e9;
        var verdict: []const u8 = "";
        if (expect) |x| blk: {
            const want_v = (x.object.get("results") orelse break :blk).object.get(name) orelse break :blk;
            const want = try tokenList(a, want_v.object.get("tokens").?);
            const same = std.mem.eql(u32, want, out.items);
            if (!same) bad += 1;
            verdict = try std.fmt.allocPrint(a, " {s}; python rounds {d} accepted {d}", .{ if (same) "MATCH" else "DIFFER", want_v.object.get("rounds").?.integer, want_v.object.get("accepted").?.integer });
        }
        std.debug.print("{s}: {d} tokens sha {s}, rounds {d} accepted {d}, {d:.1} tok/s{s}\n", .{ name, out.items.len, try sha12(a, out.items), stats.rounds, stats.accepted, @as(f64, @floatFromInt(out.items.len)) / secs, verdict });
    }
    return if (bad == 0) 0 else 1;
}

/// One stream of the multi check: its state, drafting, context (prompt and reply) and reply.
const MultiStream = struct {
    seq: qwen.state.Seq,
    d: qwen.drafts.Stream,
    context: std.ArrayList(u32) = .empty,
    out: std.ArrayList(u32) = .empty,
    done: bool = false,
    rounds: usize = 0,
    accepted: usize = 0,
};

/// MultiDecoder's rounds as capture_multi.py runs them (greedy, no costs: whole trees, blocks of 16 rows).
fn runMulti(gpa: std.mem.Allocator, a: std.mem.Allocator, io: std.Io, e: *qwen.Engine, o: Options, doc: std.json.Value) !u8 {
    const clock = std.Io.Clock.awake;
    const f0 = e.forward();
    const d = try qwen.draft.DFlash2.init(gpa, io, f0.ops, f0.t, &e.w, o.drafter orelse return error.NoDrafter, o.model);
    defer d.deinit();
    const prompts = doc.object.get("prompts").?.array.items;
    const count: usize = @intCast(doc.object.get("max_tokens").?.integer);
    const want_rounds = doc.object.get("rounds").?.array.items;
    const S = prompts.len;
    const ms = try a.alloc(MultiStream, S);
    var made: usize = 0;
    defer for (ms[0..made]) |*m| {
        m.seq.deinit();
        m.d.deinit();
        m.context.deinit(gpa);
        m.out.deinit(gpa);
    };
    for (prompts, ms) |pv, *m| {
        const ids = try tokenList(a, pv);
        m.* = .{ .seq = try e.sequence(o.context), .d = qwen.drafts.Stream.init(gpa, try qwen.draft.Context.init(f0.ops.k.d)) };
        made += 1;
        try qwen.lone.prefill(e, &m.seq, &m.d.ctx, d, ids);
        var f = e.forward();
        try f.head(&m.seq);
        var pick: [1]u32 = undefined;
        try f.picks(1, &pick);
        try m.context.appendSlice(gpa, ids);
        try m.context.append(gpa, pick[0]);
        try m.out.append(gpa, pick[0]);
    }
    var differ: usize = 0;
    var round: usize = 0;
    const t0 = clock.now(io).toNanoseconds();
    var tokens: [qwen.state.round_rows]u32 = undefined;
    var parents: [qwen.state.round_rows]i32 = undefined;
    var picks: [qwen.state.round_rows]u32 = undefined;
    var path_rows: [qwen.state.round_rows]u32 = undefined;
    var kept_rows: [qwen.state.round_rows]u32 = undefined;
    while (true) : (round += 1) {
        var live: [qwen.state.max_streams]usize = undefined;
        var n: usize = 0;
        for (ms, 0..) |m, k| if (!m.done) {
            live[n] = k;
            n += 1;
        };
        if (n == 0) break;
        var asks: [qwen.state.max_streams]qwen.drafts.Ask = undefined;
        for (live[0..n], 0..) |k, j| asks[j] = .{ .x = &ms[k].d, .context = ms[k].context.items, .sampling = null };
        try qwen.drafts.propose(d, asks[0..n]);
        // the windows: the pending token, then the drafts (tree parents shifted under it)
        var parts: [qwen.state.max_streams]qwen.Part = undefined;
        var starts: [qwen.state.max_streams + 1]usize = undefined;
        var total: usize = 0;
        const want = if (round < want_rounds.len) want_rounds[round].array.items else &.{};
        for (live[0..n], 0..) |k, j| {
            const m = &ms[k];
            starts[j] = total;
            tokens[total] = m.out.items[m.out.items.len - 1];
            parents[total] = -1;
            for (m.d.tokens.items, m.d.parents.items, 1..) |t, p, r| {
                tokens[total + r] = t;
                parents[total + r] = if (p < 0) 0 else p + 1;
            }
            const rows = 1 + m.d.tokens.items.len;
            parts[j] = .{ .seq = &m.seq, .tokens = tokens[total..][0..rows], .parents = parents[total..][0..rows] };
            if (j < want.len) {
                const w = want[j].object;
                const wt = try tokenList(a, w.get("tokens").?);
                const wp = w.get("parents").?.array.items;
                var same = @as(usize, @intCast(w.get("stream").?.integer)) == k and std.mem.eql(u32, wt, tokens[total..][0..rows]) and wp.len == rows;
                if (same) for (wp, parents[total..][0..rows]) |x, y| {
                    if (x.integer != y) same = false;
                };
                if (!same) {
                    differ += 1;
                    if (differ == 1) std.debug.print("round {d} stream {d}: first window that differs from Python's ({s} {d} rows, python {s} {d})\n", .{ round, k, @tagName(m.d.kind), rows, w.get("mode").?.string, wt.len });
                }
            } else differ += 1;
            total += rows;
        }
        starts[n] = total;
        var f = e.forward();
        f.taps = e.scratch.taps;
        try f.round(parts[0..n]);
        try f.picks(total, &picks);
        var paths: [qwen.state.max_streams][]const u32 = undefined;
        var kept: [qwen.state.max_streams]qwen.drafts.Kept = undefined;
        for (live[0..n], 0..) |k, j| {
            const m = &ms[k];
            const base = starts[j];
            const rows = starts[j + 1] - base;
            const room = count - m.out.items.len;
            // accept: children equal to their parent's draw, within the room, none past an end token
            var len: usize = 1;
            path_rows[base] = 0;
            var terminal = picks[base];
            while (len < room and !e.w.isEos(terminal)) {
                const child = for (1..rows) |r| {
                    if (parents[base + r] == @as(i32, @intCast(path_rows[base + len - 1])) and tokens[base + r] == terminal) break r;
                } else break;
                path_rows[base + len] = @intCast(child);
                len += 1;
                terminal = picks[base + child];
            }
            paths[j] = path_rows[base..][0..len];
            for (paths[j], kept_rows[base..][0..len]) |r, *g| g.* = @intCast(base + r);
            kept[j] = .{ .x = &m.d, .rows = kept_rows[base..][0..len] };
            for (paths[j][1..]) |r| {
                try m.out.append(gpa, tokens[base + r]);
                try m.context.append(gpa, tokens[base + r]);
            }
            try m.out.append(gpa, terminal);
            try m.context.append(gpa, terminal);
            m.rounds += 1;
            m.accepted += len - 1;
        }
        try f.commit(parts[0..n], paths[0..n]);
        try qwen.drafts.absorb(d, f.ops, e.scratch.taps, e.scratch.copy_items, kept[0..n]);
        for (live[0..n]) |k| {
            const m = &ms[k];
            if (m.out.items.len >= count or e.w.isEos(m.out.items[m.out.items.len - 1])) m.done = true;
        }
    }
    const secs = @as(f64, @floatFromInt(clock.now(io).toNanoseconds() - t0)) / 1e9;
    if (round != want_rounds.len) differ += 1;
    var bad: usize = 0;
    var produced: usize = 0;
    const results = doc.object.get("results").?.array.items;
    for (ms, results, 0..) |m, rv, k| {
        const want = try tokenList(a, rv.object.get("tokens").?);
        const same = std.mem.eql(u32, want, m.out.items);
        if (!same or m.rounds != rv.object.get("rounds").?.integer) bad += 1;
        produced += m.out.items.len;
        std.debug.print("stream {d}: {d} tokens sha {s} rounds {d} accepted {d} {s}; python rounds {d} accepted {d}\n", .{ k, m.out.items.len, try sha12(a, m.out.items), m.rounds, m.accepted, if (same) "MATCH" else "DIFFER", rv.object.get("rounds").?.integer, rv.object.get("accepted").?.integer });
    }
    // replies and every stream's rounds must match; a tree may order its nodes otherwise where drafts round otherwise
    std.debug.print("{d} rounds (python {d}), {d} windows differ from Python's, {d} replies differ; {d:.1} tok/s\n", .{ round, want_rounds.len, differ, bad, @as(f64, @floatFromInt(produced)) / secs });
    return if (bad == 0 and round == want_rounds.len) 0 else 1;
}
