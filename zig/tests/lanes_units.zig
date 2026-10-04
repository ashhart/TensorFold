//! The lane core's pieces against tools/zig/record_lanes.py units vectors from Python's own functions (no model).
const std = @import("std");
const lanes = @import("lanes");
const fx = @import("fixture.zig");
const depth = lanes.depth;
const Table = lanes.table.Table;

const gpa = std.testing.allocator;
const t = std.testing;

fn tableOf(a: std.mem.Allocator, v: fx.Value) !Table {
    var out: Table = .{};
    for (try fx.pairs(a, v)) |p| try out.put(a, p.key, p.value);
    return out;
}

const expectBits = fx.expectBits;
const expectTable = fx.expectTable;

const Stub = struct {
    state: ?depth.State = null,
    draft_room: i64 = 12,
    finished: bool = false,
    forced: bool = false,
    copy: bool = false,

    fn who(s: *Stub) depth.Who {
        return .{ .state = &s.state, .draft_room = s.draft_room, .finished = s.finished, .forced = s.forced, .stream = s };
    }

    fn copyAhead(_: *anyopaque, w: *const depth.Who) bool {
        const s: *Stub = @ptrCast(@alignCast(w.stream.?));
        return s.copy;
    }
};

fn checkState(a: std.mem.Allocator, rule: *depth.Rule, stubs: []Stub, want: fx.Value) !void {
    try expectTable(a, fx.get(want, "round_ms"), rule.round_ms);
    const over = try fx.pairs(a, fx.get(want, "overhead_ms"));
    try t.expectEqual(over.len, rule.overhead.items.len);
    for (over, rule.overhead.items) |p, o| {
        try t.expectEqual(p.key, @as(i64, o.streams));
        try expectBits(p.value, o.ms);
    }
    for (fx.get(want, "streams").array.items, stubs) |w, *s| {
        if (fx.isNull(w)) {
            try t.expect(s.state == null);
            continue;
        }
        const st = s.state.?;
        try t.expectEqualSlices(f64, try fx.floats(a, fx.get(w, "p")), st.p);
        try t.expectEqual(fx.int(fx.get(w, "rounds")), @as(i64, @intCast(st.rounds)));
        const plain = fx.get(w, "plain");
        try t.expectEqual(if (fx.isNull(plain)) null else @as(?u64, @intCast(fx.int(plain))), st.plain);
        const wait = fx.get(w, "wait");
        try t.expectEqual(if (fx.isNull(wait)) null else @as(?u64, @intCast(fx.int(wait))), st.wait);
        const ms = fx.get(w, "ms");
        try t.expectEqual(fx.isNull(ms), st.ms == null);
        if (st.ms) |table| try expectTable(a, ms, table);
    }
}

fn depthCase(a: std.mem.Allocator, case: fx.Value) !void {
    const c = fx.get(case, "config");
    const family = try tableOf(a, fx.get(c, "family_costs"));
    const shared = try tableOf(a, fx.get(c, "shared_costs"));
    var rule: depth.Rule = .{
        .gpa = gpa,
        .prior = try fx.floats(a, fx.get(c, "depth_prior")),
        .most_drafts = @intCast(fx.int(fx.get(c, "most_drafts"))),
        .family_costs = &family,
        .shared_costs = &shared,
        .mtp_step_ms = fx.float(fx.get(c, "mtp_step_ms")),
        .plain_guard = fx.boolean(fx.get(c, "plain_guard")),
        .node_probabilities = false,
        .batch_rows = @intCast(fx.int(fx.get(c, "batch_rows"))),
    };
    defer rule.deinit();
    const ids = fx.get(c, "streams").array.items;
    const stubs = try a.alloc(Stub, ids.len);
    for (stubs) |*s| s.* = .{};
    defer for (stubs) |*s| if (s.state) |*st| st.deinit(gpa);
    const probe: depth.Probe = .{ .ptr = undefined, .copy = Stub.copyAhead };
    const index = struct {
        fn of(names: []const fx.Value, name: []const u8) usize {
            for (names, 0..) |n, i| if (std.mem.eql(u8, fx.str(n), name)) return i;
            unreachable;
        }
    }.of;
    for (fx.get(case, "ops").array.items) |op| {
        const kind = fx.str(fx.get(op, "op"));
        const sv = op.object.get("stream");
        const s: ?*Stub = if (sv == null or fx.isNull(sv.?)) null else &stubs[index(ids, fx.str(sv.?))];
        if (std.mem.eql(u8, kind, "set")) {
            s.?.draft_room = fx.int(fx.get(op, "draft_room"));
            s.?.forced = fx.int(fx.get(op, "force")) > 0;
            s.?.copy = fx.boolean(fx.get(op, "copy"));
            s.?.finished = fx.boolean(fx.get(op, "finished"));
        } else if (std.mem.eql(u8, kind, "depth")) {
            try t.expectEqual(fx.int(fx.get(op, "result")), try rule.depth(s.?.who()));
        } else if (std.mem.eql(u8, kind, "head")) {
            const b = fx.get(op, "budget");
            const w = s.?.who();
            try t.expectEqual(fx.int(fx.get(op, "result")), try rule.headDepth(&w, if (fx.isNull(b)) null else fx.int(b), probe));
        } else if (std.mem.eql(u8, kind, "observe_depth")) {
            try rule.observeDepth(s.?.who(), fx.int(fx.get(op, "proposed")), fx.int(fx.get(op, "accepted")));
        } else if (std.mem.eql(u8, kind, "observe_cost")) {
            try rule.observeCost(fx.int(fx.get(op, "drafts")), fx.float(fx.get(op, "ms")), fx.boolean(fx.get(op, "init")), if (s) |x| x.who() else null);
        } else if (std.mem.eql(u8, kind, "round_cost")) {
            const got = try rule.roundCost(fx.int(fx.get(op, "drafts")), s.?.who());
            const want = fx.get(op, "result");
            try t.expectEqual(fx.isNull(want), got == null);
            if (got) |g| try expectBits(fx.float(want), g);
        } else if (std.mem.eql(u8, kind, "budgets")) {
            const names = fx.get(op, "streams").array.items;
            const whos = try a.alloc(depth.Who, names.len);
            for (whos, names) |*w, n| w.* = stubs[index(ids, fx.str(n))].who();
            const out = try a.alloc(i64, names.len);
            try rule.budgets(whos, probe, out);
            try t.expectEqualSlices(i64, try fx.i64s(a, fx.get(op, "result")), out);
        } else if (std.mem.eql(u8, kind, "observe_overhead")) {
            try rule.observeOverhead(@intCast(fx.int(fx.get(op, "streams"))), @intCast(fx.int(fx.get(op, "rows"))), fx.float(fx.get(op, "ms")));
        } else if (std.mem.eql(u8, kind, "overhead")) {
            try expectBits(fx.float(fx.get(op, "result")), rule.overheadMs(@intCast(fx.int(fx.get(op, "streams")))));
        } else unreachable;
        try checkState(a, &rule, stubs, fx.get(op, "state"));
    }
}

fn allocateCase(a: std.mem.Allocator, case: fx.Value) !void {
    const costs = try tableOf(a, fx.get(case, "costs"));
    const rows: u32 = @intCast(fx.int(fx.get(case, "rows")));
    if (!costs.empty()) {
        var timed: std.ArrayList(lanes.allocate.Cost) = .empty;
        for (try fx.pairs(a, fx.get(case, "timed"))) |p| try timed.append(a, .{ .width = @intCast(p.key), .ms = p.value });
        var extended = try lanes.allocate.extendCosts(gpa, timed.items, rows);
        defer extended.deinit(gpa);
        for (1..rows + 1) |w| try expectBits(costs.get(@intCast(w)).?, extended.get(@intCast(w)).?);
    }
    const probs_v = fx.get(case, "probs").array.items;
    const probs = try a.alloc([]const f64, probs_v.len);
    for (probs, probs_v) |*p, v| p.* = try fx.floats(a, v);
    const counts = try lanes.allocate.allocate(gpa, try fx.u32s(a, fx.get(case, "fixed")), probs, &costs, fx.float(fx.get(case, "overhead")), @intCast(fx.int(fx.get(case, "max_rows"))));
    defer gpa.free(counts);
    try t.expectEqualSlices(u32, try fx.u32s(a, fx.get(case, "result")), counts);
}

test "depth rule, costs and allocation (depth.jsonl)" {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var cases: [2]usize = .{ 0, 0 };
    var ops: usize = 0;
    for (try fx.lines(a, try fx.read(a, "depth.jsonl"))) |l| {
        const kind = fx.str(fx.get(l.value, "kind"));
        if (std.mem.eql(u8, kind, "depth")) {
            try depthCase(a, l.value);
            ops += fx.get(l.value, "ops").array.items.len;
            cases[0] += 1;
        } else {
            try allocateCase(a, l.value);
            cases[1] += 1;
        }
    }
    std.debug.print("depth rule: {d} cases, {d} ops equal; allocation: {d} cases equal\n", .{ cases[0], ops, cases[1] });
}

fn proposerState(a: std.mem.Allocator, s: *const lanes.SuffixLookup, want: fx.Value) !void {
    try t.expectEqual(fx.int(fx.get(want, "silent_for")), s.silent_for);
    try t.expectEqualSlices(i64, try fx.i64s(a, fx.get(want, "recent")), s.recent.items);
    try t.expectEqual(fx.int(fx.get(want, "proposals")), s.proposals);
    try t.expectEqual(fx.int(fx.get(want, "proposed_tokens")), s.proposed_tokens);
    try t.expectEqual(fx.int(fx.get(want, "judged_tokens")), s.judged_tokens);
    try t.expectEqual(fx.int(fx.get(want, "accepted_tokens")), s.accepted_tokens);
    try t.expectEqual(fx.int(fx.get(want, "silenced_rounds")), s.silenced_rounds);
}

test "suffix proposer (proposer.jsonl)" {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var proposals: usize = 0;
    const all = try fx.lines(a, try fx.read(a, "proposer.jsonl"));
    for (all) |l| {
        const p = fx.get(l.value, "params");
        var s = try lanes.SuffixLookup.init(gpa, .{ .ngram = @intCast(fx.int(fx.get(p, "ngram"))), .min_match = fx.int(fx.get(p, "min_match")), .max_extension = fx.int(fx.get(p, "max_extension")), .silence_rounds = fx.int(fx.get(p, "silence_rounds")), .window = @intCast(fx.int(fx.get(p, "window"))) });
        defer s.deinit();
        var context: std.ArrayList(u32) = .empty;
        try context.appendSlice(a, try fx.u32s(a, fx.get(l.value, "context")));
        for (fx.get(l.value, "ops").array.items) |op| {
            const kind = fx.str(fx.get(op, "op"));
            if (std.mem.eql(u8, kind, "grow")) {
                try context.appendSlice(a, try fx.u32s(a, fx.get(op, "tokens")));
            } else if (std.mem.eql(u8, kind, "reset")) {
                context.clearRetainingCapacity();
                try context.appendSlice(a, try fx.u32s(a, fx.get(op, "tokens")));
            } else if (std.mem.eql(u8, kind, "propose")) {
                const got = try s.propose(context.items, fx.int(fx.get(op, "max_draft")));
                const want = try fx.u32s(a, fx.get(op, "result"));
                if (!std.mem.eql(u32, want, got)) std.debug.print("proposer case {d}, context {d} tokens\n", .{ fx.int(fx.get(l.value, "case")), context.items.len });
                try t.expectEqualSlices(u32, want, got);
                try t.expectEqual(fx.int(fx.get(op, "last_match")), s.last_match);
                try t.expectEqual(fx.boolean(fx.get(op, "confident")), s.last_confident);
                try proposerState(a, &s, fx.get(op, "state"));
                proposals += 1;
            } else {
                try s.observe(fx.int(fx.get(op, "proposed")), fx.int(fx.get(op, "accepted")));
                try proposerState(a, &s, fx.get(op, "state"));
            }
        }
    }
    std.debug.print("proposer: {d} cases, {d} proposals equal\n", .{ all.len, proposals });
}

fn row(a: std.mem.Allocator, values: fx.Value, ids: fx.Value) !struct { []f64, []u64 } {
    const vs = values.array.items;
    const out_v = try a.alloc(f64, vs.len);
    for (out_v, vs) |*o, x| o.* = @as(f32, @bitCast(@as(u32, @intCast(fx.int(x)))));
    const is = ids.array.items;
    const out_i = try a.alloc(u64, is.len);
    for (out_i, is) |*o, x| o.* = @intCast(fx.int(x));
    return .{ out_v, out_i };
}

test "keyed host sampler (sampling.jsonl)" {
    const sampling = lanes.sampling;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var counts: [4]usize = .{ 0, 0, 0, 0 };
    for (try fx.lines(a, try fx.read(a, "sampling.jsonl"))) |l| {
        const v = l.value;
        const kind = fx.str(fx.get(v, "kind"));
        if (std.mem.eql(u8, kind, "seed")) {
            try t.expectEqual(fx.uint(fx.get(v, "result")), sampling.seedFor(try fx.u32s(a, fx.get(v, "tokens")), fx.int(fx.get(v, "salt"))));
            counts[0] += 1;
        } else if (std.mem.eql(u8, kind, "uniform")) {
            const want = fx.get(v, "result").array.items;
            for (fx.get(v, "ids").array.items, want) |id, w| try expectBits(fx.float(w), sampling.uniform(fx.uint(fx.get(v, "seed")), @intCast(fx.int(fx.get(v, "position"))), @intCast(fx.int(id))));
            counts[1] += 1;
        } else if (std.mem.eql(u8, kind, "choose")) {
            const vals, const ids = try row(a, fx.get(v, "values"), fx.get(v, "ids"));
            try t.expectEqual(fx.uint(fx.get(v, "result")), try sampling.choose(a, vals, ids, @intCast(fx.int(fx.get(v, "position"))), fx.sampling(fx.get(v, "sampling"))));
            counts[2] += 1;
        } else {
            const s = fx.sampling(fx.get(v, "sampling"));
            const want = fx.get(v, "result").array.items;
            for (fx.get(v, "values").array.items, fx.get(v, "ids").array.items, fx.get(v, "positions").array.items, want) |vv, iv, p, w| {
                const vals, const ids = try row(a, vv, iv);
                try t.expectEqual(fx.uint(w), try sampling.choose(a, vals, ids, @intCast(fx.int(p)), s));
            }
            counts[3] += 1;
        }
    }
    std.debug.print("host sampler: {d} seeds, {d} uniform sets, {d} rows, {d} row batches equal\n", .{ counts[0], counts[1], counts[2], counts[3] });
}

test "stream commits, thinking cuts and room (stream.jsonl)" {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var ops: usize = 0;
    const all = try fx.lines(a, try fx.read(a, "stream.jsonl"));
    for (all) |l| {
        const spec = fx.get(l.value, "spec");
        var s = try lanes.Stream.init(gpa, .{ .id = "t", .prompt = &.{ 1, 2, 3 }, .max_new = @intCast(fx.int(fx.get(spec, "max_new"))), .eos = try fx.u32s(a, fx.get(spec, "eos")), .think_budget = @intCast(fx.int(fx.get(spec, "think_budget"))), .think_close = try fx.u32s(a, fx.get(spec, "think_close")), .think_end = fx.int(fx.get(spec, "think_end")), .think_open = fx.boolean(fx.get(spec, "think_open")) });
        defer s.deinit(gpa);
        for (fx.get(l.value, "ops").array.items) |op| {
            const kind = fx.str(fx.get(op, "op"));
            const result = fx.get(op, "result");
            if (std.mem.eql(u8, kind, "commit")) {
                const tokens = try fx.u32s(a, fx.get(op, "tokens"));
                const landed = try s.commit(gpa, tokens);
                try t.expectEqualSlices(u32, try fx.u32s(a, result), tokens[0..landed]);
            } else if (std.mem.eql(u8, kind, "cut")) {
                const got = s.thinkCut(try fx.i64s(a, fx.get(op, "tokens")));
                try t.expectEqual(fx.isNull(result), got == null);
                if (got) |g| try t.expectEqual(fx.int(result), @as(i64, @intCast(g)));
            } else if (std.mem.eql(u8, kind, "close")) {
                try t.expectEqual(fx.int(result), @as(i64, try s.startClose(gpa)));
            } else {
                try t.expectEqual(fx.int(result), s.draftRoom());
            }
            const st = fx.get(op, "state");
            try t.expectEqualSlices(u32, try fx.u32s(a, fx.get(st, "emitted")), s.emitted());
            try t.expectEqual(fx.boolean(fx.get(st, "finished")), s.finished);
            try t.expectEqualStrings(fx.str(fx.get(st, "reason")), s.reason.name());
            try t.expectEqual(fx.boolean(fx.get(st, "think_open")), s.think_open);
            try t.expectEqualSlices(u32, try fx.u32s(a, fx.get(st, "force")), s.force.items);
            ops += 1;
        }
    }
    std.debug.print("streams: {d} cases, {d} ops equal\n", .{ all.len, ops });
}
