//! Actual CUDA family copy policies exercised through the real lane planner and an instrumented target.
const std = @import("std");
const lanes = @import("lanes");
const a = std.testing.allocator;
const be = lanes.backend;
const Family = enum { gemma4, spark25, kolibri1, dory, minimax, minimax_head, glm_mtp, glm_block };
fn facts(family: Family, enabled: bool) lanes.Model {
    switch (family) {
        .glm_mtp, .glm_block => {
            const f = @import("glm_cuda");
            var e: f.engine.Engine = undefined;
            e.mtp = true;
            e.drafter = if (family == .glm_block) @ptrFromInt(16) else null;
            var b: f.lanes.Lanes = .{ .x = .{ .e = &e, .link = undefined } };
            return b.factsWithCopies(enabled);
        },
        .gemma4 => {
            const f = @import("gemma4_cuda");
            var m: f.forward.Model = undefined;
            m.paged = false;
            m.slots = 8;
            var b: f.backend.Gemma = undefined;
            b.m = &m;
            b.priority_copies = enabled;
            return b.facts();
        },
        .spark25 => {
            const f = @import("spark25");
            var m: f.forward.Model = undefined;
            m.paged = false;
            m.slots = 8;
            var b: f.backend.Spark = undefined;
            b.m = &m;
            b.priority_copies = enabled;
            return b.facts();
        },
        .dory => {
            const f = @import("dory");
            var m: f.cuda_model.Model = undefined;
            m.paged = false;
            m.slots = 8;
            var b: f.cuda_backend.Dory = undefined;
            b.m = &m;
            b.priority_copies = enabled;
            return b.facts();
        },
        .kolibri1 => {
            const f = @import("kolibri1");
            var m: f.forward.Model = undefined;
            m.slots = 8;
            var b: f.backend.Kolibri = undefined;
            b.m = &m;
            b.priority_copies = enabled;
            return b.facts();
        },
        .minimax, .minimax_head => {
            const f = @import("minimax_m3_cuda");
            var e: f.engine.Engine = undefined;
            e.drafter = if (family == .minimax_head) @ptrFromInt(16) else null;
            var b: f.lanes.Lanes = undefined;
            b.x.e = &e;
            b.priority_copies = enabled;
            return b.facts();
        },
    }
}
const families = std.enums.values(Family);

// Literal main facts for eight slots; independent of the candidate adapters' facts.
fn mainFacts(family: Family) lanes.Model {
    if (family == .glm_mtp or family == .glm_block) {
        const block = family == .glm_block;
        return .{ .allocate_shared_drafts = !block, .exact_width = 32, .first_copy_rows = 4, .gpu_tokens = false, .mtp = true, .speculate = true, .speculate_early = false, .drafts = if (block) 7 else 15, .draft_streams = true, .mtp_step_ms = if (block) 1.6 / @as(f64, 7) else 0, .draft_probabilities = block, .window_costs = if (block) &glm_window_costs else &.{}, .shared_costs = if (block) &glm_shared_costs else &.{}, .shared_drafts = if (block) 0 else 3, .hidden_rows = true, .batch_rows = 32, .max_streams = 16 };
    }
    if (family == .minimax or family == .minimax_head) {
        const head = family == .minimax_head;
        return .{ .exact_width = 16, .gpu_tokens = false, .mtp = head, .speculate = head, .speculate_early = false, .drafts = if (head) 7 else 0, .draft_streams = head, .mtp_step_ms = if (head) 1.5 else 0, .window_costs = if (head) &main_window_costs else &.{}, .shared_costs = if (head) &main_shared_costs else &.{}, .hidden_rows = true, .batch_rows = 32, .max_streams = 16 };
    }
    return .{ .exact_width = 16, .gpu_tokens = false, .mtp = false, .speculate = false, .hidden_rows = true, .batch_rows = if (family == .kolibri1) 64 else 128, .max_streams = 8 };
}
fn glmCosts(comptime n: usize, comptime base: f64, comptime row: f64) [n]lanes.config.Cost {
    var costs: [n]lanes.config.Cost = undefined;
    for (&costs, 0..) |*c, i| c.* = .{ .width = i + 1, .ms = base + row * @as(f64, @floatFromInt(i)) };
    return costs;
}
const glm_window_costs = glmCosts(32, 52.0, 6.8);
const glm_shared_costs = glmCosts(32, 76.0, 4.1);

const main_ms = [_]f64{ 54.5, 64.4, 74.1, 84.1, 94.6, 102.6, 108.5, 114.2, 120.8, 124.9, 130.1, 136.2, 140.6, 145.2, 149.7, 152.7, 166.2, 173.4, 175.9, 182.2, 188.6, 195.5, 199.6, 203.1, 207.6, 211.8, 213.4, 215.3, 223.2, 224.9, 226.0, 244.8 };
fn mainCosts(comptime n: usize) [n]lanes.config.Cost {
    var costs: [n]lanes.config.Cost = undefined;
    for (&costs, main_ms[0..n], 0..) |*c, ms, i| c.* = .{ .width = i + 1, .ms = ms };
    return costs;
}
const main_window_costs = mainCosts(16);
const main_shared_costs = mainCosts(32);

test "CUDA lane copies: default and serving drafts off keep literal main facts" {
    for (families) |family| {
        const main = mainFacts(family);
        try std.testing.expectEqualDeep(main, facts(family, false));
        try std.testing.expect(main.copy_drafts and !main.root_copies and !main.context_copies);
        for ([_]bool{ false, true }) |on| {
            const m = facts(family, on);
            var cfg = try lanes.Config.init(a, m, m.exact_width, m.exact_width - 1);
            defer cfg.deinit(a);
            try std.testing.expectEqual(on, cfg.root_copies);
            try std.testing.expectEqual(on, cfg.context_copies);
            try std.testing.expectEqual(@as(i64, if (on) 4 else 8), cfg.enter_match);
            try std.testing.expectEqual(m.exact_width - 1, cfg.max_copy);
            for ([_]?[]const u8{ null, "0", "1" }) |value| {
                const off = try @import("cuda").priority_copies.enabled(false, value);
                try std.testing.expectEqualDeep(main, facts(family, off));
            }
        }
    }
}

const Copy = struct {
    matched: i64,
    fn propose(_: *anyopaque, _: []const u32, most: i64) ![]const u32 {
        const ids = &[_]u32{ 11, 12, 13, 14 };
        return ids[0..@intCast(@min(4, most))];
    }
    fn match(p: *anyopaque) i64 {
        return @as(*Copy, @ptrCast(@alignCast(p))).matched;
    }
};

test "CUDA lane copies: three matches refuse, four enter, disabled copies and miss refuse" {
    for (families) |family| {
        const m = facts(family, true);
        var cfg = try lanes.Config.init(a, m, m.exact_width, m.exact_width - 1);
        defer cfg.deinit(a);
        var target: lanes.fake.Fake = .{ .gpa = a };
        defer target.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var e = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
        defer e.deinit();
        e.alone = true;
        var copy: Copy = .{ .matched = 3 };
        var s = try lanes.Stream.init(a, .{ .id = "gate", .prompt = &.{ 1, 2, 3, 4 }, .max_new = 64, .proposer = .{ .ptr = &copy, .vtable = &.{ .propose = Copy.propose, .last_match = Copy.match } } });
        defer s.deinit(a);
        try std.testing.expectEqual(@as(usize, 0), (try lanes.windows.copyProposal(&e, &s)).tokens.len);
        copy.matched = 4;
        try std.testing.expect((try lanes.windows.copyProposal(&e, &s)).tokens.len > 0);
        s.root_copy_miss = true;
        try std.testing.expectEqual(@as(usize, 0), (try lanes.windows.copyProposal(&e, &s)).tokens.len);
        s.root_copy_miss = false;
        cfg.max_copy = 0;
        try std.testing.expectEqual(@as(usize, 0), (try lanes.windows.copyProposal(&e, &s)).tokens.len);
    }
}

const Stats = struct { zeros: usize = 0, heads: usize = 0, copied: usize = 0, rejected: usize = 0, trace: u64 = 0 };
threadlocal var current: ?*Stats = null;

fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) !void {
    for (requests) |r| {
        if (r.depth == 0) current.?.zeros += 1 else current.?.heads += 1;
        current.?.trace = std.hash.Wyhash.hash(current.?.trace, std.mem.asBytes(&r.depth));
    }
    const t: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
    try t.backend().vtable.draft(ptr, requests);
}

fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
    const t: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
    try t.backend().vtable.verify(ptr, windows, out);
    for (windows, out) |w, o| {
        current.?.trace = std.hash.Wyhash.hash(current.?.trace, std.mem.sliceAsBytes(w.positions));
        current.?.trace = std.hash.Wyhash.hash(current.?.trace, std.mem.sliceAsBytes(w.tokens));
        if (w.held == 0 and w.tokens.len > 0) {
            current.?.copied += 1;
            if (!std.mem.eql(u32, w.tokens, o.sampled[0..w.tokens.len])) current.?.rejected += 1;
        }
    }
}

const Result = struct { tokens: [8][64]u32 = undefined, stats: Stats = .{} };

fn run(m: lanes.Model, count: usize, sampled: bool, drafted: bool, breaks: bool, restored: bool) !Result {
    var out: Result = .{};
    current = &out.stats;
    defer current = null;
    var prompt: [80]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(11 + i % 4);
    const pattern = if (breaks) &[_]u32{ 11, 12, 13, 14, 11, 12, 13, 14, 91 } else &[_]u32{ 11, 12, 13, 14 };
    var target: lanes.fake.Fake = .{ .gpa = a, .probe_prompt = prompt.len, .cycle_after = 0, .pattern = pattern };
    defer target.deinit();
    var backend = target.backend();
    var vt = backend.vtable.*;
    vt.draft = draft;
    vt.verify = verify;
    backend.vtable = &vt;
    var cfg = try lanes.Config.init(a, m, m.exact_width, m.exact_width - 1);
    defer cfg.deinit(a);
    cfg.timed = false;
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(a, &cfg, backend, clock.clock());
    defer e.deinit();
    var streams: [8]lanes.Stream = undefined;
    var proposers: [8]lanes.SuffixLookup = undefined;
    var made: usize = 0;
    defer for (streams[0..made], proposers[0..made]) |*s, *p| {
        s.deinit(a);
        p.deinit();
    };
    var saved: ?*anyopaque = null;
    defer if (saved) |p| target.drop(p);
    if (restored) {
        var early = try lanes.Stream.init(a, .{ .id = "prefix", .prompt = prompt[0..32], .max_new = 1 });
        defer early.deinit(a);
        try backend.prefill(&early);
        saved = try target.save(&early, 32);
        backend.release(&early);
    }
    for (streams[0..count], proposers[0..count]) |*s, *p| {
        p.* = try lanes.SuffixLookup.init(a, .{ .min_match = 4 });
        s.* = try lanes.Stream.init(a, .{ .id = "copy", .prompt = &prompt, .max_new = 64, .drafts = drafted, .proposer = p.proposer(), .sampling = if (sampled) .{ .seed = 19, .temperature = 0.7 } else null, .reuse = .{ .saved = saved, .at = if (restored) 32 else 0 } });
        made += 1;
        try e.addStream(s);
    }
    var steps: usize = 0;
    while (e.activeCount() > 0 and steps < 512) : (steps += 1) try e.step();
    try std.testing.expectEqual(@as(usize, 0), e.activeCount());
    for (streams[0..count], out.tokens[0..count]) |*s, *ids| {
        try std.testing.expectEqual(@as(usize, 64), s.emitted().len);
        @memcpy(ids, s.emitted());
    }
    return out;
}

test "CUDA lane copies: real core copies equal plain tokens solo shared sampled and restored" {
    for (families) |family| for ([_]bool{ false, true }) |sampled| for ([_]bool{ false, true }) |restored| for ([_]bool{ false, true }) |breaks| {
        const m = facts(family, true);
        const plain = try run(m, 1, sampled, false, breaks, restored);
        var no_copies = mainFacts(family);
        no_copies.copy_drafts = false;
        const disabled = try run(no_copies, 1, sampled, true, breaks, restored);
        try std.testing.expectEqualSlices(u32, &plain.tokens[0], &disabled.tokens[0]);
        try std.testing.expectEqual(@as(usize, 0), disabled.stats.copied);
        for ([_]usize{ 1, 8 }) |count| {
            const got = try run(m, count, sampled, true, breaks, restored);
            for (got.tokens[0..count]) |ids| try std.testing.expectEqualSlices(u32, &plain.tokens[0], &ids);
            try std.testing.expect(got.stats.copied > 0);
            if (m.mtp) try std.testing.expect(got.stats.zeros > 0) else try std.testing.expectEqual(@as(usize, 0), got.stats.heads);
            if (breaks) {
                if (count == 1) try std.testing.expect(got.stats.rejected > 0);
                if (m.mtp) try std.testing.expect(got.stats.heads > 0);
            }
            const before = try run(mainFacts(family), count, sampled, false, breaks, restored);
            const after = try run(m, count, sampled, false, breaks, restored);
            try std.testing.expectEqualDeep(before, after);
        }
    };
}

test "CUDA lane copies: cancel a copied round and the next request equals fresh plain" {
    for (families) |family| {
        const m = facts(family, true);
        const plain = try run(m, 1, false, false, false, false);
        var cfg = try lanes.Config.init(a, m, m.exact_width, m.exact_width - 1);
        defer cfg.deinit(a);
        cfg.timed = false;
        var prompt: [80]u32 = undefined;
        for (&prompt, 0..) |*t, i| t.* = @intCast(11 + i % 4);
        var target: lanes.fake.Fake = .{ .gpa = a, .probe_prompt = prompt.len, .cycle_after = 0, .pattern = &.{ 11, 12, 13, 14 } };
        defer target.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var e = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
        defer e.deinit();
        for (0..2) |request| {
            var suffix = try lanes.SuffixLookup.init(a, .{ .min_match = 4 });
            defer suffix.deinit();
            var s = try lanes.Stream.init(a, .{ .id = "cancel-next", .prompt = &prompt, .max_new = 64, .proposer = suffix.proposer() });
            defer s.deinit(a);
            try e.addStream(&s);
            try e.step();
            try std.testing.expect(e.drafted > 0);
            if (request == 0) {
                e.discard(&s);
                try std.testing.expectEqual(lanes.stream.Reason.cancelled, s.reason);
            } else {
                while (e.activeCount() > 0) try e.step();
                try std.testing.expectEqualSlices(u32, &plain.tokens[0], s.emitted());
            }
            try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
        }
    }
}

fn glmGrant(model: lanes.Model, expensive: bool, count: usize) ![4]usize {
    var m = model;
    var costs: [32]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, rows| c.* = .{ .width = @intCast(rows), .ms = if (expensive and rows > count) 1000 else 10 + 0.01 * @as(f64, @floatFromInt(rows)) };
    m.shared_costs = &costs;
    var cfg = try lanes.Config.init(a, m, m.exact_width, m.drafts);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    var streams: [4]lanes.Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(a);
    var plans: [4]lanes.windows.Plan = undefined;
    for (streams[0..count], plans[0..count]) |*s, *p| {
        s.* = try lanes.Stream.init(a, .{ .id = "priced-copy", .prompt = &.{ 1, 2 }, .max_new = 64 });
        made += 1;
        p.* = .{ .stream = s, .position = 2, .kind = .copy, .tokens = &.{ 11, 12, 13, 14, 15, 16, 17 } };
    }
    try lanes.windows.allocate(&e, plans[0..count]);
    var grants: [4]usize = @splat(0);
    for (plans[0..count], grants[0..count]) |p, *g| g.* = p.count();
    return grants;
}

test "CUDA lane copies: GLM uses paid target costs and expensive copies get zero rows with or without confidence" {
    for ([_]Family{ .glm_mtp, .glm_block }) |family| {
        const m = facts(family, true);
        try std.testing.expect(m.own_copy_rates and m.plain_guard);
        try std.testing.expectEqual(@as(usize, @import("glm_cuda").lanes.max_rows), m.window_costs.len);
        try std.testing.expectEqual(@as(usize, 32), m.shared_costs.len);
        for ([_]usize{ 1, 4 }) |count| {
            const cheap = try glmGrant(m, false, count);
            const costly = try glmGrant(m, true, count);
            for (cheap[0..count]) |g| try std.testing.expectEqual(@as(usize, 7), g);
            for (costly[0..count]) |g| try std.testing.expectEqual(@as(usize, 0), g);
        }
    }
    const f = @import("glm_cuda");
    var target: f.engine.Engine = undefined;
    target.gpa = a;
    target.mtp = true;
    target.drafter = null;
    var b: f.lanes.Lanes = .{ .x = .{ .e = &target, .link = undefined }, .mtp_confidence = 0.7 };
    defer b.lanes.deinit(a);
    var s = try lanes.Stream.init(a, .{ .id = "confidence-cost", .prompt = &.{1}, .max_new = 16 });
    defer s.deinit(a);
    const l = try b.claim(&s);
    l.held_n = 3;
    @memcpy(l.chances[0..3], &[_]f64{ 0.8, 0.8, 0.8 });
    var probabilities: [3]f64 = undefined;
    const backend = b.backend();
    try std.testing.expect(try backend.vtable.probabilities.?(backend.ptr, &s, &probabilities));
    try std.testing.expect(probabilities[1] > 0.6 and probabilities[2] > 0.5);
    const on = b.factsWithCopies(true);
    try std.testing.expect(try backend.vtable.probabilities.?(backend.ptr, &s, &probabilities));
    try std.testing.expectEqualSlices(f64, &.{ 0.8, 0, 0 }, &probabilities);
    try std.testing.expectEqual(@as(?f64, 0.7), b.facts().draft_confidence);
    const costly = try glmGrant(on, true, 1);
    try std.testing.expectEqual(@as(usize, 0), costly[0]);
    try std.testing.expectEqual(@as(?f64, null), on.draft_confidence);
    target.mtp = false;
    try std.testing.expectEqualDeep(b.facts(), b.factsWithCopies(true));
    try std.testing.expectEqualStrings("TF_GLM_COPIES: expected 0 or 1 for verified root/context copies", f.native.explain(null, error.BadCopyPolicy).?);
}

test "CUDA lane copies: GLM seed costs and observed rejections cut an unprofitable copy to plain" {
    for ([_]Family{ .glm_mtp, .glm_block }) |family| {
        const m = facts(family, true);
        var cfg = try lanes.Config.init(a, m, m.exact_width, m.drafts);
        defer cfg.deinit(a);
        var target: lanes.fake.Fake = .{ .gpa = a };
        defer target.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var e = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
        defer e.deinit();
        e.alone = true;
        var s = try lanes.Stream.init(a, .{ .id = "broken-copy", .prompt = &.{ 1, 2 }, .max_new = 64 });
        defer s.deinit(a);
        var before = [_]lanes.windows.Plan{.{ .stream = &s, .position = 2, .kind = .copy, .tokens = &.{ 11, 12, 13, 14, 15, 16, 17 } }};
        try lanes.windows.allocate(&e, &before);
        try std.testing.expect(before[0].count() > 0);
        for (0..24) |_| lanes.windows.observeCopied(&e, &s, false, 0, 7);
        var after = [_]lanes.windows.Plan{.{ .stream = &s, .position = 2, .kind = .copy, .tokens = &.{ 11, 12, 13, 14, 15, 16, 17 } }};
        try lanes.windows.allocate(&e, &after);
        try std.testing.expectEqual(@as(usize, 0), after[0].count());
    }
}

const GlmHead = struct {
    fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) !void {
        const b: *@import("glm_cuda").lanes.Lanes = @ptrCast(@alignCast(ptr));
        for (requests) |r| {
            const l = try b.claim(r.stream);
            l.held_n = r.depth;
            @memset(l.chances[0..l.held_n], 0.8);
        }
    }
};

test "CUDA lane copies: GLM confidence bounds the actual held offer even when wider rows are cheaper" {
    const f = @import("glm_cuda");
    var target: f.engine.Engine = undefined;
    target.gpa = a;
    target.mtp = true;
    target.drafter = null;
    var b: f.lanes.Lanes = .{ .x = .{ .e = &target, .link = undefined }, .mtp_confidence = 0.7 };
    defer b.lanes.deinit(a);
    var m = b.factsWithCopies(true);
    m.shared_costs = &.{ .{ .width = 1, .ms = 10 }, .{ .width = 2, .ms = 10 }, .{ .width = 3, .ms = 0.1 }, .{ .width = 4, .ms = 0.1 } };
    var cfg = try lanes.Config.init(a, m, 8, 3);
    defer cfg.deinit(a);
    var backend = b.backend();
    var vt = backend.vtable.*;
    vt.draft = GlmHead.draft;
    backend.vtable = &vt;
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(a, &cfg, backend, clock.clock());
    defer e.deinit();
    e.alone = true;
    var s = try lanes.Stream.init(a, .{ .id = "held-cap", .prompt = &.{1}, .max_new = 64 });
    defer s.deinit(a);
    for ([_]f64{ 0, 0.7, 1 }, [_]u32{ 3, 1, 1 }) |floor, count| {
        b.mtp_confidence = floor;
        try lanes.Engine.testing.askDrafts(&e, &.{.{ .stream = &s, .depth = 3, .position = 1, .follow = &.{}, .rows = null, .start = 0 }});
        try std.testing.expectEqual(count, s.next.?.count);
        var plans = [_]lanes.windows.Plan{try lanes.windows.plan(&e, &s, null)};
        try lanes.windows.allocate(&e, &plans);
        try std.testing.expect(plans[0].count() <= count);
    }
    b.mtp_confidence = 0.7;
    _ = b.factsWithCopies(false);
    try lanes.Engine.testing.askDrafts(&e, &.{.{ .stream = &s, .depth = 3, .position = 1, .follow = &.{}, .rows = null, .start = 0 }});
    try std.testing.expectEqual(@as(u32, 3), s.next.?.count);
}

test "GLM window prices: every width from the timed ones, straight between them and never falling" {
    const f = @import("glm_cuda").lane_costs;
    var k: f.Kept = .{ .ms = undefined, .step_ms = 0.25, .fact = 1 };
    for (&k.ms, f.widths) |*m, w| m.* = 10 + 0.5 * @as(f64, @floatFromInt(w));
    k.ms[4] = k.ms[5] + 1; // a dip the sweep's smoothing let through
    try std.testing.expect(f.usable(k));
    const t = f.table(k);
    for (t.window, 1..) |c, w| try std.testing.expectEqual(@as(u32, @intCast(w)), c.width);
    for (t.window[1..], t.window[0 .. t.window.len - 1]) |c, before| try std.testing.expect(c.ms >= before.ms);
    try std.testing.expectEqual(k.ms[0], t.window[0].ms);
    try std.testing.expectApproxEqAbs(@as(f64, 10 + 0.5 * 20), t.window[19].ms, 1e-9); // between 16 and 24
    try std.testing.expectEqual(k.ms[f.widths.len - 1], t.window[t.window.len - 1].ms);
    k.ms[2] = 0;
    try std.testing.expect(!f.usable(k));
    k.ms[2] = 1;
    k.step_ms = std.math.nan(f64);
    try std.testing.expect(!f.usable(k));
}

test "GLM window prices: MTP takes timed prices and DFlash2 keeps every consumed seed price" {
    const f = @import("glm_cuda");
    var k: f.lane_costs.Kept = .{ .ms = undefined, .step_ms = 0.25, .fact = 0 };
    for (&k.ms, f.lane_costs.widths) |*m, w| m.* = 9 + 0.5 * @as(f64, @floatFromInt(w));
    var e: f.engine.Engine = undefined;
    e.mtp = true;
    for ([_]?*@TypeOf(e.drafter.?.*){ null, @ptrFromInt(16) }) |drafter| {
        e.drafter = drafter;
        var b: f.lanes.Lanes = .{ .x = .{ .e = &e, .link = undefined } };
        const seeded = b.factsWithCopies(true);
        try std.testing.expect(seeded.window_costs[0].ms > 50);
        b.costs = f.lane_costs.table(k);
        for ([_]bool{ false, true }) |copies| {
            const m = b.factsWithCopies(copies);
            try std.testing.expectEqual(@as(usize, f.lanes.max_rows), m.window_costs.len);
            const family: Family = if (drafter == null) .glm_mtp else .glm_block;
            const main = mainFacts(family);
            try std.testing.expectEqual(if (drafter == null) k.ms[0] else main.window_costs[0].ms, m.window_costs[0].ms);
            if (drafter == null) try std.testing.expectEqual(@intFromPtr(b.costs.?.window[0..].ptr), @intFromPtr(m.window_costs.ptr));
            try std.testing.expectEqualDeep(main.shared_costs, m.shared_costs);
            var cfg = try lanes.Config.init(a, m, 32, 31);
            defer cfg.deinit(a);
            var fake: lanes.fake.Fake = .{ .gpa = a };
            defer fake.deinit();
            var clock: lanes.fake.FixedClock = .{};
            var core = lanes.Engine.init(a, &cfg, fake.backend(), clock.clock());
            defer core.deinit();
            try std.testing.expectEqual(main.mtp_step_ms, core.rule.sharedStepMs());
            const steps: f64 = 7;
            try std.testing.expectApproxEqAbs(main.mtp_step_ms * steps, core.rule.headMs(&.{ 7, 7 }), 1e-12);
            var main_cfg = try lanes.Config.init(a, main, 32, 31);
            defer main_cfg.deinit(a);
            if (!copies) for (1..33) |width| {
                try std.testing.expectEqual(main_cfg.shared_costs.get(@intCast(width)), cfg.shared_costs.get(@intCast(width)));
            };
            try std.testing.expectEqual(if (drafter == null) k.step_ms else main.mtp_step_ms, m.mtp_step_ms);
            try std.testing.expectEqual(copies, m.root_copies);
            try std.testing.expectEqual(!copies and drafter == null, m.window_costs_lone);
        }
    }
}
