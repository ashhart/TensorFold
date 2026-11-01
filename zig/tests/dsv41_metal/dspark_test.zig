//! The DSpark drafter on Metal: every stage within its float64 bound, rows and slots at once == alone, bit for bit.
const std = @import("std");
const pkg = @import("dsv41_metal");
const host = @import("dsv41");
const fixture = @import("fixture.zig");
const fixture_ds = @import("fixture_ds.zig");
const probe = @import("probe.zig");
const check = @import("check_dspark.zig");

const testing = std.testing;
const Ds = fixture_ds.Ds;
const Out = fixture_ds.Out;
const st = pkg.dspark_stage;
const limit = 1280;

fn copy(comptime T: type, a: std.mem.Allocator, r: probe.Ref, n: usize) ![]T {
    return a.dupe(T, @as([*]const T, @ptrCast(@alignCast(r.buf.contents() + r.off)))[0..n]);
}

fn probeOf(a: std.mem.Allocator, ds: *Ds, stage: usize, rows: usize) probe.Probe {
    const c = &ds.r.c;
    return .{ .a = a, .r = ds.r, .w = &ds.w.stages[stage], .rows = rows, .p0 = 0, .moe = .{ c.dspark_experts, c.dspark_topk } };
}

/// Main rows from `start` into `slot`, each stage's step checked after its own launch; returns the failures.
fn ingestChecked(a: std.mem.Allocator, ds: *Ds, slot: u32, start: u64, n: u32, seed: u64) !usize {
    const c = &ds.r.c;
    const buf = try ds.taps(start, n, seed);
    defer buf.deinit();
    try ds.pass.plan.writeIngest(c, slot, start, n);
    const refs = ds.pass.plan.ingestRefs();
    const cx = ds.pass.ctx(n);
    var at: [16]i32 = undefined;
    var dest: [16]i32 = undefined;
    for (0..n) |r| {
        at[r] = @intCast(start + r);
        dest[r] = @intCast(pkg.dspark_plan.ringRow(c, slot, start + r));
    }
    var bad: usize = 0;
    {
        const pass = fixture.Pass.begin(ds.r.queue);
        try st.mainRows(&cx, pass.e, &ds.w, .{ .buf = buf }, .{ .buf = ds.pass.quant });
        try pass.finish();
        var pb = probeOf(a, ds, 0, n);
        defer pb.deinit();
        try check.mainRows(&pb, ds, .{ .buf = buf });
        bad += pb.report();
    }
    for (ds.w.stages, 0..) |*w, i| {
        const hq = try copy(u16, a, ds.r.s.hq, n * c.hidden);
        defer a.free(hq);
        const pass = fixture.Pass.begin(ds.r.queue);
        try st.keys(&cx, pass.e, w, ds.r.s.hq, ds.pass.rings[i], refs.pos, refs.dest);
        try pass.finish();
        var pb = probeOf(a, ds, i, n);
        defer pb.deinit();
        try check.keysOf(&pb, ds, i, at[0..n], dest[0..n]);
        bad += pb.report();
    }
    ds.pass.runs[slot].land(start, n);
    return bad;
}

/// 1 when the named stage went red (a planted fault caught), printed either way.
fn red(pb: *const probe.Probe, name: []const u8) usize {
    for (pb.stages.items) |s| if (std.mem.eql(u8, s.name, name)) {
        std.debug.print("control {s} (layer {d}): {d} of {d} values red\n", .{ name, pb.at orelse pb.w.layer, s.bad, s.n });
        return @intFromBool(s.bad > 0);
    };
    return 0;
}

/// A launch run stage by stage, each checked after its pass, then the chain; `controls` plants three faults instead.
fn stagewise(a: std.mem.Allocator, ds: *Ds, asks: []const pkg.dspark.Ask, cx: check.Ctx, controls: bool) !usize {
    const c = &ds.r.c;
    const H = c.hidden;
    var l = try ds.pass.prepare(asks);
    const R = l.rows;
    var bad: usize = 0;
    {
        const pass = fixture.Pass.begin(ds.r.queue);
        ds.pass.encodeEmbed(pass.e, &l);
        try pass.finish();
        var pb = probeOf(a, ds, 0, R);
        defer pb.deinit();
        try check.embed(&pb, ds, &l);
        bad += pb.report();
    }
    for (0..ds.w.stages.len) |i| {
        const s = i % 2;
        const x = try copy(u16, a, ds.pass.b.streams[s], R * 4 * H);
        defer a.free(x);
        const pre = try copy(f32, a, ds.pass.b.pres[s], R * 4);
        defer a.free(pre);
        const pass = fixture.Pass.begin(ds.r.queue);
        try ds.pass.encodeStage(pass.e, &l, i);
        try pass.finish();
        if (controls and i == 1) {
            const ring = ds.pass.rings[i].slice(u16, ds.pass.opt.slots * pkg.dspark_plan.poolRows(c) * c.head_dim);
            const row = pkg.dspark_plan.ringRow(c, asks[0].slot, asks[0].start - 1) * c.head_dim;
            for (ring[row..][0..c.head_dim]) |*v| v.* ^= 0x8000;
        }
        const sink: [*]f32 = @ptrCast(@alignCast(ds.w.stages[i].arena.contents() + ds.w.stages[i].sink.off));
        const sink0 = sink[0];
        if (controls and i == 2) sink[0] = sink0 + 4;
        var pb = probeOf(a, ds, i, R);
        defer pb.deinit();
        try check.stage(&pb, ds, &l, i, x, pre, cx);
        sink[0] = sink0;
        bad += if (controls) red(&pb, "drafter attention") else pb.report();
    }
    const last = ds.w.stages.len % 2;
    const xl = try copy(u16, a, ds.pass.b.streams[last], R * 4 * H);
    defer a.free(xl);
    const pl = try copy(f32, a, ds.pass.b.pres[last], R * 4);
    defer a.free(pl);
    {
        const pass = fixture.Pass.begin(ds.r.queue);
        ds.pass.encodeHead(pass.e, &l);
        try pass.finish();
        var pb = probeOf(a, ds, 0, R);
        pb.at = c.layers + c.mtp_layers;
        defer pb.deinit();
        try check.head(&pb, ds, xl, pl);
        bad += pb.report();
    }
    const outs = try a.alloc(Out, asks.len);
    defer a.free(outs);
    const props = try a.alloc(pkg.dspark.Proposal, asks.len);
    defer a.free(props);
    for (outs, props) |*o, *pr| {
        o.* = try Out.init(a, c);
        pr.* = o.proposal();
    }
    defer for (outs) |o| o.deinit(a);
    try ds.pass.finish(&l, props);
    if (controls) outs[0].drafts[2] = (outs[0].drafts[2] + 1) % c.vocab;
    var pb = probeOf(a, ds, 0, R);
    defer pb.deinit();
    try check.chain(&pb, ds, &l, outs);
    bad += if (controls) red(&pb, "chain drafts") else pb.report();
    return bad;
}

fn history(a: std.mem.Allocator, ds: *Ds, seed: u64) !usize {
    var bad: usize = 0;
    // slot 0: 200 rows, more than a window (the ring wraps; the first 72 are never read)
    try ds.ingest(0, 0, 200, seed);
    // slot 1: 37 rows through the production ingest, then 3 more with every stage checked
    try ds.ingest(1, 0, 37, seed);
    bad += try ingestChecked(a, ds, 1, 37, 3, seed);
    // slot 2: a context that restarts at position 500 (rows before it are not on the device)
    try ds.ingest(2, 500, 20, seed);
    return bad;
}

test "dsv41 metal dspark: every stage of the drafter within its float64 bound (ingest of main rows, 3 slots in a launch)" {
    const a = testing.allocator;
    const ds = try Ds.init(a, testing.io, limit, 61, 3);
    defer ds.deinit();
    var bad = try history(a, ds, 5);
    const asks = [_]pkg.dspark.Ask{ Ds.ask(0, 17, 200, null), Ds.ask(1, 90, 40, null), Ds.ask(2, 33, 520, null) };
    const first = [_]u64{ 72, 0, 500 };
    const count = [_]u32{ 128, 40, 20 };
    for (asks, first, count) |k, f, n| {
        const got = ds.pass.context(k.slot, k.start);
        try testing.expectEqual(f, got.first);
        try testing.expectEqual(n, got.count);
    }
    bad += try stagewise(a, ds, &asks, .{ .first = &first, .count = &count }, false);
    try testing.expectEqual(@as(usize, 0), bad);
}

test "dsv41 metal dspark: planted faults turn their checks red (a context key, a sink, a draft)" {
    const a = testing.allocator;
    const ds = try Ds.init(a, testing.io, limit, 62, 3);
    defer ds.deinit();
    _ = try history(a, ds, 6);
    const asks = [_]pkg.dspark.Ask{Ds.ask(0, 21, 200, null)};
    const caught = try stagewise(a, ds, &asks, .{ .first = &.{72}, .count = &.{128} }, true);
    std.debug.print("{s} dsv41 metal dspark controls: {d} of 3 planted faults red\n", .{ if (caught == 3) "PASS" else "FAIL", caught });
    try testing.expectEqual(@as(usize, 3), caught);
}

fn ringBytes(ds: *const Ds, slot: u32, stage: usize) []const u8 {
    const c = &ds.r.c;
    const row = c.head_dim * 2;
    const pool = pkg.dspark_plan.poolRows(c);
    return ds.pass.rings[stage].contents()[slot * pool * row ..][0 .. c.window * row];
}

fn proposeOne(a: std.mem.Allocator, ds: *Ds, k: pkg.dspark.Ask) !Out {
    const o = try Out.init(a, &ds.r.c);
    var p = [1]pkg.dspark.Proposal{o.proposal()};
    try ds.pass.propose(&.{k}, &p);
    return o;
}

test "dsv41 metal dspark: R rows ingested in one launch equal the same rows one at a time, R = 1..16" {
    const a = testing.allocator;
    const ds = try Ds.init(a, testing.io, limit, 63, 2);
    defer ds.deinit();
    var bad: usize = 0;
    for (1..17) |rows| {
        const at: u64 = 300 + 7 * rows;
        for ([_]u32{ 0, 1 }) |slot| {
            ds.pass.reset(slot);
            try ds.ingest(slot, at - 60, 60, 70 + rows);
        }
        // slot 0 takes the rows in one launch, slot 1 one launch a row
        try ds.ingest(0, at, rows, 70 + rows);
        for (0..rows) |i| try ds.ingest(1, at + i, 1, 70 + rows);
        var differ: usize = 0;
        for (0..ds.w.stages.len) |s| for (ringBytes(ds, 0, s), ringBytes(ds, 1, s)) |p, q| {
            differ += @intFromBool(p != q);
        };
        const p = at + rows;
        const o0 = try proposeOne(a, ds, Ds.ask(0, 11, p, null));
        defer o0.deinit(a);
        const o1 = try proposeOne(a, ds, Ds.ask(1, 11, p, null));
        defer o1.deinit(a);
        differ += o0.diff(o1);
        std.debug.print("{s} dsv41 metal dspark ingest: {d} rows a launch vs one: {d} values differ (ring bytes, then the drafts after)\n", .{ if (differ == 0) "PASS" else "FAIL", rows, differ });
        bad += differ;
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

fn sampled(p: u64) host.dspark.Params {
    return .{ .seed = 0x5eed + p, .pos0 = p + 1, .top_k = 40, .sampled = true, .temperature = 0.9, .top_p = 0.95, .min_p = 0.01 };
}

/// `n` slots proposed in one pass and each alone: results and the block keys' rings bit for bit.
fn together(a: std.mem.Allocator, ds: *Ds, asks: []const pkg.dspark.Ask) !usize {
    const c = &ds.r.c;
    const n = asks.len;
    const outs = try a.alloc(Out, 2 * n);
    defer a.free(outs);
    for (outs) |*o| o.* = try Out.init(a, c);
    defer for (outs) |o| o.deinit(a);
    const props = try a.alloc(pkg.dspark.Proposal, n);
    defer a.free(props);
    for (props, outs[0..n]) |*p, o| p.* = o.proposal();
    try ds.pass.propose(asks, props);
    // the block rows of every slot after the joint pass
    const pool = pkg.dspark_plan.poolRows(c);
    const row = c.head_dim * 2;
    const blocks = try a.alloc(u8, n * ds.w.stages.len * c.dspark_block * row);
    defer a.free(blocks);
    var w: usize = 0;
    for (asks) |k| {
        for (0..ds.w.stages.len) |s| {
            const src = ds.pass.rings[s].contents()[(k.slot * pool + c.window) * row ..][0 .. c.dspark_block * row];
            @memcpy(blocks[w..][0..src.len], src);
            w += src.len;
        }
    }
    var differ: usize = 0;
    for (asks, 0..) |k, i| {
        const alone = try proposeOne(a, ds, k);
        defer alone.deinit(a);
        differ += alone.diff(outs[i]);
        var at: usize = 0;
        for (asks, 0..) |k2, j2| {
            for (0..ds.w.stages.len) |s| {
                const len = c.dspark_block * row;
                if (j2 == i) {
                    const now = ds.pass.rings[s].contents()[(k2.slot * pool + c.window) * row ..][0..len];
                    for (now, blocks[at..][0..len]) |p, q| differ += @intFromBool(p != q);
                }
                at += len;
            }
        }
    }
    return differ;
}

test "dsv41 metal dspark: 2 and 3 slots in one pass equal each alone (greedy and sampled)" {
    const a = testing.allocator;
    const ds = try Ds.init(a, testing.io, limit, 64, 3);
    defer ds.deinit();
    _ = try history(a, ds, 7);
    var bad: usize = 0;
    const greedy = [_]pkg.dspark.Ask{ Ds.ask(0, 17, 200, null), Ds.ask(1, 90, 40, null), Ds.ask(2, 33, 520, null) };
    const samp = [_]pkg.dspark.Ask{ Ds.ask(0, 17, 200, sampled(200)), Ds.ask(1, 90, 40, sampled(40)), Ds.ask(2, 33, 520, sampled(520)) };
    const cases = [_]struct { []const pkg.dspark.Ask, []const u8 }{
        .{ greedy[0..2], "2 slots greedy" }, .{ greedy[0..3], "3 slots greedy" },
        .{ samp[0..2], "2 slots sampled" },  .{ samp[1..3], "2 slots sampled (slots 1, 2)" },
        .{ samp[0..3], "3 slots sampled" },
    };
    for (cases) |cs| {
        const d = try together(a, ds, cs[0]);
        std.debug.print("{s} dsv41 metal dspark {s}: {d} values differ from each slot alone\n", .{ if (d == 0) "PASS" else "FAIL", cs[1], d });
        bad += d;
    }
    // more asks than a launch holds: two launches, still each alone
    try testing.expectError(error.DuplicateSlot, ds.pass.prepare(&.{ greedy[0], greedy[0] }));
    try testing.expectError(error.Slots, ds.pass.prepare(&.{ greedy[0], greedy[1], greedy[2], greedy[0] }));
    try testing.expectEqual(@as(usize, 0), bad);
}

test "dsv41 metal dspark: a pass of more asks than a launch holds splits into launches, each ask the same as alone" {
    const a = testing.allocator;
    const ds = try Ds.init(a, testing.io, limit, 65, 4);
    defer ds.deinit();
    for (0..4) |s| try ds.ingest(@intCast(s), 0, 30 + 11 * s, 8);
    var asks: [4]pkg.dspark.Ask = undefined;
    for (&asks, 0..) |*k, s| k.* = Ds.ask(@intCast(s), @intCast(40 + s), 30 + 11 * s, null);
    const outs = try a.alloc(Out, 8);
    defer a.free(outs);
    for (outs) |*o| o.* = try Out.init(a, &ds.r.c);
    defer for (outs) |o| o.deinit(a);
    var props: [4]pkg.dspark.Proposal = undefined;
    for (&props, outs[0..4]) |*p, o| p.* = o.proposal();
    try ds.pass.propose(&asks, &props);
    var differ: usize = 0;
    for (asks, 0..) |k, i| {
        const alone = try proposeOne(a, ds, k);
        defer alone.deinit(a);
        differ += alone.diff(outs[i]);
    }
    std.debug.print("{s} dsv41 metal dspark 4 asks (launches of {d} slots): {d} values differ from each alone\n", .{ if (differ == 0) "PASS" else "FAIL", ds.pass.group(), differ });
    try testing.expectEqual(@as(usize, 0), differ);
}

test "dsv41 metal dspark: refusals (rows past the taps, a position past the tables, a draft block with no drafter)" {
    const a = testing.allocator;
    const ds = try Ds.init(a, testing.io, limit, 66, 2);
    defer ds.deinit();
    const buf = try ds.taps(0, 4, 1);
    defer buf.deinit();
    try testing.expectError(error.NotAPrefix, ds.pass.ingest(0, 0, .{ .buf = buf, .rows = 4 }, &.{ 0, 2 }));
    try testing.expectError(error.NotAPrefix, ds.pass.ingest(0, 0, .{ .buf = buf, .rows = 4 }, &.{ 3, 4 }));
    try testing.expectError(error.Slots, ds.pass.ingest(5, 0, .{ .buf = buf, .rows = 4 }, &.{0}));
    try testing.expectError(error.PastLimit, ds.pass.prepare(&.{Ds.ask(0, 1, limit - 2, null)}));
    var c = fixture.dsparkConfig();
    try pkg.pipelines.check(&c);
    c.dspark_experts = 12;
    try testing.expectError(error.UnsupportedDspark, pkg.pipelines.check(&c));
}
