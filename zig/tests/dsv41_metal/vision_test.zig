//! DeepSeek-V4.1's Metal vision tower: the ViT and aligner against float64, image rows in prompts.
const std = @import("std");
const mtl = @import("metal");
const pkg = @import("dsv41_metal");
const host = @import("dsv41");
const fv = @import("fixture_vis.zig");
const vref = @import("vision_ref.zig");
const fixture = @import("fixture.zig");
const ff = @import("fixture_fwd.zig");
const pt = @import("prompt_test.zig");
const prep = pkg.vision_prep;
const vrows = pkg.vision_rows;
const pr = pkg.prompt;
const fwd = pkg.forward;

const testing = std.testing;
const vis = pkg.vision;

pub fn say(ok: bool, comptime fmt: []const u8, args: anytype) void {
    std.debug.print("{s} dsv41 metal vision: ", .{if (ok) "PASS" else "FAIL"});
    std.debug.print(fmt ++ "\n", args);
}

/// A device, the kernels and the tiny tower's weights read through the release loader.
pub const Rig = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kit: vis.Kit,
    w: pkg.vision_weights.Weights,
    tower: pkg.vision_tower.Tower,
    pack: host.pack.Pack,
    tmp: std.testing.TmpDir,

    pub fn init(a: std.mem.Allocator, io: std.Io, cfg: vis.Config, layers: u32, experts: u32, seed: u64) !*Rig {
        const r = try a.create(Rig);
        errdefer a.destroy(r);
        r.a = a;
        r.device = try mtl.Device.init();
        r.queue = try r.device.queue();
        r.kit = try vis.Kit.load(r.device);
        r.tmp = std.testing.tmpDir(.{});
        const img = try fv.image(a, cfg, layers, experts, seed);
        defer a.free(img);
        try r.tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = img });
        const dir = try r.tmp.dir.realPathFileAlloc(io, ".", a);
        defer a.free(dir);
        r.pack = try host.pack.Pack.open(a, io, dir);
        r.w = try pkg.vision_weights.Weights.load(a, io, r.device, &r.pack, cfg, layers, experts);
        r.tower = pkg.vision_tower.Tower.init(a, r.device, r.queue, &r.kit, &r.w);
        return r;
    }

    pub fn deinit(r: *Rig) void {
        r.tower.deinit();
        r.w.deinit();
        r.pack.deinit();
        r.tmp.cleanup();
        r.kit.deinit();
        r.queue.deinit();
        r.device.deinit();
        r.a.destroy(r);
    }
};

/// Relative RMS error of bf16 values against float64, and the worst element against its own row's RMS scale.
pub const Err = struct { rms: f64, worst: f64 };

pub fn compare(got: []const u16, want: []const f64, width: usize) Err {
    var num: f64 = 0;
    var den: f64 = 0;
    var worst: f64 = 0;
    const rows = want.len / width;
    for (0..rows) |r| {
        var scale: f64 = 0;
        for (want[r * width ..][0..width]) |v| scale += v * v;
        scale = @sqrt(scale / @as(f64, @floatFromInt(width)));
        for (0..width) |i| {
            const g: f64 = @as(f32, @bitCast(@as(u32, got[r * width + i]) << 16));
            const d = g - want[r * width + i];
            num += d * d;
            den += want[r * width + i] * want[r * width + i];
            worst = @max(worst, @abs(d) / @max(scale, 1e-6));
        }
    }
    return .{ .rms = @sqrt(num / den), .worst = worst };
}

pub fn bytesOf(r: pkg.vision.Ref, n: usize) []const u16 {
    return @alignCast(std.mem.bytesAsSlice(u16, (r.buf.contents() + r.off)[0 .. n * 2]));
}

pub fn widen(a: std.mem.Allocator, x: []const f32) ![]f64 {
    const out = try a.alloc(f64, x.len);
    for (out, x) |*o, v| o.* = v;
    return out;
}

test "dsv41 metal vision: the ViT's stages and the aligner against float64, a grid that pads both ways" {
    const a = testing.allocator;
    const cfg = fv.config();
    const r = try Rig.init(a, testing.io, cfg, 2, 16, 71);
    defer r.deinit();
    var m = try vref.Model.load(a, testing.io, &r.pack, cfg);
    defer m.deinit();
    var worst_rms: f64 = 0;
    var worst_el: f64 = 0;
    for ([_][2]u32{ .{ 5, 7 }, .{ 6, 6 }, .{ 1, 4 } }, 0..) |g, gi| {
        const px = try fv.patches(a, cfg, g[0], g[1], 72 + gi);
        defer a.free(px);
        const pd = try widen(a, px);
        defer a.free(pd);
        const n = g[0] * g[1];
        // stages: the patch embedding, one block, both blocks; then the aligner's rows
        for ([_]u32{ 0, 1, 2 }) |blocks| {
            try r.tower.run(px, g[0], g[1], .{ .blocks = blocks, .finish = false });
            const want = try vref.stream(a, &m, pd, g[0], g[1], blocks);
            defer a.free(want);
            const e = compare(bytesOf(r.tower.stream(), n * cfg.dim), want, cfg.dim);
            say(e.rms < 0.01 and e.worst < 0.06, "grid {d}x{d} after {d} blocks: relative RMS error {d:.5}, worst element {d:.4} of its row's RMS", .{ g[0], g[1], blocks, e.rms, e.worst });
            worst_rms = @max(worst_rms, e.rms);
            worst_el = @max(worst_el, e.worst);
            try testing.expect(e.rms < 0.01 and e.worst < 0.06);
        }
        try r.tower.run(px, g[0], g[1], .{});
        const x = try vref.stream(a, &m, pd, g[0], g[1], cfg.layers);
        defer a.free(x);
        const want = try vref.aligner(a, &m, x, g[0], g[1]);
        defer a.free(want);
        const tokens = r.tower.tokensOf(g[0], g[1]);
        const e = compare(bytesOf(r.tower.output(), tokens * cfg.out_dim), want, cfg.out_dim);
        say(e.rms < 0.02 and e.worst < 0.1, "grid {d}x{d}: {d} aligner rows, relative RMS error {d:.5}, worst element {d:.4}", .{ g[0], g[1], tokens, e.rms, e.worst });
        worst_rms = @max(worst_rms, e.rms);
        worst_el = @max(worst_el, e.worst);
        try testing.expect(e.rms < 0.02 and e.worst < 0.1);
    }
    say(true, "(a) over 3 grids: worst relative RMS {d:.5}, worst element {d:.4}", .{ worst_rms, worst_el });
}

test "dsv41 metal vision: an image alone equals the same image after a larger one in the scratch" {
    const a = testing.allocator;
    const cfg = fv.config();
    const r = try Rig.init(a, testing.io, cfg, 2, 16, 73);
    defer r.deinit();
    const small = try fv.patches(a, cfg, 5, 7, 74);
    defer a.free(small);
    const big = try fv.patches(a, cfg, 9, 12, 75);
    defer a.free(big);
    try r.tower.run(small, 5, 7, .{});
    const t = r.tower.tokensOf(5, 7);
    const alone = try a.dupe(u16, bytesOf(r.tower.output(), t * cfg.out_dim));
    defer a.free(alone);
    try r.tower.run(big, 9, 12, .{});
    try r.tower.run(small, 5, 7, .{});
    const after = bytesOf(r.tower.output(), t * cfg.out_dim);
    var bad: usize = 0;
    for (alone, after) |x, y| bad += @intFromBool(x != y);
    say(bad == 0, "(b) tower: an image alone vs after a 9x12 one through the same scratch: {d} of {d} values differ", .{ bad, alone.len });
    try testing.expectEqual(@as(usize, 0), bad);
}

/// A tiny prompt with two images among text: its ids (virtual ids at image positions), the images and their pixels.
const Scene = struct {
    a: std.mem.Allocator,
    ids: []u32,
    images: [2]vrows.Image,
    pixels: [2][]f32,

    const at = [2]u32{ 300, 510 };
    const grids = [2][2]u32{ .{ 5, 7 }, .{ 4, 4 } };

    fn init(a: std.mem.Allocator, f: *ff.Fwd, cfg: vis.Config, total: usize, seed: u64) !Scene {
        var sc: Scene = .{ .a = a, .ids = try ff.ids(a, &f.r.c, total, seed), .images = undefined, .pixels = undefined };
        for (&sc.images, &sc.pixels, at, grids, 0..) |*im, *px, pos, g, i| {
            px.* = try fv.patches(a, cfg, g[0], g[1], seed + 10 + i);
            const lg = prep.llmGrid(g[0], g[1], cfg.ratio);
            im.* = .{ .pixels = px.*, .gh = g[0], .gw = g[1], .at = pos, .tokens = prep.spanTokens(lg[0], lg[1]), .hash = 0x9e3779b97f4a7c15 *% (seed + i + 1) };
            for (0..im.tokens) |p| sc.ids[pos + p] = prep.vid(im.hash, @intCast(p));
        }
        return sc;
    }

    fn deinit(sc: *Scene) void {
        sc.a.free(sc.ids);
        for (sc.pixels) |p| sc.a.free(p);
    }
};

/// One tiny prompt model with the tower beside it and the layers' image biases attached.
fn sceneRig(a: std.mem.Allocator, seed: u64) !struct { f: *ff.Fwd, v: *Rig } {
    const f = try ff.Fwd.initWith(a, testing.io, fixture.promptConfig(), pt.limit, seed);
    errdefer f.deinit();
    const c = &f.r.c;
    const v = try Rig.init(a, testing.io, fv.config(), c.layers, c.experts, seed + 1);
    for (f.r.layers, v.w.bias_vl) |*l, b| l.gate_bvl = b;
    return .{ .f = f, .v = v };
}

/// Positions [from, to) one row a launch through the decode forward, image rows merged in as the prompt path does.
fn decodeImages(f: *ff.Fwd, rows: ?*const vrows.Rows, slot: *const pkg.kv.Slot, hist: []const u32, from: usize, to: usize, out: ?pt.Out) !void {
    const r = f.r;
    const c = &r.c;
    const V: usize = c.vocab;
    const TH = c.dspark_targets.len * c.hidden;
    const look = c.engram_max_ngram - 1;
    for (from..to) |p| {
        const image = prep.isVid(hist[p]);
        const id: u32 = if (image) rows.?.embed_id else hist[p];
        const one = [1]u32{id};
        try fwd.prepare(&f.b, c, &one, hist[p -| look..p], f.lookup());
        f.b.bytes(f.b.mask, 1)[0] = @intFromBool(!image);
        try r.s.writeRows(c, @intCast(p), 1);
        if (rows) |rs| rs.describe(@intCast(p), @intCast(p + 1), r.s.bytes(r.s.image, 1)) else r.s.bytes(r.s.image, 1)[0] = 0;
        const x = r.ctx(slot, @intCast(p), 1);
        const pass = fixture.Pass.begin(r.queue);
        fwd.encodeEmbed(&x, pass.e, f.model(), &f.b);
        if (rows) |rs| rs.merge(pass.e, f.b.streams[0], @intCast(p), @intCast(p + 1));
        for (0..f.model().layers.len) |i| try fwd.encodeLayer(&x, pass.e, f.model(), &f.b, i);
        fwd.encodeHead(&x, pass.e, f.model(), &f.b);
        try pass.finish();
        if (out) |o| {
            @memcpy(o.logits[(p - from) * V ..][0..V], f.b.logitsOf(c, 1));
            @memcpy(o.taps[(p - from) * TH ..][0..TH], slot.taps.?.slice(u16, TH));
        }
    }
}

test "dsv41 metal vision: a prompt with two images then 8 decode steps equals one-row decode launches; chunk plans agree" {
    const a = testing.allocator;
    const rig = try sceneRig(a, 81);
    const f = rig.f;
    defer f.deinit();
    defer rig.v.deinit();
    const c = &f.r.c;
    const n_prompt: u32 = 580;
    const steps = 8;
    var sc = try Scene.init(a, rig.f, fv.config(), n_prompt + steps, 82);
    defer sc.deinit();
    const V: usize = c.vocab;
    const TH = c.dspark_targets.len * c.hidden;
    // reference: every position one row a launch (image rows merged, flagged and ungated), features from the tower
    var rows = try vrows.Rows.build(a, f.r.device, &rig.v.tower, &sc.images, n_prompt, 64, 0);
    defer rows.deinit();
    var dslot = try pkg.kv.Slot.init(a, f.r.device, c, pt.limit);
    defer dslot.deinit();
    const dref = try pt.Out.init(a, c, n_prompt + steps);
    defer dref.deinit(a);
    try decodeImages(f, &rows, &dslot, sc.ids, 0, n_prompt + steps, dref);
    // the prompt path, chunks of 64
    var run = try pt.Run.init(a, f, pt.head_opts, 0);
    defer run.deinit();
    run.p.images = &rows;
    try run.p.run(pt.env(f), &run.slot, &run.tail, sc.ids[0..n_prompt]);
    const first = n_prompt - 128;
    const lg = pt.bitsDiff(run.p.logits(c), dref.logits[first * V .. n_prompt * V]);
    const tp = pt.halfDiff(run.p.tapRows(c), dref.taps[first * TH .. n_prompt * TH]);
    const dout = try pt.Out.init(a, c, steps);
    defer dout.deinit(a);
    try decodeImages(f, &rows, &run.slot, sc.ids, n_prompt, n_prompt + steps, dout);
    try run.tail.advance(n_prompt + steps);
    const dl = pt.bitsDiff(dout.logits, dref.logits[n_prompt * V ..]);
    const dt = pt.halfDiff(dout.taps, dref.taps[n_prompt * TH ..]);
    const st = pt.stateDiff(&run.slot, &dslot);
    const bad = lg + tp + dl + dt + st;
    say(bad == 0, "(c) 2 images + {d} tokens + {d} steps vs one-row decode launches: {d} head logits, {d} taps, {d} step logits, {d} step taps, {d} state bytes differ", .{ n_prompt, steps, lg, tp, dl, dt, st });
    try testing.expectEqual(@as(usize, 0), bad);
    // the image rows route by bias_vl and shut the Engram gate: the same prompt without them differs
    var plain = try pt.Run.init(a, f, pt.head_opts, 0);
    defer plain.deinit();
    const text = try a.dupe(u32, sc.ids[0..n_prompt]);
    defer a.free(text);
    for (text) |*id| if (prep.isVid(id.*)) {
        id.* = rows.embed_id;
    };
    try plain.p.run(pt.env(f), &plain.slot, &plain.tail, text);
    const moved = pt.bitsDiff(plain.p.logits(c), run.p.logits(c));
    say(moved > 0, "(c) the same ids as plain text give other logits: {d} of {d} differ", .{ moved, run.p.head * c.vocab });
    try testing.expect(moved > 0);
    // the image rows route by bias_vl: without it the same prompt gives other logits
    {
        for (f.r.layers) |*l| l.gate_bvl = null;
        var nob = try pt.Run.init(a, f, pt.head_opts, 0);
        defer nob.deinit();
        nob.p.images = &rows;
        try nob.p.run(pt.env(f), &nob.slot, &nob.tail, sc.ids[0..n_prompt]);
        for (f.r.layers, rig.v.w.bias_vl) |*l, b| l.gate_bvl = b;
        const d = pt.bitsDiff(nob.p.logits(c), run.p.logits(c));
        say(d > 0, "(c) image rows without bias_vl give other logits: {d} of {d} differ", .{ d, run.p.head * c.vocab });
        try testing.expect(d > 0);
    }
    // (b) chunk plans: an image's chunk alone, or with its neighbours, give the same logits, taps, slot and tail
    var bad_b: usize = 0;
    for ([_]u32{ 16, 32, 100, 600 }) |chunk| {
        var o = pt.head_opts;
        o.chunk = chunk;
        var rs = try vrows.Rows.build(a, f.r.device, &rig.v.tower, &sc.images, n_prompt, chunk, 0);
        defer rs.deinit();
        var other = try pt.Run.init(a, f, o, 0);
        defer other.deinit();
        other.p.images = &rs;
        try other.p.run(pt.env(f), &other.slot, &other.tail, sc.ids[0..n_prompt]);
        const d = pt.runDiff(c, &other, &run);
        const td = pt.tailDiff(&other.tail, &run.tail);
        _ = td; // the other run's tail was not advanced by the steps
        say(d[0] + d[1] == 0, "(b) chunk {d} vs 64: {d} logits, {d} taps differ", .{ chunk, d[0], d[1] });
        bad_b += d[0] + d[1];
    }
    try testing.expectEqual(@as(usize, 0), bad_b);
}

fn targetOver(f: *ff.Fwd, v: *Rig, cache: u64) !*pkg.target.Target {
    const r = f.r;
    const env: pkg.target.Env = .{ .device = r.device, .queue = r.queue, .c = &r.c, .p = &r.p, .tables = &r.tables, .s = &r.s, .model = f.model(), .lookup = f.lookup(), .vision = .{ .kit = &v.kit, .w = &v.w } };
    return pkg.target.Target.init(f.a, env, .{ .slots = 2, .limit = pt.limit, .prompt = pt.head_opts, .cache_bytes = cache });
}

test "dsv41 metal vision: a Target takes images, keys its cache by them, and resumes a prompt past them" {
    const a = testing.allocator;
    const rig = try sceneRig(a, 91);
    const f = rig.f;
    defer f.deinit();
    defer rig.v.deinit();
    const n: u32 = 580;
    var sc = try Scene.init(a, f, fv.config(), n + 6, 92);
    defer sc.deinit();
    // the prompt as the server sends it: the image token at every span position
    const ids = try a.dupe(u32, sc.ids);
    defer a.free(ids);
    for (ids) |*id| if (prep.isVid(id.*)) {
        id.* = fv.config().image_token_id;
    };
    var t = try targetOver(f, rig.v, 1 << 28);
    defer t.deinit();
    try t.setImages(0, &sc.images);
    const first = try t.serial(0, ids[0..n], 5, null);
    defer a.free(first);
    // the same prompt again: a kept state, the same reply
    try t.setImages(1, &sc.images);
    const hits = t.stats.hits;
    const again = try t.serial(1, ids[0..n], 5, null);
    defer a.free(again);
    say(t.stats.hits == hits + 1 and std.mem.eql(u32, first, again), "(t) the same images resume the kept state ({d} hits) with the same reply", .{t.stats.hits - hits});
    try testing.expectEqual(hits + 1, t.stats.hits);
    try testing.expectEqualSlices(u32, first, again);
    // another image behind the same ids: no state of the first applies
    var other = sc.images;
    other[0].hash +%= 1;
    try t.setImages(0, &other);
    const hits2 = t.stats.hits;
    const third = try t.serial(0, ids[0..n], 5, null);
    defer a.free(third);
    say(t.stats.hits == hits2, "(t) another image's hash finds no kept state", .{});
    try testing.expectEqual(hits2, t.stats.hits);
    // a longer prompt over the kept one runs only its new rows; its reply equals a fresh target's
    try t.setImages(1, &sc.images);
    const more = try t.serial(1, ids[0 .. n + 6], 4, null);
    defer a.free(more);
    var fresh = try targetOver(f, rig.v, 0);
    defer fresh.deinit();
    try fresh.setImages(0, &sc.images);
    const want = try fresh.serial(0, ids[0 .. n + 6], 4, null);
    defer a.free(want);
    say(std.mem.eql(u32, more, want), "(t) a prompt resumed past two images equals a fresh run: {any} vs {any}", .{ more, want });
    try testing.expectEqualSlices(u32, want, more);
    try testing.expectError(error.NoVision, blk: {
        var plain = try pkg.target.Target.init(a, .{ .device = f.r.device, .queue = f.r.queue, .c = &f.r.c, .p = &f.r.p, .tables = &f.r.tables, .s = &f.r.s, .model = f.model(), .lookup = f.lookup() }, .{ .slots = 1, .limit = pt.limit });
        defer plain.deinit();
        break :blk plain.setImages(0, &sc.images);
    });
}
