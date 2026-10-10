//! tf-glm53 SETTINGS.json: full GLM-5.3 decode on one Mac or N Macs.
//!   oracle mode ("oracle": m2 dump): teacher-forced steps over its tokens, hidden states and logits compared;
//!   generate mode ("prompt": [ids], "max": n): greedy tokens, timed.
//! One Mac computes `slices` canonical slices; N Macs ("ranks" > 1, "links") compute one each and swap partials.
const std = @import("std");
const mtl = @import("metal");
const fabric = @import("fabric");
const m = @import("model.zig");
const fwd = @import("forward.zig");
const Exchange = @import("exchange.zig").Exchange;
const CopyIndex = @import("copy_index.zig").CopyIndex;
const Lw = @import("lw_gpu.zig").Lw;
const lw_train = @import("lw_train.zig");

const Settings = struct {
    model: []const u8,
    pack: []const u8,
    layers: [2]u32 = .{ 0, m.LAYERS },
    cap: u32 = 4096,
    threads: u32 = 16,
    rank: u32 = 0,
    ranks: u32 = 1,
    slices: u32 = 1,
    library: []const u8 = "",
    links: []const fabric.mcdma.Link = &.{},
    oracle: []const u8 = "",
    out: []const u8 = "", // raw f32 dump of every captured hidden state (for N-way == 1-way checks)
    prompt: []const u32 = &.{},
    max: u32 = 0,
    steps: u32 = 0, // oracle mode: run only this many positions (0 = all)
    profile: u32 = 0, // > 0: time each launch class alone this many times (no exchange), at profile_pos
    profile_pos: u32 = 100,
    profile_layer: u32 = 3,
    rows: u32 = 1, // the largest prompt block (and profile rows)
    prefill: bool = false, // generate: read the prompt in blocks of `rows` (else one row at a time)
    kv16: bool = false, // fp16 KV caches (latent + indexer keys): half the memory a position
    serve: u16 = 0, // > 0: take jobs on this TCP port (one JSON line each; see serveJobs)
    bind_local: bool = false, // serve: listen on 127.0.0.1 only (tests)
    prompt_random: u32 = 0, // generate: > 0 = a deterministic pseudo-random prompt of this many tokens (long-context benchmarks)
    mtp: []const u8 = "", // the MTP head's safetensors (layer 78 in the trunk's layout): drafts verified by the trunk
    mtp_depth: i32 = -1, // drafts a round: -1 = chosen each round by the controller, else fixed (0..mtp_max)
    mtp_max: u32 = 3,
    mtp_cost: []const f64 = &.{}, // a round's relative cost at k = 0, 1, 2, .. drafts (the same on every rank)
    mtp_force: []const u8 = "", // generate tests: "oracle" (drafts = mtp_ref's next tokens) or "garbage" (always wrong)
    mtp_ref: []const u32 = &.{}, // generate tests: the plain run's tokens
    // Living Weights at layers.77's shared expert: the sidecar file (living_weights.safetensors in a COPY's
    // folder or any path; absent file = no change yet). Empty: no change and no Lw at all (the stock engine, bit for bit).
    lw: []const u8 = "",
    slide: bool = false, // serve: take learn lines (needs lw; learning writes only this sidecar, never the KV or weights)
};

/// Drafts a round, decided the same way on every rank (acceptance counts and a fixed cost table, no clocks): the depth
/// with the most expected tokens per unit cost, with a probe one deeper every 16 rounds.
const Ctl = struct {
    max: usize,
    fixed: i32,
    cost: [fwd.verify_max]f64,
    tried: [fwd.verify_max]f64 = @splat(0),
    hit: [fwd.verify_max]f64 = @splat(0),
    rounds: u64 = 0,
    drafted: u64 = 0,
    kept: u64 = 0,
    by_k: [fwd.verify_max]u64 = @splat(0),
    // copy drafts (G53_COPY=<min match>, 0 = off): rounds, tokens offered, tokens kept, rounds cut short
    copy_min: usize = 0,
    copy_row: f64 = 0.30,
    c_rounds: u64 = 0,
    c_offered: u64 = 0,
    c_kept: u64 = 0,
    c_cut: u64 = 0,

    fn init(s: Settings) Ctl {
        var c: Ctl = .{ .max = @min(s.mtp_max, fwd.verify_max - 1), .fixed = s.mtp_depth, .cost = undefined };
        for (0..fwd.verify_max) |k| c.cost[k] = if (k < s.mtp_cost.len) s.mtp_cost[k] else 1.0 + 0.35 * @as(f64, @floatFromInt(k));
        if (std.c.getenv("G53_COPY")) |v| c.copy_min = std.fmt.parseInt(usize, std.mem.span(v), 10) catch 0;
        if (std.c.getenv("G53_COPY_ROW")) |v| c.copy_row = std.fmt.parseFloat(f64, std.mem.span(v)) catch 0.30;
        return c;
    }
    fn acc(c: *const Ctl, i: usize) f64 {
        const prior = 0.8 * std.math.pow(f64, 0.9, @floatFromInt(i));
        return (c.hit[i] + 4 * prior) / (c.tried[i] + 4);
    }
    fn pick(c: *Ctl) usize {
        if (c.fixed >= 0) return @min(@as(usize, @intCast(c.fixed)), c.max);
        var best: usize = 0;
        var best_v: f64 = 0;
        var p: f64 = 1;
        var ex: f64 = 1;
        for (0..c.max + 1) |k| {
            if (k > 0) {
                p *= c.acc(k - 1);
                ex += p;
            }
            const v = ex / c.cost[k];
            if (v > best_v) {
                best_v = v;
                best = k;
            }
        }
        if (best < c.max and c.rounds % 16 == 15) best += 1;
        return best;
    }
    /// Expected tokens a unit of cost of an MTP round at depth k (the controller's own estimate).
    fn rate(c: *const Ctl, k: usize) f64 {
        var p: f64 = 1;
        var ex: f64 = 1;
        for (0..k) |i| {
            p *= c.acc(i);
            ex += p;
        }
        return ex / c.cost[k];
    }
    /// Copy drafts: the window w (0 = none, never 1) with the most expected tokens a unit of cost, if it beats the MTP
    /// round's rate at depth k. Acceptance a = (kept + n) / (kept + n + cut + 1) from this rank's copy history (Ash Hart,
    /// TensorFold glm/engine.zig copy rounds); a copy round costs 1 + copy_row a row (no draft chain).
    fn copyWidth(c: *const Ctl, n: usize, avail: usize, k: usize, has_head: bool) usize {
        const nf: f64 = @floatFromInt(n);
        const kept: f64 = @floatFromInt(c.c_kept);
        const cut: f64 = @floatFromInt(c.c_cut);
        const a = (kept + nf) / (kept + nf + cut + 1);
        var best = if (has_head) c.rate(k) else 1.0;
        var w_best: usize = 0;
        var ex: f64 = 1;
        var p: f64 = 1;
        for (1..avail + 1) |w| {
            p *= a;
            ex += p;
            const r = ex / (1.0 + c.copy_row * @as(f64, @floatFromInt(w)));
            if (r > best) {
                best = r;
                w_best = w;
            }
        }
        return if (w_best < 2) 0 else w_best;
    }
    /// k drafts went out, the first j were kept.
    fn update(c: *Ctl, k: usize, j: usize) void {
        c.rounds += 1;
        c.by_k[k] += 1;
        c.drafted += k;
        c.kept += j;
        for (0..fwd.verify_max) |i| {
            c.tried[i] *= 0.98;
            c.hit[i] *= 0.98;
        }
        for (0..@min(j + 1, k)) |i| {
            c.tried[i] += 1;
            if (i < j) c.hit[i] += 1;
        }
    }
};

/// The trunk's greedy reply with drafts: rounds of [head absorbs + drafts] + [verify]. Tokens go to `emit` (false = stop);
/// returns the position the last emitted token would be fed at. Plain greedy and this give the same tokens.
const Spec = struct {
    t: u32, // the pending token (emitted, not yet fed)
    p: u32, // its position
    r0: usize, // xn row holding the trunk's state at position hp0
    next: [fwd.verify_max]u32 = undefined, // the head's pending absorb: next tokens of xn rows r0..
    nn: usize,
    hp0: u32,
    hist: ?*CopyIndex = null, // copy drafts: the rank's known context (tokens up to and including t)
};

fn specInit(e: *fwd.Engine, p_end: u32) Spec {
    var sp: Spec = .{ .t = e.token(), .p = p_end, .r0 = e.last_rows - 1, .nn = 1, .hp0 = p_end - 1 };
    sp.next[0] = sp.t;
    return sp;
}

/// One round: picks emitted through `emit` until it says stop. Returns false when stopped (sp.p = the stop position).
fn specRound(e: *fwd.Engine, c: *Ctl, sp: *Spec, force: ?[]const u32, emit: anytype) !bool {
    var k = if (e.w.mtp != null) c.pick() else 0;
    // copy drafts: what followed the context's last tokens earlier, verified by the trunk like any draft (exact)
    var copied: [fwd.verify_max]u32 = undefined;
    var cw: usize = 0;
    if (force == null and c.copy_min > 0) if (sp.hist) |h| {
        const mt = h.longest(8);
        if (mt.n >= c.copy_min) {
            const room: usize = if (e.cap > sp.p + 2) e.cap - (sp.p + 2) else 0;
            const avail = @min(h.ctx.items.len - mt.at, fwd.verify_max - 1, room);
            cw = c.copyWidth(mt.n, avail, k, e.w.mtp != null);
            if (cw > 0) @memcpy(copied[0..cw], h.ctx.items[mt.at .. mt.at + cw]);
            var ok: usize = 0; // only real token ids (the index holds no sentinels; belt and braces)
            while (ok < cw and copied[ok] < m.V) ok += 1;
            cw = if (ok < 2) 0 else ok;
        }
    };
    if (cw > 0) k = cw;
    _ = try e.specRoundCopy(sp.r0, sp.next[0..sp.nn], sp.hp0, sp.t, sp.p, k, if (cw > 0) copied[0..cw] else force, cw == 0);
    const g = e.picksSlice(k + 1);
    const d = if (cw > 0) copied[0..cw] else if (force) |f| f[0..k] else e.draftsSlice(k);
    var j: usize = 0;
    while (j < k and d[j] == g[j]) j += 1;
    if (cw > 0) {
        c.c_rounds += 1;
        c.c_offered += cw;
        c.c_kept += j;
        if (j < cw) c.c_cut += 1;
    } else c.update(k, j);
    const p0 = sp.p;
    for (0..j + 1) |i| {
        sp.next[i] = g[i];
        if (sp.hist) |h| h.extend(g[i .. i + 1]) catch @panic("glm53: copy index out of memory"); // never diverge silently from the other ranks
        if (!emit.token(g[i])) {
            sp.nn = i + 1;
            sp.r0 = 0;
            sp.hp0 = p0;
            sp.t = g[i];
            sp.p = p0 + @as(u32, @intCast(i)) + 1;
            return false;
        }
    }
    sp.nn = j + 1;
    sp.r0 = 0;
    sp.hp0 = p0;
    sp.t = g[j];
    sp.p = p0 + @as(u32, @intCast(j)) + 1;
    return true;
}

/// A safetensors file's tensors by name: (dtype, shape, bytes).
const Tensors = struct {
    map: mtl.MappedFile,
    names: std.StringHashMapUnmanaged(struct { off: usize, len: usize, shape: [4]usize }) = .empty,

    fn open(a: std.mem.Allocator, path: []const u8) !Tensors {
        const p = try a.dupeSentinel(u8, path, 0);
        var t: Tensors = .{ .map = try mtl.MappedFile.open(p) };
        const hlen = std.mem.readInt(u64, t.map.bytes[0..8], .little);
        const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, t.map.bytes[8 .. 8 + hlen], .{});
        var it = doc.object.iterator();
        while (it.next()) |kv| {
            if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) continue;
            const o = kv.value_ptr.object;
            const offs = o.get("data_offsets").?.array.items;
            var shape: [4]usize = @splat(1);
            for (o.get("shape").?.array.items, 0..) |d, j| shape[j] = @intCast(d.integer);
            const b: usize = @intCast(offs[0].integer);
            const e: usize = @intCast(offs[1].integer);
            try t.names.put(a, kv.key_ptr.*, .{ .off = 8 + hlen + b, .len = e - b, .shape = shape });
        }
        return t;
    }

    fn get(t: *const Tensors, comptime T: type, name: []const u8) ![]const T {
        const e = t.names.get(name) orelse {
            std.log.err("oracle has no {s}", .{name});
            return error.MissingTensor;
        };
        return @as([*]const T, @ptrCast(@alignCast(t.map.bytes.ptr + e.off)))[0 .. e.len / @sizeOf(T)];
    }
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 2) {
        std.debug.print("usage: tf-glm53 SETTINGS.json\n", .{});
        std.process.exit(2);
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], a, .limited(1 << 20));
    const s = try std.json.parseFromSliceLeaky(Settings, a, text, .{ .allocate = .alloc_always });
    const device = try mtl.Device.init();
    defer device.deinit();
    std.debug.print("glm53: {s} rank {d}/{d} slices {d} layers [{d},{d}) cap {d}\n", .{ device.name(), s.rank, s.ranks, s.slices, s.layers[0], s.layers[1], s.cap });

    // the plan: which share this process loads, which slices it sums
    var slices: std.ArrayList(m.Share) = .empty;
    var plan: fwd.Plan = undefined;
    var share: m.Share = undefined;
    if (s.profile > 0) { // one rank's share, its own slice, no exchange
        share = m.Share.of(s.rank, s.ranks);
        try slices.append(a, share);
        plan = .{ .slices = slices.items, .slot_of_first = 0, .nslots = 1 };
    } else if (s.ranks > 1) {
        share = m.Share.of(s.rank, s.ranks);
        try slices.append(a, share);
        plan = .{ .slices = slices.items, .slot_of_first = s.rank, .nslots = s.ranks };
    } else {
        share = m.Share.whole();
        for (0..s.slices) |i| try slices.append(a, m.Share.of(i, s.slices));
        plan = .{ .slices = slices.items, .slot_of_first = 0, .nslots = s.slices };
    }
    const t0 = mtl.clock.seconds();
    const w = try m.load(gpa, device, s.model, s.pack, share, s.layers[0], s.layers[1], s.threads, s.mtp);
    defer {
        w.deinit();
        gpa.destroy(w);
    }
    std.debug.print("glm53: loaded {d:.1} GB in {d:.1} s (heads {any}, moe rows {any}, vocab {any})\n", .{ @as(f64, @floatFromInt(w.bytes)) / 1e9, mtl.clock.seconds() - t0, share.heads, share.moe, share.vocab });

    var lw: ?*Lw = null;
    if (s.slide and s.lw.len == 0) {
        std.debug.print("glm53: \"slide\" needs \"lw\" (the sidecar path to learn into)\n", .{});
        return error.SlideNeedsLw;
    }
    if (s.lw.len > 0 and s.profile == 0) lw = try Lw.open(gpa, device, s.model, s.lw, s.rows);
    var lw_tag: u64 = 0; // every rank must load the same change and agree on learning (0 = none: the old start word)
    if (lw) |l| lw_tag = std.hash.Wyhash.hash(l.tag, if (s.slide) "slide" else "serve") | 1;
    var xc: ?*Exchange = null;
    if (s.ranks > 1 and s.profile == 0) {
        const lib = try a.dupeSentinel(u8, s.library, 0);
        xc = try Exchange.create(gpa, device, lib, s.rank, s.ranks, s.links, s.rows, lw_tag);
        std.debug.print("glm53: rank {d} connected to {d} peers\n", .{ s.rank, s.links.len });
    }
    defer if (xc) |x| x.deinit(gpa);
    const e = try fwd.Engine.init(gpa, device, w, plan, xc, s.cap, s.rows, s.kv16);
    e.lw = lw;
    const nl = s.layers[1] - s.layers[0];
    if (s.profile > 0) {
        for ([_]u32{ s.profile_pos, 2300 }) |pp| {
            if (pp >= s.cap) continue;
            e.setToken(1000);
            for (0..pp + 1) |p| _ = p; // caches hold zeros: the timing does not depend on their values
            std.debug.print("glm53 profile: rank share {d}/{d}, layer {d}, pos {d}, {d} reps\n", .{ s.rank, s.ranks, s.profile_layer, pp, s.profile });
            if (std.c.getenv("G53_PROFILE_LAYERS") != null) {
                try e.profileLayers(pp, s.rows);
                try e.profileLayers(pp, s.rows);
                continue;
            }
            try e.profile(s.profile_layer, pp, s.profile, s.rows);
            try e.profile(2, pp, s.profile, s.rows);
        }
        return;
    }

    if (s.serve > 0) {
        var host: lw_train.Host = .{ .gpa = gpa, .device = device, .serve = e, .lw = undefined, .kv16 = s.kv16 };
        if (lw) |l| host.lw = l;
        return serveJobs(e, s, a, if (s.slide) &host else null);
    }

    if (s.oracle.len > 0) {
        var o = try Tensors.open(a, s.oracle);
        defer o.map.deinit();
        const toks = try o.get(i32, "tokens");
        const pos_list = try o.get(i32, "positions");
        const per = @as(usize, nl + 1) * m.D; // embedding + every layer
        try e.addDump(pos_list.len * per);
        e.idx_dump = (try e.addBuf(m.LAYERS * m.KEYS * 4));
        const n: usize = if (s.steps > 0) @min(s.steps, toks.len) else toks.len;
        var gpu: f64 = 0;
        var gpu_n: usize = 0;
        var pi: usize = 0;
        const t1 = mtl.clock.seconds();
        for (0..n) |p| {
            e.setToken(@intCast(toks[p]));
            const keep = pi < pos_list.len and pos_list[pi] == p;
            const dt = try e.run(@intCast(p), if (keep) pi * per else null, keep);
            if (p >= 16) {
                gpu += dt;
                gpu_n += 1;
            }
            if (keep) {
                // the slice's argmax and logits against the oracle's
                var name: [64]u8 = undefined;
                const want = try o.get(f32, try std.fmt.bufPrint(&name, "logits.{d}", .{p}));
                const v0 = w.share.vocab[0];
                const got = e.logits.buf.slice(f32, w.share.vocab[1] - v0);
                var worst: f32 = 0;
                var top: f32 = 0;
                for (want[v0..][0..got.len], got) |x, y| {
                    worst = @max(worst, @abs(x - y));
                    top = @max(top, @abs(x));
                }
                const am = (try o.get(i32, try std.fmt.bufPrint(&name, "argmax.{d}", .{p})))[0];
                std.debug.print("pos {d}: argmax ours {d} oracle {d} {s} | logits max|diff| {e:.3} (max |logit| {d:.2})\n", .{ p, e.token(), am, if (@as(i32, @intCast(e.token())) == am) "SAME" else "DIFF", worst, top });
                if (p >= m.KEYS) for (s.layers[0]..s.layers[1]) |li_full| { // every full layer's picks against the oracle's set
                    if (!m.fullIndexer(li_full)) continue;
                    const ours = e.idx_dump.?.buf.slice(u32, m.LAYERS * m.KEYS)[li_full * m.KEYS ..][0..m.KEYS];
                    if (o.get(i32, try std.fmt.bufPrint(&name, "topk.{d}.{d}", .{ li_full, p }))) |theirs| {
                        var same: usize = 0;
                        var a2: usize = 0;
                        var b2: usize = 0;
                        while (a2 < ours.len and b2 < theirs.len) {
                            const x: i64 = ours[a2];
                            const y: i64 = theirs[b2];
                            if (x == y) {
                                same += 1;
                                a2 += 1;
                                b2 += 1;
                            } else if (x < y) a2 += 1 else b2 += 1;
                        }
                        std.debug.print("   top-k layer {d}: {d}/{d} keys shared with the oracle (ours first {d} last {d})\n", .{ li_full, same, m.KEYS, ours[0], ours[m.KEYS - 1] });
                    } else |_| {}
                };
                pi += 1;
            }
            if (p % 256 == 0) std.debug.print("  step {d}: {d:.2} ms GPU\n", .{ p, dt * 1e3 });
        }
        std.debug.print("glm53: {d} steps in {d:.1} s, mean GPU {d:.2} ms/step (after 16)\n", .{ n, mtl.clock.seconds() - t1, gpu / @as(f64, @floatFromInt(@max(gpu_n, 1))) * 1e3 });
        // hidden states, every captured position and layer
        const dump = e.dump.?.buf.slice(f32, pos_list.len * per);
        for (0..pi) |k| { // per position: the worst layer's rel-L2
            var wr: f64 = 0;
            var wl: usize = 0;
            for (0..nl + 1) |li| {
                var name: [64]u8 = undefined;
                const want = try o.get(f32, try std.fmt.bufPrint(&name, "hidden.{d}", .{@as(i64, @intCast(li)) - 1}));
                var dn: f64 = 0;
                var nn: f64 = 0;
                for (0..m.D) |d| {
                    const x: f64 = want[k * m.D + d];
                    const y: f64 = dump[k * per + li * m.D + d];
                    dn += (x - y) * (x - y);
                    nn += x * x;
                }
                const r = @sqrt(dn / @max(nn, 1e-30));
                if (r > wr) {
                    wr = r;
                    wl = li;
                }
            }
            if (wr > 1e-5) std.debug.print("  pos {d}: worst rel-L2 {e:.3} at capture {d}\n", .{ pos_list[k], wr, wl });
        }
        for (0..nl + 1) |li| {
            var name: [64]u8 = undefined;
            const want = try o.get(f32, try std.fmt.bufPrint(&name, "hidden.{d}", .{@as(i64, @intCast(li)) - 1}));
            var worst: f64 = 0;
            var worst_rel: f64 = 0;
            for (0..pi) |k| {
                var dn: f64 = 0;
                var nn: f64 = 0;
                for (0..m.D) |d| {
                    const x: f64 = want[k * m.D + d];
                    const y: f64 = dump[k * per + li * m.D + d];
                    worst = @max(worst, @abs(x - y));
                    dn += (x - y) * (x - y);
                    nn += x * x;
                }
                worst_rel = @max(worst_rel, @sqrt(dn / @max(nn, 1e-30)));
            }
            std.debug.print("hidden {s}{d}: max|diff| {e:.3}  worst row rel-L2 {e:.3}\n", .{ if (li == 0) "embed " else "layer ", if (li == 0) 0 else li - 1, worst, worst_rel });
        }
        if (s.out.len > 0) {
            var f = try std.Io.Dir.cwd().createFile(init.io, s.out, .{});
            defer f.close(init.io);
            var wb: [65536]u8 = undefined;
            var fw = f.writerStreaming(init.io, &wb);
            try fw.interface.writeAll(std.mem.sliceAsBytes(dump[0 .. pi * per]));
            try fw.interface.flush();
            std.debug.print("glm53: dump {s} ({d} floats), hash {x}\n", .{ s.out, pi * per, std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(dump[0 .. pi * per])) });
        }
        return;
    }

    // generate: the prompt one row at a time, then greedy picks
    var prompt: []const u32 = s.prompt;
    if (s.prompt_random > 0) {
        const pr = try a.alloc(u32, s.prompt_random);
        var x: u64 = 0x2545F4914F6CDD1D;
        for (pr) |*t| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            t.* = @intCast(x % m.V);
        }
        prompt = pr;
    }
    var out: std.ArrayList(u32) = .empty;
    var pos: u32 = 0;
    const t1 = mtl.clock.seconds();
    var pgpu: f64 = 0;
    if (s.prefill) {
        pgpu = try e.prefill(0, prompt);
        pos = @intCast(prompt.len);
    } else for (prompt) |t| {
        e.setToken(t);
        pgpu += try e.run(pos, null, pos + 1 == prompt.len);
        pos += 1;
    }
    const t2 = mtl.clock.seconds();
    {
        const lr = if (s.prefill) (prompt.len - 1) % s.rows else 0; // the block row that held the prompt's last token
        const xs = e.xn.buf.slice(f32, (lr + 1) * m.D)[lr * m.D ..];
        std.debug.print("glm53: after the prompt: first pick {d}, final-norm row hash {x}\n", .{ e.token(), std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(xs)) });
    }
    var gpu: f64 = 0;
    var ctl = Ctl.init(s);
    if (s.mtp.len > 0) { // rounds: drafts verified by the trunk
        try out.append(a, e.token());
        var sp = specInit(e, pos);
        var gidx: ?CopyIndex = null; // G53_COPY: copy drafts over the prompt and the reply (generate tests)
        defer if (gidx) |*x| x.deinit();
        if (ctl.copy_min > 0) {
            gidx = try CopyIndex.init(a, prompt);
            try gidx.?.extend(&.{sp.t});
            sp.hist = &gidx.?;
        }
        const Em = struct {
            out: *std.ArrayList(u32),
            a: std.mem.Allocator,
            want: usize,
            fn token(x: @This(), t: u32) bool {
                x.out.append(x.a, t) catch return false;
                return x.out.items.len < x.want;
            }
        };
        const em: Em = .{ .out = &out, .a = a, .want = s.max + 1 };
        var fbuf: [fwd.verify_max]u32 = undefined;
        while (out.items.len < s.max + 1) {
            var force: ?[]const u32 = null;
            if (s.mtp_force.len > 0) { // drafts from the reference (oracle) or always wrong (garbage)
                const at = out.items.len; // the reference's index of the token after the pending one
                for (0..fwd.verify_max) |i| {
                    const r: u32 = if (at + i < s.mtp_ref.len) s.mtp_ref[at + i] else 0;
                    fbuf[i] = if (std.mem.eql(u8, s.mtp_force, "oracle")) r else (r + 1) % m.V;
                }
                force = &fbuf;
            }
            if (!try specRound(e, &ctl, &sp, force, em)) break;
        }
        pos = sp.p;
        std.debug.print("glm53 copy: rounds {d} offered {d} kept {d} cut {d}\n", .{ ctl.c_rounds, ctl.c_offered, ctl.c_kept, ctl.c_cut });
        std.debug.print("glm53 mtp: rounds {d}, drafts {d}, kept {d} ({d:.1}%), depth use {any}, tokens/round {d:.2}\n", .{ ctl.rounds, ctl.drafted, ctl.kept, 100.0 * @as(f64, @floatFromInt(ctl.kept)) / @as(f64, @floatFromInt(@max(ctl.drafted, 1))), ctl.by_k[0 .. ctl.max + 1], @as(f64, @floatFromInt(out.items.len - 1)) / @as(f64, @floatFromInt(@max(ctl.rounds, 1))) });
    } else for (0..s.max) |_| {
        const t = e.token();
        try out.append(a, t);
        gpu += try e.run(pos, null, true);
        pos += 1;
    }
    const t3 = mtl.clock.seconds();
    if (s.mtp.len == 0) try out.append(a, e.token());
    std.debug.print("glm53: prompt {d} rows in {d:.2} s (GPU {d:.2} s, {s}, {d:.1} tok/s); {d} tokens in {d:.2} s = {d:.2} tok/s (GPU {d:.2} ms/token)\n", .{ prompt.len, t2 - t1, pgpu, if (s.prefill) "blocks" else "row by row", @as(f64, @floatFromInt(prompt.len)) / (t2 - t1), s.max, t3 - t2, @as(f64, @floatFromInt(s.max)) / (t3 - t2), gpu / @as(f64, @floatFromInt(@max(s.max, 1))) * 1e3 });
    std.debug.print("tokens {any}\nhash {x}\n", .{ out.items, std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(out.items)) });
}

const Job = struct { p0: u32 = 0, tokens: []const u32 = &.{}, max: u32 = 256, stop: []const u32 = &.{}, save: []const u8 = "", load: []const u8 = "", n: u32 = 0, cancel: bool = false, stop_at: u32 = 0, learn: ?LearnJob = null, learn_step: u32 = 0, learn_abort: bool = false };

/// Living Weights: a lesson for every rank's learner ({"learn": {...}}), then {"learn_step": k} lines run up to k bounded
/// units each (one capture, one Adam step, one held check...) so a chat request waits at most one unit; rank 0 sends
/// {"learned": ...} / {"failed": ...} events, every rank a done line with the change's state.
const LearnJob = struct {
    train: []const ExampleJ = &.{},
    held: []const ExampleJ = &.{},
    near: []const ExampleJ = &.{},
    keep: []const ExampleJ = &.{},
    undo: bool = false,
    steps: u32 = 400,
    more: bool = false,
    commit: bool = false,
    save: bool = false,
};
const ExampleJ = struct { ids: []const u32, start: u32 };

/// The learner and the lesson it reads (owned across lines).
const Slide = struct {
    host: *lw_train.Host,
    learner: lw_train.Learner,
    arena: std.heap.ArenaAllocator,
    busy: bool = false,

    fn examples(al: std.mem.Allocator, xs: []const ExampleJ) ![]const lw_train.Example {
        const out = try al.alloc(lw_train.Example, xs.len);
        for (xs, out) |x, *o| o.* = .{ .ids = try al.dupe(u32, x.ids), .start = x.start };
        return out;
    }

    fn begin(sl: *Slide, j: LearnJob) !void {
        if (sl.busy) sl.learner.abort();
        _ = sl.arena.reset(.retain_capacity);
        const al = sl.arena.allocator();
        try sl.learner.begin(.{ .train = try examples(al, j.train), .held = try examples(al, j.held), .near = try examples(al, j.near), .keep = try examples(al, j.keep), .undo = j.undo, .steps = j.steps, .more = j.more, .commit = j.commit, .save = j.save });
        sl.busy = true;
    }
};

/// Tokens past the request a rank still generates after rank 0 takes a cancel: room for the front door to relay
/// {"stop_at": k} to the other ranks while they are still short of k (they run within a step of rank 0).
const cancel_margin: u32 = 16; // a round can emit up to verify_max tokens

fn isStopLine(j: Job) bool {
    return j.tokens.len == 0 and j.save.len == 0 and j.load.len == 0 and (j.cancel or j.stop_at > 0);
}

/// During a job's decode: take any {"cancel": true} / {"stop_at": k} lines already waiting on the connection (never
/// blocks). Rank 0 alone turns a cancel into a stop step (n + cancel_margin) and announces it as {"stopping_at": k};
/// the front door relays {"stop_at": k} to the other ranks, so every rank stops after the same token and the caches
/// stay in step. A stop_at a rank has already passed is ignored (it runs to max; the front door then resets).
fn pollStop(c: c_int, buf: []u8, have: *usize, rank: u32, n: u32, limit: *u32) void {
    if (have.* < buf.len) {
        const k = std.c.recv(c, buf.ptr + have.*, buf.len - have.*, std.c.MSG.DONTWAIT);
        if (k > 0) have.* += @intCast(k);
    }
    while (std.mem.indexOfScalar(u8, buf[0..have.*], '\n')) |nl| {
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        const j = std.json.parseFromSliceLeaky(Job, arena.allocator(), buf[0..nl], .{ .ignore_unknown_fields = true, .allocate = .alloc_always }) catch return;
        if (!isStopLine(j)) return; // the next job: leave it for the job loop
        std.mem.copyForwards(u8, buf, buf[nl + 1 .. have.*]);
        have.* -= nl + 1;
        if (j.cancel and rank == 0) {
            const k = @min(limit.*, n + cancel_margin);
            limit.* = k;
            var line: [64]u8 = undefined;
            _ = sendAll(c, std.fmt.bufPrint(&line, "{{\"stopping_at\":{d}}}\n", .{k}) catch return);
        } else if (j.stop_at > 0) {
            if (j.stop_at >= n) {
                limit.* = @min(limit.*, j.stop_at);
            } else std.debug.print("glm53: stop_at {d} arrived at token {d}: ignored\n", .{ j.stop_at, n });
        }
    }
}

/// Every layer's KV rows [0, n) (latent cache, then the indexer cache on full layers) to or from one file: the exact
/// bits, so a restored conversation continues as if it had never stopped.
fn lsTag(e: *const fwd.Engine, buf: *[32]u8) []const u8 {
    if (!e.ls) return "";
    // the round-robin split with no replicated prefix keeps the plain ".ls"
    const ov = std.c.getenv("G53_LSPLIT_OWNER");
    if (e.ls_from == 0 and ov == null) return ".ls";
    const o: []const u8 = if (ov) |v| std.mem.span(v) else "rr";
    return std.fmt.bufPrint(buf, ".ls{d}o{s}", .{ e.ls_from, o }) catch ".ls";
}

fn snapshot(e: *fwd.Engine, path: []const u8, n: u32, write: bool) !usize {
    var pbuf: [1024]u8 = undefined;
    // the element size in the name: an fp16 engine never reads an fp32 snapshot, nor the other way
    // a layer-split rank holds only its layers' latent rows (+ every layer's rows < ls_from): its own file
    // name per split, never a replicated one's
    var tagbuf: [32]u8 = undefined;
    const tagged = try std.fmt.bufPrint(pbuf[0 .. pbuf.len - 1], "{s}.kv{d}{s}", .{ path, e.kv_bytes, lsTag(e, &tagbuf) });
    pbuf[tagged.len] = 0;
    const pz: [*:0]const u8 = @ptrCast(&pbuf);
    const fd = if (write) std.c.open(pz, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644)) else std.c.open(pz, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return error.OpenFailed;
    defer _ = std.c.close(fd);
    var off: usize = 0;
    for (e.w.first..e.w.last) |i| {
        const parts = [_]?m.Ref{ e.kvc[i], e.ic[i] };
        const widths = [_]usize{ m.CROW, m.ID };
        const held = [_]u32{ @min(n, e.kvRows(i)), n };
        for (parts, widths, held) |pr, wd, hn| {
            const r = pr orelse continue;
            const bytes = r.addr()[0 .. @as(usize, hn) * wd * e.kv_bytes];
            var done: usize = 0;
            while (done < bytes.len) {
                const k = if (write) std.c.pwrite(fd, bytes.ptr + done, bytes.len - done, @intCast(off + done)) else std.c.pread(fd, bytes.ptr + done, bytes.len - done, @intCast(off + done));
                if (k <= 0) return error.ShortIo;
                done += @intCast(k);
            }
            off += bytes.len;
        }
    }
    // the draft head's caches in their own file (older snapshots have none: the head then starts cold, replies unchanged)
    if (e.w.mtp != null) {
        const mt = try std.fmt.bufPrint(pbuf[0 .. pbuf.len - 1], "{s}.kv{d}{s}.mtp", .{ path, e.kv_bytes, lsTag(e, &tagbuf) });
        pbuf[mt.len] = 0;
        const fm = if (write) std.c.open(pz, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644)) else std.c.open(pz, .{ .ACCMODE = .RDONLY });
        if (fm < 0) {
            std.debug.print("glm53: no head snapshot {s} (the draft head starts cold)\n", .{mt});
            return off;
        }
        defer _ = std.c.close(fm);
        var moff: usize = 0;
        const parts = [_]?m.Ref{ e.kvc[m.MTP], e.ic[m.MTP] };
        const widths = [_]usize{ m.CROW, m.ID };
        const held = [_]u32{ @min(n, e.kvRows(m.MTP)), n };
        for (parts, widths, held) |pr, wd, hn| {
            const r = pr orelse continue;
            const bytes = r.addr()[0 .. @as(usize, hn) * wd * e.kv_bytes];
            var done: usize = 0;
            while (done < bytes.len) {
                const k = if (write) std.c.pwrite(fm, bytes.ptr + done, bytes.len - done, @intCast(moff + done)) else std.c.pread(fm, bytes.ptr + done, bytes.len - done, @intCast(moff + done));
                if (k <= 0) {
                    std.debug.print("glm53: head snapshot short ({s}); the draft head starts cold\n", .{mt});
                    return off;
                }
                done += @intCast(k);
            }
            moff += bytes.len;
        }
    }
    return off;
}

/// One learn line on every rank (lockstep: every rank gets the same lines in the same order).
fn serveLearn(c: c_int, sl: *Slide, job: Job, rank: u32) void {
    var line: [512]u8 = undefined;
    if (job.learn_abort) {
        sl.learner.abort();
        sl.busy = false;
        _ = sendAll(c, std.fmt.bufPrint(&line, "{{\"done\":true,\"aborted\":true}}\n", .{}) catch return);
        return;
    }
    if (job.learn) |lj| {
        sl.begin(lj) catch |err| {
            sl.busy = false;
            _ = sendAll(c, std.fmt.bufPrint(&line, "{{\"error\":\"learn {t}\"}}\n", .{err}) catch return);
            return;
        };
        if (job.learn_step == 0) {
            _ = sendAll(c, std.fmt.bufPrint(&line, "{{\"done\":true,\"begun\":true}}\n", .{}) catch return);
            return;
        }
    }
    const t0 = mtl.clock.seconds();
    var units: u32 = 0;
    var changed = false;
    var finished = !sl.busy;
    while (!finished and units < @max(job.learn_step, 1)) : (units += 1) {
        const st = sl.learner.step();
        changed = changed or st.changed;
        if (st.report) |r| if (rank == 0) switch (r) {
            .learned => |x| _ = sendAll(c, std.fmt.bufPrint(&line, "{{\"learned\":{{\"recalled\":{},\"steps\":{d},\"loss\":{d:.5}}}}}\n", .{ x.recalled, x.steps, x.loss }) catch return),
            .failed => |msg| _ = sendAll(c, std.fmt.bufPrint(&line, "{{\"failed\":\"{s}\"}}\n", .{msg}) catch return),
        };
        if (st.done) {
            finished = true;
            sl.busy = false;
        }
    }
    const lw = sl.host.lw;
    const rk: usize = if (sl.learner.trainer) |t| t.sites.rank else lw.served;
    // logits_changed: the served change moved (drop pending picks; the trunk KV stays valid)
    _ = sendAll(c, std.fmt.bufPrint(&line, "{{\"done\":true,\"learn_done\":{},\"logits_changed\":{},\"kv_changed\":false,\"rank\":{d},\"served\":{d},\"units\":{d},\"s\":{d:.3}}}\n", .{ finished, changed, rk, lw.served, units, mtl.clock.seconds() - t0 }) catch return);
}

fn sendAll(fd: c_int, bytes: []const u8) bool {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = std.c.write(fd, bytes.ptr + done, bytes.len - done);
        if (n <= 0) return false;
        done += @intCast(n);
    }
    return true;
}

/// Jobs over TCP, one JSON line each: {"p0": first position (the KV caches hold every position before it), "tokens":
/// [...], "max": n, "stop": [...]}. Every rank runs every job in lockstep (the exchanges keep them together); rank 0
/// answers a line per token {"t": id} and then {"done": ...}; other ranks answer only the done line.
fn serveJobs(e: *fwd.Engine, s: Settings, a: std.mem.Allocator, lw_host: ?*lw_train.Host) !void {
    var slide: ?Slide = null;
    if (lw_host) |h| slide = .{ .host = h, .learner = lw_train.Learner.init(h.gpa, h), .arena = std.heap.ArenaAllocator.init(h.gpa) };
    const fd = std.c.socket(std.c.AF.INET, std.c.SOCK.STREAM, 0);
    if (fd < 0) return error.Socket;
    var one: c_int = 1;
    _ = std.c.setsockopt(fd, std.c.SOL.SOCKET, std.c.SO.REUSEADDR, std.mem.asBytes(&one), @sizeOf(c_int));
    var addr: std.c.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, s.serve), .addr = if (s.bind_local) std.mem.nativeToBig(u32, 0x7f000001) else 0 };
    if (std.c.bind(fd, @ptrCast(&addr), @sizeOf(std.c.sockaddr.in)) != 0) return error.Bind;
    if (std.c.listen(fd, 4) != 0) return error.Listen;
    std.debug.print("glm53: rank {d} serving jobs on :{d}\n", .{ s.rank, s.serve });
    const buf = try a.alloc(u8, 64 << 20);
    var ctl = Ctl.init(s); // acceptance carries across jobs (every rank sees the same jobs)
    // copy drafts' source: the token at every cached position this rank has seen (index = position); rows a snapshot
    // load brought in are unknown (a sentinel that matches no real token)
    var known: std.ArrayList(u32) = .empty;
    const unknown_tok: u32 = 0xffff_ffff;
    while (true) {
        const c = std.c.accept(fd, null, null);
        if (c < 0) continue;
        defer _ = std.c.close(c);
        var have: usize = 0;
        conn: while (true) {
            // one line
            const nl = while (true) {
                if (std.mem.indexOfScalar(u8, buf[0..have], '\n')) |i| break i;
                if (have == buf.len) break :conn;
                const n = std.c.read(c, buf.ptr + have, buf.len - have);
                if (n <= 0) break :conn;
                have += @intCast(n);
            };
            var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena.deinit();
            const job = std.json.parseFromSliceLeaky(Job, arena.allocator(), buf[0..nl], .{ .ignore_unknown_fields = true, .allocate = .alloc_always }) catch {
                _ = sendAll(c, "{\"error\":\"bad job\"}\n");
                std.mem.copyForwards(u8, buf, buf[nl + 1 .. have]);
                have -= nl + 1;
                continue;
            };
            std.mem.copyForwards(u8, buf, buf[nl + 1 .. have]);
            have -= nl + 1;
            if (isStopLine(job)) continue; // a cancel for a job that already ended: nothing to answer
            var line: [256]u8 = undefined;
            if (job.learn != null or job.learn_step > 0 or job.learn_abort) {
                const sl = if (slide) |*x| x else {
                    _ = sendAll(c, "{\"error\":\"learning is off (settings: slide + lw)\"}\n");
                    continue;
                };
                serveLearn(c, sl, job, s.rank);
                continue;
            }
            if (job.save.len > 0 or job.load.len > 0) {
                const t0 = mtl.clock.seconds();
                if (job.load.len > 0) known.clearRetainingCapacity();
                const bytes = snapshot(e, if (job.save.len > 0) job.save else job.load, @min(job.n, s.cap), job.save.len > 0) catch |err| {
                    _ = sendAll(c, try std.fmt.bufPrint(&line, "{{\"error\":\"snapshot {t}\"}}\n", .{err}));
                    continue;
                };
                if (job.load.len > 0 and std.c.getenv("G53_TOUCH") != null) e.touchKv(@min(job.n, s.cap)); // pay the fresh-page cost now
                _ = sendAll(c, try std.fmt.bufPrint(&line, "{{\"done\":true,\"bytes\":{d},\"s\":{d:.2}}}\n", .{ bytes, mtl.clock.seconds() - t0 }));
                continue;
            }
            if (job.tokens.len == 0 or job.p0 + job.tokens.len + job.max > s.cap) {
                _ = sendAll(c, try std.fmt.bufPrint(&line, "{{\"error\":\"needs {d} positions, cap {d}\"}}\n", .{ job.p0 + job.tokens.len + job.max, s.cap }));
                continue;
            }
            const t0 = mtl.clock.seconds();
            _ = e.prefill(job.p0, job.tokens) catch |err| {
                _ = sendAll(c, try std.fmt.bufPrint(&line, "{{\"error\":\"prefill {t}\"}}\n", .{err}));
                continue;
            };
            const t1 = mtl.clock.seconds();
            if (e.pf_blocks > 0) std.debug.print("glm53 prefill timing: {d} rows from p0 {d}, {d} blocks, encode {d:.3} ms, gpu {d:.3} ms, commit->done {d:.3} ms, total {d:.3} ms, link fences {d}\n", .{ job.tokens.len, job.p0, e.pf_blocks, e.pf_enc * 1e3, e.pf_gpu * 1e3, e.pf_wall * 1e3, (t1 - t0) * 1e3, if (e.xc) |x| x.fences else 0 });
            e.pf_enc = 0;
            e.pf_gpu = 0;
            e.pf_wall = 0;
            e.pf_blocks = 0;
            var pos: u32 = job.p0 + @as(u32, @intCast(job.tokens.len));
            var n: u32 = 0;
            var failed = false;
            var limit: u32 = job.max;
            if (e.w.mtp != null) { // rounds of drafts verified by the trunk: the same tokens as the plain loop below
                const Em = struct {
                    c: c_int,
                    rank: u32,
                    n: *u32,
                    limit: *u32,
                    stop: []const u32,
                    fn token(x: @This(), t: u32) bool {
                        x.n.* += 1;
                        var l: [64]u8 = undefined;
                        if (x.rank == 0) _ = sendAll(x.c, std.fmt.bufPrint(&l, "{{\"t\":{d}}}\n", .{t}) catch return false);
                        return !(x.n.* >= x.limit.* or std.mem.indexOfScalar(u32, x.stop, t) != null);
                    }
                };
                const em: Em = .{ .c = c, .rank = s.rank, .n = &n, .limit = &limit, .stop = job.stop };
                var sp = specInit(e, pos);
                var idx: ?CopyIndex = null;
                defer if (idx) |*x| x.deinit();
                if (ctl.copy_min > 0) blk: {
                    if (job.p0 < known.items.len) known.shrinkRetainingCapacity(job.p0);
                    known.ensureTotalCapacity(a, job.p0 + job.tokens.len + 1) catch @panic("glm53: copy history out of memory");
                    while (known.items.len < job.p0) known.appendAssumeCapacity(unknown_tok);
                    known.appendSliceAssumeCapacity(job.tokens);
                    known.appendAssumeCapacity(sp.t);
                    // the index skips unknown rows (a match may span a gap: only a guess, verified like any draft)
                    idx = CopyIndex.init(a, &.{}) catch @panic("glm53: copy index out of memory");
                    for (known.items) |t| if (t != unknown_tok) idx.?.extend(&.{t}) catch @panic("glm53: copy index out of memory");
                    sp.hist = &idx.?;
                    break :blk;
                }
                if (em.token(sp.t)) while (true) {
                    pollStop(c, buf, &have, s.rank, n, &limit);
                    if (n >= limit) break;
                    const more = specRound(e, &ctl, &sp, null, em) catch {
                        failed = true;
                        break;
                    };
                    if (!more) break;
                };
                if (sp.hist) |h| { // the reply joins the known context (positions stay aligned: index = position)
                    // the reply joins the known positions: known ends with the first pending token (position pos);
                    // the n tokens emitted after it (positions pos+1 .. sp.p) are the index's last n
                    const n_new: usize = sp.p - pos;
                    known.appendSlice(a, h.ctx.items[h.ctx.items.len - n_new ..]) catch @panic("glm53: copy history out of memory");
                }
                // the head keeps every kept row (its cache stays whole for the next job)
                if (!failed) e.absorb(sp.r0, sp.next[0..sp.nn], sp.hp0) catch {
                    failed = true;
                };
                pos = sp.p;
            } else while (true) {
                const t = e.token();
                n += 1;
                if (s.rank == 0) _ = sendAll(c, try std.fmt.bufPrint(&line, "{{\"t\":{d}}}\n", .{t}));
                if (n >= limit or std.mem.indexOfScalar(u32, job.stop, t) != null) break;
                pollStop(c, buf, &have, s.rank, n, &limit);
                if (n >= limit) break;
                _ = e.run(pos, null, true) catch {
                    failed = true;
                    break;
                };
                pos += 1;
            }
            const t2 = mtl.clock.seconds();
            if (e.w.mtp != null) std.debug.print("glm53 mtp job: {d} tokens, rounds {d}, drafts {d}, kept {d}, depth use {any}, copy rounds {d} offered {d} kept {d} cut {d}\n", .{ n, ctl.rounds, ctl.drafted, ctl.kept, ctl.by_k[0 .. ctl.max + 1], ctl.c_rounds, ctl.c_offered, ctl.c_kept, ctl.c_cut });
            if (e.st_rounds > 0) std.debug.print("glm53 round timing: {d} rounds, per round encode {d:.3} ms, gpu {d:.3} ms, commit->done {d:.3} ms\n", .{ e.st_rounds, e.st_enc * 1e3 / @as(f64, @floatFromInt(e.st_rounds)), e.st_gpu * 1e3 / @as(f64, @floatFromInt(e.st_rounds)), e.st_wall * 1e3 / @as(f64, @floatFromInt(e.st_rounds)) });
            e.st_enc = 0;
            e.st_gpu = 0;
            e.st_wall = 0;
            e.st_rounds = 0;
            _ = sendAll(c, try std.fmt.bufPrint(&line, "{{\"done\":true,\"n\":{d},\"pos\":{d},\"prompt_s\":{d:.3},\"gen_s\":{d:.3},\"failed\":{},\"rounds\":{d},\"kept\":{d},\"drafted\":{d}}}\n", .{ n, pos, t1 - t0, t2 - t1, failed, ctl.rounds, ctl.kept, ctl.drafted }));
        }
    }
}
