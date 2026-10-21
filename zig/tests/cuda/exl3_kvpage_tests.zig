//! exl3-kvpage: dynamic KV paging for long contexts (docs/plans/exl3-kv-paging.md) on the family's Engine with the
//! EXL3 offload pool, as serve attaches it (one card).
//!
//!   exl3-kvpage DIR trace IDS N[,N..] DECODE OUT [prompt]
//!       the first N ids of IDS (u32 little-endian, real text) prefilled, then DECODE greedy rounds; every attention
//!       row's selected blocks recorded (cuda_exl3_kvpage.Trace) into OUT-N.bin (prompt rows only with `prompt`)
//!   exl3-kvpage none kernel [SLOTS]
//!       the paging kernels on synthetic key lists: every remapped key reads its own host row, misses against an
//!       LRU model on the host, the plan's cost a call
//!   exl3-kvpage DIR cmp IDS N[,N..] DECODE WINDOW SLOTS [NEEDLE_AT]
//!       each prompt through the all-on-device cache, then through the paged one (window WINDOW tokens a layer, SLOTS
//!       4-key block slots shared by the 12 layers): the prefill's last logits and the DECODE greedy tokens must be
//!       bit-equal, and every K/V row of every attention layer equal after the decode (paged rows flushed to host);
//!       ms, copies and device bytes.
//!       NEEDLE_AT (a fraction, e.g. 0.25): a sentence with a 5-digit code inserted that far into the text and a
//!       question about it appended; the greedy answer must contain the code (both caches)
//!
//! BF16 paged snapshot acceptance plan (the format/order contract is in cuda_snap.zig):
//! 1. Host-driver fakes: allocate/save/restore failures at every section, queued copies drained before freeing,
//!    truncated/unknown formats and checked-size overflow, exact byte admission, incompatible capacity/geometry,
//!    no mutation on preflight rejection, and no successful publication after a partial copy failure.
//! 2. Kernel lifecycle: save at 0, 1, 3, 4, dense/sparse boundaries, and immediately before/after several rebases.
//!    Seed distinct BF16 bit patterns (including NaNs/signed zero) per layer/K/V/row, dirty window tails and occupied
//!    LRU slots; compare canonical rows, device windows/slots, both mapping directions, marks, epochs and stamps.
//!    Test same-sequence rewind, restore into a distinct compatible sequence, source drop, repeated restore and
//!    snapshot drop while two other sequences remain live. Host snapshots must remain immutable through each case.
//! 3. Model continuation: save, run a suffix, restore, then require bit-identical eager logits and flushed BF16 K/V
//!    for that suffix; repeat across a window rebase and with partial pooled blocks and PLE n-gram history.
//! 4. MTP: require a settled head mark; reject outstanding drafts. Populate verification cache before restore and
//!    inject a delayed pre-restore publication; neither may survive the generation change. Check restored head
//!    rows, len/drafted/rows, recurrent rollback state and drafted/serial continuation equality across rebases.
//! 5. TP2/graphs: two distinct live histories, staggered rebases, capture/replay before AND after restore, and drop/
//!    recreation of one lane while the other remains live. Compare every rank's canonical K/V and outputs to eager;
//!    require both attention and GDN graph reuse where compatible, and recapture when origin arguments differ.
//! 6. Inject one-rank overflow, copy failure, delayed instantiation and peer failure. Require bounded failure,
//!    joined producers, no failed-batch results or stale verification results, no live-buffer reuse and clean drop.
//! 7. After those gates, two concurrent 200k TP2 requests with BF16 RAM paging, snapshot branch/reuse and MTP, graph
//!    prefill/decode parity, host/device peak accounting and cold/warm timings. No gate above has been run here.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const flash = @import("flashnext");
const Gpu = check.Gpu;

const prompt = flash.prompt;
const xpass = flash.exl3.pass;
const xpre = flash.exl3.prefill;
const kvp = flash.exl3.kvpage;
const vocab = prompt.vocabulary;
const core = @import("core");

const Eng = struct {
    gpu: Gpu,
    stream: cuda.Stream,
    glue: flash.Glue,
    e: *prompt.Engine,
    seq: prompt.Seq,
    attached: xpass.Attached,

    fn init(gpu: Gpu, dir: []const u8, cap: usize, deep_rows: usize) !*Eng {
        const g = try gpu.gpa.create(Eng);
        errdefer gpu.gpa.destroy(g);
        g.gpu = gpu;
        g.stream = try cuda.Stream.init(gpu.d, true);
        g.glue = try flash.Glue.load(gpu.d);
        const t0 = check.now(gpu.io);
        g.e = try prompt.Engine.initRank(gpu.gpa, gpu.io, @constCast(gpu.d), &g.stream, dir, 1, 0);
        _ = try g.e.reserve(@min(cap, g.e.chunk));
        _ = try g.e.reserveScores(cap, @min(cap, g.e.chunk));
        g.seq = try g.e.newSeq(cap);
        if (deep_rows > 0) try xpre.reserve(gpu.gpa, gpu.d, g.e, deep_rows);
        g.attached = try xpass.attach(gpu.gpa, gpu.io, gpu.d, g.stream, g.e.layers[0..g.e.n_layers], null, g.e.chunk, try xpass.budgetFromEnv(0));
        var free: usize = 0;
        var total: usize = 0;
        try gpu.d.check(gpu.d.api.cuMemGetInfo_v2(&free, &total), "cuMemGetInfo");
        std.debug.print("engine: loaded in {d:.1} s; cap {d}; pool {d} lines a layer; {d:.2} GiB free\n", .{ @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e9, cap, g.attached.lines, @as(f64, @floatFromInt(free)) / (1 << 30) });
        return g;
    }

    fn deinit(g: *Eng) void {
        xpre.release();
        g.seq.deinit(g.gpu.gpa);
        g.e.deinit();
        g.glue.deinit();
        g.stream.deinit();
        g.gpu.gpa.destroy(g);
    }

    fn tri(g: *Eng) flash.Tri {
        return .{ .own = &g.glue, .s = g.stream };
    }

    fn reset(g: *Eng) !void {
        const s = &g.seq;
        for ([_][]cuda.DeviceBuffer{ s.conv, s.state, s.kc, s.vc, s.ikc, s.pooled }) |bs| for (bs) |b| try b.fill8(0, g.stream.handle);
        try s.tail.fill8(0, g.stream.handle);
        try g.stream.synchronize();
        s.fed = 0;
        s.history = .{ g.e.eos, g.e.eos };
    }

    fn prefill(g: *Eng, ids: []const u32) !f64 {
        try g.reset();
        const t0 = check.now(g.gpu.io);
        try g.e.forward(flash.Tri, g.tri(), &g.seq, ids, .{ .logits = .last });
        return @as(f64, @floatFromInt(check.now(g.gpu.io) - t0)) / 1e6;
    }

    /// `n` greedy rounds from the prefill's logits; the tokens into `out`; ms a round.
    fn decode(g: *Eng, out: []u32) !f64 {
        var dev = try cuda.DeviceBuffer.alloc(g.gpu.d, vocab * 2);
        defer dev.free();
        var tok = prompt.argmax(g.seq.logits);
        const t0 = check.now(g.gpu.io);
        for (out) |*o| {
            o.* = tok;
            const q = [_]*prompt.Seq{&g.seq};
            try g.e.round(flash.Tri, g.tri(), &q, (&tok)[0..1], dev.ptr);
            tok = prompt.argmax(g.seq.logits);
        }
        return @as(f64, @floatFromInt(check.now(g.gpu.io) - t0)) / 1e6 / @as(f64, @floatFromInt(@max(out.len, 1)));
    }
};

fn readIds(gpu: Gpu, path: []const u8) ![]u32 {
    const a = gpu.gpa;
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, a, .limited(1 << 28));
    defer a.free(text);
    const ids = try a.alloc(u32, text.len / 4);
    for (ids, 0..) |*x, i| x.* = std.mem.readInt(u32, text[4 * i ..][0..4], .little);
    return ids;
}

fn parseList(s: []const u8, out: []usize) !usize {
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, s, ',');
    while (it.next()) |t| {
        out[n] = try std.fmt.parseInt(usize, t, 10);
        n += 1;
    }
    return n;
}

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 2) return error.MissingArgument;
    const dir = rest[0];
    const mode = rest[1];
    if (std.mem.eql(u8, mode, "trace")) return traceRun(gpu, dir, rest[2..]);
    if (std.mem.eql(u8, mode, "cmp")) return cmpRun(gpu, dir, rest[2..]);
    if (std.mem.eql(u8, mode, "kernel")) return kernelRun(gpu, if (rest.len > 2) try std.fmt.parseInt(usize, rest[2], 10) else 6144);
    if (std.mem.eql(u8, mode, "kernel-graph")) return kernelReplay(gpu, 49152, true, false);
    if (std.mem.eql(u8, mode, "window-graph")) return kernelReplay(gpu, 49152, true, true);
    if (std.mem.eql(u8, mode, "overflow-graph")) return overflowReplay(gpu);
    if (std.mem.eql(u8, mode, "snapshot")) return snapshotLifecycle(gpu);
    return error.UnknownMode;
}

fn snapshotSequence(gpu: Gpu, stream: cuda.Stream) !prompt.Seq {
    var engine: prompt.Engine = undefined;
    engine.gpa = gpu.gpa;
    engine.driver = @constCast(gpu.d);
    var owned_stream = stream;
    engine.stream = &owned_stream;
    engine.chunk = 256;
    engine.eos = 0;
    engine.round_graph = .{};
    engine.page_config = .{ .window = 2052, .slots = 16, .rows = 256 };
    return engine.newSeq(8192);
}

fn snapshotLifecycle(gpu: Gpu) !void {
    const snaps = @import("flashnext").snapshot;
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var destination = try snapshotSequence(gpu, stream);
    defer destination.deinit(gpu.gpa);
    for ([_]usize{ 0, 1, 3, 4, 2051, 2052, 4096, 4099, 6147 }) |at| {
        var source: ?prompt.Seq = try snapshotSequence(gpu, stream);
        defer if (source) |*sequence| sequence.deinit(gpu.gpa);
        const pg = source.?.paged.?;
        source.?.fed = at;
        source.?.history = .{ @intCast(at), 73 };
        for (0..12) |layer| {
            const origin = (at -| 2052) / 4 * 4;
            pg.win0[layer] = origin;
            pg.rebases[layer] = layer + 3;
            pg.written[layer] = origin;
            for (0..2) |side| {
                const host = pg.hostMut(layer, side == 1);
                const words: []u16 = std.mem.bytesAsSlice(u16, @as([]align(2) u8, @alignCast(host)));
                for (words, 0..) |*word, index| word.* = @truncate(index + layer * 997 + side * 32768);
            }
        }
        try pg.dev.fill8(0xFF, stream.handle);
        const words = try gpu.gpa.alloc(u32, pg.tabs.len / 4);
        defer gpu.gpa.free(words);
        try pg.tabs.download(0, std.mem.sliceAsBytes(words));
        const ctl_at = 24 * pg.nblocks;
        const inverse_at = ctl_at + 12 * 8;
        for (0..12) |layer| {
            words[ctl_at + layer * 8] = 73;
            words[ctl_at + layer * 8 + 6] = @intCast(31 + layer);
            if (at >= 4) {
                words[layer * pg.nblocks] = @intCast(layer);
                words[12 * pg.nblocks + layer * pg.nblocks] = @intCast(31 + layer);
                words[inverse_at + layer] = @intCast(layer * pg.nblocks);
                words[inverse_at + pg.cfg.slots + layer] = @intCast(19 + layer);
            }
        }
        try pg.tabs.upload(0, std.mem.sliceAsBytes(words));
        const saved = try snaps.save(gpu.gpa, gpu.d, stream, &source.?, null);
        defer snaps.drop(gpu.gpa, stream, saved);
        source.?.deinit(gpu.gpa);
        source = null;
        for (0..2) |_| {
            try snaps.restore(stream, &destination, null, saved);
            const again = try snaps.save(gpu.gpa, gpu.d, stream, &destination, null);
            defer snaps.drop(gpu.gpa, stream, again);
            try check.expect(std.meta.eql(saved.paged.?, again.paged.?), "snapshot pager counters at {d}", .{at});
            try check.expect(std.mem.eql(u8, saved.host.?.bytes[0..saved.buf.len], again.host.?.bytes[0..again.buf.len]), "snapshot exact BF16, windows, slots, tables and ordinary state at {d}", .{at});
        }
    }
    check.pass("BF16 paged snapshot source-drop, distinct destination, repeated restore at nine boundaries", .{});
}

fn traceRun(gpu: Gpu, dir: []const u8, rest: []const [:0]const u8) !void {
    const a = gpu.gpa;
    if (rest.len < 4) return error.MissingArgument;
    const all = try readIds(gpu, rest[0]);
    defer a.free(all);
    var ns_buf: [16]usize = undefined;
    const ns = ns_buf[0..try parseList(rest[1], &ns_buf)];
    const steps = try std.fmt.parseInt(usize, rest[2], 10);
    const out_prefix = rest[3];
    const with_prompt = rest.len > 4 and std.mem.eql(u8, rest[4], "prompt");
    var max_n: usize = 0;
    for (ns) |n| max_n = @max(max_n, n);
    if (max_n > all.len) return error.NotEnoughIds;
    const o = xpre.optionsFromEnv();
    const g = try Eng.init(gpu, dir, max_n + steps + 64, if (o.enabled) @min(o.rows, (max_n + 255) / 256 * 256) else 0);
    defer g.deinit();
    const toks = try a.alloc(u32, steps);
    defer a.free(toks);
    for (ns) |n| {
        var t = try kvp.Trace.init(a, with_prompt);
        defer t.deinit();
        kvp.trace = &t;
        defer kvp.trace = null;
        const ms_p = try g.prefill(all[0..n]);
        const prompt_rows = t.rows;
        const ms_d = try g.decode(toks);
        std.debug.print("TRACE {d} tokens: prefill {d:.0} ms (traced, {d} rows recorded), {d} decode rounds {d:.1} ms/round (traced: one sync a layer); {d} rows recorded; {d:.1} MiB\n", .{ n, ms_p, prompt_rows, steps, ms_d, t.rows, @as(f64, @floatFromInt(t.out.items.len)) / (1 << 20) });
        std.debug.print("  greedy: {any}\n", .{toks[0..@min(steps, 48)]});
        var buf: [512]u8 = undefined;
        const path = try std.fmt.bufPrint(&buf, "{s}-{d}.bin", .{ out_prefix, n });
        try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = path, .data = t.out.items });
    }
    check.pass("exl3-kvpage trace written", .{});
}

fn freeMem(gpu: Gpu) !usize {
    var free: usize = 0;
    var total: usize = 0;
    try gpu.d.check(gpu.d.api.cuMemGetInfo_v2(&free, &total), "cuMemGetInfo");
    return free;
}

fn gib(x: usize) f64 {
    return @as(f64, @floatFromInt(x)) / (1 << 30);
}

const Run = struct {
    prefill_ms: f64 = 0,
    decode_ms: f64 = 0,
    logits: []u8 = &.{},
    toks: []u32 = &.{},
    /// [layer][K|V] rows 0..fed (all-on-device: downloaded; paged: its host copy after a flush)
    kv: [][]u8 = &.{},
    stats_prefill: kvp.Stats = .{},
    stats_decode: kvp.Stats = .{},
};

fn oneRun(g: *Eng, ids: []const u32, steps: usize, keep_kv: bool) !Run {
    const a = g.gpu.gpa;
    var r: Run = .{};
    r.prefill_ms = try g.prefill(ids);
    r.logits = try a.dupe(u8, g.seq.logits);
    if (g.seq.paged) |pg| r.stats_prefill = try pg.total();
    r.toks = try a.alloc(u32, steps);
    r.decode_ms = try g.decode(r.toks);
    if (g.seq.paged) |pg| {
        const t = try pg.total();
        r.stats_decode = .{ .misses = t.misses - r.stats_prefill.misses, .needed = t.needed - r.stats_prefill.needed, .calls = t.calls - r.stats_prefill.calls, .overflow = t.overflow, .rebases = t.rebases - r.stats_prefill.rebases, .written_rows = t.written_rows - r.stats_prefill.written_rows };
    }
    if (!keep_kv) return r;
    const fed = g.seq.fed;
    const row = kvp.row_bytes;
    r.kv = try a.alloc([]u8, 24);
    for (0..12) |ai| for (0..2) |v| {
        const dst = try a.alloc(u8, fed * row);
        if (g.seq.paged) |pg| {
            if (v == 0 and ai == 0) {
                try pg.flush(g.stream, fed);
                try g.stream.synchronize();
            }
            @memcpy(dst, pg.hostRows(ai, v == 1)[0 .. fed * row]);
        } else {
            const src = if (v == 0) g.seq.kc[ai].ptr else g.seq.vc[ai].ptr;
            try g.gpu.d.check(g.gpu.d.api.cuMemcpyDtoH_v2(dst.ptr, src, fed * row), "cuMemcpyDtoH");
        }
        r.kv[2 * ai + v] = dst;
    };
    return r;
}

fn freeRun(a: std.mem.Allocator, r: *Run) void {
    a.free(r.logits);
    a.free(r.toks);
    for (r.kv) |x| a.free(x);
    if (r.kv.len > 0) a.free(r.kv);
}

/// The first n ids, or (needle) the text with a code sentence at `at` and a question about it at the end.
fn promptOf(a: std.mem.Allocator, tok: *const core.tokenizer.Tokenizer, all: []const u32, n: usize, needle_at: ?f64, code: []const u8) ![]u32 {
    if (needle_at == null) return a.dupe(u32, all[0..n]);
    var buf: [256]u8 = undefined;
    const fact = try tok.encodePlain(a, try std.fmt.bufPrint(&buf, "\n\nIMPORTANT NOTE: the secret access code for the blue vault is {s}. Remember this code.\n\n", .{code}));
    defer a.free(fact);
    const q = try tok.encodePlain(a, "\n\nQuestion: what is the secret access code for the blue vault, given in the important note above?\nAnswer: the secret access code for the blue vault is");
    defer a.free(q);
    const body = n - fact.len - q.len;
    const at: usize = @intFromFloat(@as(f64, @floatFromInt(body)) * needle_at.?);
    var out: std.ArrayList(u32) = .empty;
    try out.appendSlice(a, all[0..at]);
    try out.appendSlice(a, fact);
    try out.appendSlice(a, all[at..body]);
    try out.appendSlice(a, q);
    return out.toOwnedSlice(a);
}

fn cmpRun(gpu: Gpu, dir: []const u8, rest: []const [:0]const u8) !void {
    const a = gpu.gpa;
    if (rest.len < 5) return error.MissingArgument;
    const all = try readIds(gpu, rest[0]);
    defer a.free(all);
    var ns_buf: [16]usize = undefined;
    const ns = ns_buf[0..try parseList(rest[1], &ns_buf)];
    const steps = try std.fmt.parseInt(usize, rest[2], 10);
    const window = try std.fmt.parseInt(usize, rest[3], 10);
    const slots = try std.fmt.parseInt(usize, rest[4], 10);
    const needle_at: ?f64 = if (rest.len > 5) try std.fmt.parseFloat(f64, rest[5]) else null;
    var max_n: usize = 0;
    for (ns) |n| max_n = @max(max_n, n);
    if (max_n > all.len) return error.NotEnoughIds;
    var path_buf: [512]u8 = undefined;
    var tok = try core.tokenizer.loadTokenizer(gpu.io, a, try std.fmt.bufPrint(&path_buf, "{s}/tokenizer.json", .{dir}));
    defer tok.deinit();
    const o = xpre.optionsFromEnv();
    const cap = max_n + steps + 64;
    kvp.force_off = true;
    const g = try Eng.init(gpu, dir, cap, if (o.enabled) @min(o.rows, (max_n + 255) / 256 * 256) else 0);
    defer g.deinit();
    kvp.force_off = false;
    const cfg: kvp.Config = .{ .window = window, .slots = slots, .rows = g.e.chunk };
    var all_ok = true;
    const codes = [_][]const u8{ "48613", "70259", "31846", "95072", "62418", "27530" };
    for (ns, 0..) |n, ni| {
        const code = codes[ni % codes.len];
        const ids = try promptOf(a, &tok, all, n, needle_at, code);
        defer a.free(ids);
        // all on the device (the engine's own sequence)
        var d = try oneRun(g, ids, steps, true);
        defer freeRun(a, &d);
        const free_dense = try freeMem(gpu);
        g.seq.deinit(a);
        const free_none = try freeMem(gpu);
        kvp.force = cfg;
        g.seq = try g.e.newSeq(cap);
        kvp.force = null;
        const free_paged = try freeMem(gpu);
        const pg = g.seq.paged orelse return error.KvPageOff;
        const pg_dev = pg.deviceBytes();
        const pg_host = pg.host.bytes.len;
        var p = try oneRun(g, ids, steps, true);
        defer freeRun(a, &p);
        // back to an all-on-device sequence for the next prompt
        g.seq.deinit(a);
        kvp.force_off = true;
        g.seq = try g.e.newSeq(cap);
        kvp.force_off = false;
        const logits_eq = std.mem.eql(u8, d.logits, p.logits);
        const toks_eq = std.mem.eql(u32, d.toks, p.toks);
        var kv_eq: usize = 0;
        for (d.kv, p.kv) |x, y| kv_eq += @intFromBool(std.mem.eql(u8, x, y));
        const dense_seq = free_none - free_dense;
        const paged_seq = free_none - free_paged;
        std.debug.print("CMP {d} tokens{s}: prefill logits bit-equal {}; {d} greedy tokens equal {}; K/V rows 0..{d} equal in {d}/24 layer-sides\n", .{ ids.len, if (needle_at != null) " (needle)" else "", logits_eq, steps, toks_eq, ids.len + steps, kv_eq });
        std.debug.print("    device bytes of the sequence: all-on-device {d:.3} GiB, paged {d:.3} GiB (window {d} tokens a layer, {d} block slots shared by the layers: paged side {d:.3} GiB), freed {d:.3} GiB; host {d:.2} GiB pinned\n", .{ gib(dense_seq), gib(paged_seq), window, slots, gib(pg_dev), gib(dense_seq -| paged_seq), gib(pg_host) });
        std.debug.print("    prefill: all-on-device {d:.0} ms, paged {d:.0} ms; decode: all-on-device {d:.2} ms/token, paged {d:.2} ms/token\n", .{ d.prefill_ms, p.prefill_ms, d.decode_ms, p.decode_ms });
        const sp = p.stats_prefill;
        const sd = p.stats_decode;
        std.debug.print("    paged prefill: {d} calls, {d} blocks needed outside the window, {d} misses ({d:.1}% hit, {d:.1} MiB fetched, {d:.0} B a row a layer), {d} window moves, {d} rows written out; overflow {}\n", .{ sp.calls, sp.needed, sp.misses, 100 * (1 - @as(f64, @floatFromInt(sp.misses)) / @as(f64, @floatFromInt(@max(sp.needed, 1)))), @as(f64, @floatFromInt(sp.misses * 8192)) / (1 << 20), @as(f64, @floatFromInt(sp.misses * 8192)) / @as(f64, @floatFromInt(ids.len * 12)), sp.rebases, sp.written_rows, sp.overflow });
        std.debug.print("    paged decode: {d} calls, {d} needed outside the window, {d} misses ({d:.1}% hit), {d:.1} KiB fetched a token ({d:.2} KiB a layer); overflow {}\n", .{ sd.calls, sd.needed, sd.misses, 100 * (1 - @as(f64, @floatFromInt(sd.misses)) / @as(f64, @floatFromInt(@max(sd.needed, 1)))), @as(f64, @floatFromInt(sd.misses * 8192)) / 1024 / @as(f64, @floatFromInt(@max(steps, 1))), @as(f64, @floatFromInt(sd.misses * 8192)) / 1024 / @as(f64, @floatFromInt(@max(steps, 1) * 12)), sd.overflow });
        const td = try tok.decode(a, d.toks, false);
        defer a.free(td);
        const tp = try tok.decode(a, p.toks, false);
        defer a.free(tp);
        std.debug.print("    greedy (all-on-device): {any}\n    text: \"{f}\"\n", .{ d.toks[0..@min(steps, 32)], std.zig.fmtString(td[0..@min(td.len, 240)]) });
        if (!toks_eq) std.debug.print("    greedy (paged): {any}\n    text: \"{f}\"\n", .{ p.toks[0..@min(steps, 32)], std.zig.fmtString(tp[0..@min(tp.len, 240)]) });
        if (needle_at != null) {
            const hit_d = std.mem.indexOf(u8, td, code) != null;
            const hit_p = std.mem.indexOf(u8, tp, code) != null;
            std.debug.print("    NEEDLE code {s} at {d:.2} of the text: all-on-device {s}, paged {s}\n", .{ code, needle_at.?, if (hit_d) "FOUND" else "missed", if (hit_p) "FOUND" else "missed" });
            if (!hit_p) all_ok = false;
        }
        if (!logits_eq or !toks_eq or kv_eq != 24 or sp.overflow or sd.overflow) all_ok = false;
    }
    try check.expect(all_ok, "paged K/V bit-equal to all-on-device (logits, greedy tokens, rows){s}", .{if (needle_at != null) " and the needle found" else ""});
    check.pass("exl3-kvpage cmp", .{});
}

/// The host's model of kvp_plan: exact LRU at call granularity, ties to the lower slot.
const Model = struct {
    stamp: []u32,
    slot_of: std.AutoHashMap(u32, u32),
    blk: []i64,
    epoch: u32 = 0,

    fn use(mo: *Model, a: std.mem.Allocator, need: []const u32) !usize {
        mo.epoch += 1;
        var misses: std.ArrayList(u32) = .empty;
        defer misses.deinit(a);
        for (need) |b| {
            if (mo.slot_of.get(b)) |sl| mo.stamp[sl] = mo.epoch else try misses.append(a, b);
        }
        const order = try a.alloc(u32, mo.stamp.len);
        defer a.free(order);
        for (order, 0..) |*o, i| o.* = @intCast(i);
        const Ctx = struct {
            st: []const u32,
            ep: u32,
            fn key(c: @This(), i: u32) u64 {
                const t = c.st[i];
                return if (t == c.ep) std.math.maxInt(u64) else (@as(u64, t) << 32) | i;
            }
            fn less(c: @This(), x: u32, y: u32) bool {
                return c.key(x) < c.key(y);
            }
        };
        std.mem.sort(u32, order, Ctx{ .st = mo.stamp, .ep = mo.epoch }, Ctx.less);
        for (misses.items, 0..) |b, i| {
            const sl = order[i];
            if (mo.stamp[sl] == mo.epoch) return error.ModelOverflow;
            if (mo.blk[sl] >= 0) _ = mo.slot_of.remove(@intCast(mo.blk[sl]));
            mo.blk[sl] = b;
            try mo.slot_of.put(b, sl);
            mo.stamp[sl] = mo.epoch;
        }
        return misses.items.len;
    }
};

fn kernelRun(gpu: Gpu, slots: usize) !void {
    return kernelReplay(gpu, slots, false, false);
}

fn kernelReplay(gpu: Gpu, slots: usize, graphed: bool, moving: bool) !void {
    const a = gpu.gpa;
    const idw = flash.exl3.prompt_wt.index_idw;
    const row = kvp.row_bytes;
    const cap: usize = 60_000;
    const w0: usize = 40_000;
    const nb_before = w0 / 4;
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const pg = try kvp.Paged.init(a, gpu.d, cap, .{ .window = 2052, .slots = slots, .rows = 256 });
    defer pg.destroy();
    pg.setWindow(0, w0);
    // each host row's 256 words unique to (position, K or V)
    for (0..2) |v| {
        const h = std.mem.bytesAsSlice(u32, pg.hostMut(0, v == 1));
        for (0..cap) |p| for (0..row / 4) |i| {
            h[p * (row / 4) + i] = @intCast((p *% 2654435761 +% i *% 40503 +% v *% 977) & 0xFFFF_FFFF);
        };
        if (moving) try gpu.d.check(gpu.d.api.cuMemcpyHtoD_v2(pg.devRows(0, v == 1), pg.hostRows(0, v == 1).ptr + w0 * row, pg.wr * row), "cuMemcpyHtoD");
    }
    var ids_d = try cuda.DeviceBuffer.alloc(gpu.d, 256 * idw * 4);
    defer ids_d.free();
    var nk_d = try cuda.DeviceBuffer.alloc(gpu.d, 256 * 4);
    defer nk_d.free();
    var sp_d = try cuda.DeviceBuffer.alloc(gpu.d, 256 * 4);
    defer sp_d.free();
    const ids = try a.alloc(i32, 256 * idw);
    defer a.free(ids);
    const back = try a.alloc(i32, 256 * idw);
    defer a.free(back);
    const nk = try a.alloc(i32, 256);
    defer a.free(nk);
    const sp = try a.alloc(i32, 256);
    defer a.free(sp);
    @memset(sp, 1);
    const lay = pg.slotRows(0);
    const dk = try a.alloc(u8, lay[0] * row);
    defer a.free(dk);
    const dv = try a.alloc(u8, lay[0] * row);
    defer a.free(dv);
    var model: Model = .{ .stamp = try a.alloc(u32, slots), .slot_of = .init(a), .blk = try a.alloc(i64, slots) };
    defer {
        a.free(model.stamp);
        a.free(model.blk);
        model.slot_of.deinit();
    }
    @memset(model.stamp, 0);
    @memset(model.blk, -1);
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    // a drifting selection: 512 distinct blocks of [0, nb_before), 15% replaced a call
    var base: [512]u32 = undefined;
    var used = std.AutoHashMap(u32, void).init(a);
    defer used.deinit();
    var k: usize = 0;
    while (k < 512) {
        const b = rng.uintLessThan(u32, @intCast(nb_before));
        if ((try used.getOrPut(b)).found_existing) continue;
        base[k] = b;
        k += 1;
    }
    var gpu_miss: usize = 0;
    var model_miss: usize = 0;
    var bad: usize = 0;
    var keys_checked: usize = 0;
    var plan_ns: i96 = 0;
    const calls: usize = 60;
    var execs: [2]?cuda.graph.Exec = @splat(null);
    defer for (&execs) |*exec| if (exec.*) |*e| e.deinit();
    var captured_window = w0;
    var captures: usize = 0;
    var replays: usize = 0;
    for (0..calls) |c| {
        const m: usize = if (c % 10 == 9) 64 else 1;
        const pos0 = w0 + pg.wr - 70 + c;
        if (moving) {
            try pg.prepareLayer(stream, 0, pos0, m);
            const win = pg.win0[0];
            for (0..2) |v| try gpu.d.check(gpu.d.api.cuMemcpyHtoDAsync_v2(pg.devRows(0, v == 1) + (pos0 - win) * row, pg.hostRows(0, v == 1).ptr + pos0 * row, m * row, stream.handle), "cuMemcpyHtoDAsync");
            if (win != captured_window) {
                try stream.synchronize();
                for (&execs) |*exec| if (exec.*) |*e| {
                    e.deinit();
                    exec.* = null;
                };
                captured_window = win;
            }
        }
        var need = std.AutoHashMap(u32, void).init(a);
        defer need.deinit();
        for (0..m) |r| {
            // drift: replace ~15% of the base blocks
            for (0..77) |_| {
                const j = rng.uintLessThan(usize, 512);
                while (true) {
                    const b = rng.uintLessThan(u32, @intCast(nb_before));
                    if ((try used.getOrPut(b)).found_existing) continue;
                    _ = used.remove(base[j]);
                    base[j] = b;
                    break;
                }
            }
            var sorted = base;
            std.mem.sort(u32, &sorted, {}, std.sort.asc(u32));
            const out = ids[r * idw ..][0..idw];
            for (sorted, 0..) |b, j| for (0..4) |t| {
                out[j * 4 + t] = @intCast(b * 4 + t);
            };
            // 3 tail keys inside the window
            for (0..3) |t| out[2048 + t] = @intCast(if (moving) pos0 - 3 + t else w0 + 100 + c * 4 + t);
            nk[r] = 2051;
            for (sorted) |b| try need.put(b, {});
        }
        var need_list: std.ArrayList(u32) = .empty;
        defer need_list.deinit(a);
        var it = need.keyIterator();
        while (it.next()) |b| try need_list.append(a, b.*);
        model_miss += try model.use(a, need_list.items);
        try gpu.d.check(gpu.d.api.cuMemcpyHtoD_v2(ids_d.ptr, ids.ptr, m * idw * 4), "cuMemcpyHtoD");
        try gpu.d.check(gpu.d.api.cuMemcpyHtoD_v2(nk_d.ptr, nk.ptr, m * 4), "cuMemcpyHtoD");
        try gpu.d.check(gpu.d.api.cuMemcpyHtoD_v2(sp_d.ptr, sp.ptr, m * 4), "cuMemcpyHtoD");
        const before = try pg.stats(0);
        const t0 = check.now(gpu.io);
        if (graphed) {
            const slot: usize = @intFromBool(m == 64);
            if (execs[slot] == null) {
                try cuda.graph.beginCapture(stream, .thread_local);
                try pg.page(stream, 0, ids_d.ptr, nk_d.ptr, sp_d.ptr, m);
                var template = try cuda.graph.endCapture(stream);
                defer template.deinit();
                execs[slot] = try template.instantiate();
                captures += 1;
            } else replays += 1;
            try execs[slot].?.launchOn(stream);
        } else try pg.page(stream, 0, ids_d.ptr, nk_d.ptr, sp_d.ptr, m);
        try stream.synchronize();
        plan_ns += check.now(gpu.io) - t0;
        const after = try pg.stats(0);
        try check.expect(after.epoch == c + 1 and after.calls == c + 1, "pager device epoch/calls advance exactly once on launch {d}", .{c + 1});
        gpu_miss += after.misses - before.misses;
        try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(back.ptr, ids_d.ptr, m * idw * 4), "cuMemcpyDtoH");
        try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(dk.ptr, pg.devRows(0, false), dk.len), "cuMemcpyDtoH");
        try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(dv.ptr, pg.devRows(0, true), dv.len), "cuMemcpyDtoH");
        const hk = pg.hostRows(0, false);
        const hv = pg.hostRows(0, true);
        for (0..m) |r| for (0..2051) |ki| {
            const p: usize = @intCast(ids[r * idw + ki]);
            const q: usize = @intCast(back[r * idw + ki]);
            keys_checked += 1;
            if (p >= pg.win0[0]) {
                bad += @intFromBool(q != p - pg.win0[0]);
                continue;
            }
            if (q < lay[1] or q >= lay[0]) {
                bad += 1;
                continue;
            }
            bad += @intFromBool(!std.mem.eql(u8, dk[q * row ..][0..row], hk[p * row ..][0..row]) or !std.mem.eql(u8, dv[q * row ..][0..row], hv[p * row ..][0..row]));
        };
    }
    const st = try pg.stats(0);
    if (graphed) {
        std.debug.print("pager graphs: captures {d}, replays {d}, device epoch {d}, calls {d}, rebases {d}, written rows {d}\n", .{ captures, replays, st.epoch, st.calls, st.rebases, st.written_rows });
        try check.expect(replays > 0 and st.epoch == calls and st.calls == calls, "captured pager executes fresh epochs", .{});
    }
    if (moving) {
        const fed = w0 + pg.wr - 70 + calls + 63;
        try pg.flushLayer(stream, 0, fed);
        try stream.synchronize();
        for (0..2) |v| {
            const words = std.mem.bytesAsSlice(u32, pg.hostRows(0, v == 1));
            for (0..fed) |position| for (0..row / 4) |column| {
                const expected: u32 = @truncate(position *% 2654435761 +% column *% 40503 +% v *% 977);
                if (words[position * (row / 4) + column] != expected) bad += 1;
            };
        }
        try check.expect(st.rebases > 0 and st.written_rows > 0 and captures > 2, "window moves flush rows and recapture pager graphs", .{});
    }
    std.debug.print("kernel: {d} calls ({d} keys checked): bad {d}; misses GPU {d}, host LRU model {d}; needed {d}; overflow {}; plan+fetch+remap {d:.1} us a call (incl. launch + sync)\n", .{ calls, keys_checked, bad, gpu_miss, model_miss, st.needed, st.overflow, @as(f64, @floatFromInt(plan_ns)) / 1e3 / @as(f64, @floatFromInt(calls)) });
    try check.expect(bad == 0 and !st.overflow, "every remapped key reads its own host row", .{});
    const diff = @as(f64, @floatFromInt(@max(gpu_miss, model_miss) - @min(gpu_miss, model_miss))) / @as(f64, @floatFromInt(@max(model_miss, 1)));
    try check.expect(diff < 0.05, "misses within 5% of the LRU model ({d:.3})", .{diff});
    check.pass("exl3-kvpage kernel", .{});
}

fn overflowReplay(gpu: Gpu) !void {
    const idw = flash.exl3.prompt_wt.index_idw;
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const pg = try kvp.Paged.init(gpu.gpa, gpu.d, 4096, .{ .window = 2052, .rows = 1, .slots = 1 });
    defer pg.destroy();
    pg.setWindow(0, 2052);
    var ids = try cuda.DeviceBuffer.alloc(gpu.d, idw * 4);
    defer ids.free();
    var nk = try cuda.DeviceBuffer.alloc(gpu.d, 4);
    defer nk.free();
    var sp = try cuda.DeviceBuffer.alloc(gpu.d, 4);
    defer sp.free();
    var input: [idw]i32 = @splat(0);
    for (0..2048) |index| input[index] = @intCast(index);
    try ids.upload(0, std.mem.sliceAsBytes(&input));
    const count: i32 = 2048;
    const sparse: i32 = 1;
    try nk.upload(0, std.mem.asBytes(&count));
    try sp.upload(0, std.mem.asBytes(&sparse));
    try cuda.graph.beginCapture(stream, .thread_local);
    try pg.page(stream, 0, ids.ptr, nk.ptr, sp.ptr, 1);
    var template = try cuda.graph.endCapture(stream);
    defer template.deinit();
    var exec = try template.instantiate();
    defer exec.deinit();
    for (0..2) |_| {
        try nk.upload(0, std.mem.asBytes(&count));
        try exec.launchOn(stream);
        try stream.synchronize();
        var actual: i32 = -1;
        try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(&actual, nk.ptr, 4), "cuMemcpyDtoH");
        const stats = try pg.stats(0);
        try check.expect(actual == 0 and stats.overflow and stats.misses == 0, "overflow suppresses fetch and attention key reads on every replay", .{});
    }
    check.pass("exl3-kvpage graph overflow fails closed before key reads", .{});
}

test "exl3: invalid paged snapshot geometry rejects before allocation, CUDA access or TP peer dispatch" {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    var driver: cuda.Driver = undefined;
    var stream: cuda.Stream = .{ .d = &driver, .handle = null };
    var engine: prompt.Engine = undefined;
    engine.driver = &driver;
    engine.stream = &stream;
    var session: flash.rank_engine.Session = undefined;
    session.gpa = failing.allocator();
    session.engine = &engine;
    session.next_object = 77;
    var pager: kvp.Paged = undefined;
    pager.cfg = .{ .window = 0 };
    pager.cap = 200_016;
    var seq: prompt.Seq = undefined;
    seq.invalidated = false;
    seq.paged = &pager;
    var draft: flash.mtp.Draft = undefined;
    draft.verify_cache = .{ .generation = 17, .ready = 17, .rows = 4, .position = 200_000 };
    const before = draft.verify_cache;
    for ([_]usize{ 0, 1, 3, 4, 2051, 2052, 4096, 200_000 }) |at| {
        seq.fed = at;
        seq.cap = 200_016;
        seq.history = .{ 81, 93 };
        try std.testing.expectError(error.KvPageConfig, session.save(&seq, null));
        try std.testing.expectError(error.KvPageConfig, session.save(&seq, &draft));
        try std.testing.expectEqual(at, seq.fed);
        try std.testing.expectEqual([2]i64{ 81, 93 }, seq.history);
        try std.testing.expectEqual(@as(u64, 77), session.next_object);
        try std.testing.expectEqualDeep(before, draft.verify_cache);
    }
    try std.testing.expectEqual(@as(usize, 0), failing.alloc_index);
    try std.testing.expect(!failing.has_induced_failure);
}

test "exl3: paged flush orders all pending BF16 window rows before host snapshot reads" {
    const Fake = struct {
        const Copy = struct { dst: u64, src: u64, len: usize };
        var pending: [24]Copy = undefined;
        var count: usize = 0;

        fn copy(dst: u64, src: u64, len: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            std.debug.assert(count < pending.len);
            pending[count] = .{ .dst = dst, .src = src, .len = len };
            count += 1;
            return cuda.abi.success;
        }

        fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            for (pending[0..count]) |item| {
                const dst: [*]u8 = @ptrFromInt(item.dst);
                const src: [*]const u8 = @ptrFromInt(item.src);
                @memcpy(dst[0..item.len], src[0..item.len]);
            }
            count = 0;
            return cuda.abi.success;
        }

        fn pattern(ai: usize, side: usize, position: usize) u8 {
            return @intCast(ai * 7 + side * 3 + position);
        }
    };
    Fake.count = 0;
    var driver: cuda.Driver = undefined;
    driver.api.cuMemcpyDtoDAsync_v2 = Fake.copy;
    driver.api.cuStreamSynchronize = Fake.sync;
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    var pg: kvp.Paged = undefined;
    pg.d = &driver;
    pg.cfg = .{ .window = 2052, .rows = 1, .slots = 1 };
    pg.cap = 32;
    pg.wr = pg.cfg.window + pg.cfg.rows + 4;
    const layout = try kvp.SnapshotLayout.init(pg.cap, pg.cfg, 20);
    pg.nblocks = layout.nblocks;
    pg.dev.len = layout.device;
    pg.tabs.len = layout.tables;
    pg.rebases = @splat(7);
    pg.written = @splat(12);
    const a = std.testing.allocator;
    const device = try a.alloc(u8, pg.devBytes());
    defer a.free(device);
    const host = try a.alignedAlloc(u8, .@"16", 24 * pg.cap * kvp.row_bytes);
    defer a.free(host);
    @memset(host, 0xFF);
    pg.dev.ptr = @intFromPtr(device.ptr);
    pg.host = .{ .d = &driver, .bytes = @alignCast(host) };
    pg.host_dev = @intFromPtr(host.ptr);
    const fed: usize = 20;
    for (0..12) |ai| {
        pg.win0[ai] = ai % 3 * 4;
        for (0..2) |side| {
            const canonical = pg.hostMut(ai, side == 1);
            for (0..pg.win0[ai]) |position| @memset(canonical[position * kvp.row_bytes ..][0..kvp.row_bytes], Fake.pattern(ai, side, position));
            const window: [*]u8 = @ptrFromInt(pg.devRows(ai, side == 1));
            for (pg.win0[ai]..fed) |position| @memset(window[(position - pg.win0[ai]) * kvp.row_bytes ..][0..kvp.row_bytes], Fake.pattern(ai, side, position));
        }
    }
    const origins = pg.win0;
    try pg.flush(stream, fed);
    try std.testing.expectEqual(@as(usize, 24), Fake.count);
    for (0..12) |ai| for (0..2) |side| {
        try std.testing.expectEqual(@as(u8, 0xFF), pg.hostRows(ai, side == 1)[pg.win0[ai] * kvp.row_bytes]);
    };
    try stream.synchronize();
    try std.testing.expectEqual(@as(usize, 0), Fake.count);
    for (0..12) |ai| for (0..2) |side| {
        const canonical = pg.hostRows(ai, side == 1);
        for (0..fed) |position| for (canonical[position * kvp.row_bytes ..][0..kvp.row_bytes]) |byte| {
            try std.testing.expectEqual(Fake.pattern(ai, side, position), byte);
        };
        for (canonical[fed * kvp.row_bytes ..]) |byte| try std.testing.expectEqual(@as(u8, 0xFF), byte);
    };
    try std.testing.expectEqual(origins, pg.win0);
    try std.testing.expectEqual(@as([12]u64, @splat(7)), pg.rebases);
    try std.testing.expectEqual(@as([12]u64, @splat(12)), pg.written);
}
