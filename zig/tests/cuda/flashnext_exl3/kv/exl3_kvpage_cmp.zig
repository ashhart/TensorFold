//! exl3-kvpage cmp: the paged cache against all K/V on the device, answers and timings.

const std = @import("std");
const check = @import("../../check.zig");
const flash = @import("flashnext");
const Gpu = check.Gpu;
const xpre = flash.exl3.prefill;
const kvp = flash.exl3.kvpage;
const core = @import("core");
const engs = @import("exl3_kvpage_eng.zig");

const Eng = engs.Eng;
const parseList = engs.parseList;
const readIds = engs.readIds;

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

pub fn cmpRun(gpu: Gpu, dir: []const u8, rest: []const [:0]const u8) !void {
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
