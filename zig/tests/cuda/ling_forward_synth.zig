//! `ling-forward-synth DIR REF`: a synthetic checkpoint's logits vs ling_forward.py's; cuts, windows, batches exact.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const ling = @import("ling_cuda");
const rig = @import("ling_forward_rig.zig");

const F = rig.F;
const Rig = rig.Rig;
const P = 80; // prompt rows: KDA's prompt takes the wide chain (64 rows or more)

/// ling_forward.py's file: ids, float64 and bf16-point logits [T][V], the picks it routed on [layers][T][top].
pub const Ref = struct { T: usize, V: usize, layers: usize, top: usize, forced: bool, gap: f64, ids: []i32, l64: []f64, lbf: []f64, picks: []i32 };

fn take(comptime T: type, a: std.mem.Allocator, bytes: []const u8, at: *usize, n: usize) ![]T {
    const out = try a.alloc(T, n);
    @memcpy(std.mem.sliceAsBytes(out), bytes[at.*..][0 .. n * @sizeOf(T)]);
    at.* += n * @sizeOf(T);
    return out;
}

pub fn readRef(a: std.mem.Allocator, io: std.Io, path: []const u8) !Ref {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 30));
    if (bytes.len < 36 or !std.mem.eql(u8, bytes[0..8], "TFLINGF2")) return error.BadFixture;
    var at: usize = 8;
    const dims = try take(u32, a, bytes, &at, 5);
    const gap = (try take(f64, a, bytes, &at, 1))[0];
    const T: usize = dims[0];
    const V: usize = dims[1];
    const L: usize = dims[2];
    const K: usize = dims[3];
    const r: Ref = .{ .T = T, .V = V, .layers = L, .top = K, .forced = dims[4] != 0, .gap = gap, .ids = try take(i32, a, bytes, &at, T), .l64 = try take(f64, a, bytes, &at, T * V), .lbf = try take(f64, a, bytes, &at, T * V), .picks = try take(i32, a, bytes, &at, L * T * K) };
    if (at != bytes.len) return error.BadFixture;
    return r;
}

/// The serial run: the prompt in one chunk, then one decode row at a time; logits rows and sorted picks.
fn serial(r: *Rig, pb: *const F.Buffers, db: *const F.Buffers, st: *F.State, ids: []const i32, nm: usize) !struct { logits: []u16, picks: []i32 } {
    const c = &r.g.w.config;
    const V: usize = c.vocab;
    const T = ids.len;
    const top: usize = c.topk;
    const logits = try r.a.alloc(u16, T * V);
    const picks = try r.a.alloc(i32, nm * T * top);
    const pp = try r.zeros(nm * P * (top + 1) * 4);
    @memcpy(logits[0 .. P * V], try r.run(pb, &.{.{ .st = st, .row0 = 0, .rows = P }}, ids[0..P], pp));
    const sp = try rig.sortedPicks(r.a, try r.back(pp, nm * P * (top + 1) * 4), nm, P, top);
    for (0..nm) |l| @memcpy(picks[l * T * top ..][0 .. P * top], sp[l * P * top ..][0 .. P * top]);
    try r.fwd(pb, null).commit(st, P);
    const dp = try r.zeros(nm * (top + 1) * 4);
    for (P..T) |t| {
        @memcpy(logits[t * V ..][0..V], try r.run(db, &.{.{ .st = st, .row0 = 0, .rows = 1 }}, ids[t .. t + 1], dp));
        const one = try rig.sortedPicks(r.a, try r.back(dp, nm * (top + 1) * 4), nm, 1, top);
        for (0..nm) |l| @memcpy(picks[(l * T + t) * top ..][0..top], one[l * top ..][0..top]);
        try r.fwd(db, null).commit(st, 1);
    }
    return .{ .logits = logits, .picks = picks };
}

/// A committed sequence's caches (KDA states and conv windows, MLA rows below pos) equal another's, byte for byte.
fn sameState(r: *Rig, what: []const u8, a: *const F.State, b: *const F.State) !void {
    var name: [96]u8 = undefined;
    try check.expect(a.pos == b.pos, "{s}: positions {d} and {d}", .{ what, a.pos, b.pos });
    for (0..a.kda_layers) |li| {
        try check.sameBytes(try std.fmt.bufPrint(&name, "{s}: KDA layer {d} state", .{ what, li }), try r.back(a.recAt(a.cur[li], li), a.rec_layer), try r.back(b.recAt(b.cur[li], li), b.rec_layer));
        try check.sameBytes(try std.fmt.bufPrint(&name, "{s}: KDA layer {d} conv window", .{ what, li }), try r.back(a.convAt(li), 3 * a.convWidth() * 2), try r.back(b.convAt(li), 3 * b.convWidth() * 2));
    }
    for (0..a.mla_layers) |mi| try check.sameBytes(try std.fmt.bufPrint(&name, "{s}: MLA layer {d} cache rows", .{ what, mi }), try r.back(a.cacheAt(mi), a.pos * F.cache_width * 2), try r.back(b.cacheAt(mi), b.pos * F.cache_width * 2));
}

/// `ling-forward-synth DIR REF [PICKS]`: PICKS gets the serial run's picks, for ling_forward.py --picks.
pub fn run(gpu: check.Gpu, dir: []const u8, ref_path: []const u8, picks_out: ?[]const u8) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    const ref = try readRef(a, gpu.io, ref_path);
    var why: ling.config.Why = .{};
    var o = rig.W.open(gpu.gpa, gpu.io, dir, &why) catch |e| {
        std.debug.print("{s}: {s}\n", .{ dir, why.text() });
        return e;
    };
    defer o.ck.close();
    var m = try F.Modules.load(gpu.d, gpu.ctx);
    defer m.deinit();
    const backbone = try a.alloc(u32, o.c.layers);
    for (backbone, 0..) |*i, j| i.* = @intCast(j);
    var pack = try cuda.modelopt.Packer.load(gpu.d);
    defer pack.unload();
    var w = try rig.W.load(gpu.gpa, gpu.d, s, &pack, &o.ck, o.c, backbone, .{});
    defer w.deinit();
    var g = try F.Graph.init(gpu.gpa, gpu.d, &m, s, &w);
    defer g.deinit();
    const c = &w.config;
    const V: usize = c.vocab;
    const T = ref.T;
    try check.expect(ref.V == V and T > P + 8 - 1 and ref.top == c.topk, "reference: {d} tokens of vocabulary {d}, top {d}", .{ T, ref.V, ref.top });
    var r: Rig = .{ .gpu = gpu, .a = a, .s = s, .m = &m, .g = &g };
    defer r.deinit();
    var pb = try F.Buffers.init(gpu.d, &m, &w, 2 * P, true, 128);
    defer pb.deinit();
    var db = try F.Buffers.init(gpu.d, &m, &w, 2 * F.window_rows, false, 128);
    defer db.deinit();
    var nm: usize = 0;
    for (w.layers) |l| nm += @intFromBool(l.ffn == .moe);
    try check.expect(nm == ref.layers, "MoE layers {d}, the reference's {d}", .{ nm, ref.layers });
    const ids = ref.ids;
    // A: the serial run, against the references
    const sa = try r.state(128);
    const A = try serial(&r, &pb, &db, sa, ids, nm);
    const flipped = try a.alloc(bool, T);
    @memset(flipped, false);
    const nf = rig.flips(A.picks, ref.picks, nm, T, 0, T, c.topk, flipped);
    var rows_flipped: usize = 0;
    for (flipped) |x| rows_flipped += @intFromBool(x);
    const d64 = rig.distance(A.logits, ref.l64, T, V, flipped);
    const dbf = rig.distance(A.logits, ref.lbf, T, V, flipped);
    if (picks_out) |path| try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = path, .data = std.mem.sliceAsBytes(A.picks) });
    const how = if (ref.forced) "the reference routed on these picks" else "the reference's own float64 picks";
    std.debug.print("RESULT ling-forward-synth {t}: {d} layers, {d} prompt + {d} decode rows; picks vs {s}: {d} of {d} differ ({d} rows)\n", .{ c.quant, c.layers, P, T - P, how, nf, nm * T, rows_flipped });
    if (ref.forced) std.debug.print("RESULT   float64's own top-k vs these: the worst pick ranks {e:.3} below its k-th biased score\n", .{ref.gap});
    std.debug.print("RESULT   logits vs float64: rel L2 {e:.3} ({e:.3} without flipped rows), worst row {e:.3}, argmax {d}/{d}\n", .{ d64.rel, d64.rel_kept, d64.worst_row, d64.argmax, T });
    std.debug.print("RESULT   logits vs bf16 points: rel L2 {e:.3} ({e:.3} without flipped rows), worst row {e:.3}, argmax {d}/{d}\n", .{ dbf.rel, dbf.rel_kept, dbf.worst_row, dbf.argmax, T });
    var bd: f64 = 0;
    var bw: f64 = 0;
    for (ref.lbf, ref.l64) |x, e| {
        bd += (x - e) * (x - e);
        bw += e * e;
    }
    const base = @sqrt(bd / bw); // HF's bf16 rounding points alone, against float64
    std.debug.print("RESULT   the bf16-point reference itself vs float64: rel L2 {e:.3}\n", .{base});
    if (ref.forced) try check.expect(nf == 0 and d64.rel <= 1.5 * base and dbf.rel <= base, "on the same picks: logits within 1.5x the bf16 points' rel L2 of float64, and within it of the bf16 points", .{});
    // B: the prompt cut at 48
    const sb = try r.state(128);
    try rig.same("prompt cut at 48 == uncut (rows 0-47)", try r.run(&pb, &.{.{ .st = sb, .row0 = 0, .rows = 48 }}, ids[0..48], null), A.logits[0 .. 48 * V]);
    try r.fwd(&pb, null).commit(sb, 48);
    try rig.same("prompt cut at 48 == uncut (rows 48-79)", try r.run(&pb, &.{.{ .st = sb, .row0 = 0, .rows = P - 48 }}, ids[48..P], null), A.logits[48 * V .. P * V]);
    try r.fwd(&pb, null).commit(sb, P - 48);
    // C: one 8-row verify window == 8 decode rows
    const sc = try r.state(128);
    _ = try r.run(&pb, &.{.{ .st = sc, .row0 = 0, .rows = P }}, ids[0..P], null);
    try r.fwd(&pb, null).commit(sc, P);
    try rig.same("8-row verify window == 8 decode rows", try r.run(&db, &.{.{ .st = sc, .row0 = 0, .rows = 8 }}, ids[P .. P + 8], null), A.logits[P * V .. (P + 8) * V]);
    try r.fwd(&db, null).commit(sc, 3);
    try rig.same("after keeping 3 of 8: the next 5-row window == decode rows", try r.run(&db, &.{.{ .st = sc, .row0 = 0, .rows = 5 }}, ids[P + 3 .. P + 8], null), A.logits[(P + 3) * V .. (P + 8) * V]);
    try r.fwd(&db, null).commit(sc, 5);
    try sameState(&r, "8-row window, keep 3, 5-row window == 8 decode rows", sc, sa);
    // the fused KDA chain on the same window: bit-equal to the three-kernel one
    const sf = try r.state(128);
    _ = try r.run(&pb, &.{.{ .st = sf, .row0 = 0, .rows = P }}, ids[0..P], null);
    try r.fwd(&pb, null).commit(sf, P);
    r.kda_wide = false;
    try rig.same("8-row window on the fused KDA chain == on the three-kernel chain", try r.run(&db, &.{.{ .st = sf, .row0 = 0, .rows = 8 }}, ids[P .. P + 8], null), A.logits[P * V .. (P + 8) * V]);
    r.kda_wide = true;
    sf.deinit();
    // D and E: two sequences batched, in a prompt chunk and in a decode window
    const sd = try r.state(128);
    const se = try r.state(128);
    const both = try a.alloc(i32, P + 48);
    @memcpy(both[0..P], ids[0..P]);
    @memcpy(both[P..], ids[0..48]);
    const pl = try r.run(&pb, &.{ .{ .st = sd, .row0 = 0, .rows = P }, .{ .st = se, .row0 = P, .rows = 48 } }, both, null);
    try rig.same("batched prompt: sequence D == alone", pl[0 .. P * V], A.logits[0 .. P * V]);
    try rig.same("batched prompt: sequence E (48 rows) == alone", pl[P * V ..], A.logits[0 .. 48 * V]);
    try r.fwd(&pb, null).commit(sd, P);
    try r.fwd(&pb, null).commit(se, 48);
    _ = try r.run(&pb, &.{.{ .st = se, .row0 = 0, .rows = P - 48 }}, ids[48..P], null);
    try r.fwd(&pb, null).commit(se, P - 48);
    const win = try a.alloc(i32, 10);
    @memcpy(win[0..4], ids[P .. P + 4]);
    @memcpy(win[4..], ids[P .. P + 6]);
    const wl = try r.run(&db, &.{ .{ .st = sd, .row0 = 0, .rows = 4 }, .{ .st = se, .row0 = 4, .rows = 6 } }, win, null);
    try rig.same("batched window: sequence D (4 rows) == alone", wl[0 .. 4 * V], A.logits[P * V .. (P + 4) * V]);
    try rig.same("batched window: sequence E (6 rows) == alone", wl[4 * V ..], A.logits[P * V .. (P + 6) * V]);
    for ([_]*F.State{ sa, sb, sc, sd, se }) |x| x.deinit();
    check.pass("ling-forward-synth {t}: logits rel L2 {e:.3} of float64 ({s}); cut == uncut, windows == decode rows, a partial commit replays, batched == alone", .{ c.quant, d64.rel, if (ref.forced) "same picks" else "free routing" });
}
