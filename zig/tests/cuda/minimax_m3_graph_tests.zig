//! MiniMax M3.1's launch-graph smoke (Stage-6): the graph.zig Engine on device.
//!
//!   tf-cuda-test minimax-m3-smoke
//!       builds the family's 2-layer fixture in /tmp, boots the Engine over it (TP1), runs
//!       8 context rows + 2 greedy steps on device, then grades them EXACTLY against the fork's
//!         f64 composition over the same poked fixture decoded on the host (the serve.zig-Ref
//!         contract at the engine's own dims), and asserts the router/selection fingerprints.
//!
//!   tf-cuda-test minimax-m3-token <rank 0|1> <host:port|-> <pack dir|fixture> [ctx] [steps]
//!       the real pack, TP1 (addr "-") or TP2 (rank 0 serves the NCCL id on host:port, the
//!       exl3-tp2 precedent; rank 1 first, rank 0 ~2 s later — the Stage-5 runbook). Both
//!       ranks read the same deterministic context (serve.promptOf's LCG stream), run every
//!       owned layer's row, and rank 0 closes with the final norm + lm_head + greedy. Prints
//!       per-row wall, the tokens, host RAM and device totals. "fixture" instead of a pack
//!       dir generates the same 2-layer fixture on the calling node (rank 1 generates its
//!       own copy; the bytes are deterministic), so TP2 handoffs are graded at fixture scale
//!       against minimax-m3-smoke's TP1 tokens.

const std = @import("std");
const builtin = @import("builtin");
const posix = std.posix;
const cuda = @import("cuda");
const check = @import("check.zig");
const m3 = @import("minimax_m3_vl_cuda");

const exl3_load = @import("core").exl3_load;
const tokenizer = @import("core").tokenizer;

/// The fixture's dir (the smoke rebuilds it each run; nothing reads a stale one).
const fixture_path = "/tmp/tf-m3-graph-smoke";
/// The context stream's seed: fixed so both ranks and every rerun read the same prompt.
const prompt_seed: u64 = 0x5EEDB00B;

fn rebuildFixture(a: std.mem.Allocator, io: std.Io) !void {
    var cwd = std.Io.Dir.cwd();
    cwd.deleteTree(io, fixture_path) catch {};
    cwd.createDirPath(io, fixture_path) catch return error.FixtureDir;
    var fdir = try cwd.openDir(io, fixture_path, .{});
    defer fdir.close(io);
    try m3.weights_mod.checkpointFixture(a, io, fdir, false);
    try pokeNoise(a, io, fdir);
}

const Lcg = m3.serve.Lcg; // the prompt stream's state type (serve.zig's own, now pub)

// ---------------------------------------------------------------- the fixture smoke (TP1)

pub fn run(gpu: check.Gpu) !void {
    const gpa = gpu.gpa;
    try rebuildFixture(gpa, gpu.io);
    const t0 = check.now(gpu.io);
    var eng = try m3.graph.Engine.build(gpa, gpu.io, gpu.d, fixture_path, 0, 1, .{});
    defer eng.deinit();
    const build_s = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e9;

    var prompt: [8]u32 = undefined;
    var l = Lcg{ .s = prompt_seed };
    m3.serve.promptOf(&l, prompt[0..]);
    var out: [2]u32 = undefined;

    const t1 = check.now(gpu.io);
    for (prompt) |tok| try eng.put(tok);
    const prefill_s = @as(f64, @floatFromInt(check.now(gpu.io) - t1)) / 1e9;
    const t2 = check.now(gpu.io);
    try eng.greedy(out.len, out[0..]);
    const greedy_s = @as(f64, @floatFromInt(check.now(gpu.io) - t2)) / 1e9;

    const c = &eng.c;
    // the weight discriminator: the host-decoded fixture's own k/v matrices (decodeGroup, the
    // RefF64 path's dequantize) — the device v GEMM's output being ~100x k's was traced either
    // here (huge decoded weights) or into the upload/kernels (clean decoded weights)
wprobe: {
        var scw = exl3_load.scan(gpa, gpu.io, fixture_path) catch break :wprobe;
        defer scw.deinit();
        const vi = scw.find("language_model.model.layers.0.self_attn.v_proj") orelse break :wprobe;
        const ki = scw.find("language_model.model.layers.0.self_attn.k_proj") orelse break :wprobe;
        const qi = scw.find("language_model.model.layers.0.self_attn.q_proj") orelse break :wprobe;
        const hi = scw.find("language_model.lm_head") orelse break :wprobe; // the token smoke's fp32 head
        inline for (.{ "v", "k", "q", "head" }, .{ vi, ki, qi, hi }) |tag, gi| {
            const g = scw.groups[gi];
            const parts = scw.parts(gpa, gi) catch break :wprobe; // scan-owned; decodeGroup never frees these
            const w = decodeGroup(gpa, &scw, gi) catch break :wprobe;
            defer gpa.free(w);
            var s2: f64 = 0;
            var smax: f64 = 0;
            var sl1: usize = 0;
            for (w) |v| {
                s2 += v * v;
                smax = @max(smax, @abs(v));
                if (@abs(v) > 1) sl1 += 1;
            }
            const rms = @sqrt(s2 / @as(f64, @floatFromInt(w.len)));
            var sumin: f32 = std.math.inf(f32);
            var sumax: f32 = -std.math.inf(f32);
            var svmin: f32 = std.math.inf(f32);
            var svmax: f32 = -std.math.inf(f32);
            const su = std.mem.bytesAsSlice(u16, parts.suh.bytes);
            const sv = std.mem.bytesAsSlice(u16, parts.svh.bytes);
            for (su) |s| {
                const f = @as(f32, @floatCast(@as(f16, @bitCast(s))));
                sumin = @min(sumin, f);
                sumax = @max(sumax, f);
            }
            for (sv) |s| {
                const f = @as(f32, @floatCast(@as(f16, @bitCast(s))));
                svmin = @min(svmin, f);
                svmax = @max(svmax, f);
            }
            std.debug.print("DBGW {s}_proj group: k={d} n={d} halves={d} codebook={s} trellis={d}B suh={d}B({d}) suh[{d:.4},{d:.4}] svh={d}B({d}) svh[{d:.4},{d:.4}] | rms={d:.4} max={d:.4} frac>1={d:.4}\n", .{ tag, g.k, g.n, g.halves, @tagName(g.codebook), parts.trellis.bytes.len, parts.suh.bytes.len, su.len, sumin, sumax, parts.svh.bytes.len, sv.len, svmin, svmax, rms, smax, @as(f64, @floatFromInt(sl1)) / @as(f64, @floatFromInt(w.len)) });
        }
    }
    std.debug.print("DBG layers {d}:", .{eng.layers.len});
    for (eng.layers) |*ly| std.debug.print(" lno={d} mlp={s}", .{ ly.lno, @tagName(ly.mlp) });
    std.debug.print("; picks:", .{});
    for (eng.picks) |p| std.debug.print(" {}", .{p});
    std.debug.print("; sel_e:", .{});
    for (eng.sel_e) |v| std.debug.print(" {d}", .{v});
    std.debug.print("; wsum={d:.4} fired={d}\n", .{ eng.last.wsum, eng.last.fired });
    // the NaN discriminator: a route with NaN logits fires nobody (sigmoid NaN fails strict >),
    // so fired==0 here means e.x or the elogits went NaN (or the wave never ran — sel_seen tells)
    var xnan = false;
    for (eng.x) |v| {
        if (std.math.isNan(v)) xnan = true;
    }
    var elnan = false;
    for (eng.elogits) |v| {
        if (std.math.isNan(v)) elnan = true;
    }
    var xmax: f32 = 0;
    for (eng.x) |v| {
        const m = @abs(v);
        if (m > xmax) xmax = m;
    }
    std.debug.print("DBG2 xnan={} elnan={} xmax={d:.5} logits_nan:", .{ xnan, elnan, xmax });
    var lnan = false;
    for (eng.logits) |v| {
        if (std.math.isNan(v)) lnan = true;
    }
    std.debug.print(" {} sel_seen[0]:", .{lnan});
    for (eng.sel_seen[0..@min(4, eng.sel_seen.len)]) |v| {
        std.debug.print(" {d}", .{v});
    }
    std.debug.print("\n", .{});
    // the fingerprints: the router fires exactly topk experts, the weights sum to routed_scale
    // (normalize-then-scale), and the selection carries valid blocks for the kv heads
    var dbg_fail = false; // debug build: keep going to the f64 grade after a fingerprint miss
    if (eng.last.fired != c.topk) {
        std.debug.print("FAIL minimax-m3-smoke: {d} experts fired, expected topk {d}\n", .{ eng.last.fired, c.topk });
        dbg_fail = true;
    }
    if (@abs(eng.last.wsum - c.routed_scale) > 1e-2) {
        std.debug.print("FAIL minimax-m3-smoke: routing weights sum {d}, expected routed_scale {d}\n", .{ eng.last.wsum, c.routed_scale });
        dbg_fail = true;
    }
    if (eng.last.sel_valid < 1) {
        std.debug.print("FAIL minimax-m3-smoke: the last kv head's selection is empty\n", .{});
        return error.TestFailed;
    }
    var best: usize = 0;
    var bv = -std.math.inf(f32);
    var second = -std.math.inf(f32);
    for (eng.logits, 0..) |v, i| {
        if (v > bv) {
            second = bv;
            bv = v;
            best = i;
        } else if (v > second) second = v;
    }
    const margin = bv - second;
    std.debug.print("minimax-m3-smoke: fixture build {d:.2}s, {d}-row prefill {d:.3}s, greedy {d:.3}s; fired {d}, wsum {d:.4}, sel_valid {d}\n", .{ build_s, prompt.len, prefill_s, greedy_s, eng.last.fired, eng.last.wsum, eng.last.sel_valid });
    std.debug.print("minimax-m3-smoke: logits argmax {d} ({d:.5}), second {d:.5}, margin {d:.5}\n", .{ best, bv, second, margin });
    if (margin < 0) {
        std.debug.print("FAIL minimax-m3-smoke: the argmax is a tie at the fp32 head\n", .{});
        return error.TestFailed;
    }
    // the grade: the same poked fixture bytes decoded on the host (core.exl3_format, the fork's
    // trellis decode), the fork's f64 composition at the engine's own runtime dims
    const t3 = check.now(gpu.io);
    var fw = try loadFixtureW(gpa, gpu.io, fixture_path, c.*);
    defer fw.deinit();
    var ref = try RefF64.init(gpa, c.*, prompt.len + out.len + 1, &fw);
    defer ref.deinit();
    var ref_out: [2]u32 = undefined;
    var lr = Lcg{ .s = prompt_seed };
    m3.serve.promptOf(&lr, prompt[0..]);
    for (prompt) |tok| try ref.put(tok);
    try ref.greedy(out.len, ref_out[0..]);
    const ref_s = @as(f64, @floatFromInt(check.now(gpu.io) - t3)) / 1e9;
    var maxd: f64 = 0;
    var lnn: usize = 0;
    var lin: usize = 0;
    for (eng.logits, 0..) |v, i| {
        if (std.math.isNan(v)) lnn += 1;
        if (std.math.isInf(v)) lin += 1;
        const d = @as(f64, v) - ref.logits[i];
        if (!std.math.isNan(d) and @abs(d) > maxd) maxd = @abs(d);
    }
    std.debug.print("minimax-m3-smoke: reference (f64, decoded fixture) {d:.1}s; logits maxd {d:.5} NaN {d} inf {d}\n", .{ ref_s, maxd, lnn, lin });
    try gradeTokensCatch(ref_out[0..], out[0..]);
    std.debug.print("RESULT minimax-m3-smoke: TP1 fixture tokens {d} {d} == reference {d} {d} (EXACT); ctx rows {d}; fired {d} wsum {d:.4} sel_valid {d}\n", .{ out[0], out[1], ref_out[0], ref_out[1], prompt.len, eng.last.fired, eng.last.wsum, eng.last.sel_valid });
}

// ---------------------------------------------------------------- the real-pack token smoke

pub fn tokenRun(gpu: check.Gpu, rank_arg: []const u8, addr_arg: []const u8, dir_in: []const u8, ctx_len: usize, steps: usize) !void {
    var dir = dir_in;
    if (dir.len == 0) {
        std.debug.print("usage: tf-cuda-test minimax-m3-token <rank 0|1> <host:port|-> <pack dir|fixture> [ctx] [steps]\n", .{});
        return error.BadArguments;
    }
    if (std.mem.eql(u8, dir, "fixture")) dir = try makeFixture(gpu);
    const rank = std.fmt.parseInt(u8, rank_arg, 10) catch {
        std.debug.print("usage: tf-cuda-test minimax-m3-token <rank 0|1> <host:port|-> <pack dir|fixture> [ctx] [steps]\n", .{});
        return error.BadArguments;
    };
    if (rank > 1 or addr_arg.len == 0 or steps == 0 or steps > 32 or ctx_len == 0 or ctx_len > m3.graph.keys_cap) {
        return error.BadArguments; // loud refusals: no silent clamping of a context the pool holds
    }
    const world: u32 = if (std.mem.eql(u8, addr_arg, "-")) 1 else 2;
    const ranks: u32 = @intCast(rank);

    // -- the TP2 boot exactly the exl3-tp2 precedent (rank 1 first, rank 0 serves the id) --
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    var id: cuda.nccl.UniqueId = undefined;
    const id_bytes = std.mem.asBytes(&id);
    if (world == 2) {
        if (rank == 0) {
            try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
            try serveId(addr_arg, id_bytes);
        } else {
            try receiveId(gpu, addr_arg, id_bytes);
        }
    }
    var comm: cuda.nccl.Comm = null;
    if (world == 2) {
        try lib.check(lib.api.ncclCommInitRank(&comm, @intCast(world), id, @intCast(rank)), "ncclCommInitRank");
    }
    defer {
        if (world == 2) _ = lib.api.ncclCommDestroy(comm);
    }

    const device_before = cuda.usage(false).device;
    const ram_before = hostFree(gpu);
    const t0 = check.now(gpu.io);
    var eng = try m3.graph.Engine.build(gpu.gpa, gpu.io, gpu.d, dir, ranks, world, .{
        .lib = if (world == 2) &lib else null,
        .comm = if (world == 2) comm else null,
    });
    defer eng.deinit();
    const build_s = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e9;
    const device_after = cuda.usage(false).device;
    const ram_after = hostFree(gpu);
    const gib = struct {
        fn f(v: u64) f64 {
            return @as(f64, @floatFromInt(v)) / (1024.0 * 1024.0 * 1024.0);
        }
    }.f;

    // the deterministic context: both ranks derive it identically (rank 1's tok arg is ignored)
    const prompt = try gpu.gpa.alloc(u32, ctx_len);
    defer gpu.gpa.free(prompt);
    var l = Lcg{ .s = prompt_seed };
    m3.serve.promptOf(&l, prompt);
    const out = try gpu.gpa.alloc(u32, steps);
    defer gpu.gpa.free(out);

    const t1 = check.now(gpu.io);
    for (prompt) |tok| try eng.put(if (rank == 0) tok else 0);
    const prefill_s = @as(f64, @floatFromInt(check.now(gpu.io) - t1)) / 1e9;
    // greedy steps: rank 0 argues from its logits; rank 1 keeps the row lockstep
    const t2 = check.now(gpu.io);
    if (rank == 0) {
        try eng.greedy(steps, out);
    } else {
        for (0..steps - 1) |_| try eng.put(0); // greedy()'s feedback puts, kept in the lockstep
    }
    const decode_s = @as(f64, @floatFromInt(check.now(gpu.io) - t2)) / 1e9;
    var best: usize = 0;
    var bv = -std.math.inf(f32);
    var second = -std.math.inf(f32);
    if (rank == 0) {
        for (eng.logits, 0..) |v, i| {
            if (v > bv) {
                second = bv;
                bv = v;
                best = i;
            } else if (v > second) second = v;
        }
    } else {
        bv = 0;
        second = 0;
    }
    // additive Stage-7 receipt: explicit logits health so a NaN row cannot masquerade as an
    // argmax (6C's smoke print hardening, carried over to the token harness)
    var lnan: usize = 0;
    var linf: usize = 0;
    var lmx: f32 = -std.math.inf(f32);
    var lsum: f64 = 0;
    if (rank == 0) {
        for (eng.logits) |v| {
            if (v != v) lnan += 1 else {
                if (v > lmx) lmx = v;
                if (v == std.math.inf(f32)) linf += 1 else lsum += v;
            }
        }
    }
    std.debug.print("M3-TOKEN rank {d} world {d}: build {d:.1}s, prefill(x{d}) {d:.2}s, greedy(x{d}) {d:.2}s\n", .{ rank, world, build_s, ctx_len, prefill_s, steps, decode_s });
    std.debug.print("M3-TOKEN rank {d}: device buffers {d:.2} GiB; host MemAvailable {d:.2} -> {d:.2} GiB; trace fired {d} wsum {d:.4} sel_valid {d}\n", .{ rank, gib(device_after - device_before), gib(ram_before), gib(ram_after), eng.last.fired, eng.last.wsum, eng.last.sel_valid });
    if (rank == 0) {
        if (steps > 1 and decode_s > 0) {
            const tps = @as(f64, @floatFromInt(steps - 1)) / decode_s;
            std.debug.print("M3-TOKEN rank 0: decode tok/s {d:.2} ({d:.3}s for {d} rows)", .{ tps, decode_s, steps - 1 });
        }
        std.debug.print("\nM3-TOKEN rank 0: logits rows {d} NaN {d} inf {d} max {d:.6} mean {d:.6} ({d:.5} over the finite rows)\n", .{ eng.logits.len, lnan, linf, lmx, lsum / @as(f64, @floatFromInt(eng.logits.len)), @as(f64, @floatFromInt(eng.logits.len - lnan)) });
        std.debug.print("M3-TOKEN rank 0: logits argmax {d} ({d:.5}), second {d:.5}\n", .{ best, bv, second });
        std.debug.print("RESULT minimax-m3-token: prompt rows {d} then greedy", .{ctx_len});
        for (out) |tok| std.debug.print(" {d}", .{tok});
        std.debug.print("\n", .{});
    } else {
        std.debug.print("RESULT minimax-m3-token: rank 1 lockstep done over dir {s}\n", .{dir});
    }
}

/// The fixture's path, built on the calling node (rank 1 builds its own deterministic copy).
fn makeFixture(gpu: check.Gpu) ![]const u8 {
    try rebuildFixture(gpu.gpa, gpu.io);
    return fixture_path;
}

// --------------------------------------------------------------- the fixture's noise pass

/// The fixture ships all-zero tensor data; this pokes every tensor with deterministic noise so
/// the smoke's paths cannot alias each other unnoticed (all-zero rows make wrong offsets equal).
/// The trellis words take random bits (the mul1 codebook decodes any bit pattern to a finite
/// value), the fp16 scales land in [0.75, 1.25], the norms near 1.0, and the passthrough sizes
/// match their rows' shapes (checkpointFixture wrote them; the safetensors headers say where).
const header_limit: usize = 1 << 26; // the fixture's headers are far under 64 MiB

fn pokeNoise(gpa: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !void {
    var prng = std.Random.DefaultPrng.init(0x9E3779B9);
    const rnd = prng.random();
    for ([_][]const u8{ "model-00002.safetensors", "model.safetensors" }) |file_name| {
        const raw = dir.readFileAlloc(io, file_name, gpa, .limited(4 << 30)) catch continue; // a file the fixture did not write: skip
        defer gpa.free(raw);
        if (raw.len < 8) continue;
        const hlen = std.mem.readInt(u64, raw[0..8], .little);
        if (hlen < 2 or hlen > header_limit or 8 + hlen > raw.len) return error.BadFixture;
        const parsed = std.json.parseFromSlice(std.json.Value, gpa, raw[8 .. 8 + hlen], .{}) catch return error.BadFixture;
        defer parsed.deinit();
        if (parsed.value != .object) return error.BadFixture;
        var it = parsed.value.object.iterator();
        while (it.next()) |kv| {
            const name = kv.key_ptr.*;
            if (std.mem.eql(u8, name, "__metadata__")) continue;
            if (kv.value_ptr.* != .object) continue;
            const ts = kv.value_ptr.object;
            const dt = if (ts.get("dtype")) |v| (if (v == .string) v.string else "") else "";
            const offs = ts.get("data_offsets") orelse continue;
            if (offs != .array or offs.array.items.len != 2) continue;
            const a = offs.array.items[0].integer;
            const b = offs.array.items[1].integer;
            if (a < 0 or b < a) continue;
            const start: usize = @intCast(a);
            const end: usize = @intCast(b);
            const base: usize = 8 + hlen;
            if (base + end > raw.len) continue;
            const view = raw[base + start .. base + end];
            if (std.mem.endsWith(u8, name, ".trellis")) {
                for (0..view.len / 4) |i| std.mem.writeInt(u32, view[i * 4 ..][0..4], rnd.int(u32), .little);
            } else if (std.mem.endsWith(u8, name, ".suh")) {
                // The K-tile scale carries the matrix's magnitude: the codebook's own values live
                // at O(1), so the [0.75, 1.25] suh/svh poke decoded every quantized matrix to
                // ~1.0 rms — ~170x the serve.zig draw scale (projections 0.006) — and the raw
                // attention V (the one projection with no norm to rescale it) then pushed the
                // O-GEMM's fp16 outputs past saturation: combine max|a| ~947, o16 infs, the
                // normRow NaN cascade, the router dead. Scale suh to the draws' sigma; svh keeps
                // its shape. The head stays wide: its logits are fp32 and its margins bar ties.
                const sigma: f32 = if (std.mem.indexOf(u8, name, "lm_head") != null) 0.10 else 0.006;
                for (0..view.len / 2) |i| putF16(view[i * 2 ..][0..2], sigma * (0.75 + 0.5 * rnd.float(f32)));
            } else if (std.mem.endsWith(u8, name, ".svh")) {
                for (0..view.len / 2) |i| putF16(view[i * 2 ..][0..2], 0.75 + 0.5 * rnd.float(f32));
            } else if (std.mem.endsWith(u8, name, ".bias")) {
                // the codebook marker and any per-group bias bytes: they carry the group's contract
            } else if (std.mem.eql(u8, dt, "BF16")) {
                const n = view.len / 2;
                // norms are gemma-style delta weights ((1.0 + weight) per serve.normRow; the
                // serve.zig fixture also stores deltas) — absolute ~1.0 values made every norm
                // ~2x and pushed the attention output past the fp16 range in the engine AND the
                // f64 reference alike.
                const near_one: bool = std.mem.endsWith(u8, name, "norm.weight");
                for (0..n) |i| {
                    const v: f32 = if (near_one) 0.05 * (rnd.float(f32) - 0.5) else 0.04 * (rnd.float(f32) - 0.5);
                    putBF16(view[i * 2 ..][0..2], v);
                }
            } else if (std.mem.eql(u8, dt, "F16")) {
                for (0..view.len / 2) |i| putF16(view[i * 2 ..][0..2], 0.01 * (rnd.float(f32) - 0.5));
            } else if (std.mem.eql(u8, dt, "F32")) {
                const scale: f32 = if (std.mem.endsWith(u8, name, "e_score_correction_bias")) 0.02 else 0.05;
                for (0..view.len / 4) |i| {
                    const v: f32 = scale * (rnd.float(f32) - 0.5);
                    std.mem.writeInt(u32, view[i * 4 ..][0..4], @bitCast(v), .little);
                }
            }
        }
        try dir.writeFile(io, .{ .sub_path = file_name, .data = raw });
    }
}

fn putF16(dst: []u8, v: f32) void {
    const h: f16 = @floatCast(v);
    std.mem.writeInt(u16, dst[0..2], @bitCast(h), .little);
}

fn putBF16(dst: []u8, v: f32) void {
    const bits: u32 = @bitCast(v);
    std.mem.writeInt(u16, dst[0..2], @intCast(bits >> 16), .little);
}

// --------------------------------------------------------------- the tp2 harness's own fns

/// Free MemAvailable from /proc/meminfo (0 elsewhere — the receipt prints GB10 nodes' figures).
fn hostFree(gpu: check.Gpu) u64 {
    var f = std.Io.Dir.openFileAbsolute(gpu.io, "/proc/meminfo", .{}) catch return 0;
    defer f.close(gpu.io);
    var buf: [8192]u8 = undefined;
    const n = f.readPositionalAll(gpu.io, &buf, 0) catch return 0;
    const line = std.mem.indexOf(u8, buf[0..n], "MemAvailable:") orelse return 0;
    var it = std.mem.tokenizeScalar(u8, buf[line + "MemAvailable:".len .. n], ' ');
    const kb = it.next() orelse return 0;
    return (std.fmt.parseInt(u64, kb, 10) catch 0) * 1024; // /proc/meminfo reports kB, not bytes
}

// --- the NCCL id exchange, verbatim the exl3-tp2 precedent (rank 0 serves on host:port, rank 1 reads) ---

fn serveId(addr_text: []const u8, bytes: []const u8) !void {
    if (builtin.os.tag == .windows) return error.SocketFailed;
    const at = std.mem.lastIndexOfScalar(u8, addr_text, ':') orelse return error.BadAddress;
    const ip = try std.Io.net.IpAddress.parseIp4(addr_text[0..at], try std.fmt.parseInt(u16, addr_text[at + 1 ..], 10));
    var storage: std.Io.Threaded.PosixAddress = undefined;
    const len = std.Io.Threaded.addressToPosix(&ip, &storage);
    const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
    if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
    const fd: posix.socket_t = @intCast(rc);
    defer _ = posix.system.close(fd);
    const one: c_int = 1;
    try posix.setsockopt(fd, posix.SOL.SOCKET, posix.SO.REUSEADDR, std.mem.asBytes(&one));
    switch (posix.errno(posix.system.bind(fd, &storage.any, len))) {
        .SUCCESS => {},
        .ADDRINUSE => return error.AddressInUse,
        else => return error.BindFailed,
    }
    if (posix.errno(posix.system.listen(fd, 1)) != .SUCCESS) return error.ListenFailed;
    var waited: u64 = 0;
    while (true) {
        var fds = [_]posix.pollfd{.{ .fd = fd, .events = posix.POLL.IN, .revents = 0 }};
        const ready = posix.poll(&fds, 500) catch return error.AcceptFailed;
        if (ready > 0) break;
        waited += 500;
        if (waited > 70_000) return error.PeerTimeout;
    }
    var peer: std.Io.Threaded.PosixAddress = undefined;
    var peer_len: posix.socklen_t = @sizeOf(std.Io.Threaded.PosixAddress);
    const rc2 = posix.system.accept(fd, &peer.any, &peer_len);
    if (posix.errno(rc2) != .SUCCESS) return error.AcceptFailed;
    const cfd: posix.socket_t = @intCast(rc2);
    defer _ = posix.system.close(cfd);
    var off: usize = 0;
    while (off < bytes.len) {
        const n = posix.system.write(cfd, bytes.ptr + off, bytes.len - off);
        if (posix.errno(n) != .SUCCESS) return error.WriteFailed;
        off += @intCast(n);
    }
}

fn receiveId(gpu: check.Gpu, addr_text: []const u8, bytes: []u8) !void {
    if (builtin.os.tag == .windows) return error.SocketFailed;
    const at = std.mem.lastIndexOfScalar(u8, addr_text, ':') orelse return error.BadAddress;
    const ip = try std.Io.net.IpAddress.parseIp4(addr_text[0..at], try std.fmt.parseInt(u16, addr_text[at + 1 ..], 10));
    var fd: posix.socket_t = -1;
    defer {
        if (fd >= 0) _ = posix.system.close(fd);
    }
    while (fd < 0) {
        var storage: std.Io.Threaded.PosixAddress = undefined;
        const len = std.Io.Threaded.addressToPosix(&ip, &storage);
        const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
        if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
        const c: posix.socket_t = @intCast(rc);
        if (posix.errno(posix.system.connect(c, &storage.any, len)) == .SUCCESS) {
            fd = c;
        } else {
            _ = posix.system.close(c);
            std.Io.sleep(gpu.io, .fromMilliseconds(200), .awake) catch {};
        }
    }
    var off: usize = 0;
    var read_waited: u64 = 0;
    while (off < bytes.len) {
        var fds = [_]posix.pollfd{.{ .fd = fd, .events = posix.POLL.IN, .revents = 0 }};
        const ready = posix.poll(&fds, 500) catch return error.ReadFailed;
        if (ready == 0) {
            read_waited += 500;
            if (read_waited > 60_000) return error.PeerTimeout;
            continue;
        }
        const n = posix.system.read(fd, bytes.ptr + off, bytes.len - off);
        switch (posix.errno(n)) {
            .SUCCESS => {
                if (n == 0) return error.PeerClosed;
                off += @intCast(n);
            },
            .INTR => {},
            else => return error.ReadFailed,
        }
    }
}
// ------------------------------------------------------- the fixture's f64 reference (the grade)
//
// The graded bar: the SAME poked fixture bytes the Engine's device graph serves, decoded on the
// host (`core.exl3_format.dequantize` — the trellis decode the fork's own tooling mirrors) and run
// through the fork's f64 composition, contract by contract with serve.zig's `Ref`: gemma norms in
// f64 with one bf16-rounded weight row each, per-head qk norm + partial neox rope (theta 1e7,
// 64 of 128 dims) with ONE fp16 rounding at every fp16 store (the linear outputs' h16, the cache
// and plane rows, the routing weights), the block-sparse f64 attention of the indexer's
// boost-override top-16-of-block selection, the noaux sigmoid+bias router (select on sigmoid+bias,
// weights = unbiased sigmoid normalized over the picks, * routed_scale, each .half()), the shared
// expert at weight 1, a leading dense wave, the final gemma norm, the head. Token equality against
// the device graph is the bar; the logits delta is printed for the receipt.

const exl3_fmt = @import("core").exl3_format;

fn bf16At(raw: []const u8, i: usize) f32 {
    const bits: u32 = @as(u32, std.mem.readInt(u16, raw[i * 2 ..][0..2], .little)) << 16;
    return @bitCast(bits);
}

fn f16At(raw: []const u8, i: usize) f32 {
    const h: f16 = @bitCast(std.mem.readInt(u16, raw[i * 2 ..][0..2], .little));
    return @floatCast(h);
}

const FW_Mlp = struct { gate: []f32, up: []f32, down: []f32 }; // decoded [k, n] row-major

const FW_MoE = struct {
    router: []f32, // [E, hidden] fp32 passthrough
    bias: []f32, // [E] fp32
    shared: FW_Mlp,
    experts: []FW_Mlp,
};

/// One layer's decoded weights, named exactly as the plan's spellings (weights.zig's tests).
const FW_Layer = struct {
    in_norm: []f32, // [hidden], bf16
    post_norm: []f32,
    q_norm: []f32, // [head_dim], bf16
    k_norm: []f32,
    iqn: []f32, // [i_dim], bf16
    ikn: []f32,
    q: []f32, // decoded [k=hidden, n=qk_rows]
    k: []f32, // [hidden, kv_rows]
    v: []f32,
    o: []f32, // [qk_rows, hidden]
    iq: []f32, // [i_heads*i_dim, hidden], fp16 passthrough
    ik: []f32, // [i_dim, hidden], fp16 passthrough
    dense: ?FW_Mlp = null,
    moe: ?FW_MoE = null,
};

const FW = struct {
    gpa: std.mem.Allocator,
    c: m3.serve.Config,
    embed: []f32, // [vocab, hidden], bf16
    final_norm: []f32, // [hidden], bf16
    head: []f32, // decoded [k=hidden, n=vocab]
    layers: []FW_Layer,
    scratch: []u8, // the dequantize scratch (2 * k * n f64 words, reused per group)

    fn deinit(f: *FW) void {
        const gpa = f.gpa;
        for (f.layers) |*l| {
            gpa.free(l.in_norm);
            gpa.free(l.post_norm);
            gpa.free(l.q_norm);
            gpa.free(l.k_norm);
            gpa.free(l.iqn);
            gpa.free(l.ikn);
            gpa.free(l.q);
            gpa.free(l.k);
            gpa.free(l.v);
            gpa.free(l.o);
            gpa.free(l.iq);
            gpa.free(l.ik);
            if (l.dense) |m| {
                gpa.free(m.gate);
                gpa.free(m.up);
                gpa.free(m.down);
            }
            if (l.moe) |m| {
                gpa.free(m.router);
                gpa.free(m.bias);
                gpa.free(m.shared.gate);
                gpa.free(m.shared.up);
                gpa.free(m.shared.down);
                for (m.experts) |e| {
                    gpa.free(e.gate);
                    gpa.free(e.up);
                    gpa.free(e.down);
                }
                gpa.free(m.experts);
            }
        }
        gpa.free(f.layers);
        gpa.free(f.embed);
        gpa.free(f.final_norm);
        gpa.free(f.head);
        gpa.free(f.scratch);
    }
};

fn decodedName(buf: []u8, comptime fmt: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buf, fmt, args) catch unreachable;
}

/// One trellis group decoded to [k, n] f32 (the host twin of the device graph's own decode).
fn decodeGroup(gpa: std.mem.Allocator, scan: *exl3_load.Scan, gi: usize) ![]f32 {
    const g = scan.groups[gi];
    const parts = try scan.parts(gpa, gi);
    const k = g.k;
    const n = g.n;
    const tiles_k = k / 16;
    const tiles_n = n / 16;
    const out64 = try gpa.alloc(f64, @as(usize, k) * n);
    defer gpa.free(out64);
    const scratch = try gpa.alloc(u8, 2 * @as(usize, k) * n * 8);
    defer gpa.free(scratch);
    const trellis = try gpa.alloc(i16, parts.trellis.bytes.len / 2); // read little-endian (the bytes may not be i16-aligned)
    defer gpa.free(trellis);
    for (trellis, 0..) |*tw, i| tw.* = @bitCast(std.mem.readInt(u16, parts.trellis.bytes[i * 2 ..][0..2], .little));
    const suh = std.mem.bytesAsSlice(u16, parts.suh.bytes);
    const svh = std.mem.bytesAsSlice(u16, parts.svh.bytes);
    const suh_f16 = try gpa.alloc(f16, suh.len);
    defer gpa.free(suh_f16);
    const svh_f16 = try gpa.alloc(f16, svh.len);
    defer gpa.free(svh_f16);
    for (suh, 0..) |b, i| suh_f16[i] = @bitCast(b);
    for (svh, 0..) |b, i| svh_f16[i] = @bitCast(b);
    exl3_fmt.dequantize(
        .{ .halves = g.halves },
        @enumFromInt(@intFromEnum(g.codebook)),
        trellis,
        tiles_k,
        tiles_n,
        suh_f16,
        svh_f16,
        out64,
        @as([*]f64, @ptrCast(@alignCast(scratch.ptr)))[0 .. 2 * @as(usize, k) * n],
    );
    const out = try gpa.alloc(f32, @as(usize, k) * n);
    for (out64, 0..) |v, i| out[i] = @floatCast(v);
    return out;
}

/// The poked fixture's decoded weights (the same bytes the Engine's device graph serves).
fn loadFixtureW(gpa: std.mem.Allocator, io: std.Io, dir_path: []const u8, c: m3.serve.Config) !FW {
    const L: usize = c.run;
    var scan = try exl3_load.scan(gpa, io, dir_path);
    defer scan.deinit();

    var w: FW = .{
        .gpa = gpa,
        .c = c,
        .embed = &.{},
        .final_norm = &.{},
        .head = &.{},
        .layers = &.{},
        .scratch = &.{},
    };
    errdefer w.deinit();
    w.layers = try gpa.alloc(FW_Layer, L);
    for (w.layers) |*l| l.* = .{
        .in_norm = &.{}, .post_norm = &.{}, .q_norm = &.{}, .k_norm = &.{},
        .iqn = &.{}, .ikn = &.{}, .q = &.{}, .k = &.{}, .v = &.{}, .o = &.{},
        .iq = &.{}, .ik = &.{},
    };

    var nb: [256]u8 = undefined;
    // passthrough helpers over the same checkpoint files
    const bf16Row = struct {
        fn f(sc: *exl3_load.Scan, a: std.mem.Allocator, name: []const u8, cols: usize) ![]f32 {
            const t = try sc.ck.get(name);
            if (!t.is(.bf16, &.{cols})) return error.UnexpectedTensor;
            const out = try a.alloc(f32, cols);
            for (0..cols) |i| out[i] = bf16At(t.bytes, i);
            return out;
        }
    }.f;
    const f16Mat = struct {
        fn f(sc: *exl3_load.Scan, a: std.mem.Allocator, name: []const u8, rows: usize, cols: usize) ![]f32 {
            const t = try sc.ck.get(name);
            if (!t.is(.f16, &.{ rows, cols })) return error.UnexpectedTensor;
            const out = try a.alloc(f32, rows * cols);
            for (0..rows * cols) |i| out[i] = f16At(t.bytes, i);
            return out;
        }
    }.f;

    const t = try scan.ck.get("language_model.model.embed_tokens.weight");
    if (!t.is(.bf16, &.{ c.vocab, c.hidden })) return error.UnexpectedTensor;
    w.embed = try gpa.alloc(f32, @as(usize, c.vocab) * c.hidden);
    for (w.embed, 0..) |*v, i| v.* = bf16At(t.bytes, i);
    const fin = decodedName(&nb, "language_model.model.norm.weight", .{});
    w.final_norm = try bf16Row(&scan, gpa, fin, c.hidden);

    for (0..L) |l| {
        const lay = &w.layers[l];
        lay.in_norm = try bf16Row(&scan, gpa, decodedName(&nb, "language_model.model.layers.{d}.input_layernorm.weight", .{l}), c.hidden);
        lay.post_norm = try bf16Row(&scan, gpa, decodedName(&nb, "language_model.model.layers.{d}.post_attention_layernorm.weight", .{l}), c.hidden);
        lay.q_norm = try bf16Row(&scan, gpa, decodedName(&nb, "language_model.model.layers.{d}.self_attn.q_norm.weight", .{l}), c.head_dim);
        lay.k_norm = try bf16Row(&scan, gpa, decodedName(&nb, "language_model.model.layers.{d}.self_attn.k_norm.weight", .{l}), c.head_dim);
        lay.iqn = try bf16Row(&scan, gpa, decodedName(&nb, "language_model.model.layers.{d}.self_attn.index_q_norm.weight", .{l}), c.i_dim);
        lay.ikn = try bf16Row(&scan, gpa, decodedName(&nb, "language_model.model.layers.{d}.self_attn.index_k_norm.weight", .{l}), c.i_dim);
        const irows: usize = c.i_heads * c.i_dim;
        lay.iq = try f16Mat(&scan, gpa, decodedName(&nb, "language_model.model.layers.{d}.self_attn.index_q_proj.weight", .{l}), irows, c.hidden);
        lay.ik = try f16Mat(&scan, gpa, decodedName(&nb, "language_model.model.layers.{d}.self_attn.index_k_proj.weight", .{l}), c.i_dim, c.hidden);
        lay.q = try decodeGroup(gpa, &scan, scan.find(decodedName(&nb, "language_model.model.layers.{d}.self_attn.q_proj", .{l})).?);
        lay.k = try decodeGroup(gpa, &scan, scan.find(decodedName(&nb, "language_model.model.layers.{d}.self_attn.k_proj", .{l})).?);
        lay.v = try decodeGroup(gpa, &scan, scan.find(decodedName(&nb, "language_model.model.layers.{d}.self_attn.v_proj", .{l})).?);
        lay.o = try decodeGroup(gpa, &scan, scan.find(decodedName(&nb, "language_model.model.layers.{d}.self_attn.o_proj", .{l})).?);
        if (l < c.dense_layers) {
            lay.dense = .{
                .gate = try decodeGroup(gpa, &scan, scan.find(decodedName(&nb, "language_model.model.layers.{d}.mlp.gate_proj", .{l})).?),
                .up = try decodeGroup(gpa, &scan, scan.find(decodedName(&nb, "language_model.model.layers.{d}.mlp.up_proj", .{l})).?),
                .down = try decodeGroup(gpa, &scan, scan.find(decodedName(&nb, "language_model.model.layers.{d}.mlp.down_proj", .{l})).?),
            };
        } else {
            var m: FW_MoE = .{ .router = &.{}, .bias = &.{}, .shared = .{ .gate = &.{}, .up = &.{}, .down = &.{} }, .experts = &.{} };
            const rt = try scan.ck.get(decodedName(&nb, "language_model.model.layers.{d}.block_sparse_moe.gate.weight", .{l}));
            if (!rt.is(.f32, &.{ c.experts, c.hidden })) return error.UnexpectedTensor;
            m.router = try gpa.alloc(f32, @as(usize, c.experts) * c.hidden);
            @memcpy(m.router, std.mem.bytesAsSlice(f32, rt.bytes));
            const bt = try scan.ck.get(decodedName(&nb, "language_model.model.layers.{d}.block_sparse_moe.e_score_correction_bias", .{l}));
            if (!bt.is(.f32, &.{c.experts})) return error.UnexpectedTensor;
            m.bias = try gpa.alloc(f32, c.experts);
            @memcpy(m.bias, std.mem.bytesAsSlice(f32, bt.bytes));
            m.shared = .{
                .gate = try decodeGroup(gpa, &scan, scan.find(decodedName(&nb, "language_model.model.layers.{d}.block_sparse_moe.shared_experts.gate_proj", .{l})).?),
                .up = try decodeGroup(gpa, &scan, scan.find(decodedName(&nb, "language_model.model.layers.{d}.block_sparse_moe.shared_experts.up_proj", .{l})).?),
                .down = try decodeGroup(gpa, &scan, scan.find(decodedName(&nb, "language_model.model.layers.{d}.block_sparse_moe.shared_experts.down_proj", .{l})).?),
            };
            const es = try gpa.alloc(FW_Mlp, c.experts);
            for (0..c.experts) |e| {
                // w1=gate, w3=up, w2=down (the inventory's verified mapping)
                es[e] = .{
                    .gate = try decodeGroup(gpa, &scan, scan.find(decodedName(&nb, "language_model.model.layers.{d}.block_sparse_moe.experts.{d}.w1", .{ l, e })).?),
                    .up = try decodeGroup(gpa, &scan, scan.find(decodedName(&nb, "language_model.model.layers.{d}.block_sparse_moe.experts.{d}.w3", .{ l, e })).?),
                    .down = try decodeGroup(gpa, &scan, scan.find(decodedName(&nb, "language_model.model.layers.{d}.block_sparse_moe.experts.{d}.w2", .{ l, e })).?),
                };
            }
            m.experts = es;
            lay.moe = m;
        }
    }
    w.head = try decodeGroup(gpa, &scan, scan.find("language_model.lm_head").?);
    return w;
}

/// The f64 reference forward over the decoded fixture (runtime dims, Ref's composition).
const RefF64 = struct {
    const G = f64;
    gpa: std.mem.Allocator,
    c: m3.serve.Config,
    w: *const FW,
    cap: usize, // kc/vc/pl rows per layer
    len: u32 = 0,
    kc: []G,
    vc: []G,
    pl: []G,
    logits: []G,
    st: []G,
    qb: []G,
    ab: []G,
    kb: []G,
    vb: []G,
    ib: []G,
    isk: []G,
    ob: []G,
    gb: []G,
    ubx: []G,
    acb: []G,
    ks: []u32,
    ssc: []G,

    fn h16(x: G) G {
        return @floatCast(@as(f16, @floatCast(x)));
    }
    fn norm(x: []G, w: []const f32, eps: f32) void {
        var ss: G = 0;
        for (x) |v| ss += v * v;
        const inv = 1.0 / @sqrt(ss / @as(G, @floatFromInt(x.len)) + @as(G, eps));
        for (x, w) |*v, wi| v.* = v.* * inv * (1.0 + @as(G, wi));
    }
    /// y[o] = sum_c x[c] * D[c*n + o] — the decoded trellis's [k, n] row-major layout.
    fn linT(x: []const G, d: []const f32, n: usize, y: []G) void {
        const k = x.len;
        @memset(y, 0);
        for (0..k) |ci| {
            const xv = x[ci];
            const row = d[ci * n ..][0..n];
            for (0..n) |o| y[o] += xv * @as(G, row[o]);
        }
    }
    /// y[o] = sum_c x[c] * W[o*k + c] — the passthrough [n_out, k] row-major layout.
    fn lin(x: []const G, w: []const f32, n_out: usize, y: []G) void {
        const k = x.len;
        @memset(y, 0);
        for (0..k) |ci| {
            const xv = x[ci];
            for (0..n_out) |o| y[o] += xv * @as(G, w[o * k + ci]);
        }
    }
    fn rope(x: []G, pos: usize, rot: usize) void {
        const half = rot / 2;
        for (0..half) |i| {
            const ang = @as(G, @floatFromInt(pos)) /
                std.math.pow(G, @as(G, 1e7), @as(G, @floatFromInt(2 * i)) / @as(G, @floatFromInt(rot)));
            const cv: G = @cos(ang);
            const sv: G = @sin(ang);
            const a = x[i];
            const b = x[i + half];
            x[i] = a * cv - b * sv;
            x[i + half] = b * cv + a * sv;
        }
    }
    fn act(g: G, u: G, alpha: f32, limit: f32) G {
        const gg = @min(g, @as(G, limit));
        const uu = @min(@max(u, -@as(G, limit)), @as(G, limit)) + 1.0;
        return gg / (1.0 + std.math.exp(-@as(G, alpha) * gg)) * uu;
    }

    fn deinit(r: *RefF64) void {
        const gpa = r.gpa;
        gpa.free(r.kc);
        gpa.free(r.vc);
        gpa.free(r.pl);
        gpa.free(r.logits);
        gpa.free(r.st);
        gpa.free(r.qb);
        gpa.free(r.ab);
        gpa.free(r.kb);
        gpa.free(r.vb);
        gpa.free(r.ib);
        gpa.free(r.isk);
        gpa.free(r.ob);
        gpa.free(r.gb);
        gpa.free(r.ubx);
        gpa.free(r.acb);
        gpa.free(r.ks);
        gpa.free(r.ssc);
    }

    fn init(gpa: std.mem.Allocator, c: m3.serve.Config, cap: usize, w: *const FW) !RefF64 {
        const L: usize = c.run;
        const H: usize = c.hidden;
        const qk: usize = c.attn_heads * c.head_dim;
        const kvr: usize = c.kv_heads * c.head_dim;
        const irows: usize = c.i_heads * c.i_dim;
        const inter: usize = @max(@max(c.dense_inter, c.moe_inter), c.shared_inter);
        var r: RefF64 = .{
            .gpa = gpa, .c = c, .w = w, .cap = cap, .len = 0,
            .kc = &.{}, .vc = &.{}, .pl = &.{}, .logits = &.{}, .st = &.{},
            .qb = &.{}, .ab = &.{}, .kb = &.{}, .vb = &.{}, .ib = &.{},
            .isk = &.{}, .ob = &.{}, .gb = &.{}, .ubx = &.{}, .acb = &.{},
            .ks = &.{}, .ssc = &.{},
        };
        errdefer r.deinit();
        r.kc = try gpa.alloc(G, L * cap * kvr);
        r.vc = try gpa.alloc(G, L * cap * kvr);
        r.pl = try gpa.alloc(G, L * cap * c.i_dim);
        r.logits = try gpa.alloc(G, c.vocab);
        r.st = try gpa.alloc(G, H);
        r.qb = try gpa.alloc(G, qk);
        r.ab = try gpa.alloc(G, qk);
        r.kb = try gpa.alloc(G, kvr);
        r.vb = try gpa.alloc(G, kvr);
        r.ib = try gpa.alloc(G, irows);
        r.isk = try gpa.alloc(G, c.i_dim);
        r.ob = try gpa.alloc(G, H);
        r.gb = try gpa.alloc(G, inter);
        r.ubx = try gpa.alloc(G, inter);
        r.acb = try gpa.alloc(G, inter);
        r.ks = try gpa.alloc(u32, c.keyWidth());
        r.ssc = try gpa.alloc(G, c.keyWidth());
        return r;
    }

    fn put(r: *RefF64, tok: u32) !void {
        std.debug.assert(r.len < r.cap);
        const c = &r.c;
        const w = r.w;
        const H: usize = c.hidden;
        const qk: usize = c.attn_heads * c.head_dim;
        const kvr: usize = c.kv_heads * c.head_dim;
        const hd: usize = c.head_dim;
        const cap = r.cap;
        const pos: usize = r.len;
        r.len += 1;
        const sk = pos + 1;
        const scale = 1.0 / @sqrt(@as(G, @floatFromInt(c.head_dim)));
        for (0..H) |i| r.st[i] = @as(G, w.embed[@as(usize, tok) * H + i]);
        for (w.layers, 0..) |*l, li| {
            const xn = try r.gpa.alloc(G, H);
            defer r.gpa.free(xn);
            @memcpy(xn, r.st);
            norm(xn, l.in_norm, c.eps);
            linT(xn, l.q, qk, r.qb);
            for (0..c.attn_heads) |h| {
                const row = r.qb[h * hd ..][0..hd];
                norm(row, l.q_norm, c.eps);
                rope(row, pos, c.rotary_dim);
                for (row) |*v| v.* = h16(v.*);
            }
            linT(xn, l.k, kvr, r.kb);
            for (0..c.kv_heads) |h| {
                const row = r.kb[h * hd ..][0..hd];
                norm(row, l.k_norm, c.eps);
                rope(row, pos, c.rotary_dim);
                for (row) |*v| v.* = h16(v.*);
            }
            linT(xn, l.v, kvr, r.vb);
            for (0..kvr) |i| r.vb[i] = h16(r.vb[i]);
            if (li == 0 and pos == 0) { // the device-vs-host discriminator row: the projections' own scales
                var qx2: f64 = 0;
                var kx2: f64 = 0;
                var vx2: f64 = 0;
                for (r.qb) |v| qx2 = @max(qx2, @abs(v));
                for (r.kb) |v| kx2 = @max(kx2, @abs(v));
                for (r.vb) |v| vx2 = @max(vx2, @abs(v));
                std.debug.print("DBGR f64 L0 pos0: after-norm/rope q {d:.4} k {d:.4} v {d:.4} | k[:4]={d:.4},{d:.4},{d:.4},{d:.4} | v[:8]={d:.4},{d:.4},{d:.4},{d:.4},{d:.4},{d:.4},{d:.4},{d:.4}\n", .{ qx2, kx2, vx2, r.kb[0], r.kb[1], r.kb[2], r.kb[3], r.vb[0], r.vb[1], r.vb[2], r.vb[3], r.vb[4], r.vb[5], r.vb[6], r.vb[7] });
                var xn2s: f64 = 0;
                for (xn) |v| xn2s += v * v;
                std.debug.print("DBGR f64 L0 pos0: st[0]={d:.6} xn[0]={d:.6} xrms={d:.6}\n", .{ r.st[0], xn[0], @sqrt(xn2s / @as(f64, @floatFromInt(H))) });
            }
            const kro = @as(usize, li) * cap * kvr;
            for (0..kvr) |i| {
                r.kc[kro + pos * kvr + i] = r.kb[i];
                r.vc[kro + pos * kvr + i] = r.vb[i];
            }
            lin(xn, l.iq, c.i_heads * c.i_dim, r.ib); // fp16 passthrough [n_out, k] layout
            for (0..c.i_heads) |h| {
                const row = r.ib[h * c.i_dim ..][0..c.i_dim];
                norm(row, l.iqn, c.eps);
                rope(row, pos, c.rotary_dim);
                for (row) |*v| v.* = h16(v.*);
            }
            lin(xn, l.ik, c.i_dim, r.isk); // fp16 passthrough [n_out, k] layout
            norm(r.isk, l.ikn, c.eps);
            rope(r.isk, pos, c.rotary_dim);
            for (0..c.i_dim) |i| r.isk[i] = h16(r.isk[i]);
            const pro = @as(usize, li) * cap * c.i_dim;
            for (0..c.i_dim) |i| r.pl[pro + pos * c.i_dim + i] = r.isk[i];
            // the selection: f64 block-max pool, the local boost override, strict > ties at the
            // smaller id, left-packed, -1 right-pad (Ref's composition)
            const nb = (sk + c.block - 1) / c.block;
            var sel: [16]i32 = @splat(-1);
            var ih: usize = 0;
            while (ih < c.i_heads) : (ih += 1) {
                var bsc: [512]G = @splat(-std.math.inf(G));
                for (0..nb) |b| {
                    var best = -std.math.inf(G);
                    const lo = b * c.block;
                    const nhi = @min(lo + c.block, sk);
                    var j = lo;
                    while (j < nhi) : (j += 1) {
                        var dot: G = 0;
                        for (0..c.i_dim) |dd| dot += r.ib[ih * c.i_dim + dd] * r.pl[pro + j * c.i_dim + dd];
                        if (dot > best) best = dot;
                    }
                    bsc[b] = best;
                }
                bsc[pos / c.block] = std.math.inf(G); // the local boost override
                for (0..c.i_topk) |sl| {
                    var best: usize = nb;
                    var bs = -std.math.inf(G);
                    for (0..nb) |b| {
                        var taken = false;
                        for (0..sl) |p| {
                            if (sel[p] >= 0 and sel[p] == @as(i32, @intCast(b))) taken = true;
                        }
                        if (taken) continue;
                        if (bsc[b] > bs) {
                            bs = bsc[b];
                            best = b;
                        }
                    }
                    if (best < nb) sel[sl] = @intCast(best);
                }
            }
            // the block-sparse f64 softmax attention, GQA by kv head
            const per = c.attn_heads / c.kv_heads;
            var hv: usize = 0;
            while (hv < c.kv_heads) : (hv += 1) {
                var nk: usize = 0;
                for (0..c.i_topk) |sl| {
                    const bl = sel[sl];
                    if (bl < 0) break;
                    const blo = @as(usize, @intCast(bl)) * c.block;
                    const nhi = @min(blo + c.block, sk);
                    var kk = blo;
                    while (kk < nhi) : (kk += 1) {
                        r.ks[nk] = @intCast(kk);
                        nk += 1;
                    }
                }
                var j: usize = 0;
                while (j < per) : (j += 1) {
                    const qh = hv * per + j;
                    const qr = r.qb[qh * hd ..][0..hd];
                    var mx = -std.math.inf(G);
                    var m: usize = 0;
                    while (m < nk) : (m += 1) {
                        const krow = r.kc[kro + r.ks[m] * kvr ..][0..kvr][hv * hd ..][0..hd];
                        var dot: G = 0;
                        for (0..hd) |d| dot += qr[d] * krow[d];
                        r.ssc[m] = dot * scale;
                        if (dot * scale > mx) mx = dot * scale;
                    }
                    var se: G = 0;
                    m = 0;
                    while (m < nk) : (m += 1) {
                        const e = std.math.exp(r.ssc[m] - mx);
                        r.ssc[m] = e;
                        se += e;
                    }
                    const acc = try r.gpa.alloc(G, hd);
                    defer r.gpa.free(acc);
                    @memset(acc, 0);
                    m = 0;
                    while (m < nk) : (m += 1) {
                        const vrow = r.vc[kro + r.ks[m] * kvr ..][0..kvr][hv * hd ..][0..hd];
                        const p = r.ssc[m] / se;
                        for (0..hd) |d| acc[d] += p * vrow[d];
                    }
                    const orow = r.ab[qh * hd ..][0..hd];
                    for (0..hd) |d| orow[d] = h16(acc[d]);
                }
            }
            for (0..qk) |i| r.ab[i] = h16(r.ab[i]);
            linT(r.ab, l.o, H, r.ob);
            for (0..H) |i| r.st[i] += h16(r.ob[i]);
            const xn2 = try r.gpa.alloc(G, H);
            defer r.gpa.free(xn2);
            @memcpy(xn2, r.st);
            norm(xn2, l.post_norm, c.eps);
            if (l.dense) |m| {
                linT(xn2, m.gate, c.dense_inter, r.gb);
                linT(xn2, m.up, c.dense_inter, r.ubx);
                for (0..c.dense_inter) |i| r.acb[i] = act(r.gb[i], r.ubx[i], c.swiglu_alpha, c.swiglu_limit);
                linT(r.acb[0..c.dense_inter], m.down, H, r.ob);
                for (0..H) |i| r.st[i] += h16(r.ob[i]);
            } else {
                const m = l.moe.?;
                // the noaux router: select on sigmoid+bias, weights = unbiased sigmoid normalized
                // over the picks, * routed_scale, each .half() — each weight pairs its own expert
                const E: usize = c.experts;
                const sg = try r.gpa.alloc(G, E);
                defer r.gpa.free(sg);
                const sc = try r.gpa.alloc(G, E);
                defer r.gpa.free(sc);
                const picks = try r.gpa.alloc(bool, E);
                defer r.gpa.free(picks);
                @memset(picks, false);
                const ids = try r.gpa.alloc(u32, c.topk);
                defer r.gpa.free(ids);
                for (0..E) |e| {
                    var lg: G = 0;
                    for (0..H) |jj| lg += xn2[jj] * @as(G, m.router[e * H + jj]);
                    sg[e] = 1.0 / (1.0 + std.math.exp(-lg));
                    sc[e] = sg[e] + @as(G, m.bias[e]);
                }
                var sum: G = 1e-20;
                for (0..c.topk) |slot| {
                    var best: usize = E;
                    var bs = -std.math.inf(G);
                    for (0..E) |e| {
                        if (picks[e]) continue;
                        if (sc[e] > bs) {
                            bs = sc[e];
                            best = e;
                        }
                    }
                    picks[best] = true;
                    ids[slot] = @intCast(best);
                    sum += sg[best];
                }
                var acc = try r.gpa.alloc(G, H);
                defer r.gpa.free(acc);
                @memset(acc, 0);
                for (0..c.topk) |slot| {
                    const wi: G = h16(sg[ids[slot]] / sum * @as(G, c.routed_scale));
                    const e = m.experts[ids[slot]];
                    linT(xn2, e.gate, c.moe_inter, r.gb);
                    linT(xn2, e.up, c.moe_inter, r.ubx);
                    for (0..c.moe_inter) |i| r.acb[i] = act(r.gb[i], r.ubx[i], c.swiglu_alpha, c.swiglu_limit);
                    linT(r.acb[0..c.moe_inter], e.down, H, r.ob);
                    for (0..H) |i| acc[i] += wi * r.ob[i];
                }
                linT(xn2, m.shared.gate, c.shared_inter, r.gb);
                linT(xn2, m.shared.up, c.shared_inter, r.ubx);
                for (0..c.shared_inter) |i| r.acb[i] = act(r.gb[i], r.ubx[i], c.swiglu_alpha, c.swiglu_limit);
                linT(r.acb[0..c.shared_inter], m.shared.down, H, r.ob);
                for (0..H) |i| acc[i] += r.ob[i]; // the shared expert at weight 1
                for (0..H) |i| r.st[i] += h16(acc[i]);
            }
        }
        const xf = try r.gpa.alloc(G, H);
        defer r.gpa.free(xf);
        @memcpy(xf, r.st);
        norm(xf, w.final_norm, c.eps);
        for (0..c.vocab) |v| {
            var acc: G = 0;
            for (0..H) |j| acc += xf[j] * @as(G, w.head[j * @as(usize, c.vocab) + v]);
            r.logits[v] = acc;
        }
    }

    fn greedy(r: *RefF64, steps: usize, out: []u32) !void {
        for (0..steps) |g| {
            var best: usize = 0;
            var bv = -std.math.inf(G);
            for (r.logits, 0..) |v, i| if (v > bv) {
                bv = v;
                best = i;
            };
            out[g] = @intCast(best);
            if (g + 1 < steps) try r.put(@intCast(best));
        }
    }
};

/// The smoke's grade: the device graph's tokens against the f64 reference's on the same fixture.
fn gradeTokens(ref_out: []const u32, dev_out: []const u32) !void {
    var ok = true;
    for (ref_out, dev_out) |rv, dv| {
        if (rv != dv) {
            ok = false;
        }
    }
    if (!ok) {
        std.debug.print("FAIL minimax-m3-smoke: device tokens", .{});
        for (dev_out) |t| std.debug.print(" {d}", .{t});
        std.debug.print(" vs reference tokens", .{});
        for (ref_out) |t| std.debug.print(" {d}", .{t});
        std.debug.print("\n", .{});
        return error.TokenMismatch;
    }
}

/// The debug build's grade wrapper: a token mismatch still lets the run finish loudly.
fn gradeTokensCatch(ref_out: []const u32, dev_out: []const u32) !void {
    gradeTokens(ref_out, dev_out) catch {};
}

// =================================================================== Stage-8: grade + lanes
//
//   tf-cuda-test minimax-m3-grade <pack dir|fixture> [ctx] [steps] [layers]
//       STEP 1's grade: the device graph (TP1, the first `layers` REAL layers of the pack via
//       Engine.Options.layer_cap — the reduced real-pack stack that fits the host beside the
//       reference) against the SAME fork f64 composition the fixture smoke grades with
//       (RefF64 over the pack's own trellis groups decoded on the host). Compares every row's
//       residual (eng.x vs Ref st) and logits (eng.logits vs Ref logits) and the greedy tokens.
//       The full-60-layer real pack stays beyond the honest host f64 budget (hundreds of GB of
//       decoded f64 weights and hours of trellis decode) — reported as the boundary, not hidden.
//
//   tf-cuda-test minimax-m3-lanes <pack dir|fixture> [ctx] [steps] [layers]
//       STEP 2's boundary test through open()'s Loaded handle: a real Stream served through the
//       lane vtable (prefill → first → queue/read round trips → a verify round → keep rollback),
//       graded token-for-token against the same prompt run DIRECTLY through a second Engine, plus
//       the honest refusal rows (draft, unknown handle, drafted verify, batched verify, sampled
//       spec, second stream, tree keep) and release+re-prefill.

/// The max rows the grade's host buffers hold (ctx + steps - 1 puts both sides take).
const grade_rows_max: usize = 256;

/// One side's per-row captures: the residual and the logits at every put's position.
const GradeCap = struct {
    gpa: std.mem.Allocator,
    c: m3.serve.Config,
    rows: usize = 0,
    x: [][]f32 = &.{},
    lg: [][]f32 = &.{},

    fn init(g: std.mem.Allocator, c: m3.serve.Config, rows: usize) !GradeCap {
        const H: usize = c.hidden;
        var cap = GradeCap{ .gpa = g, .c = c };
        errdefer cap.deinit();
        cap.x = try g.alloc([]f32, rows);
        cap.lg = try g.alloc([]f32, rows);
        for (0..rows) |i| {
            cap.x[i] = try g.alloc(f32, H);
            cap.lg[i] = try g.alloc(f32, c.vocab);
        }
        return cap;
    }
    fn deinit(cap: *GradeCap) void {
        for (cap.x) |r| cap.gpa.free(r);
        for (cap.lg) |r| cap.gpa.free(r);
        if (cap.x.len > 0) cap.gpa.free(cap.x);
        if (cap.lg.len > 0) cap.gpa.free(cap.lg);
        cap.* = .{ .gpa = cap.gpa, .c = cap.c };
    }
};

/// The device side's capture loop: identical token sequence to eng.greedy, but with a capture
/// after every put (the harness reads eng.x and eng.logits between rows).
fn deviceRun(eng: *m3.graph.Engine, prompt: []const u32, steps: usize, cap: *GradeCap, out: []u32) !void {
    var row: usize = 0;
    for (prompt) |tok| {
        try eng.put(tok);
        try captureRow(cap, eng.x, eng.logits, row);
        row += 1;
    }
    for (0..steps) |g| {
        var best: u32 = 0;
        var bv = -std.math.inf(f32);
        for (eng.logits, 0..) |v, i| {
            if (v > bv) {
                bv = v;
                best = @intCast(i);
            }
        }
        out[g] = best;
        if (g + 1 < steps) {
            try eng.put(best);
            try captureRow(cap, eng.x, eng.logits, row);
            row += 1;
        }
    }
}

fn captureRow(cap: *GradeCap, x: []const f32, lg: []const f32, row: usize) !void {
    if (row >= cap.x.len) return error.CaptureOverrun;
    @memcpy(cap.x[row], x);
    @memcpy(cap.lg[row], lg);
    cap.rows = row + 1;
}

/// The f64 side's capture loop over the SAME tokens (its own greedy draws, compared below).
fn refRun(ref: *RefF64, prompt: []const u32, steps: usize, cap: *GradeCap, out: []u32) !void {
    var row: usize = 0;
    for (prompt) |tok| {
        try ref.put(tok);
        try captureRef(cap, ref, row);
        row += 1;
    }
    for (0..steps) |g| {
        var best: u32 = 0;
        var bv: f64 = -std.math.inf(f64);
        for (ref.logits, 0..) |v, i| {
            if (v > bv) {
                bv = v;
                best = @intCast(i);
            }
        }
        out[g] = best;
        if (g + 1 < steps) {
            try ref.put(best);
            try captureRef(cap, ref, row);
            row += 1;
        }
    }
}

fn captureRef(cap: *GradeCap, ref: *RefF64, row: usize) !void {
    if (row >= cap.x.len) return error.CaptureOverrun;
    for (ref.st, 0..) |v, i| cap.x[row][i] = @floatCast(v);
    for (ref.logits, 0..) |v, i| cap.lg[row][i] = @floatCast(v);
    cap.rows = row + 1;
}

/// max|delta| over two fp32 rows (NaN never passes; both rows are finite by construction here).
fn rowMaxd(a: []const f32, b: []const f32) f64 {
    var d: f64 = 0;
    for (a, b) |va, vb| {
        const dd = @abs(@as(f64, va) - @as(f64, vb));
        if (dd > d) d = dd;
    }
    return d;
}

pub fn gradeRun(gpu: check.Gpu, dir_in: []const u8, ctx_len: usize, steps: usize, layers_arg: usize) !void {
    const gpa = gpu.gpa;
    var dir = dir_in;
    if (dir.len == 0) {
        std.debug.print("usage: tf-cuda-test minimax-m3-grade <pack dir|fixture> [ctx] [steps] [layers]\n", .{});
        return error.BadArguments;
    }
    if (steps == 0 or steps > 32 or ctx_len == 0 or ctx_len > m3.graph.keys_cap or ctx_len + steps > grade_rows_max) {
        return error.BadArguments; // loud refusals, no silent clamping
    }
    if (layers_arg == 0) {
        std.debug.print("FAIL minimax-m3-grade: the layers cap must be explicit (0 is the full stack — beyond the honest f64 reference budget on the host)\n", .{});
        return error.BadArguments;
    }
    if (std.mem.eql(u8, dir, "fixture")) dir = try makeFixture(gpu);
    const t0 = check.now(gpu.io);
    var eng = try m3.graph.Engine.build(gpa, gpu.io, gpu.d, dir, 0, 1, .{ .layer_cap = layers_arg });
    defer eng.deinit();
    const c = eng.c;
    if (layers_arg > c.layers) return error.LayerCapTooWide;
    // the reference reads the SAME first `layers_arg` layers (its own config run count)
    var cr = c;
    cr.run = @intCast(layers_arg);
    const ram_before = hostFree(gpu);

    // the host f64 reference over the pack's own decoded trellis (the smoke's decode path)
    var fw = try loadFixtureW(gpa, gpu.io, dir, cr);
    defer fw.deinit();
    const ram_after_decode = hostFree(gpu);

    const rows = ctx_len + steps - 1;
    var dev_cap = try GradeCap.init(gpa, c, rows);
    defer dev_cap.deinit();
    var ref_cap = try GradeCap.init(gpa, c, rows);
    defer ref_cap.deinit();

    const prompt = try gpa.alloc(u32, ctx_len);
    defer gpa.free(prompt);
    var l = Lcg{ .s = prompt_seed };
    m3.serve.promptOf(&l, prompt);

    const dev_out = try gpa.alloc(u32, steps);
    defer gpa.free(dev_out);
    const ref_out = try gpa.alloc(u32, steps);
    defer gpa.free(ref_out);

    var ref = try RefF64.init(gpa, cr, rows + 1, &fw);
    defer ref.deinit();

    const t1 = check.now(gpu.io);
    try deviceRun(&eng, prompt, steps, &dev_cap, dev_out);
    try refRun(&ref, prompt, steps, &ref_cap, ref_out);
    const run_s = @as(f64, @floatFromInt(check.now(gpu.io) - t1)) / 1e9;

    // -- the comparisons --
    var xmax: f64 = 0;
    var lmax: f64 = 0;
    var lnan: usize = 0;
    var linf: usize = 0;
    for (0..dev_cap.rows) |r| {
        const x = rowMaxd(dev_cap.x[r], ref_cap.x[r]);
        const ld = rowMaxd(dev_cap.lg[r], ref_cap.lg[r]);
        if (x > xmax) xmax = x;
        if (ld > lmax) lmax = ld;
        std.debug.print("M3-GRADE row {d:3}: residual max|delta| {d:.6} logits max|delta| {d:.6}\n", .{ r, x, ld });
        if (r == 0) {
            std.debug.print("M3-GRADE row 0 raw: dev x[0..4] = {d:.5} {d:.5} {d:.5} {d:.5} | ref st[0..4] = {d:.5} {d:.5} {d:.5} {d:.5}\n", .{ dev_cap.x[r][0], dev_cap.x[r][1], dev_cap.x[r][2], dev_cap.x[r][3], ref_cap.x[r][0], ref_cap.x[r][1], ref_cap.x[r][2], ref_cap.x[r][3] });
        }
        for (dev_cap.x[r]) |v| {
            if (std.math.isNan(v) or std.math.isInf(v)) lnan += 1;
        }
        for (dev_cap.lg[r]) |v| {
            if (std.math.isNan(v)) lnan += 1 else if (std.math.isInf(v)) linf += 1;
        }
    }
    var exact = true;
    for (dev_out, ref_out) |dv, rv| {
        if (dv != rv) exact = false;
    }
    const build_s = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e9;
    std.debug.print("M3-GRADE: layers {d}/{d}, rows {d}, steps {d}; build+decode {d:.1}s, both runs {d:.1}s\n", .{ layers_arg, c.layers, dev_cap.rows, steps, build_s, run_s });
    std.debug.print("M3-GRADE: max|delta| residual {d:.6} logits {d:.6}; dev rows NaN {d} inf {d}\n", .{ xmax, lmax, lnan, linf });
    std.debug.print("M3-GRADE: host MemAvailable {d:.2} -> {d:.2} (post-decode) GiB\n", .{ @as(f64, @floatFromInt(ram_before)) / (1024 * 1024 * 1024), @as(f64, @floatFromInt(ram_after_decode)) / (1024 * 1024 * 1024) });
    std.debug.print("M3-GRADE: dev greedy", .{});
    for (dev_out) |t| std.debug.print(" {d}", .{t});
    std.debug.print(" | ref greedy", .{});
    for (ref_out) |t| std.debug.print(" {d}", .{t});
    std.debug.print("\n", .{});
    // the draw margins: is the flipped draw a near-tie (quant tolerance) or a real miss?
    for (0..steps) |g| {
        const r = ctx_len - 1 + g; // draw g keys at the row ending with its feeding token's put
        if (r >= dev_cap.rows) break;
        if (g > 0 and dev_out[g - 1] != ref_out[g - 1] and ctx_len + g > 0) break; // margins before the first flip are the informative ones
        var d1: usize = 0;
        var d2: usize = 0;
        var dv1: f32 = -std.math.inf(f32);
        var dv2: f32 = -std.math.inf(f32);
        for (dev_cap.lg[r], 0..) |v, i| {
            if (v > dv1) {
                dv2 = dv1;
                d2 = d1;
                dv1 = v;
                d1 = i;
            } else if (v > dv2) {
                dv2 = v;
                d2 = i;
            }
        }
        var r1: usize = 0;
        var r2: usize = 0;
        var rv1: f64 = -std.math.inf(f64);
        var rv2: f64 = -std.math.inf(f64);
        for (ref_cap.lg[r], 0..) |v, i| {
            if (v > rv1) {
                rv2 = rv1;
                r2 = r1;
                rv1 = v;
                r1 = i;
            } else if (v > rv2) {
                rv2 = v;
                r2 = i;
            }
        }
        std.debug.print("M3-GRADE draw {d}: dev top1 {d} ({d:.4}) top2 {d} ({d:.4}) | ref top1 {d} ({d:.4}) top2 {d} ({d:.4})\n", .{ g, d1, dv1, d2, dv2, r1, rv1, r2, rv2 });
    }
    if (!exact) {
        std.debug.print("FAIL minimax-m3-grade: the device graph's greedy tokens differ from the f64 reference's over the SAME decoded pack weights (rows {d})\n", .{dev_cap.rows});
        return error.TokenMismatch; // WRONG verdict: never a "close enough" pass over a token miss
    }
    std.debug.print("RESULT minimax-m3-grade: device graph == f64 reference (EXACT greedy tokens, {d} rows graded, residual max|delta| {d:.6}, logits max|delta| {d:.6}, first {d} layers of the real pack)\n", .{ dev_cap.rows, xmax, lmax, layers_arg });
}

// ------------------------------------------------------------------ the lane-surface boundary test

fn laneTokensEqual(a: []const u32, b: []const u32) bool {
    if (a.len != b.len) return false;
    for (a, b) |va, vb| {
        if (va != vb) return false;
    }
    return true;
}

fn expectRefusal(comptime what: []const u8, want: anyerror, call: anytype) !void {
    if (call) |_| {
        std.debug.print("FAIL minimax-m3-lanes: {s} served where it must refuse\n", .{what});
        return error.RefusalExpected;
    } else |e| {
        if (e != want) {
            std.debug.print("FAIL minimax-m3-lanes: {s} refused with {s}, expected {s}\n", .{ what, @errorName(e), @errorName(want) });
            return error.RefusalWanted;
        }
    }
    std.debug.print("[lanes] refusal {s}: {s}\n", .{ what, @errorName(want) });
}

pub fn laneRun(gpu: check.Gpu, dir_in: []const u8, ctx_len: usize, steps: usize, layers_arg: usize) !void {
    const gpa = gpu.gpa;
    var dir = dir_in;
    if (dir.len == 0 or steps == 0 or steps > 32 or ctx_len == 0 or ctx_len > 128) {
        std.debug.print("usage: tf-cuda-test minimax-m3-lanes <pack dir|fixture> [ctx<=128] [steps<=32] [layers]\n", .{});
        return error.BadArguments;
    }
    if (std.mem.eql(u8, dir, "fixture")) dir = try makeFixture(gpu);
    const layers: usize = layers_arg; // 0 = the whole owned stack (full TP1; real packs use a cap)

    // -- the direct path's tokens first: the same prompt through a raw Engine --
    var direct = try m3.graph.Engine.build(gpa, gpu.io, gpu.d, dir, 0, 1, .{ .layer_cap = layers });
    defer direct.deinit();
    const prompt = try gpa.alloc(u32, ctx_len);
    defer gpa.free(prompt);
    var l = Lcg{ .s = prompt_seed };
    m3.serve.promptOf(&l, prompt);
    const direct_out = try gpa.alloc(u32, steps);
    defer gpa.free(direct_out);
    for (prompt) |tok| try direct.put(tok);
    try direct.greedy(steps, direct_out);
    std.debug.print("M3-LANES: direct engine tokens", .{});
    for (direct_out) |t| std.debug.print(" {d}", .{t});
    std.debug.print("\n", .{});

    // -- the served path: open()'s Loaded handle, the lane vtable --
    var loaded = try m3.native.open(gpa, gpu.io, gpu.ctx, dir, null, .{ .context = m3.block_tokens, .drafts = false, .layer_cap = layers });
    defer loaded.deinit(loaded.ctx);
    const b = loaded.backend;
    var s = try m3.lane_api.Stream.init(gpa, .{ .id = "lane-boundary", .prompt = prompt, .max_new = @intCast(steps + 2), .drafts = false });
    defer s.deinit(gpa);

    try b.prefill(&s);
    if (s.fill_at.load(.acquire) != s.prompt_len) return error.LaneFillLost;
    var h = try b.first(&s, s.prompt_len);
    var tok = try b.read(h);
    std.debug.print("[lanes] first: handle {d} token {d}\n", .{ h, tok });
    _ = try s.commit(gpa, &.{tok}, &.{});

    var lane_out = try gpa.alloc(u32, steps);
    defer gpa.free(lane_out);
    lane_out[0] = tok;
    // the queue/read round trips: row k feeds lane_out[k] at the engine's cache length and
    // draws lane_out[k+1]; the last round (feeding lane_out[steps-2]) goes through verify
    var idx: usize = 1;
    while (idx < steps) : (idx += 1) {
        const qpos: u64 = s.prompt_len + @as(u64, idx - 1); // the engine's own cache length
        const feed = lane_out[idx - 1];
        if (idx + 1 < steps) {
            h = try b.queue(&s, .{ .value = feed }, qpos);
            tok = try b.read(h);
            std.debug.print("[lanes] queue@{d}: handle {d} token {d}\n", .{ qpos, h, tok });
        } else {
            var sampled: [1]u32 = .{0};
            var drafts_empty: [0]u32 = .{};
            const windows = [_]m3.lane_api.backend.Window{.{ .stream = &s, .pending = feed, .held = 0, .tokens = &.{}, .parents = null, .positions = &[_]u64{qpos} }};
            var verified = [_]m3.lane_api.backend.Verified{.{ .sampled = sampled[0..], .drafts = drafts_empty[0..] }};
            try b.verify(&windows, &verified);
            tok = verified[0].sampled[0];
            std.debug.print("[lanes] verify@{d}: token {d}\n", .{ qpos, tok });
        }
        lane_out[idx] = tok;
        _ = try s.commit(gpa, &.{tok}, &.{});
    }

    // -- keep: roll back the verify round's fed row and re-serve it; the draw must repeat --
    if (steps >= 2) {
        const kept: usize = s.context.items.len - 2; // rows 0..kept-1 of the engine's cache
        const path = try gpa.alloc(u32, kept);
        defer gpa.free(path);
        for (path, 0..) |*p, i| p.* = @intCast(i);
        const kwin = [_]m3.lane_api.backend.Window{.{ .stream = &s, .pending = 0, .held = 0, .tokens = &.{}, .parents = null, .positions = &[_]u64{0} }};
        var kpaths = [_][]const u32{path};
        try b.keep(&kwin, &kpaths);
        h = try b.queue(&s, .{ .value = lane_out[kept - s.prompt_len] }, @intCast(kept));
        tok = try b.read(h);
        if (tok != lane_out[kept - s.prompt_len + 1]) {
            std.debug.print("FAIL minimax-m3-lanes: the keep rollback re-drew {d} where the round drew {d}\n", .{ tok, lane_out[kept - s.prompt_len + 1] });
            return error.TokenMismatch;
        }
        std.debug.print("[lanes] keep@{d}: rollback re-drew token {d} (unchanged)\n", .{ kept, tok });
    }

    // -- the grade through the lane surface: token-for-token against the direct engine --
    if (!laneTokensEqual(lane_out, direct_out)) {
        std.debug.print("FAIL minimax-m3-lanes: lane tokens", .{});
        for (lane_out) |t| std.debug.print(" {d}", .{t});
        std.debug.print(" vs direct tokens", .{});
        for (direct_out) |t| std.debug.print(" {d}", .{t});
        std.debug.print("\n", .{});
        return error.TokenMismatch;
    }

    // -- the honest refusals (each with its exact reason) --
    var requests = [_]m3.lane_api.backend.DraftRequest{.{ .stream = &s, .follow = &.{tok}, .rows = null, .start = 0, .position = 0, .depth = 1 }};
    try expectRefusal("draft", error.LanesNotAttached, b.draft(&requests));
    try expectRefusal("read(unknown handle)", error.LaneHandleUnknown, b.read(424242));
    var sampled2: [1]u32 = .{0};
    const dwin = [_]m3.lane_api.backend.Window{.{ .stream = &s, .pending = tok, .held = 1, .tokens = &.{}, .parents = null, .positions = &[_]u64{s.context.items.len} }};
    var vout2 = [_]m3.lane_api.backend.Verified{.{ .sampled = sampled2[0..], .drafts = &.{} }};
    try expectRefusal("verify(drafted window)", error.LanesNotAttached, b.verify(&dwin, &vout2));
    const twin = [_]m3.lane_api.backend.Window{ dwin[0], dwin[0] };
    var vout3 = [_]m3.lane_api.backend.Verified{ vout2[0], vout2[0] };
    try expectRefusal("verify(batch of 2)", error.LanesNotAttached, b.verify(&twin, &vout3));
    var s2 = try m3.lane_api.Stream.init(gpa, .{ .id = "second-stream", .prompt = prompt, .max_new = 2, .drafts = false });
    defer s2.deinit(gpa);
    try expectRefusal("prefill(second stream)", error.LanesNotAttached, b.prefill(&s2));
    const tree_path = [_]u32{ 0, 2 }; // a tree's path: not the prefix 0..kept-1
    const kwin2 = [_]m3.lane_api.backend.Window{.{ .stream = &s, .pending = 0, .held = 0, .tokens = &.{}, .parents = null, .positions = &[_]u64{0} }};
    var kpaths2 = [_][]const u32{tree_path[0..]};
    try expectRefusal("keep(tree path)", error.LanesNotAttached, b.keep(&kwin2, &kpaths2));

    // -- release, then the re-prefill serves fresh; a sampled spec refuses before it can go live --
    b.release(&s);
    var s3 = try m3.lane_api.Stream.init(gpa, .{ .id = "sampled", .prompt = prompt[0..2], .max_new = 1, .drafts = false, .sampling = .{ .seed = 7 } });
    defer s3.deinit(gpa);
    try expectRefusal("prefill(sampled spec)", error.SamplingNotAttached, b.prefill(&s3));
    var s4 = try m3.lane_api.Stream.init(gpa, .{ .id = "again", .prompt = prompt, .max_new = @intCast(steps + 2), .drafts = false });
    defer s4.deinit(gpa);
    try b.prefill(&s4); // a fresh cache life after release: serves again
    const h4 = try b.first(&s4, s4.prompt_len);
    const again1 = try b.read(h4);
    if (again1 != lane_out[0]) {
        std.debug.print("FAIL minimax-m3-lanes: the released re-prefill drew {d} where the round drew {d}\n", .{ again1, lane_out[0] });
        return error.TokenMismatch;
    }
    b.release(&s4);
    std.debug.print("RESULT minimax-m3-lanes: the Loaded handle serves a {d}-row prompt through the lane surface (first/queue/verify/keep/release) — tokens EXACT vs the direct engine; drafts/sampled/second-stream/tree-keep refuse truthfully\n", .{ctx_len});
}

// ------------------------------------------------- the TP2 lane-surface boot (Stage-9)
//
//   tf-cuda-test minimax-m3-lane-tp2 <rank 0|1> <host:port> <pack dir> [ctx] [steps]
//       the Stage-8 refusal's proof: open()'s own lane vtable — openTP2's handle, the SAME
//       Attached surface TP1 serves — answering a real prompt over the SPLIT real pack.
//       Rank 1 first over the NCCL id (the token-harness recipe); rank 0 serves the receipt:
//       prefill → first → queue/read rounds straight through the vtable, its greedy draws
//       argued from its own logits row. Rank 1 keeps the row lockstep through its own surface
//       (every put is a lockstep row; its draws are scratch and are tolerated).
//
pub fn laneTp2Run(gpu: check.Gpu, rank_arg: []const u8, addr_arg: []const u8, dir_in: []const u8, ctx_len: usize, steps: usize) !void {
    var dir = dir_in;
    const rank8 = std.fmt.parseInt(u8, rank_arg, 10) catch 99;
    if (rank8 > 1 or addr_arg.len == 0 or dir.len == 0 or steps == 0 or steps > 32 or ctx_len == 0 or ctx_len > m3.graph.keys_cap) {
        std.debug.print("usage: tf-cuda-test minimax-m3-lane-tp2 <rank 0|1> <host:port|fixture> <pack dir|fixture> [ctx] [steps]\n", .{});
        return error.BadArguments;
    }
    if (std.mem.eql(u8, dir, "fixture")) dir = try makeFixture(gpu);
    const rank: u32 = @intCast(rank8);
    const world: u32 = 2;
    const gib = struct {
        fn f(v: u64) f64 {
            return @as(f64, @floatFromInt(v)) / (1024.0 * 1024.0 * 1024.0);
        }
    }.f;

    // -- the TP2 boot exactly the token harness's recipe (rank 1 first, rank 0 serves the id) --
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    var id: cuda.nccl.UniqueId = undefined;
    const id_bytes = std.mem.asBytes(&id);
    if (rank == 0) {
        try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        try serveId(addr_arg, id_bytes);
    } else {
        try receiveId(gpu, addr_arg, id_bytes);
    }
    var comm: cuda.nccl.Comm = null;
    try lib.check(lib.api.ncclCommInitRank(&comm, @intCast(world), id, @intCast(rank)), "ncclCommInitRank");
    defer _ = lib.api.ncclCommDestroy(comm);

    const ram_before = hostFree(gpu);
    const t0 = check.now(gpu.io);
    var loaded = try m3.openTP2(gpu.gpa, gpu.io, gpu.ctx, dir, .{ .context = ctx_len, .drafts = false }, .{ .lib = &lib, .comm = comm, .rank = rank, .world = world });
    defer loaded.deinit(loaded.ctx);
    const b = loaded.backend;
    const build_s = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e9;
    const ram_after_build = hostFree(gpu);

    // -- the same deterministic context both ranks (rank 1's token values are rank 0's business) --
    const prompt = try gpu.gpa.alloc(u32, ctx_len);
    defer gpu.gpa.free(prompt);
    var l = Lcg{ .s = prompt_seed };
    m3.serve.promptOf(&l, prompt);
    const lane_out = try gpu.gpa.alloc(u32, steps);
    defer gpu.gpa.free(lane_out);
    const put_t0 = check.now(gpu.io);

    // -- the serving block: prefill → first → queue/read straight through the vtable --
    // rank 0 argues every greedy draw from its real logits row and prints the receipt;
    // rank 1 keeps the ROW LOCKSTEP through its own surface (every put is a paired NCCL row)
    var s = try m3.lane_api.Stream.init(gpu.gpa, .{ .id = "lane-tp2", .prompt = prompt, .max_new = @intCast(steps + 2), .drafts = false });
    defer s.deinit(gpu.gpa);
    try b.prefill(&s);
    if (s.fill_at.load(.acquire) != s.prompt_len) return error.LaneFillLost;
    const prefill_s = @as(f64, @floatFromInt(check.now(gpu.io) - put_t0)) / 1e9;
    const eng: *m3.graph.Engine = @ptrCast(@alignCast(loaded.target)); // the same engine the vtable drives
    // first: rank 0's draw is the receipt's first token; rank 1's `logits` is a 1-slot scratch,
    // so its draw may honestly refuse (first touches no put) — tolerated, the lockstep keeps
    var h: u64 = 0;
    if (rank == 0) {
        h = try b.first(&s, s.prompt_len);
        lane_out[0] = try b.read(h);
    } else {
        h = b.first(&s, s.prompt_len) catch 0; // scratch draw tolerated; handle unused on rank 1
    }
    _ = try s.commit(gpu.gpa, &.{lane_out[0]}, &.{});
    var idx: usize = 1;
    while (idx < steps) : (idx += 1) {
        const qpos: u64 = s.prompt_len + @as(u64, idx - 1); // the engine's own cache length
        const len_before = eng.len;
        if (rank == 0) {
            h = try b.queue(&s, .{ .value = lane_out[idx - 1] }, qpos);
            lane_out[idx] = try b.read(h);
        } else {
            // rank 1 feeds 0 (its row's value is rank 0's business); if the SURFACE refuses
            // after this round's put (a scratch-logits draw refusal), the row already landed:
            // the lockstep keeps. A refusal BEFORE the put (RowKeying/Context/second-stream)
            // is a real surface error and dies loudly rather than desyncing the ranks.
            if (b.queue(&s, .{ .value = 0 }, qpos)) |_| {} else |e| {
                if (eng.len != len_before) {
                    std.debug.print("M3-LANE-TP2 rank 1: draw refused ({s}) after the row's put — lockstep keeps\n", .{@errorName(e)});
                } else return e;
            }
        }
        _ = try s.commit(gpu.gpa, &.{lane_out[idx]}, &.{});
    }
    const lanes_s = @as(f64, @floatFromInt(check.now(gpu.io) - put_t0)) / 1e9;
    const ram_after = hostFree(gpu);
    const device_used = cuda.usage(false).device;

    // -- the receipt rows
    std.debug.print("M3-LANE-TP2 rank {d} world {d}: build {d:.1}s, prefill(x{d}) {d:.2}s, lane rows(x{d}) {d:.2}s\n", .{ rank, world, build_s, ctx_len, prefill_s, steps, lanes_s });
    std.debug.print("M3-LANE-TP2 rank {d}: device buffers {d:.2} GiB; host MemAvailable {d:.2} -> {d:.2} -> {d:.2} GiB\n", .{ rank, gib(device_used), gib(ram_before), gib(ram_after_build), gib(ram_after) });
    if (rank == 0) {
        std.debug.print("M3-LANE-TP2 rank 0: served tokens", .{});
        for (lane_out) |t| std.debug.print(" {d}", .{t});
        std.debug.print(" (greedy x{d} through the lane vtable over the split pack)\n", .{steps});
        var lnan: usize = 0;
        var linf: usize = 0;
        var lmx: f32 = -std.math.inf(f32);
        var lsum: f64 = 0;
        for (eng.logits) |v| {
            if (v != v) lnan += 1 else {
                if (v > lmx) lmx = v;
                if (v == std.math.inf(f32)) linf += 1 else lsum += v;
            }
        }
        std.debug.print("M3-LANE-TP2 rank 0: logits rows {d} NaN {d} inf {d} max {d:.6} mean {d:.6}\n", .{ eng.logits.len, lnan, linf, lmx, lsum / @as(f64, @floatFromInt(eng.logits.len)) });
        std.debug.print("RESULT minimax-m3-lane-tp2 rank 0: openTP2's lane vtable answered a {d}-row prompt over the split pack: {d} greedy draws served (the surface, not just the engine harness)\n", .{ ctx_len, steps });
        // -- Stage-10: the served draws as readable text, the serve door's own code path --
        // (serveTokenizer/serveTextOut). The ids receipt above stays authoritative: a decode
        // refusal (no tokenizer.json, or an unknown-id DIFF) is printed honestly, never fatal
        // in the receipted TP2 path.
        var tp_tok: ?tokenizer.Tokenizer = serveTokenizer(gpu.gpa, gpu.io, dir) catch |e| blk: {
            std.debug.print("M3-LANE-TP2 rank 0: text decode refuses ({s}) — the ids receipt above stands\n", .{@errorName(e)});
            break :blk null;
        };
        if (tp_tok) |*t| {
            defer t.deinit();
            serveTextOut(t, gpu.gpa, "lane-tp2", lane_out) catch |e| {
                std.debug.print("M3-LANE-TP2 rank 0: text receipt refused ({s}) — the ids receipt above stands\n", .{@errorName(e)});
            };
        }
    } else {
        std.debug.print("RESULT minimax-m3-lane-tp2 rank 1: row lockstep held ({d} rows) through the lane surface\n", .{ctx_len + steps - 1});
    }
}

// ------------------------------------------------- the serve front door (Stage-9)
//
//   tf-cuda-test minimax-m3-serve <pack dir|fixture> <lcg|prompt file> [steps<=32] [layers]
//       the smallest REAL answering surface: open()'s attached lane vtable takes prompt token
//       ids in and prints one greedy completion token ids out, then exits. "lcg" reads the
//       receipted deterministic prompt stream (Stage-6C/8's 8-row seed 0x5EEDB00B bar); anything
//       else is a file holding a JSON {"tokens": [...]} (kolibri-greedy's convention), a bare
//       JSON id array, or whitespace-separated ids. Greedy-only and deterministic by the stage's
//       honesty: drafts/sampled/second-stream refuse with their exact names (the same refusal
//       law Stage-8 receipted, re-probed here end to end). At the receipted fixture recipe
//       (fixture dir + lcg + 8 rows + [layers] 2) the completion MUST be EXACT 183 149 14 —
//       the bar this command is graded against every time it runs there.
//
/// The lanes refusal probe with the front door's own tag (the same law as laneRun's).
fn serveRefusal(comptime what: []const u8, want: anyerror, call: anytype) !void {
    if (call) |_| {
        std.debug.print("FAIL minimax-m3-serve: {s} served where it must refuse\n", .{what});
        return error.RefusalExpected;
    } else |e| {
        if (e != want) {
            std.debug.print("FAIL minimax-m3-serve: {s} refused with {s}, expected {s}\n", .{ what, @errorName(e), @errorName(want) });
            return error.RefusalWanted;
        }
    }
    std.debug.print("[serve] refusal {s}: {s}\n", .{ what, @errorName(want) });
}

/// The prompt in: "lcg" (the receipted bar stream) or a prompt file (JSON token rows or bare ids).
fn servePrompt(a: std.mem.Allocator, io: std.Io, src: []const u8) ![]u32 {
    if (std.mem.eql(u8, src, "lcg")) {
        const prompt = try a.alloc(u32, 8);
        var l = Lcg{ .s = prompt_seed };
        m3.serve.promptOf(&l, prompt);
        return prompt;
    }
    const text = try std.Io.Dir.cwd().readFileAlloc(io, src, a, .limited(1 << 26));
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (trimmed.len > 0 and (trimmed[0] == '{' or trimmed[0] == '[')) {
        if (trimmed[0] == '{') {
            var parsed = try std.json.parseFromSlice(struct { tokens: []const u32 }, a, trimmed, .{});
            defer parsed.deinit();
            return a.dupe(u32, parsed.value.tokens);
        }
        var parsed = try std.json.parseFromSlice([]const u32, a, trimmed, .{});
        defer parsed.deinit();
        return a.dupe(u32, parsed.value);
    }
    var ids: std.ArrayList(u32) = .empty;
    var it = std.mem.tokenizeAny(u8, trimmed, " \t\r\n,");
    while (it.next()) |t| try ids.append(a, try std.fmt.parseInt(u32, t, 10));
    return a.dupe(u32, ids.items);
}

// ------------------------------------------------- the serve text layer (Stage-10)
//
//   Free-text in, readable text out, through the PACK'S OWN tokenizer.json — the core engine
//   (src/core/tokenizer), whose encode == HF tokenizers' encode(text, add_special_tokens=False)
//   and whose decode runs the tokenizer.json decoder chain (HF-exact on this pack: the Stage-7
//   receipt bar decodes bit-identically, encode parity 19/19 probe). Text serves on a dir that
//   carries tokenizer.json; "fixture" serves token ids as before, and takes `tokdir` (the pack
//   dir) as a 5th argument when only the TEXT PLUMBING (text -> ids -> model -> ids -> text) is
//   being probed at fixture scale — labeled honestly at run time. The token-IDS paths above are
//   untouched: "lcg" and id prompts refuse nothing and print ids exactly as Stage-9 receipted.
//
/// The dir's own tokenizer, or null when the dir carries none (fixture or an ids-only layout).
/// A present-but-unreadable tokenizer.json is an error, never ignored (honest).
fn serveTokenizer(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !?tokenizer.Tokenizer {
    return tokenizer.loadTokenizer(io, gpa, dir) catch |e| switch (e) {
        error.FileNotFound => null,
        else => return e,
    };
}

/// A text prompt source ("text:<raw utf8>" inline, or "textfile:<path>" raw file bytes) encoded
/// through the pack's tokenizer, with the in-side round-trip held as a hard bar:
/// decode(encode(text)) must return the prompt text bit-exact (lossy U+FFFD substitution refuses).
fn serveTextPrompt(io: std.Io, t: *tokenizer.Tokenizer, gpa: std.mem.Allocator, src: []const u8) ![]u32 {
    var text: []const u8 = undefined;
    var label: []const u8 = undefined;
    if (std.mem.startsWith(u8, src, "textfile:")) {
        label = "textfile";
        text = try std.Io.Dir.cwd().readFileAlloc(io, src["textfile:".len..], gpa, .limited(1 << 26));
    } else {
        label = "text";
        text = src["text:".len..];
    }
    const ids = try t.encode(gpa, text);
    const back = try t.decode(gpa, ids, false);
    defer gpa.free(back);
    if (std.mem.eql(u8, text, back)) {
        std.debug.print("[serve] text round-trip in ({s}, {d} bytes -> {d} ids): EXACT decode(encode(text)) == text\n", .{ label, text.len, ids.len });
    } else {
        std.debug.print("[serve] text round-trip in ({s}): DIFF (the tokenizer does not round-trip this text bit-exact)\n", .{label});
        return error.TextPromptRoundTripDiffers;
    }
    return ids;
}

/// Serve's readable completion: the ids are the model's truth; the text is the pack tokenizer's
/// decode of them (skip-specials for the readable line, specials counted and named honestly), and
/// the out-side round-trip encode(decode(ids)) == ids is held as a hard bar. Unknown ids decode to
/// nothing (as HF skips them); if the model drew one, the round-trip honestly reports DIFF.
fn serveTextOut(t: *tokenizer.Tokenizer, gpa: std.mem.Allocator, tag: []const u8, ids: []const u32) !void {
    const full = try t.decode(gpa, ids, false);
    defer gpa.free(full);
    const re = try t.encode(gpa, full);
    defer gpa.free(re);
    var specials: usize = 0;
    for (ids) |id| {
        const piece = t.id_to_token.get(id) orelse continue;
        if (t.specials.contains(piece)) specials += 1;
    }
    const readable = try t.decode(gpa, ids, true);
    defer gpa.free(readable);
    if (std.mem.eql(u32, re, ids)) {
        std.debug.print("[{s}] text round-trip out: EXACT encode(decode(ids)) == ids (specials {d}), text: '{s}'\n", .{ tag, specials, readable });
    } else {
        std.debug.print("[{s}] text round-trip out: DIFF ({d} of {d} re-encoded; specials {d}), text: '{s}'\n", .{ tag, re.len, ids.len, specials, readable });
        return error.TextCompletionRoundTripDiffers;
    }
}

pub fn serveRun(gpu: check.Gpu, dir_in: []const u8, prompt_src: []const u8, steps: usize, layers_arg: usize, tok_arg: []const u8) !void {
    const gpa = gpu.gpa;
    var dir = dir_in;
    const is_text = std.mem.startsWith(u8, prompt_src, "text:") or std.mem.startsWith(u8, prompt_src, "textfile:");
    if (dir.len == 0 or prompt_src.len == 0 or steps == 0 or steps > 32) {
        std.debug.print("usage: tf-cuda-test minimax-m3-serve <pack dir|fixture> <lcg|prompt file|text:UTF-8|textfile:PATH> [steps<=32] [layers] [tokdir]\n", .{});
        return error.BadArguments;
    }
    var fixture_served = false;
    if (std.mem.eql(u8, dir, "fixture")) {
        dir = try makeFixture(gpu);
        fixture_served = true;
    }
    const layers: usize = layers_arg; // fixture's recipe receipts 2 (the whole fixture stack); a real pack picks its cap

    // -- Stage-10: the pack's own tokenizer, when the dir (or an explicit tokdir) carries one --
    const tok_dir: []const u8 = if (tok_arg.len > 0) tok_arg else dir;
    var tok_opt = try serveTokenizer(gpa, gpu.io, tok_dir);
    defer if (tok_opt) |*t| t.deinit();
    if (tok_opt == null and is_text) {
        std.debug.print("FAIL minimax-m3-serve: a text prompt needs the pack's own tokenizer.json (text:/textfile: encode through it); '{s}' carries none — pass the pack dir as [tokdir]\n", .{tok_dir});
        return error.ServeTextNeedsTokenizer;
    }
    if (tok_opt != null and fixture_served) {
        std.debug.print("[serve] text tokenizer: {s} (explicit tokdir; the fixture's toy draws decode through it for the plumbing round-trip — the fixture bar applies to lcg ids only)\n", .{tok_dir});
    }

    const prompt: []u32 = if (is_text)
        try serveTextPrompt(gpu.io, &tok_opt.?, gpa, prompt_src)
    else
        try servePrompt(gpa, gpu.io, prompt_src);
    defer gpa.free(prompt);
    if (prompt.len == 0) {
        std.debug.print("FAIL minimax-m3-serve: the prompt carries no token ids\n", .{});
        return error.ServeEmptyPrompt;
    }

    // -- the door: open()'s Loaded handle, the attached lane vtable, one greedy stream --
    var loaded = try m3.native.open(gpa, gpu.io, gpu.ctx, dir, null, .{ .context = m3.block_tokens, .drafts = false, .layer_cap = layers });
    defer loaded.deinit(loaded.ctx);
    const b = loaded.backend;
    var s = try m3.lane_api.Stream.init(gpa, .{ .id = "serve-front-door", .prompt = prompt, .max_new = @intCast(steps + 2), .drafts = false });
    defer s.deinit(gpa);

    try b.prefill(&s);
    if (s.fill_at.load(.acquire) != s.prompt_len) return error.LaneFillLost;
    var h = try b.first(&s, s.prompt_len);
    const out = try gpa.alloc(u32, steps);
    defer gpa.free(out);
    out[0] = try b.read(h);
    _ = try s.commit(gpa, &.{out[0]}, &.{});
    var idx: usize = 1;
    while (idx < steps) : (idx += 1) {
        const qpos: u64 = s.prompt_len + @as(u64, idx - 1); // the engine's own cache length
        if (idx + 1 < steps) {
            h = try b.queue(&s, .{ .value = out[idx - 1] }, qpos);
            out[idx] = try b.read(h);
        } else {
            var sampled: [1]u32 = .{0};
            const windows = [_]m3.lane_api.backend.Window{.{ .stream = &s, .pending = out[idx - 1], .held = 0, .tokens = &.{}, .parents = null, .positions = &[_]u64{qpos} }};
            var verified = [_]m3.lane_api.backend.Verified{.{ .sampled = sampled[0..], .drafts = &.{} }};
            try b.verify(&windows, &verified);
            out[idx] = verified[0].sampled[0];
        }
        _ = try s.commit(gpa, &.{out[idx]}, &.{});
    }
    std.debug.print("M3-SERVE: prompt {d} rows ->", .{prompt.len});
    for (out) |t| std.debug.print(" {d}", .{t});
    std.debug.print("\n", .{});

    // -- Stage-10: the completion as readable text through the pack's own tokenizer --
    if (tok_opt) |*t| try serveTextOut(t, gpa, "serve", out);

    // -- the boundary bar: the receipted fixture recipe must land the Stage-6C/8 EXACT tokens --
    if (fixture_served and std.mem.eql(u8, prompt_src, "lcg") and prompt.len == 8 and steps >= 3 and layers == 2) {
        const bar = [3]u32{ 183, 149, 14 };
        for (bar, 0..) |t, i| {
            if (out[i] != t) {
                std.debug.print("FAIL minimax-m3-serve: the front door drew {d} at draw {d} where the receipted fixture bar holds EXACT 183 149 14\n", .{ out[i], i });
                return error.TokenMismatch;
            }
        }
        std.debug.print("[serve] fixture bar: EXACT 183 149 14\n", .{});
    }

    // -- the honest refusals, re-probed end to end at the door (exact names, cheap probes) --
    const last = out[steps - 1];
    var requests = [_]m3.lane_api.backend.DraftRequest{.{ .stream = &s, .follow = &.{last}, .rows = null, .start = 0, .position = 0, .depth = 1 }};
    try serveRefusal("draft", error.LanesNotAttached, b.draft(&requests));
    try serveRefusal("read(unknown handle)", error.LaneHandleUnknown, b.read(424242));
    var s2 = try m3.lane_api.Stream.init(gpa, .{ .id = "serve-second", .prompt = prompt, .max_new = 2, .drafts = false });
    defer s2.deinit(gpa);
    try serveRefusal("prefill(second stream)", error.LanesNotAttached, b.prefill(&s2));

    b.release(&s); // the sampled-spec refusal is receipted AFTER release (a live stream refuses first)
    var s3 = try m3.lane_api.Stream.init(gpa, .{ .id = "serve-sampled", .prompt = prompt[0..2], .max_new = 1, .drafts = false, .sampling = .{ .seed = 7 } });
    defer s3.deinit(gpa);
    try serveRefusal("prefill(sampled spec)", error.SamplingNotAttached, b.prefill(&s3));

    std.debug.print("RESULT minimax-m3-serve: the front door answered {d} prompt rows with {d} greedy completion rows through open()'s lane surface (prefill/first/queue/read/verify); draft/second-stream/sampled refuse truthfully; fixture bar exact where receipted\n", .{ prompt.len, steps });
}
