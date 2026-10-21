//! exl3-prefill: the pieces of a Flash Next EXL3 prefill, timed on the pack's real tensors (one idle card).
//!
//!   exl3-prefill DIR experts [LAYER]   upstream's grouped experts (Experts.runSlots) on one fully resident layer:
//!                                      ms a call at 16..1408 rows (Zipf and uniform picks), then one 1024-row block
//!                                      split into expert-major passes over masked picks (bytes equal to one call)
//!   exl3-prefill DIR copy [LAYER]      expert lines from a pinned bank: Lines.load (copy + table writes) vs one bare
//!                                      H2D of the same bytes, and copies running beside a 1024-row experts call
//!   exl3-prefill DIR prompt [LAYER]    the prefill experts kernel (flashnext_exl3_prefill.cu, 64-member tiles) vs
//!                                      runSlots: y against y at 16..1408 rows, ms at 16..8192 rows, expert-major passes
//!   exl3-prefill DIR engine IDS N,N.. [LINES] [cmp]  the family's Engine with the offload pool (as serve attaches it):
//!                                      prefill rows/s of the first N ids of IDS (u32 little-endian token ids, real text)
//!                                      through the expert-major deep chunks; `cmp` also runs the batched pass on the
//!                                      same prompt and compares the last row's logits
//!   exl3-prefill DIR agree IDS FIRST STEP COUNT [LINES]  last-row logits of COUNT prompt lengths FIRST, FIRST+STEP..:
//!                                      the batched pass vs the deep path (env options as set) and vs the text's own
//!                                      next token (top-1 agreement, next-token accuracy, |d| on the batched top-5)
//!   exl3-prefill DIR dense             exl3.Dense prompt rows: reference 128-row blocks vs #806 tiles, 16..2048 rows,
//!                                      each against the other and rows 0-1 against an exact float64 product
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const flash = @import("flashnext");
const pk_ = flash.exl3.prefill_k;

const st = core.safetensors;
const exl3fmt = core.exl3_format;
const exl3 = cuda.exl3;
const exl3e = cuda.exl3_experts;
const Gpu = check.Gpu;

const dims = 2560;
const width = 640;
const routed_n = 512;
const table_n = 513;
const top_k = 10;
const slots = 11;
const shard_count = 7;
const projs = [3][]const u8{ "gate_proj", "up_proj", "down_proj" };
/// The group kernel stages a call's picks in 64 KiB of shared memory: rows * 11 * 4 bytes.
const max_call_rows: usize = 1408;

const Pack = struct {
    files: [shard_count]st.File = undefined,
    open_n: usize = 0,

    fn open(p: *Pack, gpu: Gpu, dir: []const u8) !void {
        var buf: [512]u8 = undefined;
        for (0..shard_count) |i| {
            const path = try std.fmt.bufPrint(&buf, "{s}/model-{d:0>5}-of-{d:0>5}.safetensors", .{ dir, i + 1, shard_count });
            p.files[i] = try st.File.open(gpu.gpa, gpu.io, path);
            p.open_n += 1;
        }
    }

    fn close(p: *Pack, io: std.Io) void {
        for (p.files[0..p.open_n]) |*f| f.close(io);
    }

    fn get(p: *const Pack, comptime fmt: []const u8, args: anytype) !st.Tensor {
        var buf: [200]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, fmt, args);
        for (p.files[0..p.open_n]) |*f| if (f.get(name)) |t| return t;
        std.debug.print("missing tensor {s}\n", .{name});
        return error.MissingTensor;
    }
};

fn bf16Bits(v: f32) u16 {
    const b: u32 = @bitCast(v);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

fn bfValue(bits: u16) f64 {
    return @floatCast(@as(f32, @bitCast(@as(u32, bits) << 16)));
}

const Timer = struct {
    a: cuda.Event,
    b: cuda.Event,

    fn init(d: *const cuda.Driver) !Timer {
        return .{ .a = try cuda.Event.init(d, true), .b = try cuda.Event.init(d, true) };
    }
    fn deinit(t: *Timer) void {
        t.a.deinit();
        t.b.deinit();
    }
    /// Best of 5 batches of `reps` calls, ms a call (after 2 warm-up calls).
    fn best(t: *Timer, s: cuda.Stream, reps: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx)) anyerror!void) !f64 {
        for (0..2) |_| try f(ctx);
        try s.synchronize();
        var b: f64 = std.math.inf(f64);
        for (0..5) |_| {
            try t.a.record(s);
            for (0..reps) |_| try f(ctx);
            try t.b.record(s);
            try t.b.synchronize();
            b = @min(b, @as(f64, try t.a.elapsedMs(t.b)) / @as(f64, @floatFromInt(reps)));
        }
        return b;
    }
};

// ---------------------------------------------------------------------------------------------------- one layer

/// One layer's 513 experts: the mmapped trellis, the stacked scales and every expert resident on the device.
const LayerData = struct {
    layer: usize,
    routed: [routed_n][3][]const u8,
    shared: [3][]const u8,
    scale_bufs: [6]cuda.DeviceBuffer,
    res_mem: cuda.DeviceBuffer,
    res_tab: cuda.DeviceBuffer,
    res: exl3e.Layer,
    line_bytes: usize,

    fn load(gpu: Gpu, pack: *const Pack, layer: usize) !*LayerData {
        const gpa = gpu.gpa;
        const ld = try gpa.create(LayerData);
        errdefer gpa.destroy(ld);
        ld.layer = layer;
        const sizes = [6]usize{ dims, dims, width, width, width, dims }; // suh_g suh_u svh_g svh_u suh_d svh_d
        var host: [6][]u8 = undefined;
        for (&host, sizes) |*h, n| h.* = try gpa.alloc(u8, table_n * n * 2);
        defer for (host) |h| gpa.free(h);
        var total: usize = 0;
        ld.line_bytes = 0;
        for (0..table_n) |e| {
            var line: usize = 0;
            for (projs, 0..) |pn, p| {
                const base = "model.language_model.layers.{d}.mlp.";
                const tr = if (e < routed_n) try pack.get(base ++ "experts.{d}.{s}.trellis", .{ layer, e, pn }) else try pack.get(base ++ "shared_expert.{s}.trellis", .{ layer, pn });
                const su = if (e < routed_n) try pack.get(base ++ "experts.{d}.{s}.suh", .{ layer, e, pn }) else try pack.get(base ++ "shared_expert.{s}.suh", .{ layer, pn });
                const sv = if (e < routed_n) try pack.get(base ++ "experts.{d}.{s}.svh", .{ layer, e, pn }) else try pack.get(base ++ "shared_expert.{s}.svh", .{ layer, pn });
                const k: usize = if (p == 2) width else dims;
                const n: usize = if (p == 2) dims else width;
                if (tr.rank != 3 or tr.dim(0) * 16 != k or tr.dim(1) * 16 != n) return error.BadShape;
                if (e < routed_n) ld.routed[e][p] = tr.bytes else ld.shared[p] = tr.bytes;
                total += (tr.bytes.len + 255) / 256 * 256;
                line += (tr.bytes.len + 255) / 256 * 256;
                const su_i: usize = switch (p) {
                    0 => 0,
                    1 => 1,
                    else => 4,
                };
                const sv_i: usize = switch (p) {
                    0 => 2,
                    1 => 3,
                    else => 5,
                };
                @memcpy(host[su_i][e * k * 2 ..][0 .. k * 2], su.bytes);
                @memcpy(host[sv_i][e * n * 2 ..][0 .. n * 2], sv.bytes);
            }
            if (e < routed_n) ld.line_bytes = @max(ld.line_bytes, line);
        }
        var made: usize = 0;
        errdefer for (ld.scale_bufs[0..made]) |*b| b.free();
        for (&ld.scale_bufs, host) |*b, h| {
            b.* = try cuda.DeviceBuffer.fromHost(gpu.d, h);
            made += 1;
        }
        ld.res_mem = try cuda.DeviceBuffer.alloc(gpu.d, total);
        errdefer ld.res_mem.free();
        var ptrs: [3][table_n]u64 = undefined;
        var k2s: [3][table_n]i32 = undefined;
        var at: usize = 0;
        const tiles = (dims / 16) * (width / 16);
        for (0..table_n) |e| for (0..3) |p| {
            const bytes = if (e < routed_n) ld.routed[e][p] else ld.shared[p];
            try ld.res_mem.upload(at, bytes);
            ptrs[p][e] = ld.res_mem.ptr + at;
            k2s[p][e] = @intCast(bytes.len / (tiles * 16));
            at += (bytes.len + 255) / 256 * 256;
        };
        const tl = cuda.exl3_pool.Tables.init(table_n);
        const tab = try gpa.alloc(u8, tl.bytes);
        defer gpa.free(tab);
        @memset(tab, 0);
        for (0..3) |p| {
            @memcpy(tab[tl.ptr[p]..][0 .. table_n * 8], std.mem.sliceAsBytes(&ptrs[p]));
            @memcpy(tab[tl.k2[p]..][0 .. table_n * 4], std.mem.sliceAsBytes(&k2s[p]));
        }
        ld.res_tab = try cuda.DeviceBuffer.fromHost(gpu.d, tab);
        ld.res = .{
            .gate = ld.res_tab.ptr + tl.ptr[0],
            .up = ld.res_tab.ptr + tl.ptr[1],
            .down = ld.res_tab.ptr + tl.ptr[2],
            .gate_k2 = ld.res_tab.ptr + tl.k2[0],
            .up_k2 = ld.res_tab.ptr + tl.k2[1],
            .down_k2 = ld.res_tab.ptr + tl.k2[2],
            .suh_g = ld.scale_bufs[0].ptr,
            .suh_u = ld.scale_bufs[1].ptr,
            .svh_g = ld.scale_bufs[2].ptr,
            .svh_u = ld.scale_bufs[3].ptr,
            .suh_d = ld.scale_bufs[4].ptr,
            .svh_d = ld.scale_bufs[5].ptr,
            .experts = table_n,
            .dims = dims,
            .width = width,
            .codebook = .mul1,
            .act_mode = 1,
        };
        std.debug.print("  layer {d}: 513 experts resident ({d:.0} MiB), routed k2 gate/up/down {d}/{d}/{d}, a routed line {d:.3} MiB\n", .{ layer, @as(f64, @floatFromInt(total)) / 1048576.0, k2s[0][0], k2s[1][0], k2s[2][0], @as(f64, @floatFromInt(ld.line_bytes)) / 1048576.0 });
        return ld;
    }

    fn free(ld: *LayerData, gpa: std.mem.Allocator) void {
        ld.res_tab.free();
        ld.res_mem.free();
        for (&ld.scale_bufs) |*b| b.free();
        gpa.destroy(ld);
    }
};

/// Picks for `rows` rows: ten distinct routed ids then the shared expert; Zipf(s) over a fixed permutation (s 0: uniform).
fn genPicks(rng: std.Random, s: f64, rows: usize, out: []i32) void {
    var perm: [routed_n]u16 = undefined;
    for (&perm, 0..) |*p, i| p.* = @intCast(i);
    var prng = std.Random.DefaultPrng.init(0x9e3779b9);
    prng.random().shuffle(u16, &perm);
    var cdf: [routed_n]f64 = undefined;
    var sum: f64 = 0;
    for (&cdf, 0..) |*c, i| {
        sum += if (s == 0) 1.0 else 1.0 / std.math.pow(f64, @floatFromInt(i + 1), s);
        c.* = sum;
    }
    for (&cdf) |*c| c.* /= sum;
    for (0..rows) |r| {
        const row = out[r * slots ..][0..slots];
        var n: usize = 0;
        while (n < top_k) {
            const u = rng.float(f64);
            var i: usize = 0;
            while (i + 1 < routed_n and cdf[i] < u) i += 1;
            const id: i32 = perm[i];
            if (std.mem.indexOfScalar(i32, row[0..n], id) != null) continue;
            row[n] = id;
            n += 1;
        }
        row[top_k] = 512;
    }
}

const SlotsCall = struct {
    ex: *const exl3e.Experts,
    s: cuda.Stream,
    l: exl3e.Layer,
    x: u64,
    picks: u64,
    rows: usize,
    sc: exl3e.Scratch,
    fn go(c: SlotsCall) anyerror!void {
        try c.ex.runSlots(c.s, c.l, c.x, dims, .bf16, c.picks, c.rows, c.sc);
    }
};

/// Expert-major passes over one block: pass j runs the block's picks of experts [j*per, (j+1)*per) (others -1).
const PassesCall = struct {
    ex: *const exl3e.Experts,
    s: cuda.Stream,
    l: exl3e.Layer,
    x: u64,
    masked: u64, // int32 [passes, rows * slots]
    rows: usize,
    passes: usize,
    sc: exl3e.Scratch,
    fn go(c: PassesCall) anyerror!void {
        for (0..c.passes) |j| try c.ex.runSlots(c.s, c.l, c.x, dims, .bf16, c.masked + j * c.rows * slots * 4, c.rows, c.sc);
    }
};

fn expertsRun(gpu: Gpu, ex: *const exl3e.Experts, s: cuda.Stream, ld: *const LayerData) !void {
    const a = gpu.gpa;
    var t = try Timer.init(gpu.d);
    defer t.deinit();
    const max_rows = max_call_rows;
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const rng = prng.random();
    const xh = try a.alloc(u16, max_rows * dims);
    defer a.free(xh);
    for (xh) |*b| b.* = bf16Bits((rng.float(f32) - 0.5) * 2.0);
    var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xh));
    defer x.free();
    const picks = try a.alloc(i32, max_rows * slots);
    defer a.free(picks);
    var pk = try cuda.DeviceBuffer.alloc(gpu.d, max_rows * slots * 4);
    defer pk.free();
    var sbuf = try cuda.DeviceBuffer.alloc(gpu.d, try exl3e.scratchBytes(ld.res, max_rows, slots));
    defer sbuf.free();
    const sc = try exl3e.scratch(sbuf.ptr, ld.res, max_rows, slots);
    std.debug.print("  runSlots, one resident layer (10 routed + shared a row), ms a call (best of 5 x 5):\n", .{});
    for ([_]f64{ 0.8, 0 }) |zs| {
        genPicks(rng, zs, max_rows, picks);
        try pk.upload(0, std.mem.sliceAsBytes(picks));
        for ([_]usize{ 16, 64, 256, 512, 1024, max_rows }) |rows| {
            var seen: [table_n]bool = @splat(false);
            var uniq: usize = 0;
            for (picks[0 .. rows * slots]) |p| if (!seen[@intCast(p)]) {
                seen[@intCast(p)] = true;
                uniq += 1;
            };
            const ms = try t.best(s, 5, SlotsCall{ .ex = ex, .s = s, .l = ld.res, .x = x.ptr, .picks = pk.ptr, .rows = rows, .sc = sc }, SlotsCall.go);
            std.debug.print("    {s:<8} rows {d:5}: {d:8.3} ms  {d:7.2} us/row  {d:6.0} rows/s/layer  ({d} experts incl. shared, {d:.1} member rows each)\n", .{ if (zs == 0) "uniform" else "zipf0.8", rows, ms, 1e3 * ms / @as(f64, @floatFromInt(rows)), @as(f64, @floatFromInt(rows)) / ms * 1e3, uniq, @as(f64, @floatFromInt(rows * slots)) / @as(f64, @floatFromInt(uniq)) });
        }
    }
    // expert-major passes over one 1024-row block (uniform picks): bytes equal to one call, and the time
    const rows: usize = 1024;
    genPicks(rng, 0, rows, picks);
    try pk.upload(0, std.mem.sliceAsBytes(picks[0 .. rows * slots]));
    const one_bytes = try a.alloc(u8, rows * slots * dims * 4);
    defer a.free(one_bytes);
    try (SlotsCall{ .ex = ex, .s = s, .l = ld.res, .x = x.ptr, .picks = pk.ptr, .rows = rows, .sc = sc }).go();
    try s.synchronize();
    try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(one_bytes.ptr, sc.y, one_bytes.len), "cuMemcpyDtoH");
    const base_ms = try t.best(s, 5, SlotsCall{ .ex = ex, .s = s, .l = ld.res, .x = x.ptr, .picks = pk.ptr, .rows = rows, .sc = sc }, SlotsCall.go);
    std.debug.print("  expert-major passes over one {d}-row block (uniform picks; pass j: experts [j*512/P, (j+1)*512/P) plus the shared one in pass 0):\n", .{rows});
    for ([_]usize{ 1, 2, 4, 8, 16, 32 }) |passes| {
        const per = routed_n / passes;
        const host = try a.alloc(i32, passes * rows * slots);
        defer a.free(host);
        for (0..passes) |j| for (picks[0 .. rows * slots], host[j * rows * slots ..][0 .. rows * slots]) |p, *o| {
            const in = if (p == 512) j == 0 else @as(usize, @intCast(p)) / per == j;
            o.* = if (in) p else -1;
        };
        var masked = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host));
        defer masked.free();
        try gpu.d.check(gpu.d.api.cuMemsetD8_v2(sc.y, 0xff, rows * slots * dims * 4), "cuMemsetD8");
        const call: PassesCall = .{ .ex = ex, .s = s, .l = ld.res, .x = x.ptr, .masked = masked.ptr, .rows = rows, .passes = passes, .sc = sc };
        try call.go();
        try s.synchronize();
        const got = try a.alloc(u8, one_bytes.len);
        defer a.free(got);
        try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(got.ptr, sc.y, got.len), "cuMemcpyDtoH");
        const same = std.mem.eql(u8, got, one_bytes);
        const ms = try t.best(s, 3, call, PassesCall.go);
        std.debug.print("    {d:2} passes of {d:3} experts: {d:8.3} ms (+{d:.3} ms vs one call, {d:.3} ms a pass)  y {s}\n", .{ passes, per, ms, ms - base_ms, ms / @as(f64, @floatFromInt(passes)), if (same) "bit-identical to one call" else "DIFFERS" });
        try check.expect(same, "masked passes reproduce one call's bytes", .{});
    }
    check.pass("layer {d}: runSlots timed at 16..{d} rows; expert-major masked passes are bit-identical to one call", .{ ld.layer, max_rows });
}

// ---------------------------------------------------------------------------------------------------- prompt kernel

const PromptCall = struct {
    k: *const pk_.Kernels,
    s: cuda.Stream,
    l: exl3e.Layer,
    x: u64,
    picks: u64,
    n_members: usize,
    n_items: usize,
    dv: pk_.Device,
    y: u64,
    fn go(c: PromptCall) anyerror!void {
        try pk_.run(c.k, c.s, c.l, c.x, dims, c.picks, slots, c.n_members, c.n_items, c.dv, c.y);
    }
};

/// P expert-major passes, each its own uploaded plan (members, items) over the same picks.
const PromptPasses = struct {
    calls: []const PromptCall,
    fn go(c: PromptPasses) anyerror!void {
        for (c.calls) |call| try call.go();
    }
};

fn promptRun(gpu: Gpu, ex: *const exl3e.Experts, s: cuda.Stream, ld: *const LayerData) !void {
    const a = gpu.gpa;
    var k = try pk_.Kernels.load(gpu.d);
    defer k.unload();
    var t = try Timer.init(gpu.d);
    defer t.deinit();
    const max_rows: usize = 8192;
    const max_pairs = max_rows * slots;
    var prng = std.Random.DefaultPrng.init(0x9a11);
    const rng = prng.random();
    const xh = try a.alloc(u16, max_rows * dims);
    defer a.free(xh);
    for (xh) |*b| b.* = bf16Bits((rng.float(f32) - 0.5) * 2.0);
    var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xh));
    defer x.free();
    const picks = try a.alloc(i32, max_pairs);
    defer a.free(picks);
    genPicks(rng, 0.8, max_rows, picks);
    var pkd = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(picks));
    defer pkd.free();
    var plan = try pk_.Plan.init(a, max_pairs, table_n);
    defer plan.deinit(a);
    var counts: [table_n]u32 = undefined;
    var mem_d = try cuda.DeviceBuffer.alloc(gpu.d, max_pairs * 4);
    defer mem_d.free();
    var items_d = try cuda.DeviceBuffer.alloc(gpu.d, 3 * pk_.maxItems(max_pairs, table_n) * 4);
    defer items_d.free();
    var scr = try cuda.DeviceBuffer.alloc(gpu.d, pk_.scratchBytes(max_pairs, dims, width));
    defer scr.free();
    var yk = try cuda.DeviceBuffer.alloc(gpu.d, max_pairs * dims * 4);
    defer yk.free();
    // runSlots' reference (at most max_call_rows rows a call)
    var sbuf = try cuda.DeviceBuffer.alloc(gpu.d, try exl3e.scratchBytes(ld.res, max_call_rows, slots));
    defer sbuf.free();
    const sc = try exl3e.scratch(sbuf.ptr, ld.res, max_call_rows, slots);
    std.debug.print("  prefill experts kernel (64-member tiles, compact pairs) vs runSlots, one resident layer, zipf0.8 picks:\n", .{});
    for ([_]usize{ 16, 64, 256, 1024, 1408, 2048, 4096, 8192 }) |rows| {
        const pairs = rows * slots;
        pk_.build(picks[0..pairs], null, &counts, &plan);
        try mem_d.upload(0, plan.memberBytes());
        try items_d.upload(0, plan.itemBytes());
        const dv: pk_.Device = .{ .members = mem_d.ptr, .items = items_d.ptr, .scratch = scr.ptr };
        const call: PromptCall = .{ .k = &k, .s = s, .l = ld.res, .x = x.ptr, .picks = pkd.ptr, .n_members = plan.n_members, .n_items = plan.n_items, .dv = dv, .y = yk.ptr };
        try call.go();
        try s.synchronize();
        const ms_k = try t.best(s, 3, call, PromptCall.go);
        var ms_r: f64 = std.math.nan(f64);
        var rel: f64 = std.math.nan(f64);
        var same_words: usize = 0;
        if (rows <= max_call_rows) {
            const rc: SlotsCall = .{ .ex = ex, .s = s, .l = ld.res, .x = x.ptr, .picks = pkd.ptr, .rows = rows, .sc = sc };
            try rc.go();
            try s.synchronize();
            const want = try a.alloc(f32, pairs * dims);
            defer a.free(want);
            const got = try a.alloc(f32, pairs * dims);
            defer a.free(got);
            try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(want.ptr, sc.y, want.len * 4), "cuMemcpyDtoH");
            try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(got.ptr, yk.ptr, got.len * 4), "cuMemcpyDtoH");
            var dmax: f64 = 0;
            var peak: f64 = 0;
            for (got, want) |g, w| {
                dmax = @max(dmax, @abs(@as(f64, g) - w));
                peak = @max(peak, @abs(@as(f64, w)));
                same_words += @intFromBool(@as(u32, @bitCast(g)) == @as(u32, @bitCast(w)));
            }
            rel = dmax / peak;
            ms_r = try t.best(s, 3, rc, SlotsCall.go);
            try check.expect(rel < 1e-3, "prefill kernel within fp32 sum-order noise of runSlots ({e:.2})", .{rel});
        }
        std.debug.print("    rows {d:5}: prefill kernel {d:8.3} ms ({d:6.2} us/row, {d} items)  runSlots {d:8.3} ms ({d:6.2} us/row)  speedup {d:.2}x  max|dy|/peak {e:.2}, {d:.1}% of words bit-equal\n", .{ rows, ms_k, 1e3 * ms_k / @as(f64, @floatFromInt(rows)), plan.n_items, ms_r, 1e3 * ms_r / @as(f64, @floatFromInt(rows)), ms_r / ms_k, rel, 100 * @as(f64, @floatFromInt(same_words)) / @as(f64, @floatFromInt(pairs * dims)) });
    }
    // expert-major passes over 8192 rows: the plan of each pass alone, back to back
    const rows = max_rows;
    const pairs = rows * slots;
    std.debug.print("  expert-major passes over {d} rows (pass j: experts [j*512/P, (j+1)*512/P), the shared one in pass 0):\n", .{rows});
    for ([_]usize{ 1, 8, 16, 32 }) |passes| {
        const per = routed_n / passes;
        var calls: [32]PromptCall = undefined;
        var mems: [32]cuda.DeviceBuffer = undefined;
        var its: [32]cuda.DeviceBuffer = undefined;
        var made: usize = 0;
        defer for (mems[0..made], its[0..made]) |*m, *i| {
            m.free();
            i.free();
        };
        for (0..passes) |j| {
            var take: [table_n]bool = @splat(false);
            for (j * per..(j + 1) * per) |id| take[id] = true;
            take[512] = j == 0;
            pk_.build(picks[0..pairs], &take, &counts, &plan);
            mems[j] = try cuda.DeviceBuffer.fromHost(gpu.d, plan.memberBytes());
            its[j] = try cuda.DeviceBuffer.fromHost(gpu.d, plan.itemBytes());
            made += 1;
            calls[j] = .{ .k = &k, .s = s, .l = ld.res, .x = x.ptr, .picks = pkd.ptr, .n_members = plan.n_members, .n_items = plan.n_items, .dv = .{ .members = mems[j].ptr, .items = its[j].ptr, .scratch = scr.ptr }, .y = yk.ptr };
        }
        const ms = try t.best(s, 2, PromptPasses{ .calls = calls[0..passes] }, PromptPasses.go);
        std.debug.print("    {d:2} passes: {d:8.3} ms ({d:6.2} us/row)\n", .{ passes, ms, 1e3 * ms / @as(f64, @floatFromInt(rows)) });
    }
    check.pass("layer {d}: prefill experts kernel within fp32 order noise of runSlots, timed to {d} rows", .{ ld.layer, max_rows });
}

// ---------------------------------------------------------------------------------------------------- copies

fn copyRun(gpu: Gpu, ex: *const exl3e.Experts, s: cuda.Stream, ld: *const LayerData) !void {
    const a = gpu.gpa;
    // the layer's routed trellis in one pinned bank (what S9 keeps), lines laid out as Lines does
    var total: usize = 0;
    for (ld.routed) |e| for (e) |b| {
        total += b.len;
    };
    var bank = try cuda.HostBuffer.alloc(gpu.d, total);
    defer bank.free();
    const pinned = try a.alloc([3][]const u8, routed_n);
    defer a.free(pinned);
    var at: usize = 0;
    for (ld.routed, pinned) |e, *o| for (e, 0..) |b, p| {
        @memcpy(bank.bytes[at..][0..b.len], b);
        o[p] = bank.bytes[at..][0..b.len];
        at += b.len;
    };
    const n_lines: usize = 64;
    var bytes: [3]usize = .{ 0, 0, 0 };
    var k2: [routed_n][3]i32 = undefined;
    const tiles = (dims / 16) * (width / 16);
    for (pinned, &k2) |e, *w| for (0..3) |p| {
        bytes[p] = @max(bytes[p], e[p].len);
        w[p] = @intCast(e[p].len / (tiles * 16));
    };
    var lines = try cuda.exl3_pool.Lines.init(a, gpu.d, .{ .experts = table_n, .bytes = bytes, .lines = n_lines, .staging = 0 });
    defer lines.deinit();
    var c0 = try cuda.Event.init(gpu.d, true);
    defer c0.deinit();
    var c1 = try cuda.Event.init(gpu.d, true);
    defer c1.deinit();
    var raw = try cuda.DeviceBuffer.alloc(gpu.d, n_lines * lines.layout.line_bytes);
    defer raw.free();
    const moved = blk: {
        var m: usize = 0;
        for (pinned[0..n_lines]) |e| m += e[0].len + e[1].len + e[2].len;
        break :blk m;
    };
    std.debug.print("  {d} expert lines ({d:.2} MiB, {d:.3} MiB a line) from the pinned bank, best of 5:\n", .{ n_lines, @as(f64, @floatFromInt(moved)) / 1048576.0, @as(f64, @floatFromInt(moved)) / 1048576.0 / n_lines });
    var best_load: f64 = std.math.inf(f64);
    var best_raw: f64 = std.math.inf(f64);
    var best_raw1: f64 = std.math.inf(f64);
    for (0..5) |it| {
        try lines.drain();
        try c0.record(lines.copy);
        for (0..n_lines) |i| {
            const id = (it * n_lines + i) % routed_n;
            const prev = (id + routed_n - n_lines) % routed_n;
            try lines.load(i, id, pinned[id], k2[id], if (it == 0) null else prev, true, gpu.io);
        }
        try c1.record(lines.copy);
        try c1.synchronize();
        best_load = @min(best_load, try c0.elapsedMs(c1));
        // the same bytes as three H2D a line, no table writes
        try c0.record(lines.copy);
        for (0..n_lines) |i| for (0..3) |p| {
            const e = pinned[i];
            try gpu.d.check(gpu.d.api.cuMemcpyHtoDAsync_v2(raw.ptr + i * lines.layout.line_bytes + lines.layout.off[p], e[p].ptr, e[p].len, lines.copy.handle), "cuMemcpyHtoDAsync");
        };
        try c1.record(lines.copy);
        try c1.synchronize();
        best_raw = @min(best_raw, try c0.elapsedMs(c1));
        // one H2D of the bank's first n_lines experts (contiguous in the bank)
        try c0.record(lines.copy);
        try gpu.d.check(gpu.d.api.cuMemcpyHtoDAsync_v2(raw.ptr, bank.bytes.ptr, moved, lines.copy.handle), "cuMemcpyHtoDAsync");
        try c1.record(lines.copy);
        try c1.synchronize();
        best_raw1 = @min(best_raw1, try c0.elapsedMs(c1));
    }
    const gbs = struct {
        fn f(b: usize, ms: f64) f64 {
            return @as(f64, @floatFromInt(b)) / ms / 1e6;
        }
    }.f;
    std.debug.print("    Lines.load (3 H2D + 6-9 table memsets + event a line): {d:.3} ms = {d:.3} ms a line, {d:.2} GB/s\n", .{ best_load, best_load / n_lines, gbs(moved, best_load) });
    std.debug.print("    3 bare H2D a line:                                       {d:.3} ms = {d:.3} ms a line, {d:.2} GB/s\n", .{ best_raw, best_raw / n_lines, gbs(moved, best_raw) });
    std.debug.print("    one H2D of the same bytes:                               {d:.3} ms, {d:.2} GB/s\n", .{ best_raw1, gbs(moved, best_raw1) });

    // copies beside compute: a 1024-row experts call on the resident layer while the copy stream moves 64 lines
    const rows: usize = 1024;
    var prng = std.Random.DefaultPrng.init(0xc0b1);
    const rng = prng.random();
    const xh = try a.alloc(u16, rows * dims);
    defer a.free(xh);
    for (xh) |*b| b.* = bf16Bits((rng.float(f32) - 0.5) * 2.0);
    var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xh));
    defer x.free();
    const picks = try a.alloc(i32, rows * slots);
    defer a.free(picks);
    genPicks(rng, 0, rows, picks);
    var pk = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(picks));
    defer pk.free();
    var sbuf = try cuda.DeviceBuffer.alloc(gpu.d, try exl3e.scratchBytes(ld.res, rows, slots));
    defer sbuf.free();
    const sc = try exl3e.scratch(sbuf.ptr, ld.res, rows, slots);
    const call: SlotsCall = .{ .ex = ex, .s = s, .l = ld.res, .x = x.ptr, .picks = pk.ptr, .rows = rows, .sc = sc };
    var k0 = try cuda.Event.init(gpu.d, true);
    defer k0.deinit();
    var k1 = try cuda.Event.init(gpu.d, true);
    defer k1.deinit();
    var alone_c: f64 = std.math.inf(f64);
    var alone_k: f64 = std.math.inf(f64);
    var both_c: f64 = std.math.inf(f64);
    var both_k: f64 = std.math.inf(f64);
    var both_w: f64 = std.math.inf(f64);
    for (0..5) |_| {
        try k0.record(s);
        try call.go();
        try k1.record(s);
        try k1.synchronize();
        alone_k = @min(alone_k, try k0.elapsedMs(k1));
        try c0.record(lines.copy);
        try gpu.d.check(gpu.d.api.cuMemcpyHtoDAsync_v2(raw.ptr, bank.bytes.ptr, moved, lines.copy.handle), "cuMemcpyHtoDAsync");
        try c1.record(lines.copy);
        try c1.synchronize();
        alone_c = @min(alone_c, try c0.elapsedMs(c1));
        // both at once
        try s.synchronize();
        const t0 = check.now(gpu.io);
        try c0.record(lines.copy);
        try gpu.d.check(gpu.d.api.cuMemcpyHtoDAsync_v2(raw.ptr, bank.bytes.ptr, moved, lines.copy.handle), "cuMemcpyHtoDAsync");
        try c1.record(lines.copy);
        try k0.record(s);
        try call.go();
        try k1.record(s);
        try k1.synchronize();
        try c1.synchronize();
        both_w = @min(both_w, @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e6);
        both_c = @min(both_c, try c0.elapsedMs(c1));
        both_k = @min(both_k, try k0.elapsedMs(k1));
    }
    std.debug.print("  overlap: {d}-row experts call alone {d:.3} ms, 64-line H2D alone {d:.3} ms; together: experts {d:.3} ms, H2D {d:.3} ms, wall {d:.3} ms (serial would be {d:.3})\n", .{ rows, alone_k, alone_c, both_k, both_c, both_w, alone_k + alone_c });
    check.pass("layer {d}: copy cost and copy/compute overlap measured", .{ld.layer});
}

// ---------------------------------------------------------------------------------------------------- dense

const DenseCall = struct {
    de: *exl3.Dense,
    s: cuda.Stream,
    w: exl3.Weight,
    x: u64,
    y: u64,
    rows: usize,
    fn go(c: DenseCall) anyerror!void {
        try c.de.prompt(c.s, c.w, c.x, .bf16, c.rows, c.y, .bf16);
    }
};

/// y [rows, N] = x [rows, K] . diag(suh) H Wq H diag(svh) in float64.
fn exact(a: std.mem.Allocator, trellis: []const i16, suh: []const f16, svh: []const f16, tk: usize, tn: usize, halves: u16, x: []const f64, rows: usize, y: []f64) !void {
    const k = tk * 16;
    const n = tn * 16;
    const wq = try a.alloc(f16, k * n);
    defer a.free(wq);
    exl3fmt.unpack(.{ .halves = halves }, .mul1, trellis, tk, tn, wq);
    const xs = try a.alloc(f64, rows * k);
    defer a.free(xs);
    for (0..rows) |r| for (0..k) |i| {
        xs[r * k + i] = x[r * k + i] * @as(f64, @floatCast(suh[i]));
    };
    exl3fmt.rotate(xs, rows, k, 1);
    @memset(y, 0);
    for (0..k) |ki| {
        const row_w = wq[ki * n ..][0..n];
        for (0..rows) |r| {
            const xv = xs[r * k + ki];
            const yr = y[r * n ..][0..n];
            for (yr, row_w) |*o, w| o.* += xv * @as(f64, @floatCast(w));
        }
    }
    exl3fmt.rotate(y, rows, n, 1);
    for (0..rows) |r| for (0..n) |c| {
        y[r * n + c] *= @as(f64, @floatCast(svh[c]));
    };
}

fn copyAs(comptime T: type, a: std.mem.Allocator, bytes: []const u8) ![]T {
    const out = try a.alloc(T, bytes.len / @sizeOf(T));
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. out.len * @sizeOf(T)]);
    return out;
}

fn denseRun(gpu: Gpu, pack: *const Pack, s: cuda.Stream) !void {
    const a = gpu.gpa;
    var de = try exl3.Dense.load(gpu.d);
    defer de.unload();
    var t = try Timer.init(gpu.d);
    defer t.deinit();
    const names = [_][]const u8{
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.in_proj_z",
        "model.language_model.layers.0.linear_attn.out_proj",
        "model.language_model.layers.3.self_attn.q_proj",
        "model.language_model.layers.3.self_attn.k_proj",
        "model.language_model.layers.3.self_attn.o_proj",
        "model.language_model.layers.3.mlp.shared_expert.gate_proj",
        "model.language_model.layers.3.mlp.shared_expert.down_proj",
    };
    const row_counts = [_]usize{ 16, 128, 256, 1024, 2048 };
    const max_rows: usize = 2048;
    var prng = std.Random.DefaultPrng.init(0xde45e);
    const rng = prng.random();
    var sum_ref: [row_counts.len]f64 = @splat(0);
    var sum_tiles: [row_counts.len]f64 = @splat(0);
    for (names) |name| {
        const tr = try pack.get("{s}.trellis", .{name});
        const su = try pack.get("{s}.suh", .{name});
        const sv = try pack.get("{s}.svh", .{name});
        const tk = tr.dim(0);
        const tn = tr.dim(1);
        const halves: u16 = @intCast(tr.dim(2) / 8);
        const k = tk * 16;
        const n = tn * 16;
        const trellis = try copyAs(i16, a, tr.bytes);
        defer a.free(trellis);
        const suh = try copyAs(f16, a, su.bytes);
        defer a.free(suh);
        const svh = try copyAs(f16, a, sv.bytes);
        defer a.free(svh);
        var trd = try cuda.DeviceBuffer.fromHost(gpu.d, tr.bytes);
        defer trd.free();
        var sud = try cuda.DeviceBuffer.fromHost(gpu.d, su.bytes);
        defer sud.free();
        var svd = try cuda.DeviceBuffer.fromHost(gpu.d, sv.bytes);
        defer svd.free();
        const strides = exl3.strides(k, n, halves, false);
        const w: exl3.Weight = .{ .trellis = trd.ptr, .suh = sud.ptr, .svh = svd.ptr, .k = @intCast(k), .n = @intCast(n), .halves = halves, .codebook = .mul1, .stride_k = strides.k, .stride_nb = strides.nb };
        const tiles_ok = if (exl3.Dense.candidateWeight(w)) |_| true else |_| false;
        const xh = try a.alloc(u16, max_rows * k);
        defer a.free(xh);
        for (xh) |*b| b.* = bf16Bits((rng.float(f32) - 0.5) * 2.0);
        var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xh));
        defer x.free();
        var yr = try cuda.DeviceBuffer.alloc(gpu.d, max_rows * n * 2);
        defer yr.free();
        var yt = try cuda.DeviceBuffer.alloc(gpu.d, max_rows * n * 2);
        defer yt.free();
        // f64 of rows 0 and 1
        const xf = try a.alloc(f64, 2 * k);
        defer a.free(xf);
        for (xf, xh[0 .. 2 * k]) |*o, b| o.* = bfValue(b);
        const want = try a.alloc(f64, 2 * n);
        defer a.free(want);
        try exact(a, trellis, suh, svh, tk, tn, halves, xf, 2, want);
        std.debug.print("  {s} K={d} N={d} halves={d}{s}\n", .{ name, k, n, halves, if (tiles_ok) "" else " (tiles refuse it)" });
        for (row_counts, 0..) |rows, ri| {
            de.implementation = .reference;
            const rc: DenseCall = .{ .de = &de, .s = s, .w = w, .x = x.ptr, .y = yr.ptr, .rows = rows };
            const ms_ref = try t.best(s, 3, rc, DenseCall.go);
            sum_ref[ri] += ms_ref;
            var ms_tiles: f64 = std.math.nan(f64);
            var diff: f64 = std.math.nan(f64);
            var err_ref: f64 = 0;
            var err_tiles: f64 = std.math.nan(f64);
            var peak: f64 = 0;
            const got_r = try a.alloc(u16, rows * n);
            defer a.free(got_r);
            try yr.download(0, std.mem.sliceAsBytes(got_r));
            for (got_r[0 .. 2 * n], want) |g, wv| {
                err_ref = @max(err_ref, @abs(bfValue(g) - wv));
                peak = @max(peak, @abs(wv));
            }
            if (tiles_ok) {
                de.implementation = .tiles;
                const tc: DenseCall = .{ .de = &de, .s = s, .w = w, .x = x.ptr, .y = yt.ptr, .rows = rows };
                ms_tiles = try t.best(s, 3, tc, DenseCall.go);
                sum_tiles[ri] += ms_tiles;
                const got_t = try a.alloc(u16, rows * n);
                defer a.free(got_t);
                try yt.download(0, std.mem.sliceAsBytes(got_t));
                diff = 0;
                var rpeak: f64 = 0;
                for (got_r, got_t) |g, h| {
                    diff = @max(diff, @abs(bfValue(g) - bfValue(h)));
                    rpeak = @max(rpeak, @abs(bfValue(g)));
                }
                diff /= rpeak;
                err_tiles = 0;
                for (got_t[0 .. 2 * n], want) |g, wv| err_tiles = @max(err_tiles, @abs(bfValue(g) - wv));
                de.implementation = .reference;
            } else sum_tiles[ri] += ms_ref;
            std.debug.print("    rows {d:5}: reference {d:8.4} ms ({d:6.2} us/row)  tiles {d:8.4} ms ({d:6.2} us/row)  max|ref-tiles|/peak {e:.2}  rows 0-1 vs f64: ref {e:.2} tiles {e:.2} (rel to peak {d:.3})\n", .{ rows, ms_ref, 1e3 * ms_ref / @as(f64, @floatFromInt(rows)), ms_tiles, 1e3 * ms_tiles / @as(f64, @floatFromInt(rows)), diff, err_ref / peak, err_tiles / peak, peak });
        }
    }
    std.debug.print("  sum over the {d} shapes (ms): ", .{names.len});
    for (row_counts, sum_ref, sum_tiles) |r, x, y| std.debug.print("rows {d}: reference {d:.3} best {d:.3}; ", .{ r, x, y });
    std.debug.print("\n", .{});
    check.pass("dense prompt kernels timed", .{});
}

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 2) return error.MissingArgument;
    const dir = rest[0];
    const mode = rest[1];
    if (std.mem.eql(u8, mode, "engine")) return engineRun(gpu, dir, rest[2..]);
    if (std.mem.eql(u8, mode, "agree")) return agreeRun(gpu, dir, rest[2..]);
    const layer: usize = if (rest.len > 2) try std.fmt.parseInt(usize, rest[2], 10) else 17;
    var pack: Pack = .{};
    defer pack.close(gpu.io);
    try pack.open(gpu, dir);
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    if (std.mem.eql(u8, mode, "dense")) return denseRun(gpu, &pack, s);
    var ex = try exl3e.Experts.load(gpu.d);
    defer ex.unload();
    const ld = try LayerData.load(gpu, &pack, layer);
    defer ld.free(gpu.gpa);
    if (std.mem.eql(u8, mode, "experts")) return expertsRun(gpu, &ex, s, ld);
    if (std.mem.eql(u8, mode, "copy")) return copyRun(gpu, &ex, s, ld);
    if (std.mem.eql(u8, mode, "prompt")) return promptRun(gpu, &ex, s, ld);
    return error.UnknownMode;
}

// ---------------------------------------------------------------------------------------------------- engine

const prompt = flash.prompt;
const xpass = flash.exl3.pass;
const xpre = flash.exl3.prefill;
const vocab = prompt.vocabulary;

fn promote(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

const Eng = struct {
    gpu: Gpu,
    stream: cuda.Stream,
    glue: flash.Glue,
    e: *prompt.Engine,
    seq: prompt.Seq,
    attached: xpass.Attached,

    fn init(gpu: Gpu, dir: []const u8, lines: ?usize, cap: usize, deep_rows: usize) !*Eng {
        const g = try gpu.gpa.create(Eng);
        errdefer gpu.gpa.destroy(g);
        g.gpu = gpu;
        g.stream = try cuda.Stream.init(gpu.d, true);
        g.glue = try flash.Glue.load(gpu.d);
        const t0 = check.now(gpu.io);
        g.e = try prompt.Engine.initRank(gpu.gpa, gpu.io, @constCast(gpu.d), &g.stream, dir, 1, 0);
        var free0: usize = 0;
        var total: usize = 0;
        try gpu.d.check(gpu.d.api.cuMemGetInfo_v2(&free0, &total), "cuMemGetInfo");
        _ = try g.e.reserve(@min(cap, g.e.chunk));
        _ = try g.e.reserveScores(cap, @min(cap, g.e.chunk));
        var free1: usize = 0;
        try gpu.d.check(gpu.d.api.cuMemGetInfo_v2(&free1, &total), "cuMemGetInfo");
        g.seq = try g.e.newSeq(cap);
        var free2: usize = 0;
        try gpu.d.check(gpu.d.api.cuMemGetInfo_v2(&free2, &total), "cuMemGetInfo");
        if (deep_rows > 0) try xpre.reserve(gpu.gpa, gpu.d, g.e, deep_rows);
        var free3: usize = 0;
        try gpu.d.check(gpu.d.api.cuMemGetInfo_v2(&free3, &total), "cuMemGetInfo");
        var b = try xpass.budgetFromEnv(0);
        if (lines) |n| b.lines = n;
        g.attached = try xpass.attach(gpu.gpa, gpu.io, gpu.d, g.stream, g.e.layers[0..g.e.n_layers], null, g.e.chunk, b);
        var free4: usize = 0;
        try gpu.d.check(gpu.d.api.cuMemGetInfo_v2(&free4, &total), "cuMemGetInfo");
        const gib = struct {
            fn f(x: usize) f64 {
                return @as(f64, @floatFromInt(x)) / (1 << 30);
            }
        }.f;
        std.debug.print("engine: loaded in {d:.1} s; card {d:.2} GiB; after weights {d:.2} free; pass scratch (chunk {d}) {d:.2} GiB; caches for {d} tokens {d:.2} GiB; deep chunks of {d} rows {d:.2} GiB; pool {d} lines a layer {d:.2} GiB; {d:.2} GiB free after\n", .{ @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e9, gib(total), gib(free0), g.e.chunk, gib(free0 - free1), cap, gib(free1 - free2), deep_rows, gib(free2 - free3), g.attached.lines, gib(g.attached.device_bytes), gib(free4) });
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

    fn poolStats(g: *Eng) flash.exl3_pool.Stats {
        return xpass.poolStats(g.e.layers[0..g.e.n_layers], null);
    }
};

fn top5(row: []const u16) [5]u32 {
    var ids: [5]u32 = @splat(0);
    var vals: [5]f32 = @splat(-std.math.inf(f32));
    for (row, 0..) |b, i| {
        const v = promote(b);
        if (!(v > vals[4])) continue;
        var at: usize = 4;
        while (at > 0 and v > vals[at - 1]) : (at -= 1) {
            vals[at] = vals[at - 1];
            ids[at] = ids[at - 1];
        }
        vals[at] = v;
        ids[at] = @intCast(i);
    }
    return ids;
}

/// One prefill of `ids` from a fresh sequence: ms, and the last row's logits copied into `out`.
fn prefillOnce(g: *Eng, ids: []const u32, deep_on: bool, out: []u16) !f64 {
    var o = xpre.optionsFromEnv();
    o.enabled = deep_on;
    if (rows_override) |r| o.rows = r;
    xpre.options = o;
    try g.reset();
    const t0 = check.now(g.gpu.io);
    try g.e.forward(flash.Tri, g.tri(), &g.seq, ids, .{ .logits = .last });
    const ms_ = @as(f64, @floatFromInt(check.now(g.gpu.io) - t0)) / 1e6;
    @memcpy(std.mem.sliceAsBytes(out), g.seq.logits);
    return ms_;
}

/// Greedy decode rounds after each engine-mode prefill (TF_TEST_DECODE, default 32; 0: none).
var decode_steps: usize = 32;

fn decodeAfter(g: *Eng, n: usize) !void {
    var out = try cuda.DeviceBuffer.alloc(g.gpu.d, vocab * 2);
    defer out.free();
    var tok = prompt.argmax(g.seq.logits);
    const before = g.poolStats();
    const t0 = check.now(g.gpu.io);
    var first: f64 = 0;
    for (0..decode_steps) |i| {
        const q = [_]*prompt.Seq{&g.seq};
        try g.e.round(flash.Tri, g.tri(), &q, (&tok)[0..1], out.ptr);
        tok = prompt.argmax(g.seq.logits);
        if (i == 7) first = @as(f64, @floatFromInt(check.now(g.gpu.io) - t0)) / 1e6;
    }
    const t = @as(f64, @floatFromInt(check.now(g.gpu.io) - t0)) / 1e6;
    const after = g.poolStats();
    const hits = after.hit_slots - before.hit_slots;
    const all = hits + after.miss_slots - before.miss_slots;
    std.debug.print("        decode after {d} tokens: {d:.2} ms/token over {d} steps (first 8: {d:.2} ms/token; hit {d:.1}%)\n", .{ n, t / @as(f64, @floatFromInt(decode_steps)), decode_steps, first / 8, 100 * @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(@max(all, 1))) });
}

/// The deep chunk's rows for the next prefills (engine mode's ROWS list), else the environment's.
var rows_override: ?usize = null;

fn engineRun(gpu: Gpu, dir: []const u8, rest: []const [:0]const u8) !void {
    const a = gpu.gpa;
    if (rest.len < 2) return error.MissingArgument;
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, rest[0], a, .limited(1 << 28));
    defer a.free(text);
    const all_ids = try a.alloc(u32, text.len / 4);
    defer a.free(all_ids);
    for (all_ids, 0..) |*x, i| x.* = std.mem.readInt(u32, text[4 * i ..][0..4], .little);
    var ns: [16]usize = undefined;
    var n_ns: usize = 0;
    var it = std.mem.splitScalar(u8, rest[1], ',');
    while (it.next()) |tok| {
        ns[n_ns] = try std.fmt.parseInt(usize, tok, 10);
        n_ns += 1;
    }
    const lines: ?usize = if (rest.len > 2 and !std.mem.eql(u8, rest[2], "-")) try std.fmt.parseInt(usize, rest[2], 10) else null;
    const cmp = rest.len > 3 and std.mem.eql(u8, rest[3], "cmp");
    // ROWS: deep chunk sizes to sweep (rest[3] when not "cmp", or rest[4])
    var rs: [16]usize = undefined;
    var n_rs: usize = 0;
    const rows_arg: ?[]const u8 = if (rest.len > 3 and !cmp) rest[3] else if (rest.len > 4) rest[4] else null;
    if (rows_arg) |ra| {
        var rit = std.mem.splitScalar(u8, ra, ',');
        while (rit.next()) |tok| {
            rs[n_rs] = try std.fmt.parseInt(usize, tok, 10);
            n_rs += 1;
        }
    }
    var max_n: usize = 0;
    for (ns[0..n_ns]) |n| max_n = @max(max_n, n);
    if (max_n > all_ids.len) return error.NotEnoughIds;
    if (std.c.getenv("TF_TEST_DECODE")) |v| decode_steps = std.fmt.parseInt(usize, std.mem.span(v), 10) catch decode_steps;
    const o = xpre.optionsFromEnv();
    var max_r = o.rows;
    for (rs[0..n_rs]) |r| max_r = @max(max_r, r);
    if (n_rs > 0) max_r = 0;
    for (rs[0..n_rs]) |r| max_r = @max(max_r, r);
    const deep_rows = @min(max_r, (max_n + 255) / 256 * 256);
    const g = try Eng.init(gpu, dir, lines, max_n + 64, if (o.enabled) deep_rows else 0);
    defer g.deinit();
    const old = try a.alloc(u16, vocab);
    defer a.free(old);
    const new = try a.alloc(u16, vocab);
    defer a.free(new);
    // warm-up: kernels, the pool's first lines
    _ = try prefillOnce(g, all_ids[0..@min(max_n, 1024)], o.enabled, new);
    var all_ok = true;
    if (n_rs == 0) {
        rs[0] = 0;
        n_rs = 1;
    }
    for (rs[0..n_rs]) |rr| for (ns[0..n_ns]) |n| {
        rows_override = if (rr == 0) null else rr;
        if (rr != 0) std.debug.print("deep chunk rows {d}:\n", .{rr});
        const ids = all_ids[0..n];
        const p0 = g.poolStats();
        xpre.stats = .{};
        const t_new = try prefillOnce(g, ids, o.enabled, new);
        const p1 = g.poolStats();
        const xs_ = xpre.stats;
        std.debug.print("PREFILL {d:6} tokens ({s}): {d:9.1} ms = {d:7.1} rows/s; pool copies {d} ({d:.1} GiB, {d:.2} a layer per 1k rows), deep chunks {d}, passes {d}, prefetched {d}, members {d}; host: picks wait {d:.1} ms, plan {d:.1} ms\n", .{ n, if (o.enabled) "expert-major deep" else "batched pass", t_new, @as(f64, @floatFromInt(n)) / (t_new / 1e3), p1.misses - p0.misses, @as(f64, @floatFromInt(p1.misses - p0.misses)) * 1.758 / 1024, @as(f64, @floatFromInt(p1.misses - p0.misses)) / 48 / (@as(f64, @floatFromInt(n)) / 1000), xs_.chunks, xs_.passes, xs_.prefetched, xs_.members, @as(f64, @floatFromInt(xs_.host_wait_ns)) / 1e6, @as(f64, @floatFromInt(xs_.plan_ns)) / 1e6 });
        // decode after the prefill: the pools were left empty by the ring (or hold the batched pass's lines)
        if (decode_steps > 0) try decodeAfter(g, n);
        if (!cmp) continue;
        const q0 = g.poolStats();
        const t_old = try prefillOnce(g, ids, false, old);
        const q1 = g.poolStats();
        std.debug.print("        same prompt, batched pass: {d:9.1} ms = {d:7.1} rows/s; pool copies {d}\n", .{ t_old, @as(f64, @floatFromInt(n)) / (t_old / 1e3), q1.misses - q0.misses });
        if (decode_steps > 0) try decodeAfter(g, n);
        var dmax: f32 = 0;
        var peak: f32 = 0;
        var same: usize = 0;
        for (old, new) |x, y| {
            dmax = @max(dmax, @abs(promote(x) - promote(y)));
            peak = @max(peak, @abs(promote(x)));
            same += @intFromBool(x == y);
        }
        const to = top5(old);
        const tn = top5(new);
        var overlap: usize = 0;
        for (to) |x| for (tn) |y| {
            overlap += @intFromBool(x == y);
        };
        const margin_old = promote(old[to[0]]) - promote(old[to[1]]);
        std.debug.print("CMP {d:6} tokens: last-row logits max |d| {d:.4} (peak {d:.2}, {d:.1}% of logits bit-equal); top-1 {d} vs {d} ({s}, batched-pass margin {d:.4}); top-5 overlap {d}/5\n", .{ n, dmax, peak, 100 * @as(f64, @floatFromInt(same)) / vocab, to[0], tn[0], if (to[0] == tn[0]) "equal" else "DIFFERENT", margin_old, overlap });
        if (to[0] != tn[0] and margin_old > 0.25) all_ok = false;
    };
    const ps = g.poolStats();
    std.debug.print("pool overall: {d} rounds, {d} copies, hit {d:.1}%\n", .{ ps.rounds, ps.misses, 100 * @as(f64, @floatFromInt(ps.hit_slots)) / @as(f64, @floatFromInt(@max(1, ps.hit_slots + ps.miss_slots))) });
    try check.expect(all_ok, "top-1 equal wherever the batched pass's margin exceeds bf16 noise", .{});
    check.pass("engine prefill measured", .{});
}

fn readIds(gpu: Gpu, path: []const u8) ![]u32 {
    const a = gpu.gpa;
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, a, .limited(1 << 28));
    defer a.free(text);
    const ids = try a.alloc(u32, text.len / 4);
    for (ids, 0..) |*x, i| x.* = std.mem.readInt(u32, text[4 * i ..][0..4], .little);
    return ids;
}

fn agreeRun(gpu: Gpu, dir: []const u8, rest: []const [:0]const u8) !void {
    const a = gpu.gpa;
    if (rest.len < 4) return error.MissingArgument;
    const all_ids = try readIds(gpu, rest[0]);
    defer a.free(all_ids);
    const first = try std.fmt.parseInt(usize, rest[1], 10);
    const step = try std.fmt.parseInt(usize, rest[2], 10);
    const count = try std.fmt.parseInt(usize, rest[3], 10);
    const lines: ?usize = if (rest.len > 4) try std.fmt.parseInt(usize, rest[4], 10) else null;
    const max_n = first + step * (count - 1);
    if (max_n + 1 > all_ids.len) return error.NotEnoughIds;
    const o = xpre.optionsFromEnv();
    const g = try Eng.init(gpu, dir, lines, max_n + 64, @min(o.rows, (max_n + 255) / 256 * 256));
    defer g.deinit();
    const old = try a.alloc(u16, vocab);
    defer a.free(old);
    const new = try a.alloc(u16, vocab);
    defer a.free(new);
    var agree: usize = 0;
    var acc_old: usize = 0;
    var acc_new: usize = 0;
    var top5_same: usize = 0;
    var sum_d: f64 = 0;
    var max_d: f64 = 0;
    var n_d: usize = 0;
    var disagree_margin_max: f32 = 0;
    var t_old: f64 = 0;
    var t_new: f64 = 0;
    for (0..count) |i| {
        const n = first + step * i;
        const ids = all_ids[0..n];
        t_old += try prefillOnce(g, ids, false, old);
        t_new += try prefillOnce(g, ids, o.enabled, new);
        const to = top5(old);
        const tn = top5(new);
        const want = all_ids[n];
        agree += @intFromBool(to[0] == tn[0]);
        acc_old += @intFromBool(to[0] == want);
        acc_new += @intFromBool(tn[0] == want);
        top5_same += @intFromBool(std.mem.eql(u32, &to, &tn));
        for (to) |id| {
            const d = @abs(@as(f64, promote(old[id])) - promote(new[id]));
            sum_d += d;
            max_d = @max(max_d, d);
            n_d += 1;
        }
        if (to[0] != tn[0]) {
            const m = promote(old[to[0]]) - promote(old[to[1]]);
            disagree_margin_max = @max(disagree_margin_max, m);
            std.debug.print("  length {d}: top-1 {d} (batched) vs {d} (deep), batched margin {d:.4}, deep margin {d:.4}\n", .{ n, to[0], tn[0], m, promote(new[tn[0]]) - promote(new[tn[1]]) });
        }
    }
    std.debug.print("AGREE {d} prompt lengths {d}..{d}: top-1 equal {d}/{d}, top-5 identical in order {d}/{d}; next-token accuracy batched {d}/{d}, deep {d}/{d}; |d| on the batched top-5: mean {d:.4}, max {d:.4}; largest batched margin where top-1 differs {d:.4}; time batched {d:.1} s, deep {d:.1} s\n", .{ count, first, max_n, agree, count, top5_same, count, acc_old, count, acc_new, count, sum_d / @as(f64, @floatFromInt(n_d)), max_d, disagree_margin_max, t_old / 1e3, t_new / 1e3 });
    check.pass("agreement measured", .{});
}
