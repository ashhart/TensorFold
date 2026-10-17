//! tf-cuda-test kernel-contract: the registered CUDA kernels through gates 1-4, the planted controls, and gate 5.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const options = @import("contract_options");
const check = @import("check.zig");
const rows_k = @import("contract_rows.zig");
const router_k = @import("contract_router.zig");
const mm_k = @import("contract_mm.zig");
const mx_k = @import("contract_mxfp8.zig");
const wide_k = @import("contract_projection_f32.zig");
const rank_k = @import("contract_rank.zig");
const writeback_k = @import("contract_flashnext_writeback.zig");
const utility_k = @import("contract_utility.zig");
const best_k = @import("contract_row_best.zig");
const qsa_k = @import("contract_qsa_select.zig");
const h3_k = @import("contract_h3.zig");
const h3_prep = @import("contract_h3_prep.zig");
const h3_topk_k = @import("contract_h3_topk.zig");

const contract = core.contract;
const g = contract.gates;
const limits = contract.limits;
const registry = contract.registry;
const kk = cuda.kernels;

const controls = struct {
    pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
};

const h3_image = struct {
    pub const bytes align(16) = @embedFile("fatbin_h3").*;
};

/// The embedded image a registration names (the controls' only when this binary was built with -Dnvcc).
fn image(name: []const u8) []const u8 {
    if (std.mem.eql(u8, name, "h3")) return &h3_image.bytes;
    inline for (.{ "norms", "nemotron_route", "qmm_group", "lane_gemv", "prompt16", "qmmf", "rank_sum", "prefill_attention", "vit", "embed_rows", "route_rows", "flashnext_gdn", "flashnext_batch", "flashnext_hc", "flashnext_attn", "flashnext_mtp", "qwen_vision", "exl3_linear", "exl3_experts", "exl3_candidate", "experts3", "experts3_prefill" }) |n| if (std.mem.eql(u8, name, n)) return @field(kk, n);
    return if (options.controls) &controls.bytes else &.{};
}

/// Gates 1-4 on every registered CUDA kernel and each control at its own gate, then gate 5 on this GPU.
pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    if (!kk.available) return error.BuiltWithoutKernels;
    if (!options.controls) {
        std.debug.print("FAIL kernel-contract: the controls are built only with -Dnvcc; without them no gate is proven\n", .{});
        return error.TestFailed;
    }
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var norms = try cuda.norms.Norms.load(gpu.d);
    defer norms.unload();
    var mods: [6]cuda.Module = undefined;
    var loaded: usize = 0;
    defer for (mods[0..loaded]) |*m| m.unload();
    for ([_][]const u8{ kk.nemotron_route, kk.qmm_group, kk.qmm_prefill, kk.lane_gemv, kk.affine4_pack, &controls.bytes }, 0..) |img, i| {
        mods[i] = try cuda.Module.load(gpu.d, img);
        loaded += 1;
    }
    const cap = try gpu.ctx.capability();
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var affine4 = try cuda.qlinear.Affine4Kernels.resolve(gpu.d, mods[1], mods[2], mods[3], mods[4], sms, cap == 121);
    defer affine4.deinit();
    var split = try cuda.qlinear.Split.init(gpu.d); // GB10 serves without it; other GPUs take it for K slices
    defer split.deinit();
    var prompt = try cuda.nvfp4.Prompt.load(gpu.d);
    defer prompt.unload();
    var lane = try cuda.qmmf.Lane.load(gpu.d, cap / 10); // sm_90 up: K slices meet in a cluster
    defer lane.unload();
    var lane_reduce = try cuda.qmmf.Lane.load(gpu.d, 8); // no clusters: the slices meet in reduce_kernel
    defer lane_reduce.unload();
    var rank_sum = try cuda.rank_scatter.RankScatter.load(gpu.d);
    defer rank_sum.unload();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(a);
    var row_list: [7]rows_k.Row = undefined;
    var nrows: usize = 0;
    defer for (row_list[0..nrows]) |*r| r.deinit();
    for (std.enums.values(rows_k.Which)) |w| {
        row_list[nrows] = try rows_k.Row.init(gpu, s, &norms, mods[5], w);
        nrows += 1;
    }
    for (row_list[0..nrows]) |*r| try kernels.append(a, r.kernel());
    var logits = router_k.Logits{ .rig = .{ .gpu = gpu, .s = s }, .f = try mods[0].function("tf_nemo_router") };
    defer logits.deinit();
    var topk = router_k.TopK{ .rig = .{ .gpu = gpu, .s = s }, .f = try mods[0].function("tf_nemo_topk") };
    defer topk.deinit();
    try kernels.append(a, logits.kernel());
    try kernels.append(a, topk.kernel());
    const kinds = comptime std.enums.values(mm_k.Kind);
    var mm: [kinds.len]mm_k.Mm = undefined;
    for (kinds, &mm) |kind, *m| {
        const by_lane: ?*const cuda.qmmf.Lane = switch (kind) {
            .lane_f32 => &lane,
            .lane_f32_reduce => &lane_reduce,
            else => null,
        };
        m.* = .{ .rig = .{ .gpu = gpu, .s = s }, .kind = kind, .sm = cap, .affine4 = &affine4, .split = &split, .prompt = &prompt, .lane = by_lane };
    }
    defer for (&mm) |*m| m.deinit();
    for (&mm) |*m| {
        const k = m.kernel() orelse {
            std.debug.print("N/A  cuda qmm_group (cluster tile): needs sm_90 or newer, this GPU is sm_{d}\n", .{cap});
            continue;
        };
        if (cap < 90 and (m.kind == .lane_gemv or m.kind == .lane_gemv_split))
            std.debug.print("N/A  {s} replacement bits: cluster baseline needs sm_90; gate 3 uses the float64 bound\n", .{k.spec.name});
        try kernels.append(a, k);
    }
    var mx = [_]mx_k.Product{
        .{ .rig = .{ .gpu = gpu, .s = s }, .lane = &lane, .clusters = true },
        .{ .rig = .{ .gpu = gpu, .s = s }, .lane = &lane_reduce, .clusters = false },
    };
    defer for (&mx) |*m| m.deinit();
    for (&mx) |*m| try kernels.append(a, m.kernel());
    var wide: [wide_k.modes.len * 2]wide_k.Product = undefined;
    for (wide_k.modes, 0..) |mode, i| {
        wide[i * 2] = .{ .rig = .{ .gpu = gpu, .s = s }, .lane = &lane, .mode = mode, .clusters = true };
        wide[i * 2 + 1] = .{ .rig = .{ .gpu = gpu, .s = s }, .lane = &lane_reduce, .mode = mode, .clusters = false };
    }
    defer for (&wide) |*m| m.deinit();
    for (&wide) |*m| try kernels.append(a, m.kernel());
    var rank = rank_k.Rank{ .rig = .{ .gpu = gpu, .s = s }, .sum = &rank_sum };
    defer rank.deinit();
    try kernels.append(a, rank.kernel());
    var hc = try cuda.Module.load(gpu.d, kk.flashnext_hc);
    defer hc.unload();
    var writebacks: [3]writeback_k.Writeback = undefined;
    for (std.enums.values(writeback_k.Mode), &writebacks) |mode, *w| {
        const symbol: [:0]const u8 = switch (mode) {
            .sums => "fn_hc_writeback0",
            .moe => "fn_hc_writeback_moe",
            .slices => "fn_hc_writeback_slices",
        };
        w.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try hc.function(symbol), .mode = mode };
    }
    defer for (&writebacks) |*w| w.deinit();
    for (&writebacks) |*w| try kernels.append(a, w.kernel());
    var batch = try cuda.Module.load(gpu.d, kk.flashnext_batch);
    defer batch.unload();
    var sums: [2]utility_k.Sum = undefined;
    var shifts: [2]utility_k.Shift = undefined;
    for ([_]bool{ false, true }, &sums, &shifts) |variant, *sum, *shift| {
        sum.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try batch.function("fn_sum_slices"), .bf16 = variant };
        shift.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try batch.function("fn_shift_window"), .padded = variant };
    }
    defer for (&sums) |*sum| sum.deinit();
    defer for (&shifts) |*shift| shift.deinit();
    for (&sums) |*sum| try kernels.append(a, sum.kernel());
    for (&shifts) |*shift| try kernels.append(a, shift.kernel());
    var mtp = try cuda.Module.load(gpu.d, kk.flashnext_mtp);
    defer mtp.unload();
    var best = best_k.Best{ .rig = .{ .gpu = gpu, .s = s }, .f = try mtp.function("fn_row_best") };
    defer best.deinit();
    try kernels.append(a, best.kernel());
    var selections: [2]qsa_k.Select = undefined;
    for ([_]bool{ false, true }, &selections) |padded, *selection|
        selection.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try batch.function("fn_qsa_select"), .padded = padded };
    defer for (&selections) |*selection| selection.deinit();
    for (&selections) |*selection| try kernels.append(a, selection.kernel());
    var h3_mod = try cuda.Module.load(gpu.d, &h3_image.bytes);
    defer h3_mod.unload();
    const h3_which = comptime std.enums.values(h3_k.Which);
    var h3_list: [h3_which.len]h3_k.H3 = undefined;
    for (h3_which, &h3_list) |w, *m| {
        const f = try h3_mod.function(h3_k.symbol(w));
        if (w == .attention) try f.allowDynamicShared(h3_prep.attention_shared);
        m.* = .{ .rig = .{ .gpu = gpu, .s = s }, .which = w, .f = f, .arena = std.heap.ArenaAllocator.init(a) };
    }
    defer for (&h3_list) |*m| m.deinit();
    for (&h3_list) |*m| try kernels.append(a, m.kernel());
    var h3_topk = h3_topk_k.TopK{ .rig = .{ .gpu = gpu, .s = s }, .f = try h3_mod.function("h3_topk") };
    defer h3_topk.deinit();
    try kernels.append(a, h3_topk.kernel());
    var t: g.Tally = .{};
    const t0 = check.now(gpu.io);
    try g.runAll(a, kernels.items, 0x6b63, &t);
    const t1 = check.now(gpu.io);
    const lim = try live(gpu);
    const t2 = check.now(gpu.io);
    std.debug.print("SUMMARY {d} kernels: {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: {d} red as planted, {d} missed; {d} limit problems; gates {d} ms, limits {d} ms\n", .{ t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed, lim, @divTrunc(t1 - t0, std.time.ns_per_ms), @divTrunc(t2 - t1, std.time.ns_per_ms) });
    try check.expect(t.problems() + lim == 0, "kernel-contract: {d} gate failures, {d} controls missed, {d} limit problems", .{ t.failed, t.missed, lim });
}

/// Gate 5 on this GPU: the table's row for it against the driver, each record against the loaded function.
fn live(gpu: check.Gpu) !usize {
    var n = try registry.checkCuda(gpu.gpa, &registry.cuda, false);
    std.debug.print("{s} limits: {d} registered CUDA launches against sm_80/86/89/120/121, {d} problems\n", .{ if (n == 0) "PASS" else "FAIL", registry.cuda.len, n });
    const sm: u32 = @intCast(try gpu.ctx.capability());
    const at = for (limits.sms, 0..) |t, i| {
        if (t.sm == sm) break i;
    } else {
        std.debug.print("N/A  limits live: sm_{d} is not a target the records cover\n", .{sm});
        return n;
    };
    const t = limits.sms[at];
    const dev = [_]struct { []const u8, cuda.abi.DeviceAttribute, u32 }{
        .{ "most shared bytes a block (opt-in)", .max_shared_memory_per_block_optin, t.smem_optin },
        .{ "threads a block", .max_threads_per_block, t.threads },
    };
    for (dev) |d| {
        const got: u32 = @intCast(try gpu.ctx.attribute(d[1]));
        if (got != d[2]) n += 1;
        std.debug.print("{s} limits live sm_{d} {s}: the driver says {d}, the table {d}\n", .{ if (got == d[2]) "PASS" else "FAIL", sm, d[0], got, d[2] });
    }
    var said: ?limits.Nvcc = null;
    for (registry.cuda ++ registry.cuda_controls) |r| {
        const control = std.mem.startsWith(u8, r.launch.name, "control");
        const bytes = image(r.launch.image);
        const rel = try built(gpu.gpa, bytes, sm);
        const set = registry.setFor(rel);
        if (said == null or !std.meta.eql(said.?, set)) records("live", rel, set);
        said = set;
        const want = registry.usageFor(r, rel)[at];
        var mod = try cuda.Module.load(gpu.d, bytes);
        defer mod.unload();
        const sym = try gpu.gpa.dupeSentinel(u8, r.launch.symbol, 0);
        defer gpu.gpa.free(sym);
        const f = try mod.function(sym);
        const regs: u32 = @intCast(try f.attribute(.num_regs));
        const static: u32 = @intCast(try f.attribute(.shared_size_bytes));
        const most: u32 = @intCast(try f.attribute(.max_threads_per_block));
        const fits = r.launch.threads <= most and static + r.launch.dynamic <= t.smem_optin;
        const current = want.regs == regs and want.static == static;
        if (!control and !(fits and current)) n += 1;
        if (control and fits) n += 1;
        const word = if (control) (if (fits) "MISSED" else "RED ") else if (fits and current) "PASS" else "FAIL";
        std.debug.print("{s} limits live {s}: {d} threads launched, the driver allows {d}; {d} registers, {d} + {d} shared bytes (recorded {d}, {d})\n", .{ word, r.launch.name, r.launch.threads, most, regs, static, r.launch.dynamic, want.regs, want.static });
    }
    return n;
}

/// The nvcc release that built a fatbin's image for `sm`, else the one its images agree on (null: none names one).
fn built(a: std.mem.Allocator, bytes: []const u8, sm: u32) !?limits.Nvcc {
    return contract.cubin.releaseFor(a, bytes, sm);
}

/// Says which compiler's records an audit compares against, and why.
fn records(what: []const u8, rel: ?limits.Nvcc, set: limits.Nvcc) void {
    if (rel) |v| {
        const own = std.meta.eql(v, set);
        std.debug.print("INFO limits {s}: images built by nvcc {d}.{d}, checked against nvcc {d}.{d}'s records{s}\n", .{ what, v.major, v.minor, set.major, set.minor, if (own) "" else " (the newest set at or below it)" });
    } else std.debug.print("INFO limits {s}: the images name no nvcc release; checked against nvcc {d}.{d}'s records\n", .{ what, set.major, set.minor });
}

/// tf-cuda-test kernel-contract-limits DIR: each record against the fatbins in DIR, every SM image they hold.
pub fn fatbins(gpu: check.Gpu, dir: []const u8) !void {
    const a = gpu.gpa;
    var n: usize = 0;
    var said: ?limits.Nvcc = null;
    for (registry.cuda ++ registry.cuda_controls) |r| {
        const control = std.mem.startsWith(u8, r.launch.name, "control");
        const path = try std.fmt.allocPrint(a, "{s}/{s}.fatbin", .{ dir, r.launch.image });
        defer a.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, a, .limited(1 << 28));
        defer a.free(bytes);
        const imgs = try contract.cubin.images(a, bytes);
        defer a.free(imgs);
        var red: usize = 0;
        var line: std.ArrayList(u8) = .empty;
        defer line.deinit(a);
        for (limits.sms, 0..) |t, i| {
            const img = for (imgs) |im| {
                if (im.sm == t.sm) break im;
            } else {
                std.debug.print("FAIL limits {s}: {s} holds no sm_{d} image (build with -Dsm=80,86,89,120,121)\n", .{ r.launch.name, path, t.sm });
                n += 1;
                continue;
            };
            const ks = try contract.cubin.kernels(a, img.elf, t.sm);
            defer a.free(ks);
            const k = for (ks) |kv| {
                if (std.mem.eql(u8, kv.name, r.launch.symbol)) break kv;
            } else return error.SymbolNotInImage;
            const rel = try contract.cubin.release(a, img.elf);
            const set = registry.setFor(rel);
            if (said == null or !std.meta.eql(said.?, set)) records("fatbins", rel, set);
            said = set;
            var out: std.ArrayList(u8) = .empty;
            defer out.deinit(a);
            red += try limits.cuda(r.launch, k.usage, t, &out, a);
            const current = std.meta.eql(k.usage, registry.usageFor(r, rel)[i]);
            if (!control and !current) n += 1;
            std.debug.print("{s} limits sm_{d} {s}: {d} registers, {d} static shared bytes, bound {d} (record {s}){s}", .{ if (out.items.len > 0) (if (control) "RED " else "FAIL") else "PASS", t.sm, r.launch.name, k.usage.regs, k.usage.static, k.usage.bound, if (current) "current" else "stale", if (out.items.len > 0) ": " else "\n" });
            if (out.items.len > 0) std.debug.print("{s}", .{out.items});
            try line.print(a, ".{{ .regs = {d}, .static = {d}, .bound = {d} }}, ", .{ k.usage.regs, k.usage.static, k.usage.bound });
        }
        std.debug.print("RECORD {s}: .usage = .{{ {s}}}\n", .{ r.launch.name, line.items });
        if (control and red == 0) n += 1;
        if (!control) n += red;
    }
    try check.expect(n == 0, "kernel-contract-limits: {d} problems", .{n});
    check.pass("kernel-contract-limits: every registered launch fits sm_80, sm_86, sm_89, sm_120 and sm_121, every control red", .{});
}
