//! tf-cuda-test kernel-contract: the images the records name, and gate 5 on this GPU.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const options = @import("contract_options");
const check = @import("check.zig");

const contract = core.contract;
const limits = contract.limits;
const registry = contract.registry;
const kk = cuda.kernels;

const controls = struct {
    pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
};

const h3_image = struct {
    pub const bytes align(16) = @embedFile("fatbin_h3").*;
};

/// The named embedded image; only contract_controls may select the planted controls.
pub fn image(name: []const u8) error{UnknownContractImage}![]const u8 {
    if (@import("contract_deepseek_images.zig").image(name)) |bytes| return bytes;
    if (std.mem.eql(u8, name, "window_rows")) return kk.window_rows;
    if (std.mem.eql(u8, name, "fp8_experts")) return kk.fp8_experts;
    if (std.mem.eql(u8, name, "h3")) return &h3_image.bytes;
    inline for (.{ "qwen4_projection", "qwen4_hc", "qwen4_gdn", "qwen4_ple" }) |n|
        if (std.mem.eql(u8, name, n)) return @field(kk, n);
    inline for (.{ "tree_round", "act_rows", "gdn_prep", "qk_rows", "norms", "moe", "experts", "qwen36_mtp", "nemotron_attention", "nemotron_route", "nemotron_latent", "qmm_group", "qmm_prefill", "lane_gemv", "prompt16", "qmmf", "ck_experts", "nvfp4_experts", "nvfp4_experts_tile", "rank_sum", "prefill_attention", "vit", "embed_rows", "route_rows", "flashnext_gdn", "flashnext_batch", "flashnext_hc", "flashnext_attn", "flashnext_mtp", "qwen_vision", "exl3_linear", "exl3_experts", "exl3_candidate", "experts3", "experts3_prefill", "drama3", "drama3_fp8", "drama3_codec", "glm_mm", "glm_dflash", "glm_glue", "glm_ops", "glm_index", "nemotron_mamba", "sample", "qwen35", "qwen35_bf16_rows", "qwen35_keep", "qwen35_kv8", "w4a8_prefill", "qwen35_draft", "ck_lane", "gdn_train", "train_q4", "train_attn", "flashnext_train", "train", "attn_train", "qwen27_train", "tree_attention", "block_attention", "window_attention", "window_pages", "moe_glue", "dense_rows", "qmm", "flashnext_fact", "flashnext_hcf", "flashnext_fexp", "experts_rows" }) |n| if (std.mem.eql(u8, name, n)) return @field(kk, n);
    if (std.mem.eql(u8, name, "contract_controls")) return if (options.controls) &controls.bytes else &.{};
    std.debug.print("FAIL kernel-contract: unknown image name: {s}\n", .{name});
    return error.UnknownContractImage;
}

/// Gate 5 on this GPU: the table's row for it against the driver, each record against the loaded function.
pub fn live(gpu: check.Gpu) !usize {
    var n = try registry.checkCuda(gpu.gpa, &registry.all_cuda, false);
    std.debug.print("{s} limits: {d} registered CUDA launches against sm_80/86/89/120/121, {d} problems\n", .{ if (n == 0) "PASS" else "FAIL", registry.all_cuda.len, n });
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
    for (registry.all_cuda ++ registry.cuda_controls) |r| {
        const control = std.mem.startsWith(u8, r.launch.name, "control");
        if (!control and registry.targetDecision(r, t, r.usage[at]) != .applicable) continue;
        const bytes = try image(r.launch.image);
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

/// kernel-contract-limits DIR: each record against DIR's fatbins; a RECORD line keeps the record for SMs not built.
fn keep(a: std.mem.Allocator, line: *std.ArrayList(u8), r: registry.CudaRecord, compiler: ?limits.Nvcc, i: usize) !void {
    const u = registry.usageFor(r, compiler)[i];
    try line.print(a, ".{{ .regs = {d}, .static = {d}, .bound = {d} }}, ", .{ u.regs, u.static, u.bound });
}

pub fn fatbins(gpu: check.Gpu, dir: []const u8) !void {
    const a = gpu.gpa;
    var n: usize = 0;
    var said: ?limits.Nvcc = null;
    for (registry.all_cuda ++ registry.cuda_controls) |r| {
        const control = std.mem.startsWith(u8, r.launch.name, "control");
        const any_target = for (limits.sms, 0..) |target, index| {
            if (registry.targetDecision(r, target, r.usage[index]) == .applicable) break true;
        } else false;
        if (!control and !any_target) {
            std.debug.print("N/A limits {s}: no applicable target\n", .{r.launch.name});
            continue;
        }
        const path = try std.fmt.allocPrint(a, "{s}/{s}.fatbin", .{ dir, r.launch.image });
        defer a.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, a, .limited(1 << 28));
        defer a.free(bytes);
        const imgs = try contract.cubin.images(a, bytes);
        defer contract.cubin.freeImages(a, imgs);
        const compiler: ?limits.Nvcc = if (imgs.len > 0) try contract.cubin.release(a, imgs[0].elf) else null;
        var red: usize = 0;
        var line: std.ArrayList(u8) = .empty;
        defer line.deinit(a);
        for (limits.sms, 0..) |t, i| {
            const decision = registry.targetDecision(r, t, r.usage[i]);
            if (!control and decision != .applicable) {
                std.debug.print("N/A limits {s} sm_{d}: {t}\n", .{ r.launch.name, t.sm, decision });
                try keep(a, &line, r, compiler, i);
                continue;
            }
            const img = for (imgs) |im| {
                if (im.sm == t.sm) break im;
            } else {
                std.debug.print("FAIL limits {s}: {s} holds no sm_{d} image (build with -Dsm=80,86,89,120,121)\n", .{ r.launch.name, path, t.sm });
                n += 1;
                try keep(a, &line, r, compiler, i);
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
            const measured = registry.targetDecision(r, t, k.usage);
            if (!control and measured != .applicable) {
                std.debug.print("N/A limits {s} sm_{d}: {t}\n", .{ r.launch.name, t.sm, measured });
                try keep(a, &line, r, compiler, i);
                continue;
            }
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
