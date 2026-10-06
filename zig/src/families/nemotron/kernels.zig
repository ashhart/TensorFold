//! Nemotron's pipelines, compiled at run time from the embedded sources with MLX's custom-kernel options.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const simd_attention = @import("simd_attention.zig");

pub const glue_names = [_][:0]const u8{ "tf_embed_q4", "tf_rms_mlx", "tf_argmax_bf16", "tf_kv_write", "tf_attn_q", "tf_attn_out", "tf_copy_rows", "tf_copy_u32", "tf_coop_combine" };

/// Tree windows (nemotron_tree.metal): Mamba by parent for every forward, tree attention, compaction, gathers, top-k.
pub const tree_names = [_][:0]const u8{ "tf_tree_conv", "tf_tree_scan", "tf_tree_tail", "tf_tree_merge", "tf_kv_compact", "tf_gather_rows", "tf_topk_probs", "tf_lane_tokens" };

/// A lone stream's GPU-side round (nemotron_round.metal): the verify's arguments, the accept, the stop, a row gather.
pub const round_names = [_][:0]const u8{ "tf_round_args", "tf_round_accept", "tf_round_gather", "tf_round_root", "tf_round_top1", "tf_round_copy", "tf_round_tail", "tf_round_sib", "tf_round_attsel" };

/// The MTP head's fused one-row kernels (nemotron_head.metal).
pub const head_names = [_][:0]const u8{ "tf_head_prep", "tf_head_norm" };

/// Keyed draws over the whole vocabulary, for rows whose top_k is off or above tf_gpu_sample's 1,024 (nemotron_sample.metal).
pub const sample_names = [_][:0]const u8{ "tf_sample_full", "tf_sample_full_ids" };

/// Routed experts taking an expert's member rows two or four at a time (nemotron_experts.metal).
pub const rows_names = [_][:0]const u8{ "tf_xup_rows2", "tf_xdown_rows2", "tf_xup_rows4", "tf_xdown_rows4" };

/// Routed-expert kernels at other (rows a simdgroup, simdgroups a threadgroup): each row's arithmetic is unchanged.
pub const geometries = [_][2]usize{ .{ 2, 2 }, .{ 8, 2 }, .{ 4, 4 }, .{ 8, 4 }, .{ 4, 1 }, .{ 2, 4 } };

fn geoName(comptime which: []const u8, comptime g: [2]usize) [:0]const u8 {
    return std.fmt.comptimePrint("tf_x{s}_{d}_{d}", .{ which, g[0], g[1] });
}

pub const geo_names = blk: {
    var names: [2 * geometries.len][:0]const u8 = undefined;
    for (geometries, 0..) |g, i| {
        names[i] = geoName("up", g);
        names[geometries.len + i] = geoName("down", g);
    }
    break :blk names;
};

fn generated(comptime key: []const u8) sources.nemotron.Kernel {
    for (sources.nemotron.all) |k| if (std.mem.eql(u8, k.key, key)) return k;
    unreachable;
}

/// The generated expert source with explicit instantiations at each geometry (template args after RPS, SG).
fn geoSource(comptime key: []const u8, comptime which: []const u8, comptime head: []const u8, comptime tail: []const u8) []const u8 {
    const k = generated(key);
    var text: []const u8 = k.source;
    for (geometries) |g| {
        const args = std.fmt.comptimePrint("<{s}, {d}, {d}{s}>", .{ head, g[0], g[1], tail });
        text = text ++ "\ntemplate [[host_name(\"" ++ geoName(which, g) ++ "\")]] [[kernel]] decltype(" ++ k.function ++ args ++ ") " ++ k.function ++ args ++ ";";
    }
    return text ++ "\n";
}

const geo_up = geoSource("expert_up", "up", "2688, 1856, 64", ", 6");
const geo_down = geoSource("expert_down", "down", "1856, 2688, 64", "");

pub const total = sources.nemotron.all.len + glue_names.len + geo_names.len + rows_names.len + tree_names.len + round_names.len + head_names.len + sample_names.len;

/// A pipeline index's kernel key (generated) or function name (glue).
pub fn keyOf(i: usize) []const u8 {
    inline for (sources.nemotron.all, 0..) |k, j| if (i == j) return k.key;
    const g = i - sources.nemotron.all.len;
    if (g < glue_names.len) return glue_names[g];
    if (g < glue_names.len + geo_names.len) return geo_names[g - glue_names.len];
    const r = g - glue_names.len - geo_names.len;
    if (r < rows_names.len) return rows_names[r];
    const t = r - rows_names.len;
    if (t < tree_names.len) return tree_names[t];
    const u = t - tree_names.len;
    if (u < round_names.len) return round_names[u];
    const h = u - round_names.len;
    return if (h < head_names.len) head_names[h] else sample_names[h - head_names.len];
}

pub const Kernels = struct {
    pipelines: [total]mtl.Pipeline = undefined,

    /// The pipeline of a generated kernel (by its generator key) or a glue kernel (by function name).
    pub fn get(self: *const Kernels, comptime key: []const u8) mtl.Pipeline {
        return self.pipelines[comptime index(key)];
    }

    /// A generated kernel whose Metal function name starts with `prefix` (fixture checks look kernels up by name).
    pub fn byPrefix(self: *const Kernels, prefix: []const u8) ?mtl.Pipeline {
        inline for (sources.nemotron.all, 0..) |k, i| {
            if (std.mem.startsWith(u8, k.function, prefix)) return self.pipelines[i];
        }
        return null;
    }

    /// A glue kernel by function name.
    pub fn glue(self: *const Kernels, name: []const u8) ?mtl.Pipeline {
        inline for (glue_names, 0..) |n, i| {
            if (std.mem.eql(u8, n, name)) return self.pipelines[sources.nemotron.all.len + i];
        }
        return null;
    }

    pub fn deinit(self: *Kernels) void {
        for (&self.pipelines) |p| p.deinit();
    }
};

fn index(comptime key: []const u8) usize {
    inline for (sources.nemotron.all, 0..) |k, i| if (comptime std.mem.eql(u8, k.key, key)) return i;
    inline for (glue_names, 0..) |n, i| if (comptime std.mem.eql(u8, n, key)) return sources.nemotron.all.len + i;
    inline for (geo_names, 0..) |n, i| if (comptime std.mem.eql(u8, n, key)) return sources.nemotron.all.len + glue_names.len + i;
    inline for (rows_names, 0..) |n, i| if (comptime std.mem.eql(u8, n, key)) return sources.nemotron.all.len + glue_names.len + geo_names.len + i;
    inline for (tree_names, 0..) |n, i| if (comptime std.mem.eql(u8, n, key)) return sources.nemotron.all.len + glue_names.len + geo_names.len + rows_names.len + i;
    inline for (round_names, 0..) |n, i| if (comptime std.mem.eql(u8, n, key)) return sources.nemotron.all.len + glue_names.len + geo_names.len + rows_names.len + tree_names.len + i;
    inline for (head_names, 0..) |n, i| if (comptime std.mem.eql(u8, n, key)) return sources.nemotron.all.len + glue_names.len + geo_names.len + rows_names.len + tree_names.len + round_names.len + i;
    inline for (sample_names, 0..) |n, i| if (comptime std.mem.eql(u8, n, key)) return total - sample_names.len + i;
    @compileError("no Nemotron kernel " ++ key);
}

const Job = struct {
    device: mtl.Device,
    source: []const u8,
    names: []const [:0]const u8,
    out: []mtl.Pipeline,
    failed: bool = false,
};

fn compile(job: *Job) void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const lib = mtl.Library.fromSource(job.device, job.source, mtl.CompileOptions.mlx()) catch {
        job.failed = true;
        return;
    };
    defer lib.deinit();
    for (job.names, job.out) |name, *p| {
        p.* = mtl.Pipeline.init(job.device, lib, name, false) catch {
            job.failed = true;
            return;
        };
    }
}

/// Compile every source (one library each, as MLX does) on worker threads.
pub fn load(allocator: std.mem.Allocator, device: mtl.Device) !Kernels {
    var k = Kernels{};
    const jobs = try allocator.alloc(Job, sources.nemotron.all.len + 8);
    defer allocator.free(jobs);
    const names = try allocator.alloc([:0]const u8, sources.nemotron.all.len);
    defer allocator.free(names);
    inline for (sources.nemotron.all, 0..) |kernel, i| {
        names[i] = kernel.function;
        jobs[i] = .{ .device = device, .source = kernel.source, .names = names[i .. i + 1], .out = k.pipelines[i .. i + 1] };
    }
    var rewritten: [sources.nemotron.all.len]?[]u8 = @splat(null);
    defer for (rewritten) |r| if (r) |text| allocator.free(text);
    if (!device.tensorUnits()) inline for (sources.nemotron.all, 0..) |kernel, i| {
        if (comptime std.mem.startsWith(u8, kernel.key, "attn_partial")) {
            rewritten[i] = try simd_attention.rewrite(allocator, kernel.source);
            jobs[i].source = rewritten[i].?;
        }
    };
    const n = sources.nemotron.all.len;
    jobs[n] = .{ .device = device, .source = sources.nemotron_glue, .names = &glue_names, .out = k.pipelines[n .. n + glue_names.len] };
    const gp = n + glue_names.len;
    jobs[n + 1] = .{ .device = device, .source = geo_up, .names = geo_names[0..geometries.len], .out = k.pipelines[gp .. gp + geometries.len] };
    jobs[n + 2] = .{ .device = device, .source = geo_down, .names = geo_names[geometries.len..], .out = k.pipelines[gp + geometries.len .. gp + geo_names.len] };
    const xp = gp + geo_names.len;
    jobs[n + 3] = .{ .device = device, .source = sources.nemotron_experts, .names = &rows_names, .out = k.pipelines[xp .. xp + rows_names.len] };
    const tp = xp + rows_names.len;
    jobs[n + 4] = .{ .device = device, .source = sources.nemotron_tree, .names = &tree_names, .out = k.pipelines[tp .. tp + tree_names.len] };
    const rp = tp + tree_names.len;
    jobs[n + 5] = .{ .device = device, .source = sources.nemotron_round, .names = &round_names, .out = k.pipelines[rp .. rp + round_names.len] };
    const hp = rp + round_names.len;
    jobs[n + 6] = .{ .device = device, .source = sources.nemotron_head, .names = &head_names, .out = k.pipelines[hp .. hp + head_names.len] };
    jobs[n + 7] = .{ .device = device, .source = sources.nemotron_sample, .names = &sample_names, .out = k.pipelines[hp + head_names.len ..] };

    const workers = 10;
    var next = std.atomic.Value(usize).init(0);
    const Worker = struct {
        fn run(all: []Job, counter: *std.atomic.Value(usize)) void {
            while (true) {
                const i = counter.fetchAdd(1, .monotonic);
                if (i >= all.len) return;
                compile(&all[i]);
            }
        }
    };
    var threads: [workers]?std.Thread = @splat(null);
    for (&threads) |*t| t.* = std.Thread.spawn(.{}, Worker.run, .{ jobs, &next }) catch null;
    Worker.run(jobs, &next);
    for (threads) |t| if (t) |th| th.join();
    for (jobs) |j| if (j.failed) return error.KernelCompile;
    return k;
}

test "the full-vocabulary sampler compiles at this macOS's Metal language with 1,024 threads a row" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const lib = try mtl.Library.fromSource(device, sources.nemotron_sample, mtl.CompileOptions.mlx());
    defer lib.deinit();
    for (sample_names) |name| {
        const p = try mtl.Pipeline.init(device, lib, name, false);
        defer p.deinit();
        try std.testing.expect(p.maxThreads() >= 1024);
    }
}
