//! GPU sampling dispatch for DSpark draft steps: two-pass top-k (k <= 64) or full-vocab race.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const kernel_sources = @import("kernel_sources");
const state = @import("dspark_state.zig");

const gpu_rule = lanes.gpu_rule;
const topk_groups: usize = 32;
const topk_threads: usize = 512;
const topk_elem_max: usize = 16;
pub const topk_max: usize = 64;

pub const Sampler = struct {
    full: mtl.Pipeline,
    slices: mtl.Pipeline,
    race: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Sampler {
        const lib = try mtl.Library.fromSource(device, kernel_sources.nemotron_sample, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const full = try mtl.Pipeline.init(device, lib, "tf_sample_full", false);
        errdefer full.deinit();
        const slices = try mtl.Pipeline.init(device, lib, "tf_topk_slices", false);
        errdefer slices.deinit();
        const race = try mtl.Pipeline.init(device, lib, "tf_topk_race", false);
        return .{ .full = full, .slices = slices, .race = race };
    }

    pub fn deinit(s: *Sampler) void {
        s.race.deinit();
        s.slices.deinit();
        s.full.deinit();
        s.* = undefined;
    }

    /// Dispatches keyed sampling over contiguous sampled runs, using tf_topk_slices+race when 1 <= top_k <= 64.
    pub fn encodeRuns(
        sp: *const Sampler,
        e: mtl.ComputeEncoder,
        logits: mtl.Buffer,
        sample_tok: mtl.Buffer,
        sample_ck: mtl.Buffer,
        sample_ci: mtl.Buffer,
        samps: []const ?lanes.Sampling,
        base_pos: []const u32,
        step: u32,
    ) void {
        var si: usize = 0;
        while (si < samps.len) {
            if (samps[si] == null) {
                si += 1;
                continue;
            }
            var run: usize = 1;
            while (si + run < samps.len and samps[si + run] != null) run += 1;
            var seeds: [2 * state.max_streams]u32 = undefined;
            var pos: [state.max_streams]u32 = undefined;
            var cfgs: [4 * state.max_streams]f32 = undefined;
            var caps: [state.max_streams]u32 = undefined;
            var two = state.vocabulary <= topk_groups * topk_threads * topk_elem_max;
            for (0..run) |j| {
                const rule = gpu_rule.rule(samps[si + j].?);
                seeds[2 * j ..][0..2].* = rule.seedHalves();
                pos[j] = base_pos[si + j] + step;
                cfgs[4 * j ..][0..4].* = rule.cfg();
                caps[j] = rule.top_k;
                if (rule.top_k == 0 or rule.top_k > topk_max or rule.top_k >= state.vocabulary) two = false;
            }
            const v_count: u32 = state.vocabulary;
            e.setBuffer(logits, si * state.vocabulary * 2, 0);
            e.setBytes(std.mem.sliceAsBytes(seeds[0 .. 2 * run]), 1);
            e.setBytes(std.mem.sliceAsBytes(pos[0..run]), 2);
            e.setBytes(std.mem.sliceAsBytes(cfgs[0 .. 4 * run]), 3);
            e.setBytes(std.mem.sliceAsBytes(caps[0..run]), 4);
            e.setBuffer(sample_tok, si * 4, 5);
            e.setBuffer(sample_tok, 0, 6);
            e.setBytes(std.mem.asBytes(&v_count), 7);
            if (two) {
                e.setBuffer(sample_ck, 0, 8);
                e.setBuffer(sample_ci, 0, 9);
                e.setPipeline(sp.slices);
                e.dispatchGroups(mtl.Size.of(topk_groups, run, 1), mtl.Size.of(topk_threads, 1, 1));
                e.barrier();
                e.setPipeline(sp.race);
            } else {
                e.setPipeline(sp.full);
            }
            e.dispatchGroups(mtl.Size.of(run, 1, 1), mtl.Size.of(1024, 1, 1));
            si += run;
        }
    }
};
