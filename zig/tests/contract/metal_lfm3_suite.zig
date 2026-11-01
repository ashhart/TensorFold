//! The LFM3 text and shared-row kernels' probes, registered with the contract and released together.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const postnorm = @import("metal_lfm3_postnorm.zig");
const lfm3 = @import("metal_lfm3.zig");
const lfm3_controls = @import("metal_lfm3_controls.zig");
const lfm3_sums = @import("metal_lfm3_sums.zig");
const lfm3_attention_sums = @import("metal_lfm3_attention_sums.zig");
const lfm3_dual = @import("metal_lfm3_dual.zig");
const lfm3_segments = @import("metal_lfm3_segments.zig");
const lfm3_alpha = @import("metal_lfm3_alpha.zig");
const lfm3_conv_segments = @import("metal_lfm3_conv_segments.zig");
const binned_projection = @import("metal_binned_projection.zig");
const lfm3_projection = @import("metal_lfm3_projection.zig");
const g = contract.gates;

fn wanted(only: ?[]const u8, name: []const u8) bool {
    return std.mem.indexOf(u8, name, only orelse return true) != null;
}

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    lfm3_probes: [std.enums.values(lfm3.Kind).len]lfm3.Probe = undefined,
    lfm3_control_probes: [std.enums.values(lfm3.Kind).len]lfm3.Probe = undefined,
    lfm3_control_count: usize = 0,
    lfm3_count: usize = 0,
    lfm3_dual_probes: [8]lfm3_dual.Probe = undefined,
    lfm3_dual_count: usize = 0,
    postnorm_probes: [2]postnorm.Probe = undefined,
    postnorm_count: usize = 0,
    lfm3_skip_probe: ?lfm3_alpha.SkipProbe = null,
    lfm3_clear_probe: ?lfm3_alpha.ClearProbe = null,
    lfm3_overflow_probe: ?lfm3_alpha.OverflowProbe = null,
    lfm3_weight_probe: ?lfm3_alpha.NonfiniteWeightProbe = null,
    lfm3_product_probe: ?lfm3_alpha.ProductProbe = null,
    lfm3_alpha_probe: ?lfm3_alpha.Probe = null,
    lfm3_shared_probes: [3]lfm3_segments.Probe = undefined,
    lfm3_shared_count: usize = 0,
    lfm3_conv_probes: [2]lfm3_conv_segments.Probe = undefined,
    lfm3_conv_count: usize = 0,
    lfm3_sum_probes: [8]lfm3_sums.Probe = undefined,
    lfm3_sum_count: usize = 0,
    lfm3_attention_probes: [std.enums.values(lfm3_attention_sums.Kind).len * 2]lfm3_attention_sums.Probe = undefined,
    lfm3_attention_count: usize = 0,
    binned_probes: [9]binned_projection.Probe = undefined,
    binned_count: usize = 0,
    lfm3_projection_probes: [14]lfm3_projection.Probe = undefined,
    lfm3_projection_count: usize = 0,

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        const a = s.a;
        const device = s.device;
        const queue = s.queue;
        const kernels = s.kernels;
        for (std.enums.values(lfm3.Kind)) |kind| {
            if (!wanted(only, lfm3.name(kind))) continue;
            s.lfm3_probes[s.lfm3_count] = try lfm3.Probe.init(a, device, queue, kind);
            s.lfm3_count += 1;
        }
        for (s.lfm3_probes[0..s.lfm3_count]) |*probe| try kernels.append(a, probe.kernel());
        for (std.enums.values(lfm3.Kind)) |kind| {
            if (!wanted(only, lfm3_controls.name(kind))) continue;
            s.lfm3_control_probes[s.lfm3_control_count] = try lfm3_controls.init(a, device, queue, kind);
            s.lfm3_control_count += 1;
        }
        for (s.lfm3_control_probes[0..s.lfm3_control_count]) |*probe| try kernels.append(a, lfm3_controls.kernel(probe));
        for ([_]u8{ 4, 6, 8 }) |bits| for ([_]usize{ 32, 64 }) |group| {
            const name = try lfm3_dual.label(a, bits, group, .none);
            defer a.free(name);
            if (!wanted(only, name)) continue;
            s.lfm3_dual_probes[s.lfm3_dual_count] = try lfm3_dual.Probe.init(a, device, queue, bits, group, .none);
            s.lfm3_dual_count += 1;
        };
        for ([_]lfm3_dual.Fault{ .second, .tail }) |fault| {
            const name = try lfm3_dual.label(a, 4, 64, fault);
            defer a.free(name);
            if (!wanted(only, name)) continue;
            s.lfm3_dual_probes[s.lfm3_dual_count] = try lfm3_dual.Probe.init(a, device, queue, 4, 64, fault);
            s.lfm3_dual_count += 1;
        }
        for (s.lfm3_dual_probes[0..s.lfm3_dual_count]) |*probe| try kernels.append(a, probe.kernel());
        for ([_]bool{ false, true }) |bad| {
            if (!wanted(only, postnorm.name(bad))) continue;
            s.postnorm_probes[s.postnorm_count] = try postnorm.Probe.init(a, device, queue, bad);
            s.postnorm_count += 1;
        }
        for (s.postnorm_probes[0..s.postnorm_count]) |*probe| try kernels.append(a, probe.kernel());
        if (wanted(only, lfm3_alpha.skip_name)) {
            s.lfm3_skip_probe = try lfm3_alpha.SkipProbe.init(a, device, queue);
            try kernels.append(a, s.lfm3_skip_probe.?.kernel());
        }
        if (wanted(only, lfm3_alpha.clear_name)) {
            s.lfm3_clear_probe = try lfm3_alpha.ClearProbe.init(a, device, queue);
            try kernels.append(a, s.lfm3_clear_probe.?.kernel());
        }
        if (wanted(only, lfm3_alpha.overflow_name)) {
            s.lfm3_overflow_probe = try lfm3_alpha.OverflowProbe.init(a, device, queue);
            try kernels.append(a, s.lfm3_overflow_probe.?.kernel());
        }
        if (wanted(only, lfm3_alpha.nonfinite_weight_name)) {
            s.lfm3_weight_probe = try lfm3_alpha.NonfiniteWeightProbe.init(a, device, queue);
            try kernels.append(a, s.lfm3_weight_probe.?.kernel());
        }
        if (wanted(only, lfm3_alpha.product_name)) {
            s.lfm3_product_probe = try lfm3_alpha.ProductProbe.init(a, device, queue);
            try kernels.append(a, s.lfm3_product_probe.?.kernel());
        }
        if (wanted(only, lfm3_alpha.name)) {
            s.lfm3_alpha_probe = try lfm3_alpha.Probe.init(a, device, queue);
            try kernels.append(a, s.lfm3_alpha_probe.?.kernel());
        }
        for ([_]bool{ false, true }) |sliding| for ([_]bool{ false, true }) |bad| {
            if (sliding and bad or !wanted(only, lfm3_segments.name(sliding, bad))) continue;
            s.lfm3_shared_probes[s.lfm3_shared_count] = try lfm3_segments.Probe.init(a, device, queue, sliding, bad);
            s.lfm3_shared_count += 1;
        };
        for (s.lfm3_shared_probes[0..s.lfm3_shared_count]) |*probe| try kernels.append(a, probe.kernel());
        for ([_]bool{ false, true }) |bad| {
            if (!wanted(only, lfm3_conv_segments.name(bad))) continue;
            s.lfm3_conv_probes[s.lfm3_conv_count] = try lfm3_conv_segments.Probe.init(a, device, queue, bad);
            s.lfm3_conv_count += 1;
        }
        for (s.lfm3_conv_probes[0..s.lfm3_conv_count]) |*probe| try kernels.append(a, probe.kernel());
        for ([_]lfm3_sums.Kind{ .norm, .qk }) |kind| for ([_]bool{ false, true }) |bad| {
            if (!wanted(only, lfm3_sums.name(kind, bad))) continue;
            s.lfm3_sum_probes[s.lfm3_sum_count] = try lfm3_sums.Probe.init(a, device, queue, kind, bad);
            s.lfm3_sum_count += 1;
        };
        for (s.lfm3_sum_probes[0..s.lfm3_sum_count]) |*probe| try kernels.append(a, probe.kernel());
        for ([_]lfm3_attention_sums.Kind{ .dot, .full_values, .sliding_values, .full_merge, .sliding_merge }) |kind| for ([_]bool{ false, true }) |bad| {
            if (!wanted(only, lfm3_attention_sums.name(kind, bad))) continue;
            s.lfm3_attention_probes[s.lfm3_attention_count] = try lfm3_attention_sums.Probe.init(a, device, queue, kind, bad);
            s.lfm3_attention_count += 1;
        };
        for (s.lfm3_attention_probes[0..s.lfm3_attention_count]) |*probe| try kernels.append(a, probe.kernel());
        for ([_]u8{ 4, 6, 8 }) |bits| for ([_]usize{ 32, 64 }) |group| {
            const name = try binned_projection.label(a, bits, group, .none);
            defer a.free(name);
            if (!wanted(only, name)) continue;
            s.binned_probes[s.binned_count] = try binned_projection.Probe.init(a, device, queue, bits, group, .none);
            s.binned_count += 1;
        };
        for ([_]binned_projection.Fault{ .rows, .accum, .nonfinite }) |fault| {
            const name = try binned_projection.label(a, 4, 64, fault);
            defer a.free(name);
            if (!wanted(only, name)) continue;
            s.binned_probes[s.binned_count] = try binned_projection.Probe.init(a, device, queue, 4, 64, fault);
            s.binned_count += 1;
        }
        for (s.binned_probes[0..s.binned_count]) |*probe| try kernels.append(a, probe.kernel());

        for ([_]u8{ 4, 6, 8 }) |bits| for ([_]usize{ 32, 64 }) |group| for (std.enums.values(lfm3_projection.Proof)) |proof| {
            const name = try lfm3_projection.label(a, bits, group, proof, .none);
            defer a.free(name);
            if (!wanted(only, name)) continue;
            s.lfm3_projection_probes[s.lfm3_projection_count] = try lfm3_projection.Probe.init(a, device, queue, bits, group, proof, .none);
            s.lfm3_projection_count += 1;
        };
        for ([_]lfm3_projection.Fault{ .tail, .rows }) |fault| {
            const name = try lfm3_projection.label(a, 4, 64, .bytes, fault);
            defer a.free(name);
            if (!wanted(only, name)) continue;
            s.lfm3_projection_probes[s.lfm3_projection_count] = try lfm3_projection.Probe.init(a, device, queue, 4, 64, .bytes, fault);
            s.lfm3_projection_count += 1;
        }
        for (s.lfm3_projection_probes[0..s.lfm3_projection_count]) |*probe| try kernels.append(a, probe.kernel());
    }

    pub fn deinit(s: *Suite) void {
        for (s.lfm3_probes[0..s.lfm3_count]) |*probe| probe.deinit();
        for (s.lfm3_control_probes[0..s.lfm3_control_count]) |*probe| probe.deinit();
        for (s.lfm3_dual_probes[0..s.lfm3_dual_count]) |*probe| probe.deinit();
        for (s.postnorm_probes[0..s.postnorm_count]) |*probe| probe.deinit();
        if (s.lfm3_skip_probe) |*probe| probe.deinit();
        if (s.lfm3_clear_probe) |*probe| probe.deinit();
        if (s.lfm3_overflow_probe) |*probe| probe.deinit();
        if (s.lfm3_weight_probe) |*probe| probe.deinit();
        if (s.lfm3_product_probe) |*probe| probe.deinit();
        if (s.lfm3_alpha_probe) |*probe| probe.deinit();
        for (s.lfm3_shared_probes[0..s.lfm3_shared_count]) |*probe| probe.deinit();
        for (s.lfm3_conv_probes[0..s.lfm3_conv_count]) |*probe| probe.deinit();
        for (s.lfm3_sum_probes[0..s.lfm3_sum_count]) |*probe| probe.deinit();
        for (s.lfm3_attention_probes[0..s.lfm3_attention_count]) |*probe| probe.deinit();
        for (s.binned_probes[0..s.binned_count]) |*probe| probe.deinit();
        for (s.lfm3_projection_probes[0..s.lfm3_projection_count]) |*probe| probe.deinit();
    }
};
