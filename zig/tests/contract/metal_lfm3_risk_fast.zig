//! Fused branch-free range predicates must retain the public attention bytes.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const ct = @import("contract");
const base = @import("metal_lfm3.zig");
const rig = @import("rig.zig");
const g = ct.gates;

pub fn name(sliding: bool) []const u8 {
    return if (sliding) "metal LFM3 risk-fast sliding" else "metal LFM3 risk-fast full";
}

pub const Probe = struct {
    inner: base.Probe,
    original: mtl.Pipeline,
    original_merge: mtl.Pipeline,
    sliding: bool,
    bytes: std.ArrayList(u8) = .empty,
    const vt = blk: {
        var hooks = g.hooks(base.Probe);
        hooks.replaced = replaced;
        break :blk hooks;
    };

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, sliding: bool) !Probe {
        var inner = try base.Probe.init(a, device, queue, if (sliding) .attention_sliding else .attention_full);
        errdefer inner.deinit();
        const text = try std.mem.concat(a, u8, &.{ "#define L3_RISK_FAST 1\n", sources.lfm3, "\n", sources.lfm3_attention });
        defer a.free(text);
        const partial = try rig.pipeline(device, text, "lfm3_attn_partial");
        errdefer partial.deinit();
        const merge = try rig.pipeline(device, text, "lfm3_attn_merge");
        errdefer merge.deinit();
        if (partial.maxThreads() < 128 or merge.maxThreads() < 32 or
            rig.tgMemory(partial) != 2096 or rig.tgMemory(merge) != 0) return error.RiskFastLimits;
        const original = inner.pipeline;
        const original_merge = inner.merge.?;
        inner.pipeline = partial;
        inner.merge = merge;
        return .{ .inner = inner, .original = original, .original_merge = original_merge, .sliding = sliding };
    }

    pub fn deinit(p: *Probe) void {
        p.bytes.deinit(p.inner.a);
        p.original.deinit();
        p.original_merge.deinit();
        p.inner.deinit();
    }

    pub fn kernel(p: *Probe) g.Kernel {
        var out = p.inner.kernel();
        out.vt = &vt;
        out.spec.name = name(p.sliding);
        out.spec.replaces = "public LFM3 partial and merge with extended-range replay";
        return out;
    }

    fn replaced(ctx: *anyopaque) ![]const u8 {
        const inner: *base.Probe = @ptrCast(@alignCast(ctx));
        const p: *Probe = @fieldParentPtr("inner", inner);
        const candidate = inner.pipeline;
        const candidate_merge = inner.merge;
        inner.pipeline = p.original;
        inner.merge = p.original_merge;
        defer {
            inner.pipeline = candidate;
            inner.merge = candidate_merge;
        }
        try inner.run();
        p.bytes.clearRetainingCapacity();
        try p.bytes.appendSlice(inner.a, try inner.rig.outputBytes());
        return p.bytes.items;
    }
};
