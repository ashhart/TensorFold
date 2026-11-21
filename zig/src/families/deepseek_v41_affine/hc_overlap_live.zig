//! Run exact hyperconnection coefficient generation beside a layer's first projections.
const mtl = @import("metal");
const c = @import("config.zig");
const st = @import("state.zig");
const wts = @import("weights.zig");
const Buffer = mtl.Buffer;

pub const Job = struct { weights: wts.Hc, pre: Buffer };
const Args = extern struct { eps: f32 = c.norm_eps, hc_eps: f32 = c.hc_eps, iters: i32 = c.sinkhorn };

pub const Kernels = struct {
    metadata: mtl.Pipeline,
    audit: ?mtl.Pipeline,

    pub fn load(device: mtl.Device, checks: bool) !Kernels {
        const lib = try mtl.Library.fromSource(device, @embedFile("hc_coefficients.metal") ++ "\n" ++
            @embedFile("hc_overlap_audit.metal"), mtl.CompileOptions.mlx());
        defer lib.deinit();
        const metadata = try mtl.Pipeline.init(device, lib, "tf_hc_coefficients", false);
        errdefer metadata.deinit();
        const audit = if (checks) try mtl.Pipeline.init(device, lib, "tf_hc_coefficients_audit", false) else null;
        return .{ .metadata = metadata, .audit = audit };
    }

    pub fn deinit(p: Kernels) void {
        p.metadata.deinit();
        if (p.audit) |pipe| pipe.deinit();
    }

    pub fn issue(p: Kernels, e: mtl.ComputeEncoder, s: *st.Scratch, job: Job, rows: usize) void {
        e.setPipeline(p.metadata);
        e.setBuffer(s.hc_part, 0, 0);
        e.setBuffer(job.weights.base.buffer, job.weights.base.offset, 1);
        e.setBuffer(job.weights.scale.buffer, job.weights.scale.offset, 2);
        e.setValue(Args{}, 3);
        e.setBuffer(job.pre, 0, 4);
        e.setBuffer(s.post, 0, 5);
        e.setBuffer(s.comb, 0, 6);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(32, 1, 1));
    }

    pub fn check(p: mtl.Pipeline, e: mtl.ComputeEncoder, s: *st.Scratch, pre: Buffer, rows: usize) void {
        e.setPipeline(p);
        for ([_]Buffer{ s.normed, s.post, pre, s.comb, s.hc_overlap_ref_norm, s.hc_overlap_ref_post, s.hc_overlap_ref_pre, s.hc_overlap_ref_comb, s.hc_overlap_error }, 0..) |buffer, i|
            e.setBuffer(buffer, 0, i);
        e.setValue(@as(u32, @intCast(rows)), 9);
        e.dispatchThreads(mtl.Size.of(rows * c.hidden, 1, 1), mtl.Size.of(256, 1, 1));
    }
};
