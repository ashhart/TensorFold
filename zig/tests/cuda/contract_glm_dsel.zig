//! GLM's decode select on many CTAs under the contract on CUDA: scan and merge, bit-equal to tf_glm_select_rows.
const core = @import("core");
const glm = @import("glm_cuda");
const rig_ = @import("contract_rig.zig");
const select = @import("contract_glm_select.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const sparse = glm.sparse;

/// The wide select's rows and buckets, where cuda_sparse launches the decode select instead, and a 1M window's bucket.
pub fn spec() c.Spec {
    const k = sparse.topk_pools;
    var s = select.spec();
    s.name = "cuda glm decode select tf_glm_dsel_scan tf_glm_dsel_merge";
    s.served = &[_]c.Shape{ .{ .rows = 1, .n = 4096, .k = k }, .{ .rows = 4, .n = 65536, .k = k }, .{ .rows = 48, .n = 131072, .k = k }, .{ .rows = 8, .n = 260096, .k = k } };
    return s;
}

/// Each row's K pools through cuda_sparse.decodeSelect; scores, host rule and baseline are the wide select's.
pub const Dsel = struct {
    base: select.Select,
    fns: glm.glue.Kernels,
    rig: *rig_.Rig = undefined, // the base's, set by kernel() once this value sits at its final address
    bits: u64 = 0,
    short: bool = false, // the control: past two spans, the select is told its bucket is one span short

    pub fn deinit(self: *Dsel) void {
        self.base.deinit();
    }

    pub fn kernel(self: *Dsel) g.Kernel {
        self.rig = &self.base.rig;
        var s = spec();
        if (self.short) s.name = "control: GLM decode select reads its bucket one span short";
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = if (self.short) .shapes else null };
    }

    const vt = g.hooks(Dsel);

    pub fn prepare(self: *Dsel, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!try self.base.prepare(shape, fill, seed)) return false;
        self.bits = try self.base.rig.scratch(sparse.bitsBytes(shape.rows, shape.n));
        return true;
    }

    pub fn run(self: *Dsel) !void {
        const b = &self.base;
        const set: glm.glue.Set = .{ .glue = &self.fns, .dsel = true };
        const n = if (self.short and b.shape.n > 2 * sparse.dsel_span) b.shape.n - sparse.dsel_span else b.shape.n;
        try sparse.decodeSelect(set, b.rig.s, b.in, b.out, n, b.pos, b.at, b.shape.rows, self.bits, 0);
        try b.rig.s.synchronize();
    }

    pub fn replaced(self: *Dsel) ![]const u8 {
        return self.base.replaced();
    }

    pub fn count(self: *Dsel) usize {
        return self.base.count();
    }

    pub fn reference(self: *Dsel, want: []f64, mag: []f64) !void {
        return self.base.reference(want, mag);
    }

    pub fn check(self: *Dsel, gate: g.Gate) !?[]const u8 {
        return self.base.check(gate);
    }
};
