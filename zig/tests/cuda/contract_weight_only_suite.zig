//! Stable ownership for the CUDA projection cases collected by the complete contract runner.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const probes = @import("contract_weight_only.zig");
const hc = @import("contract_hc_activation.zig");
const pointwise = @import("contract_hc_pointwise.zig");
const gdn = @import("contract_qwen4_gdn.zig");
const ple = @import("contract_qwen4_ple.zig");
const dense = @import("contract_qwen4_dense.zig");
const qsa_norm = @import("contract_qwen4_qsa_norm.zig");
const qsa_scores = @import("contract_qwen4_qsa_scores.zig");
const qsa_select = @import("contract_qwen4_qsa_select.zig");
const qsa_attn = @import("contract_qwen4_qsa_attn.zig");
pub const Suite = struct {
    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, stream: cuda.Stream, out: *std.ArrayList(core.contract.gates.Kernel)) !*Suite {
        const self = try a.create(Suite);
        errdefer a.destroy(self);
        try self.open(a, gpu, stream, out);
        return self;
    }
    kernels: probes.Kernels,
    cases: [probes.count]probes.Case = undefined,
    hc_kernels: hc.Kernels,
    hc_cases: [hc.count]hc.Case = undefined,
    pointwise_kernels: pointwise.Kernels,
    pointwise_cases: [pointwise.count]pointwise.Case = undefined,
    stream: cuda.Stream,
    gdn_suite: ?*gdn.Suite = null,
    ple_suite: ?*ple.Suite = null,
    dense_suite: ?*dense.Suite = null,
    qsa_norm_suite: ?*qsa_norm.Suite = null,
    qsa_scores_suite: ?*qsa_scores.Suite = null,
    qsa_select_suite: ?*qsa_select.Suite = null,
    qsa_attn_suite: ?*qsa_attn.Suite = null,
    pub fn open(self: *Suite, a: std.mem.Allocator, gpu: check.Gpu, stream: cuda.Stream, out: *std.ArrayList(core.contract.gates.Kernel)) !void {
        const native = try gdn.Suite.create(a, gpu, stream, out);
        errdefer native.destroy(a);
        const lexical = try ple.Suite.create(a, gpu, stream, out);
        errdefer lexical.destroy(a);
        const activation = try dense.Suite.create(a, gpu, stream, out);
        errdefer activation.destroy(a);
        const frontend = try qsa_norm.Suite.create(a, gpu, stream, out);
        errdefer frontend.destroy(a);
        const scores = try qsa_scores.Suite.create(a, gpu, stream, out);
        errdefer scores.destroy(a);
        const selection = try qsa_select.Suite.create(a, gpu, stream, out);
        errdefer selection.destroy(a);
        const attention = try qsa_attn.Suite.create(a, gpu, stream, out);
        errdefer attention.destroy(a);
        try self.openImages(a, gpu, stream, cuda.kernels.qwen4_projection, cuda.kernels.qwen4_hc, out);
        self.gdn_suite = native;
        self.ple_suite = lexical;
        self.dense_suite = activation;
        self.qsa_norm_suite = frontend;
        self.qsa_scores_suite = scores;
        self.qsa_select_suite = selection;
        self.qsa_attn_suite = attention;
    }
    pub fn openImages(self: *Suite, a: std.mem.Allocator, gpu: check.Gpu, stream: cuda.Stream, projection: []const u8, activation: []const u8, out: *std.ArrayList(core.contract.gates.Kernel)) !void {
        var weights = try probes.Kernels.load(gpu.d, projection, true);
        errdefer weights.close(stream) catch {};
        var gates = try hc.Kernels.load(gpu.d, activation, true);
        errdefer gates.close(stream) catch {};
        var points = try pointwise.Kernels.load(gpu.d, activation, true);
        errdefer points.close(stream) catch {};
        self.* = .{ .kernels = weights, .hc_kernels = gates, .pointwise_kernels = points, .stream = stream };
        try probes.collect(a, gpu, stream, &self.kernels, &self.cases, out);
        try hc.collect(a, gpu, stream, &self.hc_kernels, &self.hc_cases, out);
        try pointwise.collect(a, gpu, stream, &self.pointwise_kernels, &self.pointwise_cases, out);
    }
    /// A failed drain retains the heap owner, cases and module until device teardown.
    pub fn destroy(self: *Suite, a: std.mem.Allocator) void {
        self.stream.synchronize() catch return;
        if (self.gdn_suite) |native| native.destroy(a);
        if (self.ple_suite) |lexical| lexical.destroy(a);
        if (self.dense_suite) |activation| activation.destroy(a);
        if (self.qsa_norm_suite) |frontend| frontend.destroy(a);
        if (self.qsa_scores_suite) |scores| scores.destroy(a);
        if (self.qsa_select_suite) |selection| selection.destroy(a);
        if (self.qsa_attn_suite) |attention| attention.destroy(a);
        for (&self.cases) |*case| case.deinit();
        for (&self.hc_cases) |*case| case.deinit();
        for (&self.pointwise_cases) |*case| case.deinit();
        self.pointwise_kernels.close(self.stream) catch return;
        self.hc_kernels.close(self.stream) catch return;
        self.kernels.close(self.stream) catch return;
        a.destroy(self);
    }
};
