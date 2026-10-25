//! Compile-only native CUDA placement and contract bodies on both supported Linux architectures.
const std = @import("std");
export fn qwen4CudaDeviceGate(gpu: *const @import("tests/cuda/check.zig").Gpu) bool {
    @import("tests/cuda/qwen4_tests.zig").execute(gpu.*) catch return false;
    return true;
}
const cuda = @import("cuda");
const core = @import("core");
const family = @import("qwen4").family;
const api = core.weight_only_cuda_adapter(core.weight_only, cuda);
const probes = @import("tests/cuda/contract_weight_only.zig");
const Storage = family.cuda_storage.Backend(cuda);
const Placed = family.device_weights.Placed(Storage);
export fn qwen4CudaBodiesLoad(out: *api.Kernels, driver: *const cuda.Driver, image: [*]const u8, len: usize) bool {
    out.* = api.Kernels.load(driver, image[0..len], true) catch return false;
    return true;
}
export fn qwen4CudaBodiesContract(case: *probes.Case) bool {
    return (core.contract.gates.one(std.heap.page_allocator, .shapes, case.kernel(), 0x367) catch return false) == .pass;
}
export fn qwen4CudaBodiesPlace(out: *Placed, storage: *Storage, reader: *family.reader.Reader, c: *const family.Config, budget: u64, problem: *family.Problem) bool {
    out.* = Placed.open(std.heap.page_allocator, storage, reader, c, budget, problem) catch return false;
    return true;
}
export fn qwen4CudaBodiesLookup(out: *Placed, name: [*]const u8, len: usize, id: usize, row: [*]u16, width: usize, problem: *family.Problem) bool {
    out.lookup(name[0..len], id, row[0..width], problem) catch return false;
    return true;
}
export fn qwen4CudaBodiesClose(out: *Placed) bool {
    out.close() catch return false;
    return true;
}

const Gdn = family.cuda_gdn;
const GdnKernels = Gdn.Kernels(cuda);
export fn qwen4CudaGdnLoad(out: *GdnKernels, driver: *const cuda.Driver, image: [*]const u8, len: usize) bool {
    out.* = GdnKernels.load(driver, image[0..len]) catch return false;
    return true;
}
export fn qwen4CudaGdnEncode(out: *GdnKernels, stream: *const cuda.Stream, p: *const Gdn.Params, ranges: *const [10]core.weight_only.Range) bool {
    out.encode(stream.*, p.*, ranges.*) catch return false;
    return true;
}
export fn qwen4CudaGdnClose(out: *GdnKernels, stream: *const cuda.Stream) bool {
    out.close(stream.*) catch return false;
    return true;
}

const HcCuda = @import("core").hc_activation_cuda_adapter(@import("core").hc_activation, cuda);
export fn qwen4HcCudaLoad(out: *HcCuda.Kernels, d: *const cuda.Driver, image: [*]const u8, n: usize) bool {
    out.* = HcCuda.Kernels.load(d, image[0..n], true) catch return false;
    return true;
}
export fn qwen4HcCudaEncode(k: *HcCuda.Kernels, s: *const cuda.Stream, p: *const @import("core").hc_activation.Params, x: *const HcCuda.View, y: *const HcCuda.View) bool {
    k.encode(s.*, p.*, x.*, y.*) catch return false;
    return true;
}
const FullSuite = @import("tests/cuda/contract_weight_only_suite.zig").Suite;
const GdnSuite = @import("tests/cuda/contract_qwen4_gdn.zig").Suite;
export fn qwen4NativeGdnSuiteOpen(gpu: *const @import("tests/cuda/check.zig").Gpu, s: *const cuda.Stream, image: [*]const u8, len: usize, kernels: *std.ArrayList(@import("core").contract.gates.Kernel)) ?*GdnSuite {
    return GdnSuite.createImage(std.heap.page_allocator, gpu.*, s.*, image[0..len], kernels) catch null;
}
export fn qwen4NativeGdnSuiteClose(s: *GdnSuite) void {
    s.destroy(std.heap.page_allocator);
}
export fn qwen4ContractSuiteOpen(out: *FullSuite, gpu: *const @import("tests/cuda/check.zig").Gpu, s: *const cuda.Stream, image: [*]const u8, len: usize, kernels: *std.ArrayList(@import("core").contract.gates.Kernel)) bool {
    out.openImages(std.heap.page_allocator, gpu.*, s.*, image[0..len], image[0..len], kernels) catch return false;
    return true;
}

const NativeState = family.state.State(Storage);
export fn qwen4NativeStateOpen(out: *NativeState, b: *Storage, c: *const family.Config, capacity: usize, window: usize, budget: u64) bool {
    out.* = NativeState.open(std.heap.page_allocator, b, c, capacity, window, budget) catch return false;
    return true;
}
export fn qwen4NativeStateReset(out: *NativeState) bool {
    out.reset() catch return false;
    return true;
}
export fn qwen4NativeStateBegin(out: *NativeState, rows: usize) bool {
    out.begin(rows) catch return false;
    return true;
}
export fn qwen4NativeStateKeep(out: *NativeState, rows: usize) bool {
    out.keep(rows) catch return false;
    return true;
}
export fn qwen4NativeStateAbort(out: *NativeState) bool {
    out.abort() catch return false;
    return true;
}
export fn qwen4NativeStateClose(out: *NativeState) bool {
    out.close() catch return false;
    return true;
}

const NativeSnapshot = family.snapshot.Snapshot(Storage);
export fn qwen4SnapshotOpen(out: *NativeSnapshot, state: *NativeState, key: *const [32]u8, budget: u64) bool {
    out.* = NativeSnapshot.open(std.heap.page_allocator, state, key.*, budget) catch return false;
    return true;
}
export fn qwen4SnapshotSave(out: *NativeSnapshot, state: *NativeState, key: *const [32]u8) bool {
    out.capture(state, key.*) catch return false;
    return true;
}
export fn qwen4SnapshotRestore(out: *NativeSnapshot, state: *NativeState, key: *const [32]u8) bool {
    out.restore(state, key.*) catch return false;
    return true;
}
export fn qwen4SnapshotClose(out: *NativeSnapshot) bool {
    out.close() catch return false;
    return true;
}

const Qsa = family.cuda_attention.Kernels(cuda);
export fn qwen4CudaAttentionLoad(out: *Qsa, d: *const cuda.Driver, image: [*]const u8, n: usize) bool {
    out.* = Qsa.load(d, image[0..n], image[0..n]) catch return false;
    return true;
}
export fn qwen4CudaAttentionEncode(out: *Qsa, s: *const cuda.Stream, g: *const family.cuda_attention.Geometry, p: *const family.cuda_attention.Call, views: *const [23]Qsa.View) bool {
    out.encode(s.*, g.*, p.*, views.*) catch return false;
    return true;
}
export fn qwen4CudaAttentionClose(out: *Qsa, s: *const cuda.Stream) bool {
    out.close(s.*) catch return false;
    return true;
}

const NativeModel = family.model.Model(Storage);
export fn qwen4NativeModelBind(out: *NativeModel, weights: *family.device_weights.Placed(Storage), problem: *family.Problem) bool {
    out.* = NativeModel.init(std.heap.page_allocator, weights, problem) catch return false;
    return true;
}
export fn qwen4NativeModelClose(out: *NativeModel) void {
    out.deinit();
}

const NativeArena = family.arena.Arena(Storage);
export fn qwen4NativeArenaOpen(out: *NativeArena, b: *Storage, bytes: usize, budget: u64) bool {
    out.* = NativeArena.open(b, bytes, budget) catch return false;
    return true;
}
export fn qwen4NativeArenaTake(out: *NativeArena, tensor: *NativeArena.Tensor, rows: usize, width: usize) bool {
    tensor.* = out.take(rows, width, .bf16) catch return false;
    return true;
}
export fn qwen4NativeArenaClose(out: *NativeArena) bool {
    out.close() catch return false;
    return true;
}

const NativeHc = family.cuda_hc.Kernels(cuda);
export fn qwen4CudaHcLoad(out: *NativeHc, d: *const cuda.Driver, image: [*]const u8, len: usize, g: *const family.hc_layout.Geometry) bool {
    out.* = NativeHc.load(d, image[0..len], g.*) catch return false;
    return true;
}
export fn qwen4CudaHcEncode(out: *NativeHc, s: *const cuda.Stream, op: u8, p: *const family.hc_layout.Call, views: [*]const NativeHc.View, n: usize) bool {
    if (op > 3) return false;
    out.encode(s.*, @fromBackingInt(@as(u2, @intCast(op))), p.*, views[0..n]) catch return false;
    return true;
}
export fn qwen4CudaHcClose(out: *NativeHc, s: *const cuda.Stream) bool {
    out.close(s.*) catch return false;
    return true;
}

const Pointwise = @import("core").hc_pointwise_cuda_adapter(@import("core").hc_pointwise, cuda);
export fn qwen4PointwiseCudaLoad(out: *Pointwise.Kernels, d: *const cuda.Driver, image: [*]const u8, n: usize) bool {
    out.* = Pointwise.Kernels.load(d, image[0..n], true) catch return false;
    return true;
}
export fn qwen4PointwiseCudaEncode(out: *Pointwise.Kernels, s: *const cuda.Stream, p: *const @import("core").hc_pointwise.Params, v: *const [4]Pointwise.View) bool {
    out.encode(s.*, p.*, v.*) catch return false;
    return true;
}
export fn qwen4PointwiseCudaClose(out: *Pointwise.Kernels, s: *const cuda.Stream) bool {
    out.close(s.*) catch return false;
    return true;
}

export fn qwen4ContractSuiteDestroy(out: *FullSuite) void {
    out.destroy(std.heap.page_allocator);
}
const PointwiseCudaCase = @import("tests/cuda/contract_hc_pointwise.zig").Case;
export fn qwen4PointwiseCudaContract(case: *PointwiseCudaCase) bool {
    return (@import("core").contract.gates.one(std.heap.page_allocator, .shapes, case.kernel(), 0x367) catch return false) == .pass;
}

const CudaGpu = family.cuda_gpu.Gpu(cuda);
export fn qwen4CudaGpuLoad(out: *CudaGpu, b: *Storage, c: *const family.Config, p: *family.Problem, images: *const CudaGpu.Images) bool {
    out.* = CudaGpu.load(std.heap.page_allocator, b, c, p, images.*, .hf) catch return false;
    return true;
}
export fn qwen4CudaGpuClose(out: *CudaGpu) bool {
    out.close() catch return false;
    return true;
}
export fn qwen4CudaGraph(out: *family.arena.Arena(Storage).Tensor, gpu: *CudaGpu, model: *family.model.Model(Storage), scratch: *family.arena.Arena(Storage), state: *family.state.State(Storage), tokens: [*]const u32, n: usize, problem: *family.Problem) bool {
    var ls: [family.config.max_layers]family.graph.Layer = undefined;
    const geometry = family.graph.geometry(model.c, family.graph.layers(model.c, &ls) catch return false) catch return false;
    var ops: family.gpu_ops.Ops(CudaGpu) = .{ .gpu = gpu, .model = model, .scratch = scratch, .host = std.heap.page_allocator, .problem = problem };
    out.* = family.graph.forward(&ops, geometry, state, tokens[0..n]) catch return false;
    return true;
}
const CudaBatch = family.batch_state.Batch(NativeState);
export fn qwen4CudaBatchGraph(out: *NativeArena.Tensor, gpu: *CudaGpu, model: *NativeModel, scratch: *NativeArena, state: *CudaBatch, tokens: [*]const u32, n: usize, problem: *family.Problem) bool {
    var ls: [family.config.max_layers]family.graph.Layer = undefined;
    const geometry = family.graph.geometry(model.c, family.graph.layers(model.c, &ls) catch return false) catch return false;
    var ops: family.gpu_ops.Ops(CudaGpu) = .{ .gpu = gpu, .model = model, .scratch = scratch, .host = std.heap.page_allocator, .problem = problem };
    out.* = family.graph.forward(&ops, geometry, state, tokens[0..n]) catch return false;
    return true;
}
const CudaEngine = family.engine.Engine(CudaGpu);
const CudaLoader = struct {
    images: CudaGpu.Images,
    pub fn load(l: @This(), a: std.mem.Allocator, b: *Storage, c: *const family.Config, p: *family.Problem) !CudaGpu {
        return CudaGpu.load(a, b, c, p, l.images, .hf);
    }
};
export fn qwen4CudaEngineOpen(out: **CudaEngine, io: *const std.Io, dir: [*]const u8, n: usize, b: *Storage, o: *const family.native_admission.Options, p: *family.Problem, images: *const CudaGpu.Images) bool {
    out.* = CudaEngine.open(std.heap.page_allocator, io.*, dir[0..n], b, o.*, p, CudaLoader{ .images = images.* }) catch return false;
    return true;
}
export fn qwen4CudaEngineForward(e: *CudaEngine, items: [*]const CudaEngine.Segment, n: usize, p: *family.Problem) bool {
    e.forward(items[0..n], CudaEngine.NoDraw{}, p) catch return false;
    return true;
}
export fn qwen4CudaEngineClose(e: *CudaEngine) bool {
    e.close() catch return false;
    return true;
}
const DrawLanes = struct {
    pub const Sampling = core.lanes_sampling.Sampling;
    pub const gpu_rule = core.lanes_gpu_rule;
};
const Drawer = family.cuda_draw.Drawer(cuda, DrawLanes, Storage);
export fn qwen4CudaDrawerLoad(out: *Drawer, b: *Storage, vocab: usize, rows: usize, streams: usize, sample: [*]const u8, sn: usize, argmax: [*]const u8, an: usize) bool {
    out.* = Drawer.load(b, vocab, rows, streams, sample[0..sn], argmax[0..an]) catch return false;
    return true;
}
export fn qwen4CudaDrawerRun(d: *Drawer, x: *const NativeArena.Tensor, off: usize, positions: [*]const u64, n: usize, sampling: ?*const DrawLanes.Sampling, bits: [*]const u32, nb: usize) bool {
    d.draw(x.*, off, positions[0..n], if (sampling) |s| s.* else null, bits[0..nb]) catch return false;
    return true;
}
export fn qwen4CudaDrawerRead(d: *Drawer, off: usize, out: [*]u32, n: usize) bool {
    d.read(off, out[0..n]) catch return false;
    return true;
}
export fn qwen4CudaDrawerClose(d: *Drawer) bool {
    d.close() catch return false;
    return true;
}

const ServeApi = @import("engine_api");
export fn qwen4CudaServe(out: *ServeApi.Opened, io: *const std.Io, o: *const ServeApi.Open) bool {
    var problem: []const u8 = "";
    out.* = (@import("src/native/qwen4_cuda.zig").open(std.heap.page_allocator, std.heap.page_allocator, io.*, o.*, &problem) catch return false) orelse return false;
    return true;
}
export fn qwen4CudaHostBodies(out: *ServeApi.Opened, io: *const std.Io, o: *const ServeApi.Open, problem: *family.Problem) bool {
    out.* = @import("src/native/qwen4_cuda.zig").Native.open(std.heap.page_allocator, io.*, o.*, problem) catch return false;
    return true;
}

const CudaPleSuite = @import("tests/cuda/contract_qwen4_ple.zig").Suite;
export fn qwen4NativeCudaPleSuiteOpen(gpu: *const @import("tests/cuda/check.zig").Gpu, stream: *const cuda.Stream, image: [*]const u8, len: usize, kernels: *std.ArrayList(@import("core").contract.gates.Kernel)) ?*CudaPleSuite {
    return CudaPleSuite.createImage(std.heap.page_allocator, gpu.*, stream.*, image[0..len], kernels) catch null;
}
export fn qwen4NativeCudaPleSuiteClose(s: *CudaPleSuite) void {
    s.destroy(std.heap.page_allocator);
}

const CudaDenseSuite = @import("tests/cuda/contract_qwen4_dense.zig").Suite;
export fn qwen4NativeCudaDenseSuiteOpen(gpu: *const @import("tests/cuda/check.zig").Gpu, stream: *const cuda.Stream, image: [*]const u8, len: usize, kernels: *std.ArrayList(@import("core").contract.gates.Kernel)) ?*CudaDenseSuite {
    return CudaDenseSuite.createImage(std.heap.page_allocator, gpu.*, stream.*, image[0..len], kernels) catch null;
}
export fn qwen4NativeCudaDenseSuiteClose(s: *CudaDenseSuite) void {
    s.destroy(std.heap.page_allocator);
}

const CudaQsaNormSuite = @import("tests/cuda/contract_qwen4_qsa_norm.zig").Suite;
export fn qwen4NativeCudaQsaNormSuiteOpen(gpu: *const @import("tests/cuda/check.zig").Gpu, stream: *const cuda.Stream, image: [*]const u8, len: usize, kernels: *std.ArrayList(@import("core").contract.gates.Kernel)) ?*CudaQsaNormSuite {
    return CudaQsaNormSuite.createImage(std.heap.page_allocator, gpu.*, stream.*, image[0..len], kernels) catch null;
}
export fn qwen4NativeCudaQsaNormSuiteClose(s: *CudaQsaNormSuite) void {
    s.destroy(std.heap.page_allocator);
}

const CudaQsaScoresSuite = @import("tests/cuda/contract_qwen4_qsa_scores.zig").Suite;
export fn qwen4NativeCudaQsaScoresSuiteOpen(gpu: *const @import("tests/cuda/check.zig").Gpu, stream: *const cuda.Stream, image: [*]const u8, len: usize, kernels: *std.ArrayList(@import("core").contract.gates.Kernel)) ?*CudaQsaScoresSuite {
    return CudaQsaScoresSuite.createImage(std.heap.page_allocator, gpu.*, stream.*, image[0..len], kernels) catch null;
}
export fn qwen4NativeCudaQsaScoresSuiteClose(s: *CudaQsaScoresSuite) void {
    s.destroy(std.heap.page_allocator);
}

const CudaQsaSelectSuite = @import("tests/cuda/contract_qwen4_qsa_select.zig").Suite;
export fn qwen4NativeCudaQsaSelectSuiteOpen(gpu: *const @import("tests/cuda/check.zig").Gpu, stream: *const cuda.Stream, image: [*]const u8, len: usize, kernels: *std.ArrayList(@import("core").contract.gates.Kernel)) ?*CudaQsaSelectSuite {
    return CudaQsaSelectSuite.createImage(std.heap.page_allocator, gpu.*, stream.*, image[0..len], kernels) catch null;
}
export fn qwen4NativeCudaQsaSelectSuiteClose(s: *CudaQsaSelectSuite) void {
    s.destroy(std.heap.page_allocator);
}

const CudaQsaAttnSuite = @import("tests/cuda/contract_qwen4_qsa_attn.zig").Suite;
export fn qwen4NativeCudaQsaAttnSuiteOpen(gpu: *const @import("tests/cuda/check.zig").Gpu, stream: *const cuda.Stream, image: [*]const u8, len: usize, kernels: *std.ArrayList(@import("core").contract.gates.Kernel)) ?*CudaQsaAttnSuite {
    return CudaQsaAttnSuite.createImage(std.heap.page_allocator, gpu.*, stream.*, image[0..len], kernels) catch null;
}
export fn qwen4NativeCudaQsaAttnSuiteClose(s: *CudaQsaAttnSuite) void {
    s.destroy(std.heap.page_allocator);
}
