//! Emits the real backend launch bodies without a shader compiler, device execution or model assets.
const std = @import("std");
const cuda = @import("cuda");
const metal = @import("metal");
const cu = @import("cuda_projection");
const mt = @import("metal_projection");
const wo = @import("core").weight_only;
export fn qwen4CudaProjectionLoad(out: *cu.Kernels, driver: *const cuda.Driver, image: [*]const u8, len: usize) bool {
    out.* = cu.Kernels.load(driver, image[0..len], true) catch return false;
    return true;
}
export fn qwen4CudaProjectionRun(k: *cu.Kernels, stream: *const cuda.Stream, p: *const wo.Params, views: *const [5]cu.View) bool {
    k.testEncode(stream.*, p.*, views[0], views[1], views[2], views[3], views[4], .production, 0) catch return false;
    return true;
}
export fn qwen4CudaProjectionClose(k: *cu.Kernels, stream: *const cuda.Stream) bool {
    k.close(stream.*) catch return false;
    return true;
}
export fn qwen4MetalProjectionLoad(out: *mt.Kernels, device: *const metal.Device) bool {
    const source = mt.source(std.heap.page_allocator, @embedFile("kernels/metal/core/weight_only.h"), @embedFile("kernels/metal/core/weight_only.metal"), true) catch return false;
    defer std.heap.page_allocator.free(source);
    out.* = mt.Kernels.load(device.*, source, true) catch return false;
    return true;
}
export fn qwen4MetalProjectionRun(k: *mt.Kernels, encoder: *const metal.ComputeEncoder, p: *const wo.Params, views: *const [5]mt.View) bool {
    k.testEncode(encoder.*, p.*, views[0], views[1], views[2], views[3], views[4], .production, 0) catch return false;
    return true;
}
export fn qwen4MetalProjectionClose(k: *mt.Kernels) void {
    k.deinit();
}

const cuda_case = @import("tests/cuda/contract_weight_only.zig");
const metal_case = @import("tests/contract/metal_weight_only.zig");
export fn qwen4CudaContractBodies(case: *cuda_case.Case) bool {
    const result = @import("core").contract.gates.one(std.heap.page_allocator, .shapes, case.kernel(), 0x367) catch return false;
    return result == .pass;
}
export fn qwen4MetalContractBodies(case: *metal_case.Case) bool {
    const result = @import("contract").gates.one(std.heap.page_allocator, .shapes, case.kernel(), 0x367) catch return false;
    return result == .pass;
}

const family = @import("qwen4").family;
const CudaStorage = family.cuda_storage.Backend(cuda);
const MetalStorage = family.metal_storage.Backend(metal);
const CudaPlaced = family.device_weights.Placed(CudaStorage);
const MetalPlaced = family.device_weights.Placed(MetalStorage);
export fn qwen4CudaPlacement(out: *CudaPlaced, backend: *CudaStorage, reader: *family.reader.Reader, config: *const family.Config, budget: u64, problem: *family.Problem) bool {
    out.* = CudaPlaced.open(std.heap.page_allocator, backend, reader, config, budget, problem) catch return false;
    return true;
}
export fn qwen4CudaPlacementClose(out: *CudaPlaced) bool {
    out.close() catch return false;
    return true;
}
export fn qwen4MetalPlacement(out: *MetalPlaced, backend: *MetalStorage, reader: *family.reader.Reader, config: *const family.Config, budget: u64, problem: *family.Problem) bool {
    out.* = MetalPlaced.open(std.heap.page_allocator, backend, reader, config, budget, problem) catch return false;
    return true;
}
export fn qwen4MetalPlacementClose(out: *MetalPlaced) bool {
    out.close() catch return false;
    return true;
}

const MetalGdn = family.metal_gdn.Kernels(metal);
export fn qwen4MetalGdnLoad(out: *MetalGdn, device: *const metal.Device, text: [*]const u8, len: usize, keys: usize, values: usize) bool {
    out.* = MetalGdn.load(device.*, text[0..len], keys, values, .silu) catch return false;
    return true;
}
export fn qwen4MetalGdnEncode(out: *MetalGdn, encoder: *const metal.ComputeEncoder, p: *const family.cuda_gdn.Params, views: *const [10]MetalGdn.View) bool {
    out.encode(encoder.*, p.*, views.*) catch return false;
    return true;
}
export fn qwen4MetalGdnClose(out: *MetalGdn) void {
    out.deinit();
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
export fn qwen4ContractSuiteOpen(out: *FullSuite, gpu: *const @import("tests/cuda/check.zig").Gpu, s: *const cuda.Stream, image: [*]const u8, len: usize, kernels: *std.ArrayList(@import("core").contract.gates.Kernel)) bool {
    out.openImages(std.heap.page_allocator, gpu.*, s.*, image[0..len], image[0..len], kernels) catch return false;
    return true;
}

const HcMetal = @import("core").hc_activation_metal_adapter(@import("core").hc_activation, metal);
export fn qwen4HcMetalLoad(out: *HcMetal.Kernels, d: *const metal.Device) bool {
    const sources = @import("kernel_sources");
    const text = HcMetal.source(std.heap.page_allocator, sources.core_weight_only_body, sources.core_hc_activation_body, sources.core_hc_activation, true) catch return false;
    defer std.heap.page_allocator.free(text);
    out.* = HcMetal.Kernels.load(d.*, text, true) catch return false;
    return true;
}
export fn qwen4HcMetalEncode(k: *HcMetal.Kernels, e: *const metal.ComputeEncoder, p: *const @import("core").hc_activation.Params, x: *const HcMetal.View, y: *const HcMetal.View) bool {
    k.encode(e.*, p.*, x.*, y.*) catch return false;
    return true;
}
const HcMetalCase = @import("tests/contract/metal_hc_activation.zig").Case;
const NativeHcSuite = @import("tests/contract/metal_qwen4_hc.zig").Suite;
export fn qwen4NativeHcSuiteOpen(device: *const metal.Device, queue: *const metal.Queue, kernels: *std.ArrayList(@import("contract").gates.Kernel)) ?*NativeHcSuite {
    return NativeHcSuite.create(std.heap.page_allocator, device.*, queue.*, kernels, null) catch null;
}
export fn qwen4NativeHcSuiteClose(s: *NativeHcSuite) void {
    s.destroy();
}
const NativeGdnSuite = @import("tests/contract/metal_qwen4_gdn.zig").Suite;
const NativeCudaGdnSuite = @import("tests/cuda/contract_qwen4_gdn.zig").Suite;
export fn qwen4NativeCudaGdnSuiteOpen(gpu: *const @import("tests/cuda/check.zig").Gpu, s: *const cuda.Stream, image: [*]const u8, len: usize, kernels: *std.ArrayList(@import("core").contract.gates.Kernel)) ?*NativeCudaGdnSuite {
    return NativeCudaGdnSuite.createImage(std.heap.page_allocator, gpu.*, s.*, image[0..len], kernels) catch null;
}
export fn qwen4NativeCudaGdnSuiteClose(s: *NativeCudaGdnSuite) void {
    s.destroy(std.heap.page_allocator);
}
export fn qwen4NativeGdnSuiteOpen(device: *const metal.Device, queue: *const metal.Queue, kernels: *std.ArrayList(@import("contract").gates.Kernel)) ?*NativeGdnSuite {
    return NativeGdnSuite.create(std.heap.page_allocator, device.*, queue.*, kernels, null) catch null;
}
export fn qwen4NativeGdnSuiteClose(s: *NativeGdnSuite) void {
    s.destroy();
}
export fn qwen4HcMetalContract(case: *HcMetalCase) bool {
    return (@import("contract").gates.one(std.heap.page_allocator, .shapes, case.kernel(), 0x367) catch return false) == .pass;
}

const NativeState = family.state.State(MetalStorage);
export fn qwen4NativeStateOpen(out: *NativeState, b: *MetalStorage, c: *const family.Config, capacity: usize, window: usize, budget: u64) bool {
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

const NativeSnapshot = family.snapshot.Snapshot(MetalStorage);
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

const NativeModel = family.model.Model(MetalStorage);
export fn qwen4NativeModelBind(out: *NativeModel, weights: *family.device_weights.Placed(MetalStorage), problem: *family.Problem) bool {
    out.* = NativeModel.init(std.heap.page_allocator, weights, problem) catch return false;
    return true;
}
export fn qwen4NativeModelClose(out: *NativeModel) void {
    out.deinit();
}

const NativeArena = family.arena.Arena(MetalStorage);
export fn qwen4NativeArenaOpen(out: *NativeArena, b: *MetalStorage, bytes: usize, budget: u64) bool {
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

const MetalHc = family.metal_hc.Kernels(metal);
export fn qwen4MetalHcLoad(out: *MetalHc, d: *const metal.Device, g: *const family.hc_layout.Geometry) bool {
    out.* = MetalHc.load(std.heap.page_allocator, d.*, g.*) catch return false;
    return true;
}
export fn qwen4MetalHcEncode(out: *MetalHc, e: *const metal.ComputeEncoder, op: u8, p: *const family.hc_layout.Call, views: [*]const MetalHc.View, n: usize) bool {
    if (op > 3) return false;
    out.encode(e.*, @fromBackingInt(@as(u2, @intCast(op))), p.*, views[0..n]) catch return false;
    return true;
}
export fn qwen4MetalHcClose(out: *MetalHc) void {
    out.deinit();
}

export fn qwen4MetalSessionBegin(out: *metal.CommandSession) bool {
    out.begin() catch return false;
    return true;
}
export fn qwen4MetalSessionEncoder(out: *metal.CommandSession, e: *metal.ComputeEncoder) bool {
    e.* = out.encoder() catch return false;
    return true;
}
export fn qwen4MetalSessionSubmit(out: *metal.CommandSession) bool {
    out.submit() catch return false;
    return true;
}
export fn qwen4MetalSessionPoll(out: *metal.CommandSession) bool {
    out.poll() catch return false;
    return true;
}
export fn qwen4MetalSessionDiscard(out: *metal.CommandSession) void {
    out.discard();
}
export fn qwen4MetalSessionClose(out: *metal.CommandSession) bool {
    out.deinit() catch return false;
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

const MetalPointwise = @import("core").hc_pointwise_metal_adapter(@import("core").hc_pointwise, metal);
export fn qwen4PointwiseMetalLoad(out: *MetalPointwise.Kernels, d: *const metal.Device) bool {
    const sources = @import("kernel_sources");
    const text = MetalPointwise.source(std.heap.page_allocator, sources.core_weight_only_body, sources.core_hc_activation_body, sources.core_hc_pointwise, true) catch return false;
    defer std.heap.page_allocator.free(text);
    out.* = MetalPointwise.Kernels.load(d.*, text, true) catch return false;
    return true;
}
export fn qwen4PointwiseMetalEncode(out: *MetalPointwise.Kernels, e: *const metal.ComputeEncoder, p: *const @import("core").hc_pointwise.Params, v: *const [4]MetalPointwise.View) bool {
    out.encode(e.*, p.*, v.*) catch return false;
    return true;
}
export fn qwen4PointwiseMetalClose(out: *MetalPointwise.Kernels) void {
    out.deinit();
}

export fn qwen4ContractSuiteDestroy(out: *FullSuite) void {
    out.destroy(std.heap.page_allocator);
}

const MetalAttention = family.metal_attention.Kernels(metal);
export fn qwen4MetalAttentionLoad(out: *MetalAttention, d: *const metal.Device, g: *const family.cuda_attention.Geometry) bool {
    out.* = MetalAttention.load(std.heap.page_allocator, d.*, g.*) catch return false;
    return true;
}
export fn qwen4MetalAttentionEncode(out: *MetalAttention, session: *metal.CommandSession, p: *const family.cuda_attention.Call, v: *const [18]MetalAttention.View) bool {
    out.encode(session, p.*, v.*) catch return false;
    return true;
}
export fn qwen4MetalAttentionClose(out: *MetalAttention) void {
    out.deinit();
}

const MetalGpu = family.metal_gpu.Gpu(metal);
export fn qwen4MetalGpuLoad(out: *MetalGpu, b: *MetalStorage, c: *const family.Config, p: *family.Problem) bool {
    out.* = MetalGpu.load(std.heap.page_allocator, b, c, p, .hf) catch return false;
    return true;
}
export fn qwen4MetalGpuClose(out: *MetalGpu) bool {
    out.close() catch return false;
    return true;
}
export fn qwen4MetalGraph(out: *NativeArena.Tensor, gpu: *MetalGpu, model: *NativeModel, scratch: *NativeArena, state: *NativeState, tokens: [*]const u32, n: usize, problem: *family.Problem) bool {
    var ls: [family.config.max_layers]family.graph.Layer = undefined;
    const geometry = family.graph.geometry(model.c, family.graph.layers(model.c, &ls) catch return false) catch return false;
    var ops: family.gpu_ops.Ops(MetalGpu) = .{ .gpu = gpu, .model = model, .scratch = scratch, .host = std.heap.page_allocator, .problem = problem };
    out.* = family.graph.forward(&ops, geometry, state, tokens[0..n]) catch return false;
    return true;
}
const MetalBatch = family.batch_state.Batch(NativeState);
export fn qwen4MetalBatchGraph(out: *NativeArena.Tensor, gpu: *MetalGpu, model: *NativeModel, scratch: *NativeArena, state: *MetalBatch, tokens: [*]const u32, n: usize, problem: *family.Problem) bool {
    var ls: [family.config.max_layers]family.graph.Layer = undefined;
    const geometry = family.graph.geometry(model.c, family.graph.layers(model.c, &ls) catch return false) catch return false;
    var ops: family.gpu_ops.Ops(MetalGpu) = .{ .gpu = gpu, .model = model, .scratch = scratch, .host = std.heap.page_allocator, .problem = problem };
    out.* = family.graph.forward(&ops, geometry, state, tokens[0..n]) catch return false;
    return true;
}
const MetalEngine = family.engine.Engine(MetalGpu);
const MetalLoader = struct {
    pub fn load(_: @This(), a: std.mem.Allocator, b: *MetalStorage, c: *const family.Config, p: *family.Problem) !MetalGpu {
        return MetalGpu.load(a, b, c, p, .hf);
    }
};
export fn qwen4MetalEngineOpen(out: **MetalEngine, io: *const std.Io, dir: [*]const u8, n: usize, b: *MetalStorage, o: *const family.native_admission.Options, p: *family.Problem) bool {
    out.* = MetalEngine.open(std.heap.page_allocator, io.*, dir[0..n], b, o.*, p, MetalLoader{}) catch return false;
    return true;
}
export fn qwen4MetalEngineForward(e: *MetalEngine, items: [*]const MetalEngine.Segment, n: usize, p: *family.Problem) bool {
    e.forward(items[0..n], MetalEngine.NoDraw{}, p) catch return false;
    return true;
}
export fn qwen4MetalEngineClose(e: *MetalEngine) bool {
    e.close() catch return false;
    return true;
}
const DrawLanes = struct {
    pub const Sampling = @import("core").lanes_sampling.Sampling;
    pub const gpu_rule = @import("core").lanes_gpu_rule;
};
const Drawer = family.metal_draw.Drawer(metal, DrawLanes, MetalStorage);
export fn qwen4MetalDrawerLoad(out: *Drawer, b: *MetalStorage, vocab: usize, rows: usize, streams: usize) bool {
    out.* = Drawer.load(std.heap.page_allocator, b, vocab, rows, streams) catch return false;
    return true;
}
export fn qwen4MetalDrawerRun(d: *Drawer, x: *const NativeArena.Tensor, off: usize, positions: [*]const u64, n: usize, sampling: ?*const DrawLanes.Sampling, bits: [*]const u32, nb: usize) bool {
    d.draw(x.*, off, positions[0..n], if (sampling) |s| s.* else null, bits[0..nb]) catch return false;
    return true;
}
export fn qwen4MetalDrawerRead(d: *Drawer, off: usize, out: [*]u32, n: usize) bool {
    d.read(off, out[0..n]) catch return false;
    return true;
}
export fn qwen4MetalDrawerClose(d: *Drawer) bool {
    d.close() catch return false;
    return true;
}
const PointwiseCudaCase = @import("tests/cuda/contract_hc_pointwise.zig").Case;
export fn qwen4PointwiseCudaContract(case: *PointwiseCudaCase) bool {
    return (@import("core").contract.gates.one(std.heap.page_allocator, .shapes, case.kernel(), 0x367) catch return false) == .pass;
}

const PointwiseMetalCase = @import("tests/contract/metal_hc_pointwise.zig").Case;
export fn qwen4PointwiseMetalContract(case: *PointwiseMetalCase) bool {
    return (@import("contract").gates.one(std.heap.page_allocator, .shapes, case.kernel(), 0x367) catch return false) == .pass;
}

const ServeApi = @import("engine_api");
export fn qwen4CudaServe(out: *ServeApi.Opened, io: *const std.Io, o: *const ServeApi.Open) bool {
    var problem: []const u8 = "";
    out.* = (@import("src/native/qwen4_cuda.zig").open(std.heap.page_allocator, std.heap.page_allocator, io.*, o.*, &problem) catch return false) orelse return false;
    return true;
}
export fn qwen4MetalServe(out: *ServeApi.Opened, io: *const std.Io, o: *const ServeApi.Open) bool {
    var problem: []const u8 = "";
    out.* = (@import("src/native/qwen4_metal.zig").open(std.heap.page_allocator, std.heap.page_allocator, io.*, o.*, &problem) catch return false) orelse return false;
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

const MetalPleSuite = @import("tests/contract/metal_qwen4_ple.zig").Suite;
export fn qwen4NativeMetalPleSuiteOpen(device: *const metal.Device, queue: *const metal.Queue, kernels: *std.ArrayList(@import("contract").gates.Kernel)) ?*MetalPleSuite {
    return MetalPleSuite.create(std.heap.page_allocator, device.*, queue.*, kernels, null) catch null;
}
export fn qwen4NativeMetalPleSuiteClose(s: *MetalPleSuite) void {
    s.destroy();
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

const MetalQsaNormSuite = @import("tests/contract/metal_qwen4_qsa_norm.zig").Suite;
export fn qwen4NativeMetalQsaNormSuiteOpen(device: *const metal.Device, queue: *const metal.Queue, kernels: *std.ArrayList(@import("contract").gates.Kernel)) ?*MetalQsaNormSuite {
    return MetalQsaNormSuite.create(std.heap.page_allocator, device.*, queue.*, kernels, null) catch null;
}
export fn qwen4NativeMetalQsaNormSuiteClose(s: *MetalQsaNormSuite) void {
    s.destroy();
}

const CudaQsaScoresSuite = @import("tests/cuda/contract_qwen4_qsa_scores.zig").Suite;
export fn qwen4NativeCudaQsaScoresSuiteOpen(gpu: *const @import("tests/cuda/check.zig").Gpu, stream: *const cuda.Stream, image: [*]const u8, len: usize, kernels: *std.ArrayList(@import("core").contract.gates.Kernel)) ?*CudaQsaScoresSuite {
    return CudaQsaScoresSuite.createImage(std.heap.page_allocator, gpu.*, stream.*, image[0..len], kernels) catch null;
}
export fn qwen4NativeCudaQsaScoresSuiteClose(s: *CudaQsaScoresSuite) void {
    s.destroy(std.heap.page_allocator);
}

const MetalQsaScoresSuite = @import("tests/contract/metal_qwen4_qsa_scores.zig").Suite;
export fn qwen4NativeMetalQsaScoresSuiteOpen(device: *const metal.Device, queue: *const metal.Queue, kernels: *std.ArrayList(@import("contract").gates.Kernel)) ?*MetalQsaScoresSuite {
    return MetalQsaScoresSuite.create(std.heap.page_allocator, device.*, queue.*, kernels, null) catch null;
}
export fn qwen4NativeMetalQsaScoresSuiteClose(s: *MetalQsaScoresSuite) void {
    s.destroy();
}

const CudaQsaSelectSuite = @import("tests/cuda/contract_qwen4_qsa_select.zig").Suite;
export fn qwen4NativeCudaQsaSelectSuiteOpen(gpu: *const @import("tests/cuda/check.zig").Gpu, stream: *const cuda.Stream, image: [*]const u8, len: usize, kernels: *std.ArrayList(@import("core").contract.gates.Kernel)) ?*CudaQsaSelectSuite {
    return CudaQsaSelectSuite.createImage(std.heap.page_allocator, gpu.*, stream.*, image[0..len], kernels) catch null;
}
export fn qwen4NativeCudaQsaSelectSuiteClose(s: *CudaQsaSelectSuite) void {
    s.destroy(std.heap.page_allocator);
}

const MetalQsaSelectSuite = @import("tests/contract/metal_qwen4_qsa_select.zig").Suite;
export fn qwen4NativeMetalQsaSelectSuiteOpen(device: *const metal.Device, queue: *const metal.Queue, kernels: *std.ArrayList(@import("contract").gates.Kernel)) ?*MetalQsaSelectSuite {
    return MetalQsaSelectSuite.create(std.heap.page_allocator, device.*, queue.*, kernels, null) catch null;
}
export fn qwen4NativeMetalQsaSelectSuiteClose(s: *MetalQsaSelectSuite) void {
    s.destroy();
}

const CudaQsaAttnSuite = @import("tests/cuda/contract_qwen4_qsa_attn.zig").Suite;
export fn qwen4NativeCudaQsaAttnSuiteOpen(gpu: *const @import("tests/cuda/check.zig").Gpu, stream: *const cuda.Stream, image: [*]const u8, len: usize, kernels: *std.ArrayList(@import("core").contract.gates.Kernel)) ?*CudaQsaAttnSuite {
    return CudaQsaAttnSuite.createImage(std.heap.page_allocator, gpu.*, stream.*, image[0..len], kernels) catch null;
}
export fn qwen4NativeCudaQsaAttnSuiteClose(s: *CudaQsaAttnSuite) void {
    s.destroy(std.heap.page_allocator);
}

const MetalQsaAttnSuite = @import("tests/contract/metal_qwen4_qsa_attn.zig").Suite;
export fn qwen4NativeMetalQsaAttnSuiteOpen(device: *const metal.Device, queue: *const metal.Queue, kernels: *std.ArrayList(@import("contract").gates.Kernel)) ?*MetalQsaAttnSuite {
    return MetalQsaAttnSuite.create(std.heap.page_allocator, device.*, queue.*, kernels, null) catch null;
}
export fn qwen4NativeMetalQsaAttnSuiteClose(s: *MetalQsaAttnSuite) void {
    s.destroy();
}
