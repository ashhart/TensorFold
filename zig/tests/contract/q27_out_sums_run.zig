//! Every real layer's producer sums and out-projection bytes, plus the actual paged and pre-tensor-unit sources.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const ct = @import("contract");
const panel = @import("metal_q27_out_sums.zig");
const q = tf.qwen27;
const K = 6144;

fn realInputs(p: *panel.Probe, norm: ?q.gpu_weights.Tensor, eps: f32, seed: u64) !void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    if (norm) |weight| {
        const rec: [*]u16 = @ptrCast(@alignCast(p.ins[0].buf.contents() + p.ins[0].off));
        const gate: [*]u16 = @ptrCast(@alignCast(p.ins[1].buf.contents() + p.ins[1].off));
        for (0..p.rows * K) |i| {
            const x = (rng.float(f32) * 2 - 1) * (if (i % 119 == 0) @as(f32, 1e10) else 1);
            rec[i] = switch (i % 127) {
                0 => 0,
                1 => 0x8000,
                2 => 1,
                else => ct.spec.toBf16(x),
            };
            gate[i] = ct.spec.toBf16(rng.float(f32) * 4 - 2);
        }
        const r = q.gpu_weights.ref(weight);
        p.ins[2] = .{ .buf = r.buffer, .off = r.offset };
        p.p.weight_flags = if (weight.dtype == .f32) 8 else if (weight.dtype == .bf16) 0 else return error.NormKind;
        p.p.eps = eps;
    } else {
        const gates = try p.a.alloc(u16, p.rows * 24 * 512);
        defer p.a.free(gates);
        for (gates) |*v| v.* = ct.spec.toBf16(rng.float(f32) * 4 - 2);
        p.gate = try p.rig.input(std.mem.sliceAsBytes(gates));
    }
    try p.baseline();
}

fn compare(p: *panel.Probe) !void {
    try p.run();
    const xs = p.rig.bytes(p.xs, 96 * p.rows * 4);
    if (!std.mem.eql(u8, xs, p.expected)) return error.FusedSumsDiffer;
    if (!try p.rig.intact() or p.rig.overlap()) return error.SumsGuards;
}

fn projection(m: *q.model.Model, p: *panel.Probe, linear: q.projection.Linear) !void {
    if (linear.k != K) return error.OutProjectionWidth;
    const bytes = @as(usize, p.rows) * linear.n * 2;
    const old = try p.rig.scratch(bytes);
    const fused = try p.rig.scratch(bytes);
    var ids: [16]u32 = @splat(1);
    var positions: [16]i32 = @splat(0);
    try m.frame.setRows(ids[0..p.rows], positions[0..p.rows]);
    const pass = p.rig.begin();
    const input = q.projection.Ref{ .buffer = p.stored[0].buf, .offset = p.stored[0].off };
    try m.kernels.quant(pass.e, linear, input, null, m.frame.get(.dims), .{ .buffer = old.buf, .offset = old.off }, p.rows);
    pass.e.barrier();
    try m.kernels.quant(pass.e, linear, .{ .buffer = p.stored[1].buf, .offset = p.stored[1].off }, .{ .ref = .{ .buffer = p.xs.buf, .offset = p.xs.off }, .stride = p.stride }, m.frame.get(.dims), .{ .buffer = fused.buf, .offset = fused.off }, p.rows);
    pass.e.barrier();
    try p.rig.finish(pass);
    if (!std.mem.eql(u8, p.rig.bytes(old, bytes), p.rig.bytes(fused, bytes))) return error.OutProjectionBytesDiffer;
    if (!try p.rig.intact()) return error.ProjectionGuards;
}

fn emit(a: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    for ([_]bool{ false, true }) |units| for ([_]bool{ false, true }) |paged| {
        const source = try q.simd_attention.grouped(a, units, true, 6, 4, paged);
        defer a.free(source);
        const path = try std.fmt.allocPrint(a, "{s}/attention-{s}-{s}.metal", .{ dir, if (units) "tensor" else "simd", if (paged) "paged" else "contiguous" });
        defer a.free(path);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = source });
    };
    const path = try std.fmt.allocPrint(a, "{s}/deltanet.metal", .{dir});
    defer a.free(path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = @import("kernel_sources").core_deltanet });
}

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len == 3 and std.mem.eql(u8, args[1], "--emit-sources")) return emit(a, init.io, args[2]);
    if (args.len != 2) return error.ExpectedCheckpoint;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = try q.model.Model.load(a, init.io, args[1], 128, null);
    defer m.deinit();
    if (!m.device.tensorUnits() or m.config.gdnValueDim() != K or m.config.heads * m.config.head_dim != K) return error.NeedQwen27TensorUnits;
    var probes = [_]panel.Probe{ try panel.Probe.init(a, m.device, m.queue, .gdn, false), try panel.Probe.init(a, m.device, m.queue, .attention, false) };
    defer for (&probes) |*p| p.deinit();
    var cells: usize = 0;
    var checked: usize = 0;
    for (m.weights.layers, 0..) |layer, index| for ([_]u32{ 1, 5, 16 }) |rows| {
        const kind: usize = if (layer.mixer == .linear) 0 else 1;
        const p = &probes[kind];
        if (!try p.prepare(.{ .rows = rows, .n = 96, .k = 64 }, .random, index * 103 + rows)) return error.BadFixture;
        const norm = switch (layer.mixer) {
            .linear => |gdn| gdn.norm,
            .attention => null,
        };
        const linear = switch (layer.mixer) {
            .linear => |gdn| gdn.out,
            .attention => |attn| attn.out,
        };
        try realInputs(p, norm, m.config.eps, index * 211 + rows);
        for (0..5) |_| try compare(p);
        try projection(m, p, linear);
        cells += 1;
        checked += 96 * rows;
        std.debug.print("PASS layer={d} kind={s} rows={d}: producer, {d} group sums and out-projection bytes equal\n", .{ index, if (kind == 0) "gdn" else "attention", rows, 96 * rows });
    };
    std.debug.print("OUT_SUMS layers={d} cells={d} groups={d}, five repeats each, every projection byte equal\n", .{ m.weights.layers.len, cells, checked });
}
