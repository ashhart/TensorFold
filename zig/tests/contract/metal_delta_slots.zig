//! Register-slot DeltaNet pipelines at the 27B geometry, checked against a parent-indexed FP64 recurrence.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const ref = contract.delta_slots_ref;
const declared = contract.delta_slots_pending;
fn expect(ok: bool) !void {
    if (!ok) return error.DeltaSlotsContractFailed;
}
pub fn run(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, only: ?[]const u8, seed: u64) !void {
    comptime {
        std.debug.assert(@sizeOf(ref.Params) == @sizeOf(tf.qwen27.gdn_contract.Params));
        std.debug.assert(@sizeOf(ref.Segment) == @sizeOf(tf.qwen27.gdn_contract.Segment));
        for (@typeInfo(ref.Params).@"struct".field_names) |name|
            std.debug.assert(@offsetOf(ref.Params, name) == @offsetOf(tf.qwen27.gdn_contract.Params, name));
        for (@typeInfo(ref.Segment).@"struct".field_names) |name|
            std.debug.assert(@offsetOf(ref.Segment, name) == @offsetOf(tf.qwen27.gdn_contract.Segment, name));
    }
    for (declared.records, declared.counts) |record, slots| {
        if (only) |filter| if (std.mem.indexOf(u8, record.name, filter) == null) continue;
        const symbol = record.name[std.mem.lastIndexOfScalar(u8, record.name, ' ').? + 1 ..];
        const pipeline = try rig_.pipeline(device, sources.core_deltanet, symbol);
        defer pipeline.deinit();
        try expect(pipeline.simdWidth() == 32 and pipeline.maxThreads() >= 128 and rig_.tgMemory(pipeline) == 0);
        for ([_]usize{ 1, 3, if (slots == 4) 15 else 31 }) |n| {
            var arena = std.heap.ArenaAllocator.init(a);
            defer arena.deinit();
            const f = try ref.make(tf.tree_slots, arena.allocator(), slots, n, n == 1, seed);
            try one(a, device, queue, pipeline, f);
        }
        std.debug.print("PASS {s}: FP64, two streams, branch reuse, high slot, in-place state and backup; corruption RED\n", .{record.name});
    }
}
fn one(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, pipeline: mtl.Pipeline, f: ref.Fixture) !void {
    var rig = rig_.Rig{ .a = a, .device = device, .queue = queue };
    defer rig.deinit();
    const inputs = [_]rig_.Ref{
        try rig.input(std.mem.sliceAsBytes(f.query)),  try rig.input(std.mem.sliceAsBytes(f.key)),
        try rig.input(std.mem.sliceAsBytes(f.value)),  try rig.input(std.mem.sliceAsBytes(f.decay)),
        try rig.input(std.mem.sliceAsBytes(f.mixing)),
    };
    const state = try rig.scratch(f.state.len * 4);
    const parents = try rig.input(std.mem.sliceAsBytes(f.parents));
    const segments = try rig.input(std.mem.asBytes(&f.segments));
    const output = try rig.output(f.output.len * 2);
    const snapshots = try rig.scratch(256);
    const steps = try rig.input(std.mem.sliceAsBytes(f.steps));
    const before = try rig.scratch(f.state.len * 4);
    const original_inputs = try a.dupe(u8, try rig.inputBytes());
    defer a.free(original_inputs);
    var first: ?[]u8 = null;
    defer if (first) |bytes| a.free(bytes);
    for (0..5) |_| {
        @memcpy(rig.bytes(state, f.state.len * 4), std.mem.sliceAsBytes(f.state));
        @memset(rig.bytes(before, f.state.len * 4), 0xa5);
        const pass = rig.begin();
        const e = pass.e;
        e.setPipeline(pipeline);
        for (inputs, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
        e.setBuffer(state.buf, state.off, 5);
        e.setBuffer(parents.buf, parents.off, 6);
        e.setBuffer(segments.buf, segments.off, 7);
        e.setValue(f.p, 8);
        e.setBuffer(output.buf, output.off, 9);
        e.setBuffer(snapshots.buf, snapshots.off, 10);
        e.setBuffer(steps.buf, steps.off, 11);
        e.setBuffer(state.buf, state.off, 12);
        e.setBuffer(before.buf, before.off, 13);
        e.dispatchThreads(mtl.Size.of(32, 128, 48 * 2), mtl.Size.of(32, 4, 1));
        try rig.finish(pass);
        const bytes = rig.bytes(output, f.output.len * 2);
        try expect(f.accepts(bytes));
        const corrupted = try a.dupe(u8, bytes);
        defer a.free(corrupted);
        @memset(corrupted[0..2], 0x40);
        try expect(!f.accepts(corrupted));
        if (first) |saved| try expect(std.mem.eql(u8, saved, bytes)) else first = try a.dupe(u8, bytes);
        for (f.advanced, f.state_bound, 0..) |want, bound, i| {
            const got = contract.spec.read(.f32, rig.bytes(state, f.state.len * 4), i);
            try expect(std.math.isFinite(got) and @abs(got - want) <= bound);
            const saved = rig.bytes(before, f.state.len * 4)[i * 4 ..][0..4];
            const touched = f.p.advance != 0 and i / (48 * 128 * 128) != 1;
            if (touched) {
                try expect(contract.spec.read(.f32, saved, 0) == f.state[i]);
            } else try expect(std.mem.readInt(u32, saved, .little) == 0xa5a5a5a5);
        }
        for (rig.bytes(snapshots, 256)) |byte| try expect(byte == 0xa5);
        try expect(try rig.intact() and !rig.overlap() and std.mem.eql(u8, original_inputs, try rig.inputBytes()));
    }
}
