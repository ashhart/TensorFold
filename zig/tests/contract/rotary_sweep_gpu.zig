//! One bounded chunk at a time, with captured scalar intermediates and poisoned tail/guard checks.
const std = @import("std");
const mtl = @import("metal");
const f = @import("tensorfold").flashnext_vision_contract;
const accuracy = f.rotary_accuracy;
const cpu = @import("rotary_sweep_cpu.zig");
const norm = @import("rotary_sweep_norm.zig");
const rig_ = @import("rig.zig");

pub fn run(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, row: *accuracy.Row) !void {
    const body = @import("kernel_sources").flashnext_rotary_sweep;
    const source = try std.mem.concat(a, u8, &.{ f.source(norm.which(row.role)), "\n", body });
    defer a.free(source);
    const pipeline = try rig_.pipeline(device, source, "tf_rotary_sweep");
    defer pipeline.deinit();
    var rig: rig_.Rig = .{ .a = a, .device = device, .queue = queue };
    defer rig.deinit();
    const table = accuracy.domain.table(row.mode);
    for (&row.table_bits, table) |*bits, value| bits.* = @bitCast(value);
    const input = try rig.input(std.mem.asBytes(&table));
    const capacity = accuracy.domain.chunk_cases;
    const output = try rig.output(capacity * @sizeOf(cpu.Sample));
    const count = accuracy.domain.count(row.role);
    var first: u64 = 0;
    while (first < count) {
        const n: u32 = @intCast(@min(capacity, count - first));
        const bytes = rig.bytes(output, capacity * @sizeOf(cpu.Sample));
        @memset(bytes, 0xff);
        const pass = rig.begin();
        pass.e.setPipeline(pipeline);
        pass.e.setBuffer(input.buf, input.off, 0);
        pass.e.setValue([4]u32{ @intCast(first), n, accuracy.domain.stride(row.role), 0 }, 1);
        pass.e.setBuffer(output.buf, output.off, 2);
        pass.e.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
        try rig.finish(pass);
        if (!try rig.intact() or !std.mem.eql(u8, rig.bytes(input, @sizeOf(@TypeOf(table))), std.mem.asBytes(&table))) return error.SweepChangedInputs;
        for (bytes[n * @sizeOf(cpu.Sample) ..]) |byte| if (byte != 0xff) return error.SweepWrotePastChunk;
        const samples: []align(1) const cpu.Sample = std.mem.bytesAsSlice(cpu.Sample, bytes[0 .. n * @sizeOf(cpu.Sample)]);
        for (samples, 0..) |sample, i| try cpu.observe(row, first + i, sample, &table);
        first += n;
    }
    if (row.cases != count) return error.IncompleteRotarySweep;
}
