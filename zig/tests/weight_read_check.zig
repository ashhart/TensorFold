//! Check the actual native loaders with a synthetic file and bounded I/O faults.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");

extern "c" fn tf_probe_config(delay: c_uint, limit: usize, inject_eof: c_int) void;
extern "c" fn tf_probe_calls() u64;
extern "c" fn tf_probe_poison(path: [*:0]const u8) c_int;

const source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\kernel void copy_words(device const uint *x [[buffer(0)]], device uint *y [[buffer(1)]],
    \\    constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    \\    if (i < n) y[i] = x[i];
    \\}
;

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 4) return error.ExpectedFamilyFileMode;
    if (!std.mem.startsWith(u8, std.fs.path.basename(args[2]), "tensorfold-loader-probe-")) return error.TestFileRequired;
    const nemotron = std.mem.eql(u8, args[1], "nemotron");
    const grouped = std.mem.eql(u8, args[1], "flashnext-group");
    if (!nemotron and !grouped and !std.mem.eql(u8, args[1], "flashnext")) return error.UnknownFamily;
    const delayed = std.mem.eql(u8, args[3], "delayed");
    const failure_kind: c_int = if (std.mem.eql(u8, args[3], "eof")) 1 else if (std.mem.eql(u8, args[3], "eio")) 2 else if (std.mem.eql(u8, args[3], "eintr")) 3 else 0;
    const eof = failure_kind != 0;
    if (!delayed and !eof and !std.mem.eql(u8, args[3], "normal")) return error.UnknownMode;
    const path = try std.fmt.allocPrintSentinel(a, "{s}", .{args[2]}, 0);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const library = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
    defer library.deinit();
    const pipeline = try mtl.Pipeline.init(device, library, "copy_words", false);
    defer pipeline.deinit();
    var checkpoint = tf.checkpoint.Checkpoint.init(a);
    defer checkpoint.deinit();
    var replay: tf.flashnext_replay.Run = .{ .arena = a, .device = device, .queue = queue };
    const start = mtl.clock.seconds();
    tf_probe_config(if (delayed) 5000 else 0, if (delayed or eof) 32768 else 0, failure_kind);
    const allocated_before = mtl.objc.msg(usize, device.id, "currentAllocatedSize", .{});
    var input: mtl.Buffer = undefined;
    var offset: usize = 0;
    var bytes: usize = 0;
    if (nemotron) {
        checkpoint.addFile(device, path, "") catch |err| {
            if (!eof or err != error.ShortRead) return err;
            const allocated_after = mtl.objc.msg(usize, device.id, "currentAllocatedSize", .{});
            std.debug.print("{{\"family\":\"{s}\",\"mode\":\"{s}\",\"error\":\"ShortRead\",\"pread_calls\":{d},\"allocated_before\":{d},\"allocated_after\":{d},\"payload_check_reached\":false}}\n", .{ args[1], args[3], tf_probe_calls(), allocated_before, allocated_after });
            if (allocated_after != allocated_before) return error.BufferLeak;
            return;
        };
        const tensor = try checkpoint.get("probe");
        input = tensor.buffer;
        offset = tensor.offset;
        bytes = tensor.bytes;
    } else {
        try replay.indexFile(path);
        const entry = try replay.entry(if (grouped) "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight" else "probe");
        defer _ = std.c.close(entry.fd);
        const result = if (grouped) replay.group(0, 2, "weight") else replay.load("probe");
        const buffer = result catch |err| {
            if (!eof or err != error.ShortRead) return err;
            const allocated_after = mtl.objc.msg(usize, device.id, "currentAllocatedSize", .{});
            std.debug.print("{{\"family\":\"{s}\",\"mode\":\"{s}\",\"error\":\"ShortRead\",\"pread_calls\":{d},\"allocated_before\":{d},\"allocated_after\":{d},\"payload_check_reached\":false}}\n", .{ args[1], args[3], tf_probe_calls(), allocated_before, allocated_after });
            if (allocated_after != allocated_before) return error.BufferLeak;
            return;
        };
        input = buffer.b;
        bytes = if (grouped) entry.len * 2 else entry.len;
    }
    defer if (!nemotron) input.deinit();
    if (eof) return error.ExpectedShortRead;
    const load_seconds = mtl.clock.seconds() - start;
    const reads = tf_probe_calls();
    if (reads == 0) return error.InterpositionInactive;
    if (delayed and reads < 64) return error.PartialReadsNotTested;
    if (bytes != 2 * 1024 * 1024) return error.WrongTestSize;
    if (tf_probe_poison(path) != 0) return error.FilePoisonFailed;
    const count: u32 = @intCast(bytes / 4);
    const output = try device.buffer(bytes, mtl.ResourceOptions.shared);
    defer output.deinit();
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    enc.setPipeline(pipeline);
    enc.setBuffer(input, offset, 0);
    enc.setBuffer(output, 0, 1);
    enc.setValue(count, 2);
    enc.dispatchThreads(mtl.Size.of(count, 1, 1), mtl.Size.of(256, 1, 1));
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |failure| {
        std.debug.print("{s}\n", .{failure});
        return error.MetalFailure;
    }
    for (output.slice(u32, count), 0..) |value, index| {
        if (value != @as(u32, @intCast(index)) + 1) return error.PayloadMismatch;
    }
    if (tf_probe_calls() != reads) return error.ReadAfterLoad;
    std.debug.print("{{\"family\":\"{s}\",\"mode\":\"{s}\",\"payload_bytes\":{d},\"pread_calls\":{d},\"load_seconds\":{d:.6},\"gpu_seconds\":{d:.6},\"source_poisoned\":true,\"exact_words\":{d},\"reads_after_load\":0}}\n", .{ args[1], args[3], bytes, reads, load_seconds, cb.gpuSeconds(), count });
}
