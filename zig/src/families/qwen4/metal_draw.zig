//! Sampling and greedy picks reuse the maintained Metal kernels; metadata copies follow the inference queue.
const std = @import("std");
const edit = @import("../flashnext/geometry_source.zig");
pub fn source(a: std.mem.Allocator, index: usize, replay: []const u8) ![]u8 {
    return switch (index) {
        0 => a.dupe(u8, @embedFile("../../../kernels/metal/nemotron_sample.metal")),
        1 => std.mem.concat(a, u8, &.{ "#include <metal_stdlib>\nusing namespace metal;\n", try edit.function(replay, "kernel void fz_argmax(") }),
        2 => a.dupe(u8, @embedFile("../../../kernels/metal/core/grammar_mask.metal")),
        else => error.Qwen4DrawSource,
    };
}
pub fn Drawer(comptime M: type, comptime L: type, comptime B: type) type {
    const Tensor = @import("arena.zig").Arena(B).Tensor;
    return struct {
        const Self = @This();
        backend: *B,
        vocab: usize,
        rows: usize,
        words: usize,
        output: B.Buffer = undefined,
        meta: B.Buffer = undefined,
        pipes: [3]M.Pipeline = undefined,
        count: usize = 0,
        buffers: usize = 0,
        pub fn bytes(vocab: usize, rows: usize, streams: usize) !usize {
            if (vocab == 0 or rows == 0 or rows > 128 or streams == 0 or streams > 128 or rows > std.math.maxInt(u32) / vocab) return error.Qwen4DrawShape;
            return std.math.mul(usize, try std.math.add(usize, try std.math.mul(usize, rows, 1 + (vocab + 31) / 32), try std.math.add(usize, rows, streams)), 4);
        }
        pub fn load(a: std.mem.Allocator, b: *B, vocab: usize, rows: usize, streams: usize) !Self {
            if (b.commands == null) return error.Qwen4CommandSessionMissing;
            _ = try bytes(vocab, rows, streams);
            var out: Self = .{ .backend = b, .vocab = vocab, .rows = rows, .words = (vocab + 31) / 32 };
            errdefer out.deinitDrained();
            const names = [_][]const u8{ "tf_sample_full", "fz_argmax", "tf_grammar_mask" };
            for (names, 0..) |name, i| {
                const text = try source(a, i, @import("core").replay_glue);
                defer a.free(text);
                const lib = try M.Library.fromSource(b.device, text, M.CompileOptions.mlx());
                defer lib.deinit();
                out.pipes[i] = try M.Pipeline.init(b.device, lib, name, false);
                out.count += 1;
                if (out.pipes[i].maxThreads() < @as(usize, if (i < 2) 1024 else 256) or out.pipes[i].simdWidth() != 32) return error.Qwen4DrawThreads;
            }
            out.meta = try b.allocate(rows * (1 + out.words) * 4);
            out.buffers = 1;
            out.output = try b.allocate((rows + streams) * 4);
            out.buffers = 2;
            return out;
        }
        pub fn deinitDrained(d: *Self) void {
            if (d.buffers == 2) d.backend.release(d.output);
            if (d.buffers > 0) d.backend.release(d.meta);
            for (d.pipes[0..d.count]) |pipe| pipe.deinit();
            d.* = undefined;
        }
        pub fn close(d: *Self) !void {
            try d.backend.drain();
            d.deinitDrained();
        }
        pub fn draw(d: *Self, logits: Tensor, offset: usize, positions: []const u64, sampling: ?L.Sampling, bits: []const u32) !void {
            const n = positions.len;
            if (n == 0 or n > d.rows or logits.rows != n or logits.width != d.vocab or logits.kind != .bf16 or offset % 4 != 0 or offset > d.output.length() or n * 4 > d.output.length() - offset) return error.Qwen4DrawShape;
            const input = try logits.bytes();
            if (logits.buffer.deviceId() != d.backend.device.id or logits.offset > logits.buffer.length() or input > logits.buffer.length() - logits.offset) return error.Qwen4DrawBuffer;
            if (bits.len != 0 and bits.len != n * d.words) return error.Qwen4DrawMask;
            try @import("draw_inputs.zig").check(d.vocab, positions, sampling, bits);
            var pos: [128]u32 = undefined;
            for (positions, 0..) |p, i| {
                if (p > std.math.maxInt(i32)) return error.Qwen4DrawPosition;
                pos[i] = @intCast(p);
            }
            if (bits.len > 0) {
                try d.backend.write(d.meta, d.rows * 4, std.mem.sliceAsBytes(bits));
                const e = try d.backend.commands.?.encoder();
                e.setPipeline(d.pipes[2]);
                e.setBuffer(logits.buffer, logits.offset, 0);
                e.setBuffer(d.meta, d.rows * 4, 1);
                const dims = [_]u32{ @intCast(d.vocab), @intCast(d.words), @intCast(n), 0 };
                e.setBytes(std.mem.asBytes(&dims), 2);
                e.dispatchThreads(M.Size.of(d.words, n, 1), M.Size.of(256, 1, 1));
                e.barrier();
            }
            const active = if (sampling) |s| s.temperature > 0 else false;
            if (active) try d.backend.write(d.meta, 0, std.mem.sliceAsBytes(pos[0..n]));
            const e = try d.backend.commands.?.encoder();
            e.setPipeline(d.pipes[if (active) 0 else 1]);
            e.setBuffer(logits.buffer, logits.offset, 0);
            const v: u32 = @intCast(d.vocab);
            if (active) {
                const rule = L.gpu_rule.rule(sampling.?);
                var seeds: [256]u32 = undefined;
                var cfgs: [512]f32 = undefined;
                var caps: [128]u32 = undefined;
                for (0..n) |i| {
                    seeds[2 * i ..][0..2].* = rule.seedHalves();
                    cfgs[4 * i ..][0..4].* = rule.cfg();
                    caps[i] = rule.top_k;
                }
                e.setBytes(std.mem.sliceAsBytes(seeds[0 .. 2 * n]), 1);
                e.setBuffer(d.meta, 0, 2);
                e.setBytes(std.mem.sliceAsBytes(cfgs[0 .. 4 * n]), 3);
                e.setBytes(std.mem.sliceAsBytes(caps[0..n]), 4);
                e.setBuffer(d.output, offset, 5);
                e.setBuffer(d.output, offset, 6);
                e.setBytes(std.mem.asBytes(&v), 7);
            } else {
                e.setBuffer(d.output, offset, 1);
                e.setBytes(std.mem.asBytes(&v), 2);
            }
            e.dispatchGroups(M.Size.of(n, 1, 1), M.Size.of(1024, 1, 1));
            e.barrier();
        }
        pub fn read(d: *Self, offset: usize, out: []u32) !void {
            if (offset % 4 != 0 or offset > d.output.length() or out.len * 4 > d.output.length() - offset) return error.Qwen4DrawBuffer;
            try d.backend.drain();
            @memcpy(std.mem.sliceAsBytes(out), d.output.contents()[offset..][0 .. out.len * 4]);
            for (out) |token| if (token >= d.vocab) return error.Qwen4DrawToken;
        }
    };
}
