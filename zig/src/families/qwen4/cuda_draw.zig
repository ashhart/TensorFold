//! Keyed CUDA draws and raw BF16 argmax reuse the maintained images; readback contains token IDs only.
const std = @import("std");
pub fn Drawer(comptime C: type, comptime L: type, comptime B: type) type {
    const Tensor = @import("arena.zig").Arena(B).Tensor;
    return struct {
        const Self = @This();
        backend: *B,
        vocab: usize,
        rows: usize,
        words: usize,
        output: B.Buffer = undefined,
        meta: B.Buffer = undefined,
        sample: C.Module = undefined,
        argmax: C.Module = undefined,
        modules: usize = 0,
        buffers: usize = 0,
        draws: C.sample.Draws = undefined,
        greedy: C.Function = undefined,
        mask: C.Function = undefined,
        pub fn bytes(vocab: usize, rows: usize, streams: usize) !usize {
            if (vocab == 0 or rows == 0 or rows > 128 or streams == 0 or streams > 128 or rows > std.math.maxInt(i32) / vocab) return error.Qwen4DrawShape;
            const metadata = try std.math.mul(usize, rows, @sizeOf(C.sample.Rule) + 4 + 4 * ((vocab + 31) / 32));
            return std.math.add(usize, metadata, try std.math.mul(usize, try std.math.add(usize, rows, streams), 4));
        }
        pub fn load(b: *B, vocab: usize, rows: usize, streams: usize, sample: []const u8, argmax: []const u8) !Self {
            _ = try bytes(vocab, rows, streams);
            try b.enter();
            var out: Self = .{ .backend = b, .vocab = vocab, .rows = rows, .words = (vocab + 31) / 32 };
            errdefer out.deinitDrained();
            out.sample = try C.Module.load(b.driver, sample);
            out.modules = 1;
            out.argmax = try C.Module.load(b.driver, argmax);
            out.modules = 2;
            out.draws = try C.sample.Draws.resolve(out.sample);
            if (out.draws.draw_rows_fn == null) return error.Qwen4DrawRowsMissing;
            out.mask = try out.sample.function("tf_grammar_mask");
            out.greedy = try out.argmax.function("tf_argmax_rows_i32_kernel");
            if (try out.mask.attribute(.max_threads_per_block) < 256 or try out.greedy.attribute(.max_threads_per_block) < 256 or try out.draws.draw_rows_fn.?.attribute(.max_threads_per_block) < 1024) return error.Qwen4DrawThreads;
            out.meta = try b.allocate(rows * (@sizeOf(C.sample.Rule) + 4 + out.words * 4));
            out.buffers = 1;
            out.output = try b.allocate((rows + streams) * 4);
            out.buffers = 2;
            return out;
        }
        pub fn deinitDrained(d: *Self) void {
            if (d.buffers == 2) d.backend.release(d.output);
            if (d.buffers > 0) d.backend.release(d.meta);
            if (d.modules == 2) d.argmax.unload();
            if (d.modules > 0) d.sample.unload();
            d.* = undefined;
        }
        pub fn close(d: *Self) !void {
            try d.backend.drain();
            d.deinitDrained();
        }
        pub fn draw(d: *Self, logits: Tensor, offset: usize, positions: []const u64, sampling: ?L.Sampling, bits: []const u32) !void {
            const n = positions.len;
            if (n == 0 or n > d.rows or logits.rows != n or logits.width != d.vocab or logits.kind != .bf16 or offset % 4 != 0 or offset > d.output.len or n * 4 > d.output.len - offset) return error.Qwen4DrawShape;
            if (logits.buffer.d != d.backend.driver or logits.offset > logits.buffer.len or try logits.bytes() > logits.buffer.len - logits.offset) return error.Qwen4DrawBuffer;
            if (bits.len != 0 and bits.len != n * d.words) return error.Qwen4DrawMask;
            try @import("draw_inputs.zig").check(d.vocab, positions, sampling, bits);
            var pos: [128]i32 = undefined;
            for (positions, 0..) |p, i| {
                if (p == 0 or p > std.math.maxInt(i32)) return error.Qwen4DrawPosition;
                pos[i] = @intCast(p - 1);
            }
            try d.backend.enter();
            const ptr = try logits.buffer.at(logits.offset);
            const output = try d.output.at(offset);
            if (bits.len > 0) {
                const off = d.rows * (@sizeOf(C.sample.Rule) + 4);
                try d.backend.write(d.meta, off, std.mem.sliceAsBytes(bits));
                var args: C.Args = .{};
                args.add(ptr);
                args.add(@as(u32, @intCast(d.vocab)));
                args.add(try d.meta.at(off));
                args.add(@as(u32, @intCast(d.words)));
                try C.launch.launch(d.mask, .{ .grid = .{ .x = @intCast(n) }, .block = .{ .x = 256 } }, d.backend.stream, &args);
            }
            const active = if (sampling) |s| s.temperature > 0 else false;
            if (active) {
                const rule = C.sample.Rule.from(L.gpu_rule.rule(sampling.?));
                const rules: [128]C.sample.Rule = @splat(rule);
                try d.backend.write(d.meta, 0, std.mem.sliceAsBytes(rules[0..n]));
                const off = d.rows * @sizeOf(C.sample.Rule);
                try d.backend.write(d.meta, off, std.mem.sliceAsBytes(pos[0..n]));
                try d.draws.drawRows(d.backend.stream, ptr, d.vocab, d.meta.ptr, try d.meta.at(off), output, n, null);
            } else {
                var args: C.Args = .{};
                args.add(ptr);
                args.add(output);
                args.add(@as(i64, @intCast(n)));
                args.add(@as(i64, @intCast(d.vocab)));
                args.add(@as(i64, @intCast(d.vocab)));
                args.add(@as(i32, 0));
                try C.launch.launch(d.greedy, .{ .grid = .{ .x = @intCast(n) }, .block = .{ .x = 256 } }, d.backend.stream, &args);
            }
        }
        pub fn read(d: *Self, offset: usize, out: []u32) !void {
            if (offset % 4 != 0 or offset > d.output.len or out.len * 4 > d.output.len - offset) return error.Qwen4DrawBuffer;
            try d.backend.drain();
            try d.output.download(offset, std.mem.sliceAsBytes(out));
            for (out) |token| if (token >= d.vocab) return error.Qwen4DrawToken;
        }
    };
}
