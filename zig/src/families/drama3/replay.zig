//! Merged-family controls adapted from #927's whole/chunk, scratch poison and reset/replay checks.
const std = @import("std");
pub fn Replay(comptime d3: type) type {
    return struct {
        const Frame = d3.model.Frame;

        const Audio = struct {
            out: []f32,
            at: usize = 0,
            sizes: [17]usize = @splat(0),
            calls: usize = 0,
            fn accept(p: *anyopaque, samples: []const f32) !void {
                const a: *Audio = @ptrCast(@alignCast(p));
                if (samples.len > a.out.len - a.at) return error.AudioOverflow;
                for (samples) |s| if (!std.math.isFinite(s)) return error.NonFiniteAudio;
                if (samples.len % 2048 != 0 or a.calls == a.sizes.len) return error.BadCodecChunk;
                a.sizes[a.calls] = samples.len / 2048;
                a.calls += 1;
                @memcpy(a.out[a.at..][0..samples.len], samples);
                a.at += samples.len;
            }
        };

        fn reset(t: *d3.tts.Tts) !void {
            if (@hasDecl(d3, "gpu")) try t.c.reset() else try t.c.reset(t.cs);
        }

        fn equal(a: []const u8, b: []const u8) !void {
            if (!std.mem.eql(u8, a, b)) return error.ReplayMismatch;
        }

        fn audio(t: *d3.tts.Tts, out: []f32, n: usize, chunk: usize) !void {
            try reset(t);
            var sink: Audio = .{ .out = out };
            var from: usize = 0;
            while (from < n) {
                const to = @min(n, from + chunk);
                try t.flush(from, to, .{ .ctx = &sink, .audio = Audio.accept });
                from = to;
            }
            if (sink.at != out.len) return error.AudioShort;
            var at: usize = 0;
            var calls: usize = 0;
            while (at < n) : (calls += 1) {
                const want = @min(chunk, n - at);
                if (calls >= sink.calls or sink.sizes[calls] != want) return error.BadCodecChunk;
                at += want;
            }
            if (sink.calls != calls) return error.BadCodecChunk;
            std.debug.print("codec requested {d}, actual sizes {any}\n", .{ chunk, sink.sizes[0..sink.calls] });
        }

        pub fn verify(gpa: std.mem.Allocator, io: std.Io, e: *d3.Engine, t: *d3.tts.Tts, rows: []const Frame, smp: d3.Sampling) !void {
            if (t.max_chunk < 16) return error.ReplayCodecCapacity;
            const baseline = try e.generate(io, rows, smp, 17, false, 1);
            defer gpa.free(baseline.frames);
            if (baseline.frames.len != 17) return error.ShortReplayFixture;
            for (baseline.frames) |frame| if (frame[0] == e.c.ids.eos) return error.ShortReplayFixture;
            const pcm = try gpa.alloc(f32, baseline.frames.len * 2048);
            defer gpa.free(pcm);
            const other = try gpa.alloc(f32, pcm.len);
            defer gpa.free(other);
            try audio(t, pcm, baseline.frames.len, 1);
            try e.poison();
            try t.c.poison();
            const replay = try e.generate(io, rows, smp, 17, true, 1);
            defer gpa.free(replay.frames);
            try equal(std.mem.sliceAsBytes(baseline.frames), std.mem.sliceAsBytes(replay.frames));
            for ([_]usize{ 1, 4, 16 }) |chunk| {
                try audio(t, other, replay.frames.len, chunk);
                try equal(std.mem.sliceAsBytes(pcm), std.mem.sliceAsBytes(other));
            }
            const second = try e.fork();
            defer second.destroy();
            const second_tts = try t.fork(second);
            defer second_tts.destroy();
            try e.prefill(rows, null);
            try second.prefill(rows, null);
            for (baseline.frames, 0..) |expected, i| {
                for ([_]*d3.Engine{ e, second }) |lane| {
                    try lane.frameStep(smp, false, null);
                    var frame: Frame = undefined;
                    try lane.download(lane.frames_out + i * 44, std.mem.asBytes(&frame));
                    try equal(std.mem.asBytes(&expected), std.mem.asBytes(&frame));
                }
            }
            try audio(second_tts, other, baseline.frames.len, 1);
            try equal(std.mem.sliceAsBytes(pcm), std.mem.sliceAsBytes(other));
            std.debug.print("Drama replay passed: eager/graphs, poison/reset, codec chunks and interleaved state\n", .{});
        }
    };
}
