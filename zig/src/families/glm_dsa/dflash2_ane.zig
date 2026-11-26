//! 32-core Apple Neural Engine offload for GLM-5.3-DFlash2 `fc` projection (overlaps with GPU L76-77).
const std = @import("std");
const mtl = @import("metal");
const ane = mtl.ane;

/// Fixed ANE spatial slice width (128 prompt rows -> 6.70 TFLOPS / 2.16 ms on M3 Ultra 32-core ANE).
pub const slice_rows: u32 = 128;
pub const max_slices: u32 = 4; // covers up to 512 rows per prefill chunk (gpu.prompt_rows)
pub const chunk_k: u32 = 9216; // TP4 fc K-width (1 chunk on TP4, 2 on TP2, 4 on TP1)
pub const max_chunks: u32 = 4;

fn bf16ToF16(bits: u16) f16 {
    const f: f32 = @bitCast(@as(u32, bits) << 16);
    return @floatCast(std.math.clamp(f, -65504.0, 65504.0));
}

pub const AneFc = struct {
    gpa: std.mem.Allocator,
    hidden: u32,
    k_loc: u32,
    chunks: u32, // k_loc / chunk_k (1 on TP4, 2 on TP2, 4 on TP1)
    programs: [max_chunks]ane.Program = undefined,
    requests: [max_chunks][max_slices]ane.Request = undefined,
    inputs: [max_chunks][max_slices]ane.Surface = undefined,
    outputs: [max_chunks][max_slices]ane.Surface = undefined,
    event: mtl.SharedEvent,
    ring: [64]std.atomic.Value(u32) = @splat(.init(0)),
    calls: u64 = 0,
    stop: std.atomic.Value(bool) = .init(false),
    failed: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,

    /// Compile and load `chunks` ANE Conv(9216->6144, S=128) programs from this rank's K-sharded BF16 `fc` weights.
    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, fc_bf16: []const u16, hidden: u32, k_loc: u32) !*AneFc {
        if (!ane.available() or k_loc == 0 or k_loc % chunk_k != 0 or k_loc / chunk_k > max_chunks) return error.NoNeuralEngine;
        const chunks: u32 = k_loc / chunk_k;
        const s = try gpa.create(AneFc);
        errdefer gpa.destroy(s);
        s.* = .{
            .gpa = gpa,
            .hidden = hidden,
            .k_loc = k_loc,
            .chunks = chunks,
            .event = try device.sharedEvent(),
        };
        errdefer s.event.deinit();

        const mil = try ane.conv(gpa, chunk_k, hidden, slice_rows);
        defer gpa.free(mil);
        var blob = try ane.Blob.init(gpa, &.{@as(usize, hidden) * chunk_k});
        defer blob.deinit(gpa);
        const w_fp16 = blob.values(0, @as(usize, hidden) * chunk_k);

        var made_c: u32 = 0;
        errdefer for (0..made_c) |c| {
            for (0..max_slices) |sl| {
                s.requests[c][sl].deinit();
                s.inputs[c][sl].deinit();
                s.outputs[c][sl].deinit();
            }
            s.programs[c].deinit();
        };

        for (0..chunks) |c| {
            const k0: usize = c * chunk_k;
            for (0..hidden) |o| {
                const src_row = fc_bf16[o * k_loc + k0 ..][0..chunk_k];
                const dst_row = w_fp16[o * chunk_k ..][0..chunk_k];
                for (src_row, dst_row) |b, *d| d.* = bf16ToF16(b);
            }
            s.programs[c] = try ane.Program.init(mil, blob.bytes);
            errdefer s.programs[c].deinit();
            var made_s: u32 = 0;
            errdefer for (0..made_s) |sl| {
                s.requests[c][sl].deinit();
                s.inputs[c][sl].deinit();
                s.outputs[c][sl].deinit();
            };
            for (0..max_slices) |sl| {
                s.inputs[c][sl] = try ane.Surface.init(device, @as(usize, chunk_k) * slice_rows * 2);
                s.outputs[c][sl] = ane.Surface.init(device, @as(usize, hidden) * slice_rows * 2) catch |err| {
                    s.inputs[c][sl].deinit();
                    return err;
                };
                s.requests[c][sl] = s.programs[c].request(s.inputs[c][sl], s.outputs[c][sl]) catch |err| {
                    s.inputs[c][sl].deinit();
                    s.outputs[c][sl].deinit();
                    return err;
                };
                made_s += 1;
            }
            made_c += 1;
        }
        s.thread = try std.Thread.spawn(.{}, worker, .{s});
        std.log.info("glm_dsa DFlash2 ANE: loaded {d}x Conv({d}->{d}, S={d}) on 32-core Neural Engine", .{ chunks, chunk_k, hidden, slice_rows });
        return s;
    }

    pub fn deinit(s: *AneFc) void {
        s.stop.store(true, .release);
        if (s.thread) |th| th.join();
        for (0..s.chunks) |c| {
            for (0..max_slices) |sl| {
                s.requests[c][sl].deinit();
                s.inputs[c][sl].deinit();
                s.outputs[c][sl].deinit();
            }
            s.programs[c].deinit();
        }
        s.event.deinit();
        s.gpa.destroy(s);
    }

    /// Queue `used_slices` (1..max_slices) on ANE; returns `{ gpu_signal_val, gpu_wait_val }`.
    pub fn queue(s: *AneFc, used_slices: u32) [2]u64 {
        const t = s.calls;
        s.calls += 1;
        s.ring[t % s.ring.len].store(used_slices, .release);
        return .{ 2 * t + 1, 2 * t + 2 };
    }

    fn worker(s: *AneFc) void {
        var t: u64 = 0;
        while (true) : (t += 1) {
            var spins: u64 = 0;
            while (s.event.value() < 2 * t + 1) : (spins += 1) {
                if (spins < 1_000_000) std.atomic.spinLoopHint() else if (s.event.wait(2 * t + 1, 50)) break else if (s.stop.load(.acquire)) return;
            }
            if (s.stop.load(.acquire)) return;
            const used = @min(s.ring[t % s.ring.len].load(.acquire), max_slices);
            for (0..s.chunks) |c| {
                for (0..used) |sl| {
                    s.requests[c][sl].run() catch s.failed.store(true, .release);
                }
            }
            s.event.set(2 * t + 2);
        }
    }
};
