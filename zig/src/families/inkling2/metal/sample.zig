//! GPU f32 sampling and argmax dispatch for Inkling 2 Mini target steps.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const sources = @import("kernel_sources");
const glue = @import("glue.zig");
const state = @import("state.zig");

pub const Ref = state.Ref;

pub const SamplePipelines = struct {
    full: mtl.Pipeline,
    slices: mtl.Pipeline,
    topk_rule: mtl.Pipeline,
    rule_1024: mtl.Pipeline,

    pub fn load(device: mtl.Device) !SamplePipelines {
        const lib_full = try mtl.Library.fromSource(device, sources.nemotron_sample, mtl.CompileOptions.mlx());
        defer lib_full.deinit();
        const full = try mtl.Pipeline.init(device, lib_full, "tf_sample_full_f32", false);
        errdefer full.deinit();
        const slices = try mtl.Pipeline.init(device, lib_full, "tf_topk_slices_f32", false);
        errdefer slices.deinit();
        const lib_ink = try mtl.Library.fromSource(device, sources.inkling2_sample, mtl.CompileOptions.mlx());
        defer lib_ink.deinit();
        const topk_rule = try mtl.Pipeline.init(device, lib_ink, "tf_ink2_topk_rule_f32", false);
        errdefer topk_rule.deinit();
        const rule_1024 = try mtl.Pipeline.init(device, lib_ink, "tf_ink2_rule_1024_f32", false);
        errdefer rule_1024.deinit();
        if (full.maxThreads() < 1024 or slices.maxThreads() < 512 or topk_rule.maxThreads() < 64 or rule_1024.maxThreads() < 1024) {
            return error.Inkling2Threads;
        }
        return .{ .full = full, .slices = slices, .topk_rule = topk_rule, .rule_1024 = rule_1024 };
    }

    pub fn deinit(s: *SamplePipelines) void {
        s.rule_1024.deinit();
        s.topk_rule.deinit();
        s.slices.deinit();
        s.full.deinit();
    }
};

pub const Draw = struct {
    how: union(enum) { greedy, sampled: struct { sampling: lanes.Sampling, position: u64 } },
    out: Ref,
};

pub const DrawMode = enum { greedy, full, topk, rule_1024 };

pub fn drawMode(d: Draw, vocab: u32) DrawMode {
    return switch (d.how) {
        .greedy => .greedy,
        .sampled => |sd| if (lanes.gpu_full.fullVocabulary(sd.sampling))
            .full
        else if (sd.sampling.top_k <= state.sample_kmax and vocab >= state.sample_slices and vocab <= state.sample_slices * 512 * 16)
            .topk
        else
            .rule_1024,
    };
}

pub fn drawHow(sampling: ?lanes.Sampling, position: u64) @FieldType(Draw, "how") {
    const t = if (sampling) |s| (if (s.temperature > 0) s else null) else null;
    return if (t) |s| .{ .sampled = .{ .sampling = s, .position = position } } else .greedy;
}

/// Encodes contiguous runs of GPU greedy argmax or f32 sampling into each draw's output buffer.
pub fn drawRows(
    sample_p: *const SamplePipelines,
    glue_p: *const glue.Pipelines,
    e: mtl.ComputeEncoder,
    s: *state.Scratch,
    logits: Ref,
    draws: []const ?Draw,
    vocab: u32,
) void {
    const row_bytes = @as(usize, vocab) * 4;
    var row_at: usize = 0;
    var topk_scratch_busy = false;
    while (row_at < draws.len) {
        const first = draws[row_at] orelse {
            row_at += 1;
            continue;
        };
        const mode = drawMode(first, vocab);
        var n: usize = 1;
        while (row_at + n < draws.len and n < state.batch_rows) {
            const next = draws[row_at + n] orelse break;
            if (drawMode(next, vocab) != mode) break;
            if (next.out.buf.id != first.out.buf.id or next.out.off != first.out.off + n * 4) break;
            n += 1;
        }
        const row_logits = Ref{ .buf = logits.buf, .off = logits.off + row_at * row_bytes };
        if (mode == .greedy) {
            glue.argmax(glue_p, e, row_logits, first.out, @intCast(n), vocab);
        } else {
            var launch: lanes.gpu_rule.Launch(state.batch_rows) = .{};
            for (draws[row_at .. row_at + n]) |d| {
                const sd = d.?.how.sampled;
                launch.put(sd.sampling, sd.position);
            }
            const padded = launch.padded();
            if (mode == .topk) {
                if (topk_scratch_busy) e.barrier();
                e.setPipeline(sample_p.slices);
                e.setBuffer(row_logits.buf, row_logits.off, 0);
                e.setBytes(std.mem.sliceAsBytes(launch.cfgs[0 .. 4 * padded]), 3);
                e.setBytes(std.mem.sliceAsBytes(launch.caps[0..padded]), 4);
                e.setValue(vocab, 7);
                e.setBuffer(s.sample_ck, 0, 8);
                e.setBuffer(s.sample_ci, 0, 9);
                e.dispatchGroups(mtl.Size.of(state.sample_slices, n, 1), mtl.Size.of(512, 1, 1));
                e.barrier();
            } else {
                e.setBuffer(row_logits.buf, row_logits.off, 0);
                e.setBytes(std.mem.sliceAsBytes(launch.cfgs[0 .. 4 * padded]), 3);
                e.setBytes(std.mem.sliceAsBytes(launch.caps[0..padded]), 4);
                e.setValue(vocab, 7);
            }
            e.setPipeline(switch (mode) {
                .topk => sample_p.topk_rule,
                .rule_1024 => sample_p.rule_1024,
                .full => sample_p.full,
                .greedy => unreachable,
            });
            e.setBytes(std.mem.sliceAsBytes(launch.seeds[0 .. 2 * padded]), 1);
            e.setBytes(std.mem.sliceAsBytes(launch.positions[0..padded]), 2);
            e.setBuffer(first.out.buf, first.out.off, 5);
            e.setBuffer(first.out.buf, first.out.off, 6);
            if (mode == .topk) {
                e.setBuffer(s.sample_ck, 0, 8);
                e.setBuffer(s.sample_ci, 0, 9);
                topk_scratch_busy = true;
            }
            const tg: usize = if (mode == .topk) 64 else 1024;
            e.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(tg, 1, 1));
        }
        row_at += n;
    }
}

/// Executes a single-row GPU draw synchronously on `queue` for unmasked first-token selection.
pub fn drawOne(
    queue: mtl.Queue,
    sample_p: *const SamplePipelines,
    glue_p: *const glue.Pipelines,
    s: *state.Scratch,
    logits: Ref,
    d: Draw,
    vocab: u32,
) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    const single = [_]?Draw{d};
    drawRows(sample_p, glue_p, e, s, logits, &single, vocab);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("inkling2_loop sample: {s}", .{msg});
        return error.GpuFailed;
    }
}
