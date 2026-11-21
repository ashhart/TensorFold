//! The eight attention output blocks are fixed across verifier rows.
const std = @import("std");
const mtl = @import("metal");
const c = @import("config.zig");
const st = @import("state.zig");
const Enc = @import("forward.zig").Enc;
const row = @import("../../core/row_projection.zig");
const Model = @import("model.zig").Model;
const indexed = @import("route_pair.zig").encode;

const max_width = 1 + c.draft_block;
const layouts = [_]struct { name: [:0]const u8, columns: usize, rows: usize }{
    .{ .name = "tf_woa_c4_r1", .columns = 4, .rows = 1 },
    .{ .name = "tf_woa_c2_r2", .columns = 2, .rows = 2 },
    .{ .name = "tf_woa_c4_r2", .columns = 4, .rows = 2 },
    .{ .name = "tf_woa_c8_r2", .columns = 8, .rows = 2 },
    .{ .name = "tf_woa_c4_r3", .columns = 4, .rows = 3 },
};

pub const Plan = [max_width]usize;

pub fn parse(spec: []const u8) !Plan {
    var fields = std.mem.splitScalar(u8, spec, ',');
    var plan: Plan = undefined;
    for (&plan) |*mode| {
        mode.* = try std.fmt.parseInt(usize, fields.next() orelse return error.InvalidWoAPlan, 10);
        if (mode.* > layouts.len) return error.InvalidWoAPlan;
    }
    if (fields.next() != null) return error.InvalidWoAPlan;
    return plan;
}

fn compatible(wt: row.Experts) bool {
    comptime std.debug.assert(c.o_groups == 8 and c.group_in == 4096 and c.o_rank == 1024);
    return wt.experts == c.o_groups and wt.repeat == 1 and wt.one.bits == 4 and wt.one.group == 64 and
        wt.one.sum == .f32 and wt.one.k == c.group_in and wt.one.n == c.o_rank;
}

pub const Kernels = struct {
    project: [layouts.len]mtl.Pipeline,
    audit: ?mtl.Pipeline,
    plan: Plan,

    pub fn load(device: mtl.Device, plan: Plan, audit: bool) !Kernels {
        const source = @import("kernel_sources").core_row_projection ++ "\n" ++ @embedFile("woa_batch.metal");
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var out: Kernels = .{ .project = undefined, .audit = null, .plan = plan };
        var made: usize = 0;
        errdefer for (out.project[0..made]) |p| p.deinit();
        for (&out.project, layouts) |*p, variant| {
            p.* = try mtl.Pipeline.init(device, lib, variant.name, false);
            made += 1;
        }
        if (audit) out.audit = try mtl.Pipeline.init(device, lib, "tf_woa_audit", false);
        return out;
    }

    pub fn deinit(out: Kernels) void {
        for (out.project) |p| p.deinit();
        if (out.audit) |p| p.deinit();
    }

    fn encode(p: Kernels, enc: mtl.ComputeEncoder, mode: usize, w: row.Weights, width: usize, x: mtl.Buffer, out: mtl.Buffer) void {
        const format = layouts[mode - 1];
        const rows: u32 = @intCast(width);
        enc.setPipeline(p.project[mode - 1]);
        enc.setBuffer(x, 0, 0);
        enc.setBuffer(w.w, w.w_off, 1);
        enc.setBuffer(w.scales, w.s_off, 2);
        enc.setBuffer(w.biases, w.b_off, 3);
        enc.setBuffer(out, 0, 4);
        enc.setBytes(std.mem.asBytes(&rows), 5);
        enc.dispatchGroups(mtl.Size.of(c.o_rank / (2 * format.columns), c.o_groups, (width + format.rows - 1) / format.rows), mtl.Size.of(64, 1, 1));
    }

    pub fn forward(p: Kernels, x: Enc, wt: row.Experts, width: usize) bool {
        if (width > max_width or !compatible(wt)) return false;
        const mode = p.plan[width - 1];
        if (mode == 0) return false;
        const s = x.s;
        if (p.audit != null) x.indexed(wt, width * c.o_groups, s.o, s.wo_ids, s.wo_reference);
        x.fence();
        p.encode(x.ep.*, mode, wt.one, width, s.o, s.oa);
        if (p.audit) |a| {
            x.fence();
            const enc = x.ep.*;
            enc.setPipeline(a);
            enc.setBuffer(s.oa, 0, 0);
            enc.setBuffer(s.wo_reference, 0, 1);
            enc.setBuffer(s.wo_mismatch, 0, 2);
            const len = width * c.o_groups * c.o_rank;
            enc.dispatchThreads(mtl.Size.of(len, 1, 1), mtl.Size.of(256, 1, 1));
            x.segs[0].cache.wo_audited += len;
        }
        return true;
    }
};

fn time(p: Kernels, queue: mtl.Queue, mode: usize, reference: row.Call, weights: []const row.Experts, width: usize, x: mtl.Buffer, ids: mtl.Buffer, out: mtl.Buffer) !f64 {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const e = cb.compute(.concurrent);
    for (weights, 0..) |wt, i| {
        if (i != 0) e.barrier();
        if (mode == 0) indexed(e, reference, wt.one, x, ids, null, out) else p.encode(e, mode, wt.one, width, x, out);
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.status() != .completed) return error.WoAProbeFailed;
    return 1e6 * cb.gpuSeconds() / @as(f64, @floatFromInt(weights.len));
}

pub fn measure(p: Kernels, m: *Model) !void {
    var weights: [10]row.Experts = undefined;
    for (&weights, 0..) |*wt, i| switch (m.weights.layers[i * 4].attn.wo_a) {
        .affine => |w| {
            if (!compatible(w)) return error.IncompatibleWoAProbe;
            wt.* = w;
        },
        .bf16 => return error.IncompatibleWoAProbe,
    };
    var storage: st.Storage = .{ .gpa = m.gpa, .device = m.device };
    defer storage.deinit();
    const input_len = max_width * c.o_groups * c.group_in;
    const output_len = max_width * c.o_groups * c.o_rank;
    const x = try storage.alloc(input_len * 2);
    const expected = try storage.alloc(output_len * 2);
    const actual = try storage.alloc(output_len * 2);
    const ids = try storage.alloc(max_width * c.o_groups * 4);
    for (ids.slice(u32, max_width * c.o_groups), 0..) |*item, i| item.* = @intCast(i % c.o_groups);
    for (x.slice(u16, input_len), 0..) |*item, i| {
        const f: f32 = @floatFromInt(@as(i32, @intCast((i * 11 + 31) % 517)) - 258);
        const scale: f32 = if (i % 69 < 13) 4096 else 32;
        item.* = @truncate(@as(u32, @bitCast(f / scale)) >> 16);
    }
    for (1..max_width + 1) |width| {
        const len = width * c.o_groups * c.o_rank;
        const control = try row.callIndexed(&m.kernels.rows, weights[0], width * c.o_groups, false);
        for (1..layouts.len + 1) |mode| {
            for (0..weights.len) |i| {
                _ = try time(p, m.queue, 0, control, weights[i..][0..1], width, x, ids, expected);
                _ = try time(p, m.queue, mode, control, weights[i..][0..1], width, x, ids, actual);
                if (!std.mem.eql(u16, expected.slice(u16, len), actual.slice(u16, len))) return error.WoAProbeOutputDiffers;
            }
            const a = try time(p, m.queue, 0, control, &weights, width, x, ids, expected);
            const b = try time(p, m.queue, mode, control, &weights, width, x, ids, actual);
            const d = try time(p, m.queue, mode, control, &weights, width, x, ids, actual);
            const e = try time(p, m.queue, 0, control, &weights, width, x, ids, expected);
            std.log.info("woa-probe width={d} layout={d} checked={d} original_us={d:.3} variant_us={d:.3}", .{ width, mode, len * weights.len, (a + e) / 2, (b + d) / 2 });
        }
    }
}
