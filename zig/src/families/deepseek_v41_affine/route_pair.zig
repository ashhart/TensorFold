//! Match repeated routed experts in decode windows and retain the existing per-row operations.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const row = @import("../../core/row_projection.zig");
const Buffer = mtl.Buffer;

pub const Kernels = struct {
    plan: mtl.Pipeline,
    q2: mtl.Pipeline,
    q3: mtl.Pipeline,
    inline_cap: u8 = 0,
    fixed: ?[4]mtl.Pipeline = null,

    pub fn load(device: mtl.Device) !?Kernels {
        if (!device.tensorUnits()) return null;
        const inline_cap = 2;
        const source = sources.core_row_projection ++ "\n" ++ @embedFile("route_pair.metal");
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const plan = try mtl.Pipeline.init(device, lib, "tf_pair_expert_slots", false);
        errdefer plan.deinit();
        const names: [2][:0]const u8 = switch (inline_cap) {
            2 => .{ "tf_row_projection_inline2_q2", "tf_row_projection_inline2_q3" },
            3 => .{ "tf_row_projection_inline3_q2", "tf_row_projection_inline3_q3" },
            else => .{ "tf_row_projection_pair_q2", "tf_row_projection_pair_q3" },
        };
        const q2 = try mtl.Pipeline.init(device, lib, names[0], false);
        errdefer q2.deinit();
        const q3 = try mtl.Pipeline.init(device, lib, names[1], false);
        errdefer q3.deinit();
        var out: Kernels = .{ .plan = plan, .q2 = q2, .q3 = q3, .inline_cap = inline_cap };
        {
            var fixed: [4]mtl.Pipeline = undefined;
            var initialized: usize = 0;
            errdefer for (fixed[0..initialized]) |p| p.deinit();
            for ([_][:0]const u8{ "tf_row_shape_gate_q2", "tf_row_shape_down_q2", "tf_row_shape_gate_q3", "tf_row_shape_down_q3" }, 0..) |name, i| {
                fixed[i] = try mtl.Pipeline.init(device, lib, name, false);
                initialized += 1;
            }
            out.fixed = fixed;
            std.log.info("deepseek_v41 bench: fixed target expert shapes", .{});
        }
        return out;
    }

    pub fn deinit(p: Kernels) void {
        p.plan.deinit();
        p.q2.deinit();
        p.q3.deinit();
        if (p.fixed) |fixed| for (fixed) |item| item.deinit();
    }

    pub fn encodePlan(p: Kernels, e: mtl.ComputeEncoder, ids: Buffer, mates: Buffer, slots: usize) void {
        e.setPipeline(p.plan);
        e.setBuffer(ids, 0, 0);
        e.setBuffer(mates, 0, 1);
        const n: u32 = @intCast(slots);
        e.setBytes(std.mem.asBytes(&n), 2);
        e.dispatchThreads(mtl.Size.of(slots, 1, 1), mtl.Size.of(@min(32, slots), 1, 1));
    }

    pub fn choice(p: Kernels, base: *const row.Pipelines, wt: row.Experts, slots: usize) ?row.Call {
        if ((p.inline_cap != 0 and slots > 36) or wt.one.sum != .f32 or (wt.one.bits != 2 and wt.one.bits != 3)) return null;
        var call = row.callIndexed(base, wt, slots, false) catch return null;
        call.pipeline = if (wt.one.bits == 2) p.q2 else p.q3;
        if (p.fixed) |fixed| if (fixedIndex(wt)) |index| {
            call.pipeline = fixed[index];
        };
        return call;
    }
};

pub fn fixedIndex(wt: row.Experts) ?usize {
    if (wt.experts != 384 or wt.one.group != 64 or wt.one.w_off % 2 != 0 or
        wt.one.sum != .f32 or (wt.one.bits != 2 and wt.one.bits != 3)) return null;
    const bits: usize = if (wt.one.bits == 2) 0 else 2;
    if (wt.one.k == 5120 and wt.one.n == 2304 and wt.repeat == 6) return bits;
    if (wt.one.k == 2304 and wt.one.n == 5120 and wt.repeat == 1) return bits + 1;
    return null;
}

pub fn encode(e: mtl.ComputeEncoder, call: row.Call, wt: row.Weights, x: Buffer, ids: Buffer, mates: ?Buffer, y: Buffer) void {
    e.setPipeline(call.pipeline);
    e.setBuffer(x, 0, 0);
    e.setBuffer(wt.w, wt.w_off, 1);
    e.setBuffer(wt.scales, wt.s_off, 2);
    e.setBuffer(wt.biases, wt.b_off, 3);
    e.setBytes(std.mem.asBytes(&call.dims), 4);
    e.setBuffer(y, 0, 5);
    e.setBuffer(ids, 0, 6);
    if (mates) |b| e.setBuffer(b, 0, 7);
    e.dispatchThreads(mtl.Size.of(call.groups * call.threads, call.slots, 1), mtl.Size.of(call.threads, 1, 1));
}

pub fn check(p: Kernels, base: *const row.Pipelines, dev: mtl.Device, queue: mtl.Queue, wt: row.Experts, rows: usize, topk: usize) !usize {
    const slots = rows * topk;
    const test_call = p.choice(base, wt, slots) orelse return 0;
    const ref_call = try row.callIndexed(base, wt, slots, false);
    const values = slots * wt.one.n;
    const inputs = slots / wt.repeat * wt.one.k;
    const opts = mtl.ResourceOptions.shared;
    const x = try dev.buffer(inputs * 2, opts);
    defer x.deinit();
    const ids = try dev.buffer(slots * 4, opts);
    defer ids.deinit();
    const mates = try dev.buffer(slots * 4, opts);
    defer mates.deinit();
    const ref = try dev.buffer(values * 2, opts);
    defer ref.deinit();
    const result = try dev.buffer(values * 2, opts);
    defer result.deinit();
    for (x.slice(u16, inputs), 0..) |*v, i| {
        const signed: f32 = @floatFromInt(@as(i32, @intCast((i * 739 + 17) % 257)) - 128);
        v.* = @truncate(@as(u32, @bitCast(signed / 32.0)) >> 16);
    }
    for (ids.slice(u32, slots), 0..) |*v, i| {
        const slot = i % topk;
        const group = if (slot % 2 == 0) i / topk else 0;
        v.* = @intCast((slot * 11 + group * 31) % wt.experts);
    }
    if (p.inline_cap != 0) {
        // Include repeated masked slots in both halves of a 36-slot window.
        ids.slice(u32, slots)[0] = @intCast(wt.experts);
        ids.slice(u32, slots)[slots - 1] = @intCast(wt.experts);
    }
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    encode(e, ref_call, wt.one, x, ids, null, ref);
    if (p.inline_cap == 0) p.encodePlan(e, ids, mates, slots);
    encode(e, test_call, wt.one, x, ids, if (p.inline_cap == 0) mates else null, result);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |message| {
        std.log.err("DeepSeek row pairing check: {s}", .{message});
        return error.ExpertPairCheckFailed;
    }
    if (!std.mem.eql(u16, ref.slice(u16, values), result.slice(u16, values))) return error.ExpertPairOutputDiffers;
    return values;
}
