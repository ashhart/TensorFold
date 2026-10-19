//! Two-rank output-column MoE: complete K reductions, then byte-preserving row gathers.
const std = @import("std");
const mtl = @import("metal");
const ex = @import("../../core/exl3_projection.zig");
const weights = @import("weights.zig");
const work = @import("workspace.zig");
const ops = @import("ops.zig");
const route = @import("route.zig");
const View = @import("view.zig").View;

pub const Exchange = struct {
    ptr: *anyopaque,
    /// Both ranks must agree on tag and length; failure invalidates the enclosing model forward.
    call: *const fn (*anyopaque, u64, []const u8, []u8) anyerror!void,
};

pub const Context = struct {
    gpa: std.mem.Allocator,
    rank: u1,
    exchange: Exchange,
    descriptors: [40][3]mtl.Buffer,
    send: []u8,
    peer: []u8,
    sequence: u64 = 0,
    valid: bool = true,
    gpu_seconds: f64 = 0,
    trace: bool = false,

    pub fn begin(c: *Context, slot: u32, position: u32, ids: []const u32) !void {
        if (!c.valid) return error.InvalidTensorParallelState;
        c.valid = false;
        var own: [48]u8 = @splat(0);
        std.mem.writeInt(u32, own[0..4], slot, .little);
        std.mem.writeInt(u32, own[4..8], position, .little);
        std.mem.writeInt(u64, own[8..16], ids.len, .little);
        std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(ids), own[16..48], .{});
        var peer: [48]u8 = undefined;
        try c.exchange.call(c.exchange.ptr, std.math.maxInt(u64), &own, &peer);
        if (!std.mem.eql(u8, &own, &peer)) return error.TensorParallelHistoryMismatch;
        c.valid = true;
    }

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, model: *const weights.Model, rank: u1, max_rows: u32, exchange: Exchange) !Context {
        if (max_rows == 0 or max_rows > 4096) return error.InvalidPrefillChunk;
        const bytes = @as(usize, max_rows) * 7 * 5120;
        const send = try gpa.alloc(u8, bytes);
        errdefer gpa.free(send);
        const peer = try gpa.alloc(u8, bytes);
        errdefer gpa.free(peer);
        var descriptors: [40][3]mtl.Buffer = undefined;
        var owned: usize = 0;
        errdefer for (0..owned) |i| descriptors[i / 3][i % 3].deinit();
        for (&model.layers, 0..) |*layer, l| for (0..3) |p| {
            const buffer = try device.buffer(385 * @sizeOf(ex.ExpertDescriptor), mtl.ResourceOptions.shared);
            descriptors[l][p] = buffer;
            owned += 1;
            const n: usize = if (p == 2) 5120 else 2304;
            const src = layer.moe.descriptors[p].slice(ex.ExpertDescriptor, 385);
            const dst = buffer.slice(ex.ExpertDescriptor, 385);
            for (src, dst) |d, *out| out.* = try d.columns(n, @as(usize, rank) * n / 2, n / 2);
        };
        const trace = if (std.c.getenv("TF_DSV41_TP_TRACE")) |value| std.mem.eql(u8, std.mem.span(value), "1") else false;
        return .{ .gpa = gpa, .rank = rank, .exchange = exchange, .descriptors = descriptors, .send = send, .peer = peer, .trace = trace };
    }

    pub fn deinit(c: *Context) void {
        for (c.descriptors) |layer| for (layer) |buffer| buffer.deinit();
        c.gpa.free(c.send);
        c.gpa.free(c.peer);
        c.* = undefined;
    }

    fn complete(c: *Context, cb: mtl.CommandBuffer, encoder: mtl.ComputeEncoder, layer: usize, phase: []const u8, encode_started: f64) !void {
        const began = if (c.trace) mtl.clock.seconds() else 0;
        encoder.end();
        const ended = if (c.trace) mtl.clock.seconds() else 0;
        cb.commit();
        const committed = if (c.trace) mtl.clock.seconds() else 0;
        cb.wait();
        const waited = if (c.trace) mtl.clock.seconds() else 0;
        if (cb.failure() != null) return error.TensorParallelGpuFailure;
        const gpu = cb.gpuSeconds();
        c.gpu_seconds += gpu;
        if (c.trace) std.debug.print("{{\"tp_gpu_rank\":{d},\"layer\":{d},\"phase\":\"{s}\",\"encode_seconds\":{d},\"end_seconds\":{d},\"commit_seconds\":{d},\"wait_seconds\":{d},\"gpu_seconds\":{d},\"scheduled_after_commit_seconds\":{d},\"returned_after_gpu_seconds\":{d}}}\n", .{ c.rank, layer, phase, if (encode_started == 0) 0 else began - encode_started, ended - began, committed - ended, waited - committed, gpu, cb.gpuStart() - committed, waited - cb.gpuEnd() });
    }

    fn gather(c: *Context, layer: usize, phase: usize, ws: *work.Workspace, field: work.Field, shared: work.Field, rows: usize, n: usize) !void {
        const half_bytes = n;
        const used = rows * 7 * half_bytes;
        if (used > c.send.len) return error.TensorParallelScratchTooSmall;
        const routed = ws.bytes(field, rows * 6 * n * 2);
        const shared_rows = ws.bytes(shared, rows * n * 2);
        const began = if (c.trace) mtl.clock.seconds() else 0;
        pack(c.rank, routed, shared_rows, n, c.send[0..used]);
        const packed_at = if (c.trace) mtl.clock.seconds() else 0;
        const tag = (try std.math.mul(u64, c.sequence, 256)) | @as(u64, @intCast(layer * 3 + phase));
        try c.exchange.call(c.exchange.ptr, tag, c.send[0..used], c.peer[0..used]);
        const received = if (c.trace) mtl.clock.seconds() else 0;
        unpack(1 - c.rank, c.peer[0..used], n, routed, shared_rows);
        if (c.trace) std.debug.print("{{\"tp_pack_rank\":{d},\"tag\":{d},\"pack_seconds\":{d},\"unpack_seconds\":{d}}}\n", .{ c.rank, tag, packed_at - began, mtl.clock.seconds() - received });
    }

    /// Finish the enclosing command before CPU exchange, then restore its encoder for the next layer.
    pub fn boundary(c: *Context, model: anytype, cb: *mtl.CommandBuffer, encoder: *mtl.ComputeEncoder, encoding: *bool, layer: usize, rows: u32) !f64 {
        encoding.* = false;
        model.valid = false;
        c.gpu_seconds = 0;
        try c.complete(cb.*, encoder.*, layer, "attention", 0);
        const prior = c.gpu_seconds;
        c.gpu_seconds = 0;
        try c.run(model, layer, rows);
        cb.* = model.queue.commandBufferUnretained();
        encoder.* = cb.compute(if (model.concurrent) .concurrent else .serial);
        encoding.* = true;
        for (model.checkpoint.shards.items) |shard| encoder.useResource(shard.buffer, mtl.ResourceUsage.read);
        return prior + c.gpu_seconds;
    }

    /// The caller has completed earlier GPU work; this returns with the full MoE branch ready on both ranks.
    pub fn run(c: *Context, model: anytype, layer_id: usize, rows: u32) !void {
        if (!c.valid) return error.InvalidTensorParallelState;
        c.valid = false;
        const ws = &model.workspace;
        const w = &model.weights.layers[layer_id].moe;
        const p = &model.pipelines;
        // Model and context own every bound buffer through completion, as in the enclosing forward.
        const gate_started = if (c.trace) mtl.clock.seconds() else 0;
        const cb = model.queue.commandBufferUnretained();
        const encoder = cb.compute(.serial);
        var encoding = true;
        errdefer if (encoding) encoder.end();
        for (model.checkpoint.shards.items) |shard| encoder.useResource(shard.buffer, mtl.ResourceUsage.read);
        ops.encodeDense(encoder, &p.ops, ws.view(.normalized), tensor(w.gate), ws.view(.router_logits), .{ .rows = rows, .k = 5120, .n = 384, .x_stride = 5120, .w_stride = 5120, .y_stride = 384, .w_dtype = if (w.gate.dtype == .bf16) .bf16 else .f16, .y_dtype = .f32 });
        encoder.barrier();
        const bias_dtype: route.Dtype = switch (w.bias.dtype) {
            .f32 => .f32,
            .f16 => .f16,
            else => .bf16,
        };
        route.encode(encoder, &p.route, ws.view(.router_logits), tensor(w.bias), ws.view(.route_ids), ws.view(.route_weights), .{ .rows = rows, .bias_dtype = bias_dtype });
        encoder.barrier();
        try c.project(encoder, model, layer_id, 0, rows, .normalized, .gate, .shared_gate);
        try c.project(encoder, model, layer_id, 1, rows, .normalized, .up, .shared_up);
        encoding = false;
        try c.complete(cb, encoder, layer_id, "gate_up", gate_started);
        try c.gather(layer_id, 0, ws, .gate, .shared_gate, rows, 2304);
        try c.gather(layer_id, 1, ws, .up, .shared_up, rows, 2304);

        const down_started = if (c.trace) mtl.clock.seconds() else 0;
        const down_cb = model.queue.commandBufferUnretained();
        const down = down_cb.compute(.serial);
        var down_encoding = true;
        errdefer if (down_encoding) down.end();
        for (model.checkpoint.shards.items) |shard| down.useResource(shard.buffer, mtl.ResourceUsage.read);
        ops.encodeSwiGLU(down, &p.ops, ws.view(.gate), ws.view(.up), ws.view(.route_weights), ws.view(.activation), .{ .rows = rows * 6, .width = 2304, .input_stride = 2304, .output_stride = 2304, .weighted = true });
        ops.encodeSwiGLU(down, &p.ops, ws.view(.shared_gate), ws.view(.shared_up), ws.view(.route_weights), ws.view(.shared_activation), .{ .rows = rows, .width = 2304, .input_stride = 2304, .output_stride = 2304 });
        down.barrier();
        try c.project(down, model, layer_id, 2, rows, .activation, .routed, .shared_down);
        down_encoding = false;
        try c.complete(down_cb, down, layer_id, "down", down_started);
        try c.gather(layer_id, 2, ws, .routed, .shared_down, rows, 5120);
        const combine_started = if (c.trace) mtl.clock.seconds() else 0;
        const combine_cb = model.queue.commandBufferUnretained();
        const combine = combine_cb.compute(.serial);
        ops.encodeCombine(combine, &p.ops, ws.view(.routed), ws.view(.shared_down), ws.view(.route_ids), ws.view(.branch), .{ .rows = rows, .width = 5120, .routed_row_stride = 6 * 5120, .slot_stride = 5120, .shared_stride = 5120, .y_stride = 5120 });
        try c.complete(combine_cb, combine, layer_id, "combine", combine_started);
        c.sequence = try std.math.add(u64, c.sequence, 1);
        c.valid = true;
    }

    fn project(c: *Context, encoder: mtl.ComputeEncoder, model: anytype, layer: usize, part: usize, rows: u32, input_field: work.Field, output_field: work.Field, shared_field: work.Field) !void {
        const ws = &model.workspace;
        const k: usize = if (part == 2) 2304 else 5120;
        const n: usize = if (part == 2) 5120 else 2304;
        const offset = @as(usize, c.rank) * n;
        try ex.encodeIndexed(&model.pipelines.projection, encoder, .{ .descriptors = .{ .buffer = c.descriptors[layer][part] }, .count = 384, .k = k, .n = n / 2 }, convert(ws.view(input_field)), convert(ws.view(output_field).shifted(offset)), ws.projection(), .{ .count = @as(usize, rows) * 6, .x_stride = k, .y_stride = n, .routes = .{ .ids = convert(ws.view(.route_ids)), .repeat = if (part == 2) 1 else 6 } });
        const matrix = try model.weights.layers[layer].moe.shared[part].columns(@as(usize, c.rank) * n / 2, n / 2);
        const shared_input = if (part == 2) work.Field.shared_activation else input_field;
        try ex.encode(&model.pipelines.projection, encoder, matrix, convert(ws.view(shared_input)), convert(ws.view(shared_field).shifted(offset)), ws.projection(), .{ .count = rows, .x_stride = k, .y_stride = n });
    }
};

fn tensor(t: weights.Tensor) View {
    return .{ .buffer = t.buffer, .offset = t.offset };
}
fn convert(v: View) ex.View {
    return .{ .buffer = v.buffer, .offset = v.offset };
}

fn pack(rank: u1, routed: []const u8, shared: []const u8, n: usize, out: []u8) void {
    var at: usize = 0;
    for ([_][]const u8{ routed, shared }) |field| {
        var row: usize = 0;
        while (row < field.len) : (row += n * 2) {
            @memcpy(out[at..][0..n], field[row + @as(usize, rank) * n ..][0..n]);
            at += n;
        }
    }
}

fn unpack(rank: u1, input: []const u8, n: usize, routed: []u8, shared: []u8) void {
    var at: usize = 0;
    for ([_][]u8{ routed, shared }) |field| {
        var row: usize = 0;
        while (row < field.len) : (row += n * 2) {
            @memcpy(field[row + @as(usize, rank) * n ..][0..n], input[at..][0..n]);
            at += n;
        }
    }
}

test "TP output row gather preserves routed/shared order and the local half" {
    var routed: [48]u8 = undefined;
    var shared: [8]u8 = undefined;
    for (&routed, 0..) |*v, i| v.* = @intCast(i);
    for (&shared, 0..) |*v, i| v.* = @intCast(100 + i);
    var half: [28]u8 = undefined;
    pack(1, &routed, &shared, 4, &half);
    try std.testing.expectEqualSlices(u8, &.{ 4, 5, 6, 7 }, half[0..4]);
    try std.testing.expectEqualSlices(u8, &.{ 104, 105, 106, 107 }, half[24..28]);
    var target_r: [48]u8 = @splat(255);
    var target_s: [8]u8 = @splat(255);
    unpack(1, &half, 4, &target_r, &target_s);
    for (0..6) |row| {
        try std.testing.expectEqualSlices(u8, &.{ 255, 255, 255, 255 }, target_r[row * 8 ..][0..4]);
        try std.testing.expectEqualSlices(u8, routed[row * 8 + 4 ..][0..4], target_r[row * 8 + 4 ..][0..4]);
    }
    try std.testing.expectEqualSlices(u8, &.{ 255, 255, 255, 255, 104, 105, 106, 107 }, &target_s);
}

test "TP history agreement includes lane, position and ordered tokens and fails closed" {
    const Peer = struct {
        expected: ?[48]u8 = null,
        calls: usize = 0,
        fn transfer(ptr: *anyopaque, tag: u64, send: []const u8, receive: []u8) !void {
            const peer: *@This() = @ptrCast(@alignCast(ptr));
            try std.testing.expectEqual(std.math.maxInt(u64), tag);
            peer.calls += 1;
            if (peer.expected == null) peer.expected = send[0..48].*;
            @memcpy(receive, &peer.expected.?);
        }
    };
    for (0..3) |change| {
        var peer = Peer{};
        var context = Context{ .gpa = std.testing.allocator, .rank = 0, .exchange = .{ .ptr = &peer, .call = Peer.transfer }, .descriptors = undefined, .send = undefined, .peer = undefined };
        try context.begin(0, 4, &.{ 10, 20 });
        try context.begin(0, 4, &.{ 10, 20 });
        try std.testing.expectError(error.TensorParallelHistoryMismatch, context.begin(if (change == 0) 1 else 0, if (change == 1) 5 else 4, if (change == 2) &.{ 20, 10 } else &.{ 10, 20 }));
        try std.testing.expectError(error.InvalidTensorParallelState, context.begin(0, 4, &.{ 10, 20 }));
        try std.testing.expectEqual(@as(usize, 3), peer.calls);
    }
}
