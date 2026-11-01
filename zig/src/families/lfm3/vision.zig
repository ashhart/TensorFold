//! LFM3-VL image patches to projected language rows on Metal; one bounded image pass owns the scratch.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const Storage = @import("state.zig").Storage;
const wts = @import("vision_weights.zig");
const Kernels = @import("vision_kernels.zig").Kernels;
pub const Ref = wts.Ref;
pub const Trace = struct {
    ctx: *anyopaque,
    take: *const fn (ctx: *anyopaque, name: []const u8, value: Ref, rows: usize, width: usize) anyerror!void,
};

/// Device scratch excluding checkpoint weights and the caller-owned projected output rows.
pub fn scratchBytes(max_tokens: u32) u64 {
    return @as(u64, max_tokens) * (4 * 2 * (1536 + 5 * 1024 + 3072 + 2 * 512 + 3 * 2560) + 4096 * 2);
}

fn ref(b: mtl.Buffer) Ref {
    return .{ .buf = b };
}

fn bf16(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    return @intCast((bits +% (0x7fff + ((bits >> 16) & 1))) >> 16);
}

pub const Vision = struct {
    gpa: std.mem.Allocator,
    weights: wts.Weights,
    kernels: Kernels,
    memory: Storage,
    max_tokens: u32,
    patches: mtl.Buffer,
    h: mtl.Buffer,
    x: mtl.Buffer,
    r: mtl.Buffer,
    y: mtl.Buffer,
    bcx: mtl.Buffer,
    q: mtl.Buffer,
    k: mtl.Buffer,
    v: mtl.Buffer,
    gate: mtl.Buffer,
    up: mtl.Buffer,
    act: mtl.Buffer,
    projected: mtl.Buffer,

    pub fn load(gpa: std.mem.Allocator, device: mtl.Device, checkpoint: *const ck.Checkpoint, max_tokens: u32) !*Vision {
        if (max_tokens == 0 or max_tokens > 16384) return error.InvalidLfm3ImageBudget;
        const out = try gpa.create(Vision);
        errdefer gpa.destroy(out);
        out.gpa = gpa;
        out.max_tokens = max_tokens;
        out.weights = try wts.load(checkpoint);
        out.kernels = try Kernels.load(gpa, device);
        errdefer out.kernels.deinit();
        out.memory = .{ .gpa = gpa, .device = device };
        errdefer out.memory.deinit();
        const p = @as(usize, max_tokens) * 4;
        const memory = &out.memory;
        out.patches = try memory.alloc(p * 1536 * 2);
        inline for (.{ "h", "x", "r", "y", "q" }) |name| @field(out, name) = try memory.alloc(p * 1024 * 2);
        out.bcx = try memory.alloc(p * 3072 * 2);
        out.k = try memory.alloc(p * 512 * 2);
        out.v = try memory.alloc(p * 512 * 2);
        inline for (.{ "gate", "up", "act" }) |name| @field(out, name) = try memory.alloc(p * 2560 * 2);
        out.projected = try memory.alloc(@as(usize, max_tokens) * 4096 * 2);
        return out;
    }

    pub fn deinit(self: *Vision) void {
        self.memory.deinit();
        self.kernels.deinit();
        self.gpa.destroy(self);
    }

    pub fn buffers(self: *const Vision) []const mtl.Buffer {
        return self.memory.buffers.items;
    }

    /// Called without another image pass in flight; the output holds one 2048-wide row per merge block.
    pub fn image(self: *Vision, queue: mtl.Queue, pixels: []const f32, gh: u32, gw: u32, out: Ref) !void {
        return self.execute(queue, pixels, gh, gw, out, null);
    }

    /// Developer tracing waits at stage boundaries; all operators and their BF16 roundpoints stay unchanged.
    pub fn imageTrace(self: *Vision, queue: mtl.Queue, pixels: []const f32, gh: u32, gw: u32, out: Ref, trace: Trace) !void {
        return self.execute(queue, pixels, gh, gw, out, trace);
    }

    fn execute(self: *Vision, queue: mtl.Queue, pixels: []const f32, gh: u32, gw: u32, out: Ref, trace: ?Trace) !void {
        if (gh == 0 or gw == 0 or gh % 2 != 0 or gw % 2 != 0) return error.Lfm3ImageShape;
        const patches = std.math.mul(usize, gh, gw) catch return error.Lfm3ImageShape;
        if (patches > @as(usize, self.max_tokens) * 4 or pixels.len != patches * 1536) return error.Lfm3ImageShape;
        const tokens = patches / 4;
        const end = std.math.add(usize, out.off, tokens * 2048 * 2) catch return error.Lfm3ImageShape;
        if (end > out.buf.length() or out.off % 4 != 0) return error.Lfm3ImageShape;
        const px = self.patches.slice(u16, pixels.len);
        for (pixels, px) |value, *dst| {
            if (!std.math.isFinite(value)) return error.NonfiniteLfm3Image;
            dst.* = bf16(value);
        }
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        var cb = queue.commandBuffer();
        var e = cb.compute(.serial);
        var active = true;
        errdefer if (active) e.end();
        self.kernels.linear(e, ref(self.patches), self.weights.patch, ref(self.r), patches, 1024, 1536);
        try stage(queue, trace, "patchifier", ref(self.r), patches, 1024, &cb, &e, &active);
        self.norm(e, self.r, self.weights.norm, self.h, patches);
        try stage(queue, trace, "embedding_norm", ref(self.h), patches, 1024, &cb, &e, &active);
        for (self.weights.blocks, 0..) |block, index| {
            self.norm(e, self.h, block.operator_norm, self.x, patches);
            if (index == 0) try stage(queue, trace, "first_operator_norm", ref(self.x), patches, 1024, &cb, &e, &active);
            switch (block.mixer) {
                .conv => |conv| self.convolution(e, conv, gh, gw, patches),
                .attention => |a| self.attention(e, a, gw, patches),
            }
            if (index == 2) {
                try stage(queue, trace, "first_attention_norm", ref(self.x), patches, 1024, &cb, &e, &active);
                try stage(queue, trace, "first_attention_q", ref(self.q), patches, 1024, &cb, &e, &active);
                try stage(queue, trace, "first_attention_k", ref(self.k), patches, 512, &cb, &e, &active);
                try stage(queue, trace, "first_attention_v", ref(self.v), patches, 512, &cb, &e, &active);
                try stage(queue, trace, "first_attention_context", ref(self.y), patches, 1024, &cb, &e, &active);
                try stage(queue, trace, "first_attention_operator", ref(self.r), patches, 1024, &cb, &e, &active);
            }
            if (index == 0) {
                try stage(queue, trace, "first_conv_in", ref(self.bcx), patches, 3072, &cb, &e, &active);
                try stage(queue, trace, "first_conv_gated", ref(self.y), patches, 1024, &cb, &e, &active);
                try stage(queue, trace, "first_operator", ref(self.r), patches, 1024, &cb, &e, &active);
            }
            self.add(e, patches);
            if (index == 0) try stage(queue, trace, "first_residual", ref(self.h), patches, 1024, &cb, &e, &active);
            self.norm(e, self.h, block.ffn_norm, self.x, patches);
            if (index == 0) try stage(queue, trace, "first_ffn_norm", ref(self.x), patches, 1024, &cb, &e, &active);
            self.kernels.linear(e, ref(self.x), block.gate, ref(self.gate), patches, 2560, 1024);
            self.kernels.linear(e, ref(self.x), block.up, ref(self.up), patches, 2560, 1024);
            if (index == 0) {
                try stage(queue, trace, "first_ffn_gate", ref(self.gate), patches, 2560, &cb, &e, &active);
                try stage(queue, trace, "first_ffn_up", ref(self.up), patches, 2560, &cb, &e, &active);
            }
            e.setPipeline(self.kernels.get("l3v_swiglu"));
            e.setBuffer(self.gate, 0, 0);
            e.setBuffer(self.up, 0, 1);
            e.setBuffer(self.act, 0, 2);
            e.setValue(@as(u32, @intCast(patches * 2560)), 3);
            e.dispatchThreads(mtl.Size.of(patches * 2560, 1, 1), mtl.Size.of(256, 1, 1));
            if (index == 0) try stage(queue, trace, "first_ffn_act", ref(self.act), patches, 2560, &cb, &e, &active);
            self.kernels.linear(e, ref(self.act), block.down, ref(self.r), patches, 1024, 2560);
            if (index == 0) try stage(queue, trace, "first_ffn_down", ref(self.r), patches, 1024, &cb, &e, &active);
            self.add(e, patches);
            var name: [24]u8 = undefined;
            try stage(queue, trace, try std.fmt.bufPrint(&name, "layer_{d:0>2}", .{index}), ref(self.h), patches, 1024, &cb, &e, &active);
        }
        e.setPipeline(self.kernels.get("l3v_layer_norm"));
        e.setBuffer(self.h, 0, 0);
        bind(e, self.weights.projector_norm, 1);
        bind(e, self.weights.projector_norm_bias, 2);
        e.setBuffer(self.x, 0, 3);
        e.dispatchGroups(mtl.Size.of(patches, 1, 1), mtl.Size.of(256, 1, 1));
        try stage(queue, trace, "projector_norm", ref(self.x), patches, 1024, &cb, &e, &active);
        self.project(e, ref(self.x), self.weights.linear1, self.weights.bias1, ref(self.projected), tokens, 4096);
        e.setPipeline(self.kernels.get("l3v_gelu"));
        e.setBuffer(self.projected, 0, 0);
        e.setValue(@as(u32, @intCast(tokens * 4096)), 1);
        e.dispatchThreads(mtl.Size.of(tokens * 4096, 1, 1), mtl.Size.of(256, 1, 1));
        try stage(queue, trace, "projector_gelu", ref(self.projected), tokens, 4096, &cb, &e, &active);
        self.project(e, ref(self.projected), self.weights.linear2, self.weights.bias2, out, tokens, 2048);
        e.end();
        active = false;
        cb.commit();
        cb.wait();
        if (cb.failure()) |message| {
            std.log.err("LFM3 vision pass: {s}", .{message});
            return error.GpuFailed;
        }
    }

    fn norm(self: *Vision, e: mtl.ComputeEncoder, x: mtl.Buffer, w: Ref, out: mtl.Buffer, rows: usize) void {
        e.setPipeline(self.kernels.get("l3v_rms"));
        e.setBuffer(x, 0, 0);
        bind(e, w, 1);
        e.setBuffer(out, 0, 2);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn add(self: *Vision, e: mtl.ComputeEncoder, rows: usize) void {
        e.setPipeline(self.kernels.get("l3v_add"));
        e.setBuffer(self.r, 0, 0);
        e.setBuffer(self.h, 0, 1);
        e.setValue(@as(u32, @intCast(rows * 1024)), 2);
        e.dispatchThreads(mtl.Size.of(rows * 1024, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn convolution(self: *Vision, e: mtl.ComputeEncoder, weights: wts.Conv, gh: u32, gw: u32, patches: usize) void {
        self.kernels.linear(e, ref(self.x), weights.in, ref(self.bcx), patches, 3072, 1024);
        e.setPipeline(self.kernels.get("l3v_conv"));
        e.setBuffer(self.bcx, 0, 0);
        bind(e, weights.spatial, 1);
        e.setBuffer(self.y, 0, 2);
        e.setValue(extern struct { gh: u32, gw: u32 }{ .gh = gh, .gw = gw }, 3);
        e.dispatchThreads(mtl.Size.of(1024, patches, 1), mtl.Size.of(256, 1, 1));
        self.kernels.linear(e, ref(self.y), weights.out, ref(self.r), patches, 1024, 1024);
    }

    fn attention(self: *Vision, e: mtl.ComputeEncoder, weights: wts.Attention, gw: u32, patches: usize) void {
        self.kernels.linear(e, ref(self.x), weights.q, ref(self.q), patches, 1024, 1024);
        self.kernels.linear(e, ref(self.x), weights.k, ref(self.k), patches, 512, 1024);
        self.kernels.linear(e, ref(self.x), weights.v, ref(self.v), patches, 512, 1024);
        e.setPipeline(self.kernels.get("l3v_qk"));
        e.setBuffer(self.q, 0, 0);
        e.setBuffer(self.k, 0, 1);
        bind(e, weights.q_norm, 2);
        bind(e, weights.k_norm, 3);
        e.setValue(gw, 4);
        e.dispatchGroups(mtl.Size.of(24, patches, 1), mtl.Size.of(32, 1, 1));
        e.setPipeline(self.kernels.get("l3v_attention"));
        e.setBuffer(self.q, 0, 0);
        e.setBuffer(self.k, 0, 1);
        e.setBuffer(self.v, 0, 2);
        e.setBuffer(self.y, 0, 3);
        e.setValue(@as(u32, @intCast(patches)), 4);
        e.dispatchGroups(mtl.Size.of(16, patches, 1), mtl.Size.of(32, 1, 1));
        self.kernels.linear(e, ref(self.y), weights.out, ref(self.r), patches, 1024, 1024);
    }

    fn project(self: *Vision, e: mtl.ComputeEncoder, x: Ref, weight: Ref, bias: Ref, out: Ref, rows: usize, n: u32) void {
        e.setPipeline(self.kernels.get("l3v_project"));
        bind(e, x, 0);
        bind(e, weight, 1);
        bind(e, bias, 2);
        bind(e, out, 3);
        e.setValue(extern struct { m: u32, n: u32, k: u32 }{ .m = @intCast(rows), .n = n, .k = 4096 }, 4);
        e.dispatchGroups(mtl.Size.of((n + 3) / 4, (rows + 3) / 4, 1), mtl.Size.of(32, 1, 1));
    }
};

fn stage(queue: mtl.Queue, trace: ?Trace, name: []const u8, value: Ref, rows: usize, width: usize, cb: *mtl.CommandBuffer, e: *mtl.ComputeEncoder, active: *bool) !void {
    const hook = trace orelse return;
    e.end();
    active.* = false;
    cb.commit();
    cb.wait();
    if (cb.failure()) |_| return error.GpuFailed;
    try hook.take(hook.ctx, name, value, rows, width);
    cb.* = queue.commandBuffer();
    e.* = cb.compute(.serial);
    active.* = true;
}

fn bind(e: mtl.ComputeEncoder, r: Ref, slot: usize) void {
    e.setBuffer(r.buf, r.off, slot);
}
