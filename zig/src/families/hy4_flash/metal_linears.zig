//! Hy4-flash's dense projection kernels: the tensor units' lane projections by shape, and the row kernels.
const std = @import("std");
const mtl = @import("metal");
const affine = @import("../../core/row_projection.zig");
const cfg = @import("config.zig");
const ihc = @import("ihc.zig");
const ihc_gpu = @import("ihc_metal.zig");
const mla_gpu = @import("mla_metal.zig");
const mha = @import("mla_attention.zig");
const shape_ = @import("mla_weights.zig");
const gpu_qkv = @import("metal_tensor_qkv.zig");
const lane = @import("../../core/lane_projection.zig");
const dense = @import("metal_dense.zig");
const Buffers = dense.Buffers;
const Ref = dense.Ref;
const Attention = dense.Attention;

/// The lane projections for every dense shape a model has, made once at load
pub const Linears = struct {
    shapes: [16][2]u32 = undefined,
    projections: [16]lane.Projection = undefined,
    /// The same projections adding their own group sums (4-bit reg layouts), for small passes.
    inlines: [16]?lane.Projection = @splat(null),
    count: usize = 0,
    sums: ?mtl.Buffer = null,
    pub const sum_rows = gpu_qkv.prompt_rows;

    pub fn load(a: std.mem.Allocator, device: mtl.Device, t: cfg.Text, bits: u8) !Linears {
        var out: Linears = .{};
        errdefer out.deinit();
        const v = shape_.value_width;
        const wanted = [_][2]u32{
            .{ t.hidden_size, t.swa_num_attention_heads * v },
            .{ t.hidden_size, t.num_attention_heads * v },
            .{ t.routed_expert_hidden_size, t.hidden_size },
            .{ t.moe_intermediate_size * t.n_shared_experts, t.hidden_size },
            .{ t.hidden_size, t.routed_expert_hidden_size },
            .{ t.hidden_size, t.moe_intermediate_size * t.n_shared_experts },
            .{ t.intermediate_size, t.hidden_size },
            .{ t.hidden_size, t.intermediate_size },
            .{ t.vocab_size, t.hidden_size },
            // Engram's value and stream keys over a lookup row: every column's slice of the n-gram embedding
            .{ t.hidden_size, @import("engram.zig").columns * (t.engram_n_embed_per_ngram / @import("engram.zig").heads) },
        };
        var groups: usize = 0;
        for (wanted) |nk| {
            if (out.find(nk[0], nk[1]) != null or nk[0] % 32 != 0) continue;
            out.projections[out.count] = try lane.Projection.init(a, device, gpu_qkv.denseLayout(bits, nk[0], nk[1]));
            out.inlines[out.count] = if (gpu_qkv.withInlineSums(gpu_qkv.denseLayout(bits, nk[0], nk[1]))) |l| try lane.Projection.init(a, device, l) else null;
            out.shapes[out.count] = nk;
            out.count += 1;
            groups = @max(groups, nk[1] / 64);
        }
        out.sums = try device.buffer(sum_rows * groups * 4, mtl.ResourceOptions.shared);
        return out;
    }

    pub fn deinit(self: *Linears) void {
        for (self.projections[0..self.count]) |p| p.deinit();
        for (self.inlines[0..self.count]) |p| if (p) |x| x.deinit();
        if (self.sums) |b| b.deinit();
        self.count = 0;
    }

    pub fn find(self: *const Linears, n: u32, k: u32) ?*const lane.Projection {
        for (self.shapes[0..self.count], self.projections[0..self.count]) |nk, *p| if (nk[0] == n and nk[1] == k) return p;
        return null;
    }

    pub fn findInline(self: *const Linears, n: u32, k: u32) ?*const lane.Projection {
        for (self.shapes[0..self.count], self.inlines[0..self.count]) |nk, *p| if (nk[0] == n and nk[1] == k) if (p.*) |*x| return x;
        return null;
    }
};

/// The row kernels for 2-8 row passes of portable weights
pub const Rows = struct {
    q4: mtl.Pipeline,
    q8: mtl.Pipeline,
    rps: u32 = 2,
    /// Greedy tokens on the GPU (first index of the largest logit), so draft columns chain without the host.
    argmax: mtl.Pipeline,

    pub fn load(a: std.mem.Allocator, device: mtl.Device) !Rows {
        const text = try std.mem.concat(a, u8, &.{ affine.source(device.tensorUnits()), @import("kernel_sources").hy4_flash_rows });
        defer a.free(text);
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var q4 = try mtl.Pipeline.init(device, lib, "tf_hy4_rows_r2_q4", false);
        errdefer q4.deinit();
        var q8 = try mtl.Pipeline.init(device, lib, "tf_hy4_rows_r2_q8", false);
        errdefer q8.deinit();
        return .{ .q4 = q4, .q8 = q8, .argmax = try mtl.Pipeline.init(device, lib, "tf_hy4_argmax", false) };
    }

    pub fn deinit(self: Rows) void {
        self.q4.deinit();
        self.q8.deinit();
        self.argmax.deinit();
    }

    /// out[at + r] = the greedy token of logits row r, for `rows` rows of `vocab`.
    pub fn pick(self: Rows, e: mtl.ComputeEncoder, logits: Ref, out: Ref, at: u32, rows: u32, vocab: u32) void {
        e.setPipeline(self.argmax);
        e.setBuffer(logits.buffer, logits.offset, 0);
        e.setBuffer(out.buffer, out.offset, 1);
        e.setValue([2]u32{ vocab, at }, 2);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(1024, 1, 1));
    }

    pub fn sg(self: Rows) u32 {
        return if (self.rps == 1) 8 else 4;
    }

    pub fn choose(self: Rows, w: affine.Weights, rows: u32) ?mtl.Pipeline {
        if (rows < 2 or w.sum != .f32 or w.n % (self.rps * self.sg()) != 0) return null;
        return switch (w.bits) {
            4 => self.q4,
            8 => self.q8,
            else => null,
        };
    }
};
