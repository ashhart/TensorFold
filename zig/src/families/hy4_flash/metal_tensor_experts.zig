//! TensorOps for Hy4-flash's active experts, retaining FFN gate and up codes byte-for-byte.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const lane = @import("lane_projection");
const tensor = lane.affine_mm;
const source = @import("kernel_sources").hy4_flash_tensor_dispatch;

const Ref = lane.Ref;
const Matrix = struct { w: Ref, s: Ref, b: Ref, n: u32, k: u32 };

pub const Weights = struct {
    gate: Matrix,
    up: Matrix,
    down: Matrix,
    pub fn upload(owner: anytype, up: core.quant.mlx.Host, down: core.quant.mlx.Host) !Weights {
        if (up.group != 64 or down.group != 64 or up.bits != down.bits) return error.BadHy4ExpertLayout;
        return .{
            .gate = try read(owner, up, 0),
            .up = try read(owner, up, 1),
            .down = try read(owner, down, null),
        };
    }
};

fn read(owner: anytype, w: core.quant.mlx.Host, part: ?usize) !Matrix {
    const count = w.words.shape[0];
    const n: u32 = @intCast(w.words.shape[1] / @as(usize, if (part == null) 1 else 2));
    const k: u32 = @intCast(w.scales.shape[2] * w.group);
    const tensors = [_][]const u8{ w.words.bytes, w.scales.bytes, w.biases.bytes };
    var packed_: [3]Ref = undefined;
    for (tensors, &packed_) |bytes, *dst| {
        if (part) |side| {
            if (count == 0 or bytes.len % (count * 2) != 0) return error.BadHy4ExpertLayout;
            const half = bytes.len / count / 2;
            const buf = try owner.alloc(half * count);
            const contents = buf.slice(u8, half * count);
            for (0..count) |i| @memcpy(contents[i * half ..][0..half], bytes[(i * 2 + side) * half ..][0..half]);
            dst.* = .{ .buf = buf };
        } else dst.* = .{ .buf = try owner.upload(bytes) };
    }
    return .{ .w = packed_[0], .s = packed_[1], .b = packed_[2], .n = n, .k = k };
}

pub const Scratch = struct {
    order: Ref,
    offsets: Ref,
    identity: Ref,
    latent_sums: Ref,
    act_sums: Ref,

    pub fn init(owner: anytype, rows: usize, picks: usize, width: usize, inner: usize) !Scratch {
        const slots = rows * picks;
        if (slots > 256) return error.BadHy4TensorRows;
        const identity = try owner.alloc((picks + 1) * @sizeOf(i32));
        for (identity.slice(i32, picks + 1), 0..) |*v, i| v.* = @intCast(i);
        return .{
            .order = .{ .buf = try owner.alloc(slots * @sizeOf(i32)) },
            .offsets = .{ .buf = try owner.alloc(slots * @sizeOf(i32)) },
            .identity = .{ .buf = identity },
            .latent_sums = .{ .buf = try owner.alloc(rows * (width / 64) * @sizeOf(f32)) },
            .act_sums = .{ .buf = try owner.alloc(slots * (inner / 64) * @sizeOf(f32)) },
        };
    }
};

pub const Route = struct { pick: Ref, ids: Ref, members: Ref, count: Ref };
pub const Jobs = struct { offsets: Ref, order: Ref, ids: Ref };

pub const Kernels = struct {
    pipes: tensor.Pipes,
    compact: mtl.Pipeline,

    pub fn load(a: std.mem.Allocator, d: mtl.Device, bits: u8) !Kernels {
        const text = try tensor.source(d, a, .{ .bits = bits, .group = 64, .out_f32 = true, .glu = .clipped_silu, .mapped_experts = true });
        defer a.free(text);
        const lib = try mtl.Library.fromSource(d, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var pipes: tensor.Pipes = undefined;
        var n: usize = 0;
        errdefer for (pipes[0..n]) |p| p.deinit();
        for (&pipes, tensor.names) |*dst, name| {
            dst.* = try mtl.Pipeline.init(d, lib, name, false);
            n += 1;
        }
        const lib_compact = try mtl.Library.fromSource(d, source, mtl.CompileOptions.mlx());
        defer lib_compact.deinit();
        return .{
            .pipes = pipes,
            .compact = try mtl.Pipeline.init(d, lib_compact, "tf_hy4_active_experts", false),
        };
    }

    pub fn deinit(self: Kernels) void {
        for (self.pipes) |p| p.deinit();
        self.compact.deinit();
    }

    /// For one row each expert is unique, so the pick list itself is the active-to-global map.
    pub fn prepare(self: Kernels, e: mtl.ComputeEncoder, s: Scratch, route: Route, x: Ref, rows: u32, pairs: u32, max_rows: u32, width: u32) Jobs {
        if (rows > 1) {
            e.setPipeline(self.compact);
            for ([_]Ref{ route.members, route.count, s.offsets, s.order }, 0..) |ref, i| e.setBuffer(ref.buf, ref.off, i);
            e.setValue([2]u32{ pairs, max_rows }, 4);
            e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(256, 1, 1));
        }
        tensor.rowSums(e, self.pipes, 64, x, s.latent_sums, rows, width);
        return if (rows == 1)
            .{ .offsets = s.identity, .order = s.identity, .ids = route.pick }
        else
            .{ .offsets = s.offsets, .order = s.order, .ids = route.ids };
    }

    pub fn apply(self: Kernels, e: mtl.ComputeEncoder, w: Weights, s: Scratch, jobs: Jobs, x: Ref, act: Ref, out: Ref, pairs: u32, picks: u32, limit: f32) void {
        tensor.gatherGluMapped(e, self.pipes, x, s.latent_sums, w.gate, w.up, jobs.offsets, jobs.order, act, jobs.ids, pairs, pairs, picks, limit);
        e.barrier();
        tensor.rowSums(e, self.pipes, 64, act, s.act_sums, pairs, w.down.k);
        e.barrier();
        tensor.gatherToMapped(e, self.pipes, act, s.act_sums, w.down, jobs.offsets, out, jobs.order, jobs.ids, pairs, pairs);
    }
};
