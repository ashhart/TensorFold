//! Saved affine pocket rows beside the original dense MLP; empty layers allocate and launch no GPU work.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
pub const config = @import("config.zig");
const Config = config.Config;
const Ref = @import("projection.zig").Ref;
const Selection = @import("pocket_rows.zig").Selection;
const max_neurons = 64;

pub const Rows = struct { weights: mtl.Buffer, cap: usize, used: usize };
pub const Pockets = struct {
    allocator: std.mem.Allocator,
    device: mtl.Device,
    hidden: usize,
    layers: []?Rows,
    kernel: ?mtl.Pipeline = null,
    selected: Selection = .{},
    residency: ?mtl.ResidencySet.Follow = null,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, hidden: usize, layers: usize) !Pockets {
        const list = try a.alloc(?Rows, layers);
        @memset(list, null);
        return .{ .allocator = a, .device = device, .hidden = hidden, .layers = list };
    }

    pub fn deinit(p: *Pockets) void {
        p.setResident(null);
        for (p.layers) |slot| if (slot) |rows| rows.weights.deinit();
        if (p.kernel) |kernel| kernel.deinit();
        p.allocator.free(p.layers);
        p.* = undefined;
    }

    /// The caller drains prior GPU work before replacing a layer's shared rows.
    pub fn set(p: *Pockets, layer: usize, gate: []const f32, up: []const f32, down: []const f32, used: usize) !void {
        if (layer >= p.layers.len or p.hidden == 0 or gate.len % p.hidden != 0) return error.BadPocketShape;
        const n = gate.len / p.hidden;
        if (n == 0 or n > max_neurons or used > n or up.len != gate.len or down.len != gate.len) return error.BadPocketShape;
        if (used == 0 and p.layers[layer] == null) return;
        const buffer = try p.device.buffer(3 * gate.len * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer buffer.deinit();
        const values = buffer.slice(f32, 3 * gate.len);
        @memcpy(values[0..gate.len], gate);
        @memcpy(values[gate.len..][0..up.len], up);
        @memcpy(values[2 * gate.len ..], down);
        if (p.kernel == null and used > 0) {
            const lib = try mtl.Library.fromSource(p.device, @import("qwen_runtime_sources").pocket, mtl.CompileOptions.mlx());
            defer lib.deinit();
            p.kernel = try mtl.Pipeline.init(p.device, lib, "tf_q27_pocket_rows", false);
        }
        if (p.residency) |hook| hook.join(hook.ctx, buffer);
        if (p.layers[layer]) |old| {
            if (p.residency) |hook| hook.leave(hook.ctx, old.weights);
            old.weights.deinit();
        }
        p.layers[layer] = .{ .weights = buffer, .cap = n, .used = used };
    }

    pub fn read(p: *const Pockets, layer: usize, gate: []f32, up: []f32, down: []f32) !usize {
        if (layer >= p.layers.len) return error.BadPocketShape;
        const slot = p.layers[layer] orelse {
            @memset(gate, 0);
            @memset(up, 0);
            @memset(down, 0);
            return 0;
        };
        if (gate.len != slot.cap * p.hidden or up.len != gate.len or down.len != gate.len) return error.BadPocketShape;
        const values = slot.weights.slice(f32, 3 * gate.len);
        @memcpy(gate, values[0..gate.len]);
        @memcpy(up, values[gate.len..][0..up.len]);
        @memcpy(down, values[2 * gate.len ..]);
        return slot.used;
    }

    pub fn any(p: *const Pockets) bool {
        for (p.layers) |slot| if (slot) |rows| if (rows.used != 0) return true;
        return false;
    }

    pub fn add(p: *const Pockets, e: mtl.ComputeEncoder, layer: usize, input: Ref, output: Ref, rows: usize) !void {
        const slot = p.layers[layer] orelse return;
        if (slot.used == 0) return;
        if (p.selected.all) return p.run(e, slot, input, output, .{ .first = 0, .count = rows });
        for (p.selected.ranges[0..p.selected.count]) |range| {
            if (range.first > rows or range.count > rows - range.first) return error.BadPocketRows;
            try p.run(e, slot, input, output, range);
        }
    }

    fn run(p: *const Pockets, e: mtl.ComputeEncoder, slot: Rows, input: Ref, output: Ref, range: Selection.Range) !void {
        if (range.count == 0) return;
        const kernel = p.kernel orelse return error.MissingPocketKernel;
        if (kernel.maxThreads() < 256) return error.UnsupportedPocketGroup;
        const offset = range.first * p.hidden * 2;
        e.setPipeline(kernel);
        e.setBuffer(input.buffer, input.offset + offset, 0);
        e.setBuffer(slot.weights, 0, 1);
        e.setBuffer(output.buffer, output.offset + offset, 2);
        e.setValue(@Vector(4, u32){ @intCast(p.hidden), @intCast(slot.used), @intCast(slot.cap), 0 }, 3);
        e.dispatchGroups(.{ .width = range.count }, .{ .width = 256 });
        e.barrier();
    }

    pub fn load(p: *Pockets, io: std.Io, dir: []const u8, c: Config) !void {
        if (c.pocket == 0) return;
        var ck = try core.Checkpoint.openModel(p.allocator, io, dir);
        defer ck.close();
        const scoped = ck.has("language_model.model.layers.0.mlp.gate_proj.weight");
        if (scoped == ck.has("model.layers.0.mlp.gate_proj.weight")) return error.BadPocketNamespace;
        const prefix: []const u8 = if (scoped) "language_model." else "";
        const n = c.pocket * c.hidden;
        const host = try p.allocator.alloc(f32, 3 * n);
        defer p.allocator.free(host);
        for (0..c.layers) |layer| {
            var buf: [128]u8 = undefined;
            const base = try std.fmt.bufPrint(&buf, "{s}model.layers.{d}.", .{ prefix, layer });
            try core.dense_pocket.read(&ck, base, "mlp.gate_proj", c.intermediate, c.hidden, host[0..n]);
            try core.dense_pocket.read(&ck, base, "mlp.up_proj", c.intermediate, c.hidden, host[n..][0..n]);
            try core.dense_pocket.read(&ck, base, "mlp.down_proj", c.hidden, c.intermediate, host[2 * n ..][0..n]);
            var used: usize = 0;
            for (0..c.pocket) |j| if (std.mem.indexOfNone(f32, host[n + j * c.hidden ..][0..c.hidden], &.{0}) != null) {
                used = j + 1;
            };
            try p.set(layer, host[0..n], host[n..][0..n], host[2 * n ..][0..n], used);
        }
    }

    pub fn appendBuffers(p: *const Pockets, a: std.mem.Allocator, list: *std.ArrayList(mtl.Buffer)) !void {
        for (p.layers) |slot| if (slot) |rows| try list.append(a, rows.weights);
    }

    pub fn setResident(p: *Pockets, follow: ?mtl.ResidencySet.Follow) void {
        if (p.residency) |old| for (p.layers) |slot| if (slot) |rows| old.leave(old.ctx, rows.weights);
        p.residency = follow;
        if (follow) |hook| for (p.layers) |slot| if (slot) |rows| hook.join(hook.ctx, rows.weights);
    }
};
