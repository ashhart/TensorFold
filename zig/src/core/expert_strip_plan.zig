//! Immutable placement of a complete tensor over devices and independent on-disk replicas.
const std = @import("std");
const types = @import("expert_strip_types.zig");
fn cyclic(index: usize, shift: usize, modulus: usize) usize {
    const base = index % modulus;
    const step = shift % modulus;
    const until_wrap = modulus - base;
    return if (step >= until_wrap) step - until_wrap else base + step;
}

pub const Replica = struct {
    disk: u32,
    base: u64,
    bytes: u64,
};

pub const DeviceSpec = struct { id: u32 };

pub const Stripe = struct {
    tensor_offset: u64,
    length: usize,
    device: usize,
    checksum: types.Digest,
    candidates: []const types.Candidate,
};

pub const Device = struct {
    id: u32,
    stripes: []const usize,
};

pub const Spec = struct {
    identity: types.Identity,
    tensor_bytes: u64,
    stripe_bytes: u64,
    replicas: []const Replica,
    devices: []const DeviceSpec,
    checksums: []const types.Digest,
};

pub const Plan = struct {
    arena: std.heap.ArenaAllocator,
    identity: types.Identity,
    tensor_bytes: u64,
    stripe_bytes: u64,
    replicas: []const Replica,
    devices: []const Device,
    stripes: []const Stripe,

    pub fn init(gpa: std.mem.Allocator, spec: Spec) !Plan {
        if (spec.tensor_bytes == 0 or spec.stripe_bytes == 0) return error.InvalidStripePlan;
        if (spec.replicas.len == 0 or spec.devices.len == 0) return error.InvalidStripePlan;
        if (spec.tensor_bytes > std.math.maxInt(usize)) return error.TensorTooLarge;
        const count64 = 1 + (spec.tensor_bytes - 1) / spec.stripe_bytes;
        if (count64 > std.math.maxInt(usize)) return error.InvalidStripePlan;
        const count: usize = @intCast(count64);
        if (spec.checksums.len != count) return error.ChecksumCountMismatch;
        for (spec.devices, 0..) |device, i| for (spec.devices[0..i]) |prior| {
            if (device.id == prior.id) return error.DuplicateDevice;
        };
        for (spec.replicas, 0..) |replica, i| {
            for (spec.replicas[0..i]) |prior| if (replica.disk == prior.disk) return error.DuplicateDisk;
            const end = std.math.add(u64, replica.base, spec.tensor_bytes) catch return error.ReplicaTooSmall;
            if (end > replica.bytes) return error.ReplicaTooSmall;
        }
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const replicas = try a.dupe(Replica, spec.replicas);
        const stripes = try a.alloc(Stripe, count);
        const candidate_count = try std.math.mul(usize, count, replicas.len);
        const candidates = try a.alloc(types.Candidate, candidate_count);
        const devices = try a.alloc(Device, spec.devices.len);
        const device_indices = try a.alloc(usize, count);
        var device_at: usize = 0;
        for (devices, spec.devices, 0..) |*out, input, di| {
            const n = if (di < count) 1 + (count - 1 - di) / devices.len else 0;
            const indices = device_indices[device_at..][0..n];
            out.* = .{ .id = input.id, .stripes = indices };
            for (indices, 0..) |*stripe, j| stripe.* = di + j * devices.len;
            device_at += n;
        }
        for (stripes, 0..) |*stripe, i| {
            const offset = try std.math.mul(u64, spec.stripe_bytes, i);
            const remaining = spec.tensor_bytes - offset;
            const length: usize = @intCast(@min(spec.stripe_bytes, remaining));
            const device = i % devices.len;
            const first_disk = cyclic(device, i / devices.len, replicas.len);
            const selected = candidates[i * replicas.len ..][0..replicas.len];
            for (selected, 0..) |*candidate, j| {
                const replica_index = cyclic(first_disk, j, replicas.len);
                const replica = replicas[replica_index];
                candidate.* = .{
                    .replica = replica_index,
                    .disk = replica.disk,
                    .offset = try std.math.add(u64, replica.base, offset),
                };
            }
            stripe.* = .{
                .tensor_offset = offset,
                .length = length,
                .device = device,
                .checksum = spec.checksums[i],
                .candidates = selected,
            };
        }
        return .{
            .arena = arena,
            .identity = spec.identity,
            .tensor_bytes = spec.tensor_bytes,
            .stripe_bytes = spec.stripe_bytes,
            .replicas = replicas,
            .devices = devices,
            .stripes = stripes,
        };
    }

    pub fn deinit(p: *Plan) void {
        p.arena.deinit();
        p.* = undefined;
    }

    pub fn deviceForStripe(p: *const Plan, stripe: usize) !u32 {
        if (stripe >= p.stripes.len) return error.InvalidStripe;
        return p.devices[p.stripes[stripe].device].id;
    }
};
