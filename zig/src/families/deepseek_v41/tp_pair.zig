//! Native lane serving over one ordered control stream and the MoE data channel.
const std = @import("std");
const ds = @import("metal_model.zig");
const wire = @import("../glm/ep_control.zig");
const control = @import("tp_control.zig");
const Transport = @import("tp_transport.zig").Transport;
const Context = @import("tp_moe.zig").Context;

pub const Pair = struct {
    gpa: std.mem.Allocator,
    model: *ds.Model,
    transport: Transport,
    moe: Context,
    failed: std.atomic.Value(bool) = .init(false),
    wire: wire.Control,
    control: control.Control,
    scratch: []u32,
    output: []const f32 = &.{},

    pub fn open(gpa: std.mem.Allocator, model: *ds.Model, json: []const u8, fingerprint: [32]u8) !*Pair {
        const p = try gpa.create(Pair);
        errdefer gpa.destroy(p);
        p.gpa = gpa;
        p.model = model;
        p.failed = .init(false);
        p.output = &.{};
        p.transport = try Transport.initWithFailure(gpa, json, model.workspace.plan.rows, wire.BYTES, &p.failed);
        errdefer p.transport.deinit();
        p.moe = try Context.init(gpa, model.device, &model.weights, @intCast(p.transport.channel.myRank()), model.workspace.plan.rows, p.transport.exchange());
        errdefer p.moe.deinit();
        p.wire = try wire.Control.init(gpa, p.transport.endpoint.rdma(), p.transport.control_base, &p.failed);
        errdefer p.wire.deinit(gpa);
        p.scratch = try gpa.alloc(u32, model.workspace.plan.rows + 3);
        errdefer gpa.free(p.scratch);
        p.control = .{ .wire = &p.wire, .failed = &p.failed, .limits = .{ .lanes = @intCast(model.lanes), .rows = model.workspace.plan.rows, .capacity = model.caches[0].plan.capacity }, .executor = .{ .ptr = p, .apply = apply }, .scratch = p.scratch };
        var peer: [32]u8 = undefined;
        const exchange = p.transport.exchange();
        try exchange.call(exchange.ptr, std.math.maxInt(u64) - 1, &fingerprint, &peer);
        if (!std.mem.eql(u8, &fingerprint, &peer)) return error.TensorParallelIdentityMismatch;
        _ = try p.wire.hello(.{ .magic = 0x44535450, .version = 1, .layers = 40, .run = model.workspace.plan.rows, .mtp = @intCast(model.lanes), .experts = 2, .own_lo = p.wire.rank, .own_hi = p.wire.rank + 1, .cap = model.caches[0].plan.capacity, .model = std.mem.readInt(u64, fingerprint[0..8], .little), .rest = fingerprint[8..24].* });
        p.moe.registerResidency(model.residency);
        model.residency.commit();
        model.residency.requestResidency();
        model.tp = &p.moe;
        return p;
    }

    pub fn follows(p: *const Pair) bool {
        return p.wire.rank == 1;
    }

    pub fn stop(p: *Pair) void {
        p.wire.bye();
        p.wire.stop.store(true, .release);
    }

    /// The host joins its follower and stops its lane worker before freeing the pair.
    pub fn deinit(p: *Pair) void {
        p.stop();
        p.model.tp = null;
        p.moe.removeResidency(p.model.residency);
        p.model.residency.commit();
        p.wire.deinit(p.gpa);
        p.moe.deinit();
        p.transport.deinit();
        p.gpa.free(p.scratch);
        p.gpa.destroy(p);
    }

    pub fn follow(p: *Pair) void {
        p.control.follow() catch |err| {
            p.poison();
            std.log.err("DeepSeek TP follower stopped: {s}", .{@errorName(err)});
        };
    }

    fn poison(p: *Pair) void {
        p.failed.store(true, .release);
        p.wire.bye();
        p.model.valid = false;
        p.moe.valid = false;
    }

    fn apply(ptr: *anyopaque, command: control.Command) !u32 {
        const p: *Pair = @ptrCast(@alignCast(ptr));
        errdefer p.poison();
        if (p.model.caches[command.slot].position != command.position) return error.TensorParallelHistoryMismatch;
        switch (command.kind) {
            .reset => try p.model.resetSlot(command.slot),
            .begin => try p.model.beginSlot(command.slot),
            .commit => try p.model.commitSlot(command.slot),
            .rollback => try p.model.rollbackSlot(command.slot),
            .forward => p.output = try p.model.forwardOn(command.slot, command.tokens, switch (command.logits) {
                0 => .none,
                1 => .last,
                2 => .all,
                else => unreachable,
            }),
        }
        return p.model.caches[command.slot].position;
    }

    pub fn slot(p: *Pair, kind: control.Kind, index: u32) !void {
        errdefer p.poison();
        if (index >= p.model.lanes) return error.InvalidLaneSlot;
        try p.control.run(.{ .kind = kind, .slot = index, .position = p.model.caches[index].position });
    }

    pub fn forward(p: *Pair, index: u32, tokens: []const u32, logits: ds.Logits) ![]const f32 {
        errdefer p.poison();
        if (index >= p.model.lanes) return error.InvalidLaneSlot;
        try p.control.run(.{ .kind = .forward, .slot = index, .position = p.model.caches[index].position, .tokens = tokens, .logits = switch (logits) {
            .none => 0,
            .last => 1,
            .all => 2,
        } });
        return p.output;
    }
};

/// Peer compatibility: executable, checkpoint metadata, token map, and arithmetic switches.
pub fn identity(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) ![32]u8 {
    // The launch recipe verifies weight and Engram SHA256 manifests separately.
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    const exe = try std.process.executablePathAlloc(io, gpa);
    defer gpa.free(exe);
    try hashFile(gpa, io, &hash, exe);
    for ([_][]const u8{ "config.json", "model.safetensors.index.json", "engram-token-map.bin" }) |name| {
        const path = try std.fs.path.join(gpa, &.{ dir, name });
        defer gpa.free(path);
        try hashFile(gpa, io, &hash, path);
    }
    for ([_][:0]const u8{ "TF_DSV41_SERIAL", "TF_EXL3_PAIRS_UP_TO_GROUPS", "TF_DSV41_PARALLEL_ATTENTION", "TF_DSV41_TP_SLOTS" }) |key| {
        const value = if (std.c.getenv(key)) |v| std.mem.span(v) else "<unset>";
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(value, &digest, .{});
        hash.update(&digest);
    }
    return hash.finalResult();
}

fn hashFile(gpa: std.mem.Allocator, io: std.Io, hash: *std.crypto.hash.sha2.Sha256, path: []const u8) !void {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30));
    defer gpa.free(bytes);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    hash.update(&digest);
}
