//! Nemotron's Sliding Weights learner on Metal: learner.zig's lessons over the Metal trainer, its weights and shards.
const std = @import("std");
const lessons = @import("learner.zig");
const train = @import("train.zig");
const backend = @import("backend.zig");
const adapters = @import("adapters.zig");
const living = @import("living.zig");
const learn_shape = @import("learn_shape.zig");
const weights = @import("weights.zig");
const mtl = @import("metal");
const ckpt = @import("../../core/checkpoint_metal.zig");
const shard_edit = @import("../../core/shard_edit.zig");

const Living = living.Io(struct {
    pub const shard_edit = @import("../../core/shard_edit.zig");
    pub const safetensors = @import("../../core/safetensors.zig");
});
const PocketShards = @import("pocket_shards.zig").Io(struct {
    pub const shard_edit = @import("../../core/shard_edit.zig");
    pub const safetensors = @import("../../core/safetensors.zig");
    pub const affine4_host = @import("../../core/affine4_host.zig");
});

pub const Example = lessons.Example;
pub const Lesson = lessons.Lesson;
pub const Report = lessons.Report;
pub const Step = lessons.Step;

/// What the lessons learn into on Metal: the backend's model, its weights and its checkpoint.
const Host = struct {
    pub const Trainer = train.Trainer;
    pub const max_rows = train.max_rows;
    pub const Ctx = *backend.Metal;

    pub fn layers(b: Ctx) usize {
        return b.m.config.layers;
    }

    pub fn trainer(gpa: std.mem.Allocator, _: std.Io, b: Ctx) !*Trainer {
        return train.Trainer.init(gpa, b);
    }

    pub fn free(t: *Trainer, gpa: std.mem.Allocator) void {
        t.deinit(gpa);
    }

    pub fn attach(b: Ctx, t: *Trainer) !void {
        t.sites.attach(&b.m.weights, true);
    }

    pub fn write(b: Ctx, gpa: std.mem.Allocator, _: std.Io, t: *Trainer, ranks: usize) !usize {
        return Living.save(gpa, b.m.dir, &t.sites, ranks / adapters.block);
    }

    pub fn seconds(_: std.Io) f64 {
        return mtl.clock.seconds();
    }

    pub const fold4 = @import("../../core/affine4_fold.zig");

    /// MoE layer l's pocket for the forward: `up`, `down` [neurons, D] f32, the first `used` in use; made on first use.
    pub fn pocket(b: Ctx, l: usize, up: []const f32, down: []const f32, used: usize) !void {
        const w = &b.m.weights;
        const m = switch (w.layers[l]) {
            .moe => |*x| x,
            else => return error.NotAnMoeLayer,
        };
        if (m.pocket == null) {
            const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
            const rows = @max(b.o.chunk, train.max_rows);
            const n = up.len / b.m.config.hidden;
            try w.owned.ensureUnusedCapacity(w.allocator, 4);
            var made: [4]mtl.Buffer = undefined;
            for (&made, [_]usize{ up.len * 4, down.len * 4, rows * n * 4, rows * 4 }, 0..) |*buf, bytes, i| {
                buf.* = b.m.device.buffer(bytes, opts) catch |e| {
                    for (made[0..i]) |x| x.deinit();
                    return e;
                };
            }
            for (made) |x| w.owned.appendAssumeCapacity(x);
            m.pocket = .{ .up = made[0], .down = made[1], .z = made[2], .zn = made[3], .used = 0 };
        }
        const p = &m.pocket.?;
        @memcpy(p.up.slice(f32, up.len), up);
        @memcpy(p.down.slice(f32, down.len), down);
        p.used = used;
    }

    /// MoE layer l's pocket as loaded from the shards, into `up` and `down`; the neurons in use (0: none).
    pub fn pocketRead(b: Ctx, l: usize, up: []f32, down: []f32) usize {
        const m = switch (b.m.weights.layers[l]) {
            .moe => |x| x,
            else => return 0,
        };
        const p = m.pocket orelse return 0;
        @memcpy(up, p.up.slice(f32, up.len));
        @memcpy(down, p.down.slice(f32, down.len));
        return p.used;
    }

    /// The pocket's neurons into the shards (widened the first time), `up` and `down` set to the 4-bit values; bytes.
    pub fn pocketSave(b: Ctx, gpa: std.mem.Allocator, io: std.Io, moe: []const usize, neurons: usize, up: []f32, down: []f32) !usize {
        return PocketShards.write(gpa, io, b.m.dir, b.m.config, moe, neurons, up, down);
    }

    /// One layer's pocket neurons rounded in place as pocketSave writes them.
    pub fn pocketRound(gpa: std.mem.Allocator, neurons: usize, d: usize, up: []f32, down: []f32) !void {
        return PocketShards.round(gpa, neurons, d, up, down);
    }

    /// MoE layer l's shared-expert down projection codes as the checkpoint holds them live, for its row kernels.
    pub fn codes(b: Ctx, l: usize) !lessons.Codes {
        const t = try downProj(b, l);
        return .{ .words = @constCast(t[0].host(u32)), .scales = t[1].host(u16), .biases = t[2].host(u16) };
    }

    /// Layer l's codes, as changed in the checkpoint, into the kernels' tiles and the shard on disk that holds them.
    pub fn commit(b: Ctx, gpa: std.mem.Allocator, io: std.Io, l: usize) !void {
        const t = try downProj(b, l);
        const lin = switch (b.m.weights.layers[l]) {
            .moe => |m| m.shared_down,
            else => return error.NotAnMoeLayer,
        };
        weights.retile(lin, t[0].host(u32));
        var name: [160]u8 = undefined;
        var full: [192]u8 = undefined;
        const weight = try std.fmt.bufPrint(&full, "{s}.weight", .{try learn_shape.moduleName(&name, b.m.config, l)});
        try shard_edit.overwrite(gpa, io, b.m.dir, &.{.{ .name = weight, .dtype = .u32, .shape = t[0].shape[0..t[0].rank], .bytes = std.mem.sliceAsBytes(t[0].host(u32)) }});
    }

    /// Layer l's shared-expert down projection: its codes, scales and biases in the checkpoint.
    fn downProj(b: Ctx, l: usize) ![3]ckpt.Tensor {
        if (b.m.config.kinds[l] != .moe) return error.NotAnMoeLayer;
        var name: [160]u8 = undefined;
        const base = try learn_shape.moduleName(&name, b.m.config, l);
        var out: [3]ckpt.Tensor = undefined;
        for (&out, [_][]const u8{ "weight", "scales", "biases" }) |*t, leaf| {
            var full: [192]u8 = undefined;
            t.* = try b.m.checkpoint.get(try std.fmt.bufPrint(&full, "{s}.{s}", .{ base, leaf }));
        }
        if (out[0].dtype != .u32 or out[1].dtype != .bf16 or out[2].dtype != .bf16) return error.UnexpectedTensor;
        return out;
    }
};

pub const Learner = lessons.Learner(Host);

/// The model's saved lessons, if it has any, into every layer's forward: gated blocks the backend holds from here.
pub fn loadLiving(gpa: std.mem.Allocator, io: std.Io, b: *backend.Metal) !void {
    var saved = (try Living.open(gpa, io, b.m.dir)) orelse return;
    defer saved.close(io);
    const c = b.m.config;
    var sites = try adapters.Sites.init(gpa, b.m.device, c, @max(b.o.chunk, 64), saved.lessons.len);
    errdefer sites.deinit();
    const blocks = try Living.fill(&saved, c, &sites);
    sites.attach(&b.m.weights, true);
    b.living = sites;
    std.log.info("slide: {d} saved lessons act on the replies their routers choose", .{blocks});
}
