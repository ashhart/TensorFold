//! Nemotron's Sliding Weights learner on Metal: learner.zig's lessons over the Metal trainer, its weights and shards.
const std = @import("std");
const lessons = @import("learner.zig");
const train = @import("train.zig");
const backend = @import("backend.zig");
const adapters = @import("adapters.zig");
const living = @import("living.zig");
const mtl = @import("metal");

const Living = living.Io(struct {
    pub const shard_edit = @import("../../core/shard_edit.zig");
    pub const safetensors = @import("../../core/safetensors.zig");
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
        return writeLiving(gpa, b.m.dir, &t.sites, ranks);
    }

    pub fn seconds(_: std.Io) f64 {
        return mtl.clock.seconds();
    }
};

pub const Learner = lessons.Learner(Host);

/// The first `ranks` of every layer's blocks, written as the model's living-weights file; its bytes.
fn writeLiving(gpa: std.mem.Allocator, dir: []const u8, sites: *const adapters.Sites, ranks: usize) !usize {
    const layers = try gpa.alloc(living.Layer, sites.list.len);
    defer gpa.free(layers);
    for (sites.list, layers) |*s, *l| l.* = .{ .layer = s.layer, .in = s.in, .out = s.out, .a = s.a.slice(f32, adapters.max_rank * s.in), .b = s.b.slice(f32, adapters.max_rank * s.out), .tau = s.tau.slice(f32, adapters.max_blocks) };
    return Living.write(gpa, dir, layers, sites.needs(), ranks);
}

/// The model's saved lessons, if it has any, into every layer's forward: gated blocks the backend holds from here.
pub fn loadLiving(gpa: std.mem.Allocator, io: std.Io, b: *backend.Metal) !void {
    var saved = (try Living.open(gpa, io, b.m.dir)) orelse return;
    defer saved.close(io);
    const c = b.m.config;
    var sites = try adapters.Sites.init(gpa, b.m.device, c, @max(b.o.chunk, 64));
    errdefer sites.deinit();
    const blocks = saved.ranks / adapters.block;
    for (sites.list) |*site| {
        const tau = site.tau.slice(f32, adapters.max_blocks);
        @memset(tau, 0);
        const got = (try saved.layer(c, site.layer)) orelse continue;
        @memcpy(site.a.slice(f32, adapters.max_rank * site.in)[0..got.a.len], got.a);
        @memcpy(site.b.slice(f32, adapters.max_rank * site.out)[0..got.b.len], got.b);
        @memcpy(tau[0..blocks], got.tau);
    }
    @memcpy(sites.needs()[0..blocks], try saved.need());
    sites.rank = saved.ranks;
    sites.attach(&b.m.weights, true);
    b.living = sites;
    std.log.info("slide: {d} saved lessons act on the replies their routers choose", .{blocks});
}
