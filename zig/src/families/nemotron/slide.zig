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
        return Living.save(gpa, b.m.dir, &t.sites, ranks / adapters.block);
    }

    pub fn seconds(_: std.Io) f64 {
        return mtl.clock.seconds();
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
