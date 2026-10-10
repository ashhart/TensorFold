//! Nemotron's Sliding Weights learner on Metal: learner.zig's lessons over the Metal trainer, its weights and shards.
const std = @import("std");
const lessons = @import("learner.zig");
const train = @import("train.zig");
const learned = @import("learned.zig");
const backend = @import("backend.zig");
const mtl = @import("metal");

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

    pub fn write(b: Ctx, gpa: std.mem.Allocator, io: std.Io, t: *Trainer, ranks: usize) !usize {
        const m = b.m;
        return learned.write(gpa, io, m.dir, &m.checkpoint, &m.weights, m.config, &t.sites, ranks);
    }

    pub fn seconds(_: std.Io) f64 {
        return mtl.clock.seconds();
    }
};

pub const Learner = lessons.Learner(Host);
