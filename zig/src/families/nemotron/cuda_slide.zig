//! Nemotron's Sliding Weights learner on CUDA: learner.zig's lessons over the CUDA trainer, the engine and its shards.
const std = @import("std");
const lessons = @import("learner.zig");
const train = @import("cuda_train.zig");
const learned = @import("cuda_learned.zig");
const engine = @import("cuda_engine.zig");

pub const Example = lessons.Example;
pub const Lesson = lessons.Lesson;
pub const Report = lessons.Report;
pub const Step = lessons.Step;

/// The first of a folder's shards, index and config that links to data another file shares: learning refuses it.
pub const linked = @import("core").shard_edit.linked;

/// What the lessons learn into on CUDA: the engine the lanes decode with, and the folder whose shards it loaded.
pub const Model = struct { e: *engine.Engine, dir: []const u8 };

const Host = struct {
    pub const Trainer = train.Trainer;
    pub const max_rows = train.max_rows;
    pub const Ctx = Model;

    pub fn layers(m: Ctx) usize {
        return m.e.c.layers;
    }

    pub fn trainer(gpa: std.mem.Allocator, io: std.Io, m: Ctx) !*Trainer {
        return train.Trainer.init(gpa, io, m.e, m.dir);
    }

    pub fn free(t: *Trainer, _: std.mem.Allocator) void {
        t.deinit();
    }

    /// The change into every layer's forward, and the engine's graphs captured again to read it.
    pub fn attach(m: Ctx, t: *Trainer) !void {
        t.sites.attach(&m.e.w, true);
        try m.e.recapture();
    }

    pub fn write(m: Ctx, gpa: std.mem.Allocator, io: std.Io, t: *Trainer, ranks: usize) !usize {
        return learned.write(gpa, io, m.dir, t, ranks);
    }

    pub fn seconds(io: std.Io) f64 {
        return @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds())) / 1e9;
    }
};

pub const Learner = lessons.Learner(Host);
