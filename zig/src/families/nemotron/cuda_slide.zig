//! Nemotron's Sliding Weights learner on CUDA: learner.zig's lessons over the CUDA trainer, the engine and its shards.
const std = @import("std");
const lessons = @import("learner.zig");
const train = @import("cuda_train.zig");
const engine = @import("cuda_engine.zig");
const sites_mod = @import("cuda_sites.zig");
const living = @import("living.zig");
const state = @import("cuda_state.zig");

const Living = living.Io(@import("core"));

pub const Example = lessons.Example;
pub const Lesson = lessons.Lesson;
pub const Report = lessons.Report;
pub const Step = lessons.Step;

/// The first of a folder's shards, index and config that links to data another file shares: learning refuses it.
pub const linked = @import("core").shard_edit.linked;

/// What lessons learn into on CUDA: the engine, its model folder, and saved blocks until the trainer takes them.
pub const Model = struct { e: *engine.Engine, dir: []const u8, living: *?sites_mod.Sites };

const Host = struct {
    pub const Trainer = train.Trainer;
    pub const max_rows = train.max_rows;
    pub const Ctx = Model;

    pub fn layers(m: Ctx) usize {
        return m.e.c.layers;
    }

    pub fn trainer(gpa: std.mem.Allocator, io: std.Io, m: Ctx) !*Trainer {
        return train.Trainer.init(gpa, io, m.e, m.dir, m.living);
    }

    pub fn free(t: *Trainer, _: std.mem.Allocator) void {
        t.deinit();
    }

    /// The change into every layer's forward, and the engine's graphs captured again to read it.
    pub fn attach(m: Ctx, t: *Trainer) !void {
        t.sites.attach(&m.e.w, true);
        try m.e.recapture();
    }

    pub fn write(m: Ctx, gpa: std.mem.Allocator, _: std.Io, t: *Trainer, ranks: usize) !usize {
        try m.e.stream.synchronize();
        return Living.save(gpa, m.dir, &t.sites, ranks / sites_mod.block);
    }

    pub fn seconds(io: std.Io) f64 {
        return @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds())) / 1e9;
    }
};

pub const Learner = lessons.Learner(Host);

/// The model's saved lessons, if it has any, into every layer's forward (graphs captured again): blocks with their
/// routers, or null.
pub fn loadLiving(gpa: std.mem.Allocator, io: std.Io, e: *engine.Engine, dir: []const u8) !?sites_mod.Sites {
    var saved = (try Living.open(gpa, io, dir)) orelse return null;
    defer saved.close(io);
    const c = e.c;
    var sites = try sites_mod.Sites.init(gpa, e.ctx.d, c, state.prefill_rows, saved.lessons.len);
    errdefer sites.deinit();
    const blocks = try Living.fill(&saved, c, &sites);
    sites.attach(&e.w, true);
    try e.recapture();
    std.log.info("slide: {d} saved lessons act on the replies their routers choose", .{blocks});
    return sites;
}
