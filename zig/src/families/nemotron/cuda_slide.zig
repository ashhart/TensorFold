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
        return writeLiving(gpa, m.dir, &t.sites, ranks);
    }

    pub fn seconds(io: std.Io) f64 {
        return @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds())) / 1e9;
    }
};

pub const Learner = lessons.Learner(Host);

/// The first `ranks` of every layer's blocks, written as the model's living-weights file; its bytes.
fn writeLiving(gpa: std.mem.Allocator, dir: []const u8, sites: *const sites_mod.Sites, ranks: usize) !usize {
    const layers = try gpa.alloc(living.Layer, sites.list.len);
    defer gpa.free(layers);
    for (sites.list, layers) |*s, *l| l.* = .{ .layer = s.layer, .in = s.in, .out = s.out, .a = s.a.slice(f32, sites_mod.max_rank * s.in), .b = s.b.slice(f32, sites_mod.max_rank * s.out), .tau = s.tau.slice(f32, sites_mod.max_blocks) };
    return Living.write(gpa, dir, layers, sites.needs(), ranks);
}

/// The model's saved lessons, if it has any, into every layer's forward (graphs captured again): blocks with their
/// routers, or null.
pub fn loadLiving(gpa: std.mem.Allocator, io: std.Io, e: *engine.Engine, dir: []const u8) !?sites_mod.Sites {
    var saved = (try Living.open(gpa, io, dir)) orelse return null;
    defer saved.close(io);
    const c = e.c;
    var sites = try sites_mod.Sites.init(gpa, e.ctx.d, c, state.prefill_rows);
    errdefer sites.deinit();
    const blocks = saved.ranks / sites_mod.block;
    for (sites.list) |*site| {
        const tau = site.tau.slice(f32, sites_mod.max_blocks);
        @memset(tau, sites_mod.shut);
        const got = (try saved.layer(c, site.layer)) orelse continue;
        @memcpy(site.a.slice(f32, sites_mod.max_rank * site.in)[0..got.a.len], got.a);
        @memcpy(site.b.slice(f32, sites_mod.max_rank * site.out)[0..got.b.len], got.b);
        @memcpy(tau[0..blocks], got.tau);
    }
    @memcpy(sites.needs()[0..blocks], try saved.need());
    sites.rank = saved.ranks;
    sites.attach(&e.w, true);
    try e.recapture();
    std.log.info("slide: {d} saved lessons act on the replies their routers choose", .{blocks});
    return sites;
}
