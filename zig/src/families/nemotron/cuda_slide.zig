//! Nemotron's Sliding Weights learner on CUDA: learner.zig's lessons over the CUDA trainer, the engine and its shards.
const std = @import("std");
const cuda = @import("cuda");
const lessons = @import("living").learner;
const train = @import("cuda_train.zig");
const engine = @import("cuda_engine.zig");
const sites_mod = @import("cuda_sites.zig");
const living = @import("living.zig");
const state = @import("cuda_state.zig");

const Living = living.Io(@import("core"));
const PocketShards = @import("pocket_shards.zig").Io(@import("core"));
const pocket_layout = @import("pocket_layout.zig");

pub const Example = lessons.Example;
pub const Lesson = lessons.Lesson;
pub const Report = lessons.Report;
pub const Step = lessons.Step;

/// The first of a folder's shards, index and config that links to data another file shares: learning refuses it.
pub const linked = @import("core").shard_edit.linked;
pub const plainShared = @import("core").shard_edit.plainShared;

/// What lessons learn into on CUDA: the engine, its model folder, and saved blocks until the trainer takes them.
pub const Model = struct { e: *engine.Engine, dir: []const u8, living: *?sites_mod.Sites };

const Host = struct {
    pub const Trainer = train.Trainer;
    pub const max_rows = train.max_rows;
    pub const Ctx = Model;
    pub const family = pocket_layout.family;

    pub fn layers(m: Ctx) usize {
        return m.e.c.layers;
    }

    /// Where the pocket sits and learns: every MoE layer, all but the last learning.
    pub fn layout(m: Ctx) @import("living").Layout {
        return pocket_layout.layout(m.e.c);
    }

    pub fn trainer(gpa: std.mem.Allocator, io: std.Io, m: Ctx) !*Trainer {
        return train.Trainer.init(gpa, io, m.e, m.dir, m.living);
    }

    pub fn drain(t: *Trainer) void {
        t.drain();
    }

    pub fn free(t: *Trainer, _: std.mem.Allocator) void {
        t.deinit();
    }

    /// The change into every layer's forward, and the engine's graphs captured again to read it.
    pub fn attach(m: Ctx, t: *Trainer) !void {
        t.sites.attach(&m.e.w, true);
        try recapture(m.e);
    }

    pub fn write(m: Ctx, gpa: std.mem.Allocator, _: std.Io, t: *Trainer, ranks: usize) !usize {
        try m.e.stream.synchronize();
        return Living.save(gpa, m.dir, &t.sites, ranks / sites_mod.block);
    }

    pub fn seconds(io: std.Io) f64 {
        return @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds())) / 1e9;
    }

    /// MoE layer l's pocket for the forward: `up`, `down` [neurons, D] f32, the first `used` in use; made on first use.
    pub fn pocket(m: Ctx, l: usize, up: []const f32, down: []const f32, used: usize) !void {
        const blk = &m.e.w.blocks[l];
        if (blk.kind != .moe) return error.NotAnMoeLayer;
        // the kernels take their bounds from the count, so it never passes the pocket's width
        const width = if (blk.moe.pocket) |p| p.n else up.len / m.e.c.hidden;
        if (used > width) return error.PocketTooWide;
        if (blk.moe.pocket == null) {
            const w = &m.e.w;
            try w.buffers.ensureUnusedCapacity(w.gpa, 3);
            var ub = try cuda.DeviceBuffer.alloc(m.e.ctx.d, up.len * 4);
            errdefer ub.free();
            var db = try cuda.DeviceBuffer.alloc(m.e.ctx.d, down.len * 4);
            errdefer db.free();
            const cb = try cuda.DeviceBuffer.alloc(m.e.ctx.d, 4);
            w.buffers.appendAssumeCapacity(ub);
            w.buffers.appendAssumeCapacity(db);
            w.buffers.appendAssumeCapacity(cb);
            blk.moe.pocket = .{ .up = ub.ptr, .down = db.ptr, .n = width, .count = cb.ptr };
        }
        const p = &blk.moe.pocket.?;
        const o = m.e.ops();
        const count: u32 = @intCast(used);
        try o.upload(p.up, std.mem.sliceAsBytes(up));
        try o.upload(p.down, std.mem.sliceAsBytes(down));
        try o.upload(p.count, std.mem.asBytes(&count));
        // the kernels read the count, so graphs follow it; a layer they skipped needs them captured again
        const launched = p.used > 0;
        p.used = used;
        if (!launched and used > 0) try recapture(m.e);
    }

    /// MoE layer l's pocket as loaded from the shards, into `up` and `down`; the neurons in use (0: none).
    pub fn pocketRead(m: Ctx, l: usize, up: []f32, down: []f32) !usize {
        const blk = m.e.w.blocks[l];
        if (blk.kind != .moe) return error.NotAnMoeLayer;
        const p = blk.moe.pocket orelse return 0;
        const d = m.e.c.hidden;
        const count = std.math.mul(usize, p.n, d) catch return error.PocketShape;
        if (count != up.len or count != down.len) return error.PocketShape;
        const o = m.e.ops();
        errdefer cuda.completion.drain(m.e.k.d, m.e.stream, "a pocket read did not drain; its candidate rows stay alive");
        try o.download(std.mem.sliceAsBytes(up), p.up);
        try o.download(std.mem.sliceAsBytes(down), p.down);
        try m.e.stream.synchronize();
        var used: usize = 0;
        for (0..p.n) |j| {
            if (std.mem.indexOfNone(f32, up[j * d ..][0..d], &.{0}) != null) used = j + 1;
        }
        return used;
    }

    /// The pocket's neurons into the shards (widened the first time), `up` and `down` set to the 4-bit values; bytes.
    pub fn pocketSaveBegin(m: Ctx, _: std.mem.Allocator, io: std.Io) !void {
        return @import("save_guard").begin(io, m.dir);
    }

    pub fn pocketSaveFinish(m: Ctx, _: std.mem.Allocator, io: std.Io) !void {
        return @import("save_guard").finish(io, m.dir);
    }

    pub fn pocketSave(m: Ctx, gpa: std.mem.Allocator, io: std.Io, moe: []const usize, neurons: usize, up: []f32, down: []f32) !usize {
        try m.e.stream.synchronize();
        return PocketShards.writePending(gpa, io, m.dir, m.e.c, moe, neurons, up, down);
    }
};

pub const Learner = lessons.Learner(Host);

/// The model's saved lesson blocks, if any, into every layer's forward (graphs captured again), or null.
pub fn loadLiving(gpa: std.mem.Allocator, io: std.Io, e: *engine.Engine, dir: []const u8) !?sites_mod.Sites {
    var saved = (try Living.open(gpa, io, dir)) orelse return null;
    defer saved.close(io);
    const c = e.c;
    var sites = try sites_mod.Sites.init(gpa, e.ctx.d, c, state.prefill_rows, saved.lessons.len);
    errdefer sites.deinit();
    const blocks = try Living.fill(&saved, c, &sites);
    sites.attach(&e.w, true);
    try recapture(e);
    std.log.info("slide: {d} saved lessons act on the replies their routers choose", .{blocks});
    return sites;
}

/// Every graph captured again on the own sequence (contents kept): a graph keeps each layer's change as captured.
pub fn recapture(e: *engine.Engine) !void {
    e.epoch += 1; // other sequences' graphs are captured again at their next launch
    if (e.serial == null) return;
    try e.stream.synchronize();
    const was = e.bound;
    const sampling = e.sampling;
    if (was != &e.own) e.bind(&e.own);
    const own = e.sampling;
    defer {
        e.sampling = own; // bind keeps the own sequence's rule, not the captures' placeholder
        if (was != &e.own) e.bind(was);
        e.sampling = sampling;
    }
    e.sampling = null;
    const serial = try e.record(.{ .window = .{ .rows = 1, .feed = true } });
    e.serial.?.deinit();
    e.serial = serial;
    for (&e.windows, 0..) |*set, m| {
        e.sampling = if (m == 0) null else .{ .seed = 0, .temperature = 1.0 }; // the sampled set reads each stream's rule
        for (1..set.len) |rows| if (set[rows] != null) {
            const fresh = try e.record(.{ .window = .{ .rows = rows, .feed = false } });
            set[rows].?.deinit();
            set[rows] = fresh;
        };
    }
}

test "nemotron learning safety: actual Nemotron reads reject download, wait and shape failures and retry original rows" {
    try @import("cuda_slide_read_test.zig").run(Host);
}
