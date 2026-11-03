//! The existing lesson gate and CUDA-compatible shard writer over the native Metal trainer.
const std = @import("std");
const living = @import("living");
const core = @import("core");
const mtl = @import("metal");
const train = @import("train.zig");
const lessons = living.learner;
pub const family = living.families.qwen3_5;
pub const Example = lessons.Example;
pub const Lesson = lessons.Lesson;
pub const Report = lessons.Report;
pub const Step = lessons.Step;
pub const Model = struct { m: *@import("model.zig").Model, dir: []const u8, follow: ?mtl.ResidencySet.Follow = null };
pub const linked = core.shard_edit.linked;

pub fn layout(c: @import("config.zig").Config) living.Layout {
    var every: living.Layers = .empty;
    for (0..c.layers) |l| every.set(l);
    return .fromExperts(c.hidden, c.intermediate, c.layers, every);
}
const Host = struct {
    pub const Trainer = train.Trainer;
    pub const max_rows = train.max_rows;
    pub const Ctx = Model;
    pub const family = living.families.qwen3_5;
    pub fn layout(m: Ctx) living.Layout {
        return @import("slide.zig").layout(m.m.config);
    }
    pub fn trainer(a: std.mem.Allocator, io: std.Io, m: Ctx) !*Trainer {
        return Trainer.init(a, io, m.m, m.dir, m.follow);
    }
    pub fn drain(t: *Trainer) void {
        t.drain();
    }
    pub fn free(t: *Trainer, _: std.mem.Allocator) void {
        t.deinit();
    }
    pub fn seconds(io: std.Io) f64 {
        return @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds())) / 1e9;
    }
    pub fn pocketSaveBegin(m: Ctx, _: std.mem.Allocator, io: std.Io) !void {
        try @import("save_guard").begin(io, m.dir);
    }
    pub fn pocketSaveFinish(m: Ctx, _: std.mem.Allocator, io: std.Io) !void {
        try @import("save_guard").finish(io, m.dir);
    }
    pub fn pocketSwiglu(m: Ctx, layer: usize, gate: []const f32, up: []const f32, down: []const f32, used: usize) !void {
        if (m.m.pockets == null) {
            m.m.pockets = try @import("pocket.zig").Pockets.init(m.m.allocator, m.m.device, m.m.config.hidden, m.m.config.layers);
            m.m.pockets.?.setResident(m.follow);
        }
        try m.m.pockets.?.set(layer, gate, up, down, used);
    }
    pub fn pocketReadSwiglu(m: Ctx, layer: usize, gate: []f32, up: []f32, down: []f32) !usize {
        if (m.m.pockets) |p| return p.read(layer, gate, up, down);
        @memset(gate, 0);
        @memset(up, 0);
        @memset(down, 0);
        return 0;
    }
    pub fn pocketSaveSwiglu(m: Ctx, a: std.mem.Allocator, io: std.Io, layers: []const usize, neurons: usize, gate: []f32, up: []f32, down: []f32) !usize {
        return @import("pocket_storage.zig").save(a, io, m.dir, Host.layout(m), layers, neurons, gate, up, down);
    }
};
pub const Learner = lessons.Learner(Host);
