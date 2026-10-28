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
        // The native loader accepts both public namespaces. Keep the folder's names and config key scope as stored.
        var ck = try core.Checkpoint.openModel(a, io, m.dir);
        defer ck.close();
        const scoped = ck.has("language_model.model.layers.0.mlp.gate_proj.weight");
        const path = try std.fs.path.join(a, &.{ m.dir, "config.json" });
        defer a.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20));
        defer a.free(text);
        var doc = try std.json.parseFromSlice(std.json.Value, a, text, .{});
        defer doc.deinit();
        if (doc.value != .object) return error.BadConfig;
        const wrapped = doc.value.object.contains("text_config");
        inline for (.{ false, true }) |scope| inline for (.{ false, true }) |wrap| {
            if (scoped == scope and wrapped == wrap) return living.shards.Io(core).writePending(a, io, m.dir, storedFamily(scope, wrap), Host.layout(m), layers, neurons, &.{ gate, up }, down);
        };
        unreachable;
    }
};
pub const Learner = lessons.Learner(Host);

fn storedFamily(comptime scoped: bool, comptime wrapped: bool) living.family.Family {
    var f = family;
    if (!scoped) f.names.layers = "model.layers.";
    if (!wrapped) f.keys.scope = null;
    return f;
}
