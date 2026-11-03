//! The native trainer supplies rows and loss adjoints; the existing shared learner owns fitting and acceptance.
const std = @import("std");
const living = @import("living");
const profile = @import("learning_profile.zig");

pub fn Slide(comptime E: type, comptime T: type) type {
    return struct {
        pub const Example = living.learner.Example;
        pub const Lesson = living.learner.Lesson;
        pub const Report = living.learner.Report;
        pub const Step = living.learner.Step;
        pub const Model = struct { e: *E, pockets: *@import("pocket.zig").Pockets(E.Storage), dir: []const u8, trainer: T.Context };

        const Host = struct {
            pub const Trainer = T;
            pub const Ctx = Model;
            pub const family = living.families.qwen4;
            pub const max_rows = T.max_rows;

            pub fn layout(m: Ctx) living.Layout {
                return profile.layout(&m.e.c);
            }
            pub fn trainer(a: std.mem.Allocator, io: std.Io, m: Ctx) !*T {
                return T.init(a, io, m.e, m.pockets, m.trainer);
            }
            pub fn drain(t: *T) void {
                t.drain();
            }
            pub fn free(t: *T, _: std.mem.Allocator) void {
                t.deinit();
            }
            pub fn seconds(io: std.Io) f64 {
                return @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds())) / 1e9;
            }
            pub fn pocketSwiglu(m: Ctx, layer: usize, gate: []const f32, up: []const f32, down: []const f32, used: usize) !void {
                try m.pockets.set(layer, gate, up, down, used);
            }
            pub fn pocketReadSwiglu(m: Ctx, layer: usize, gate: []f32, up: []f32, down: []f32) !usize {
                return m.pockets.read(layer, gate, up, down);
            }
            pub fn pocketSaveBegin(m: Ctx, _: std.mem.Allocator, io: std.Io) !void {
                try @import("save_guard").begin(io, m.dir);
            }
            pub fn pocketSaveFinish(m: Ctx, _: std.mem.Allocator, io: std.Io) !void {
                try @import("save_guard").finish(io, m.dir);
            }
            pub fn pocketSaveSwiglu(m: Ctx, a: std.mem.Allocator, io: std.Io, layers: []const usize, neurons: usize, gate: []f32, up: []f32, down: []f32) !usize {
                return @import("pocket_storage.zig").save(a, io, m.dir, m.e.reader.namespace, m.e.c.root.contains("text_config"), layout(m), layers, neurons, gate, up, down);
            }
        };
        pub const Learner = living.learner.Learner(Host);
    };
}
