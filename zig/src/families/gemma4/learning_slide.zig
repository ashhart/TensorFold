//! Both native backends use the same GeGLU family, learner, pending-save transaction and quantized row writer.
const std = @import("std");
const core = @import("core");
const living = @import("living");
pub fn Slide(comptime E: type) type {
    const T = @import("learning_train.zig").Trainer(E);
    return struct {
        pub const Example = living.learner.Example;
        pub const Lesson = living.learner.Lesson;
        pub const Report = living.learner.Report;
        pub const Step = living.learner.Step;
        pub const Model = struct { e: *E, pockets: *@TypeOf(@as(E.Runtime, undefined).pockets), dir: []const u8, trainer: *E };
        const Host = struct {
            pub const Trainer = T;
            pub const Ctx = Model;
            pub const family = living.families.gemma4;
            pub const max_rows = T.max_rows;
            pub fn layout(m: Ctx) living.Layout {
                return @import("learning_profile.zig").layout(m.e.r.shape);
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
            pub fn pocketSwiglu(m: Ctx, l: usize, g: []const f32, u: []const f32, d: []const f32, used: usize) !void {
                try m.pockets.set(l, g, u, d, used);
            }
            pub fn pocketReadSwiglu(m: Ctx, l: usize, g: []f32, u: []f32, d: []f32) !usize {
                return m.pockets.read(l, g, u, d);
            }
            pub fn pocketSaveBegin(m: Ctx, _: std.mem.Allocator, io: std.Io) !void {
                try @import("save_guard").begin(io, m.dir);
            }
            pub fn pocketSaveFinish(m: Ctx, _: std.mem.Allocator, io: std.Io) !void {
                try @import("save_guard").finish(io, m.dir);
            }
            pub fn pocketSaveSwiglu(m: Ctx, a: std.mem.Allocator, io: std.Io, layers: []const usize, neurons: usize, gate: []f32, up: []f32, down: []f32) !usize {
                inline for (.{ false, true }) |unified| inline for (.{ false, true }) |wrapped| {
                    if (m.e.unified == unified and m.e.wrapped == wrapped) {
                        const f = comptime blk: {
                            var out = if (unified) living.families.gemma4_unified else living.families.gemma4;
                            if (!wrapped) out.keys.scope = null;
                            break :blk out;
                        };
                        return living.shards.Io(core).writePending(a, io, m.dir, f, layout(m), layers, neurons, &.{ gate, up }, down);
                    }
                };
                unreachable;
            }
        };
        pub const Learner = living.learner.Learner(Host);
    };
}
