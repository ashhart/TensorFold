//! Flash Next's Living Weights learner on CUDA (--slide): the core learner over cuda_train.zig, the pocket's writer.
const std = @import("std");
const living = @import("living");
const core = @import("core");
const prompt = @import("cuda_prompt.zig");
const train = @import("cuda_train.zig");
const fam = @import("living.zig");

const lessons = living.learner;
const Shards = fam.Shards(core);

/// The layout config.json in `dir` gives (text_config's keys, a pocket written earlier left out of the width).
pub fn layoutOf(a: std.mem.Allocator, io: std.Io, dir: []const u8) !living.Layout {
    const path = try std.fs.path.join(a, &.{ dir, "config.json" });
    defer a.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20));
    defer a.free(bytes);
    var doc = try std.json.parseFromSlice(std.json.Value, a, bytes, .{});
    defer doc.deinit();
    if (doc.value != .object) return error.BadConfig;
    const text = living.families.scoped(fam.family, doc.value.object) orelse return error.BadConfig;
    return fam.layoutOf(text);
}

/// The first file a lesson rewrites in `dir` (config, index, a shard with a shared expert) that is a link, or null.
pub fn linked(arena: std.mem.Allocator, io: std.Io, dir: []const u8) !?[]const u8 {
    var d = try std.Io.Dir.cwd().openDir(io, dir, .{});
    defer d.close(io);
    var names: std.ArrayList([]const u8) = .empty;
    try names.appendSlice(arena, &.{ "config.json", "model.safetensors.index.json" });
    const text = d.readFileAlloc(io, "model.safetensors.index.json", arena, .limited(1 << 26)) catch |e| switch (e) {
        error.FileNotFound => return core.shard_edit.linked(arena, io, dir),
        else => return e,
    };
    const index = try std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{});
    if (index != .object) return error.BadIndex;
    const map = index.object.get("weight_map") orelse return error.BadIndex;
    if (map != .object) return error.BadIndex;
    var it = map.object.iterator();
    while (it.next()) |kv| {
        if (fam.family.expertLayer(kv.key_ptr.*) == null and std.mem.indexOf(u8, kv.key_ptr.*, "mtp.layers.0.mlp.shared_expert.") == null) continue;
        if (kv.value_ptr.* != .string) return error.BadIndex;
        for (names.items) |n| {
            if (std.mem.eql(u8, n, kv.value_ptr.string)) break;
        } else try names.append(arena, kv.value_ptr.string);
    }
    for (names.items) |n| {
        const info = d.statFile(io, n, .{ .follow_symlinks = false }) catch |e| if (e == error.FileNotFound) continue else return e;
        if (info.kind == .sym_link or info.nlink > 1) return try arena.dupe(u8, n);
    }
    return null;
}

/// The learner over an engine whose glue `Tri` launches.
pub fn Slide(comptime Tri: type) type {
    return struct {
        pub const Example = lessons.Example;
        pub const Lesson = lessons.Lesson;
        pub const Report = lessons.Report;
        pub const Step = lessons.Step;
        pub const Trainer = train.Trainer(Tri);

        /// What lessons learn into: the engine, its kernels, its folder and the layout its config gave at load.
        pub const Model = struct { e: *prompt.Engine, tri: Tri, dir: []const u8, lay: living.Layout };

        const Host = struct {
            pub const Trainer = train.Trainer(Tri);
            pub const max_rows = train.max_rows;
            pub const Ctx = Model;
            pub const family = fam.family;

            pub fn layout(m: Ctx) living.Layout {
                return m.lay;
            }

            pub fn trainer(gpa: std.mem.Allocator, io: std.Io, m: Ctx) !*Host.Trainer {
                return Host.Trainer.init(gpa, io, m.e, m.tri);
            }

            pub fn drain(t: *Host.Trainer) void {
                t.drain();
            }

            pub fn pocketSaveBegin(m: Ctx, _: std.mem.Allocator, io: std.Io) !void {
                try @import("save_guard").begin(io, m.dir);
            }
            pub fn pocketSaveFinish(m: Ctx, _: std.mem.Allocator, io: std.Io) !void {
                try @import("save_guard").finish(io, m.dir);
            }

            pub fn free(t: *Host.Trainer, _: std.mem.Allocator) void {
                t.deinit();
            }

            pub fn seconds(io: std.Io) f64 {
                return @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds())) / 1e9;
            }

            /// Layer `layer`'s neurons for the forward: gate rows (sharp u), up rows (u / sharp), outputs.
            pub fn pocketSwiglu(m: Ctx, layer: usize, gate: []const f32, up: []const f32, down: []const f32, used: usize) !void {
                try m.e.setPocket(layer, gate, up, down, used);
            }

            /// Layer `layer`'s neurons as loaded from the shards (or uploaded since); the neurons in use.
            pub fn pocketReadSwiglu(m: Ctx, layer: usize, gate: []f32, up: []f32, down: []f32) !usize {
                return m.e.pocketRead(layer, gate, up, down);
            }

            /// Every layer's neurons into the folder's shards (widened on the first save), handed back as written.
            pub fn pocketSaveSwiglu(m: Ctx, gpa: std.mem.Allocator, io: std.Io, layers: []const usize, neurons: usize, gate: []f32, up: []f32, down: []f32) !usize {
                try m.e.stream.synchronize();
                return Shards.writePending(gpa, io, m.dir, m.lay, layers, neurons, gate, up, down);
            }
        };

        pub const Learner = lessons.Learner(Host);
    };
}
