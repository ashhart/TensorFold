//! A single saved fact, a fresh learner, then another fact through actual affine shards.
const std = @import("std");
const living = @import("living");
const st = @import("core").safetensors;
const se = @import("core").shard_edit;
const a = std.testing.allocator;
const io = std.testing.io;
const base = "backbone.layers.0.mixer.shared_experts.";
const d = 64;
const n = 64;
pub const Store = struct {
    pub const write = living.shards.Io(struct {
        pub const safetensors = st;
        pub const shard_edit = se;
    }).write;

    pub fn init(dir: []const u8) !void {
        const words: [d * n / 8]u32 = @splat(0);
        const halves: [d * n / 64]u16 = @splat(0);
        const path = try std.fs.path.join(a, &.{ dir, "model.safetensors" });
        defer a.free(path);
        var parts: [6]se.New = undefined;
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const al = arena.allocator();
        for ([_][]const u8{ "up_proj", "down_proj" }, 0..) |proj, p| {
            for ([_][]const u8{ "weight", "scales", "biases" }, 0..) |leaf, j| parts[3 * p + j] = .{
                .name = try std.fmt.allocPrint(al, "{s}{s}.{s}", .{ base, proj, leaf }),
                .dtype = if (j == 0) .u32 else .bf16,
                .shape = try al.dupe(usize, &.{ n, if (j == 0) d / 8 else d / 64 }),
                .bytes = if (j == 0) std.mem.sliceAsBytes(&words) else std.mem.sliceAsBytes(&halves),
            };
        }
        try se.create(a, path, &.{.{ "format", "mlx" }}, &parts);
        var folder = try std.Io.Dir.cwd().openDir(io, dir, .{});
        defer folder.close(io);
        try folder.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"nemotron_h\",\"moe_shared_expert_intermediate_size\":64}" });
    }

    pub fn read(al: std.mem.Allocator, i: std.Io, dir: []const u8, up: []f32, down: []f32) !void {
        const path = try std.fs.path.join(al, &.{ dir, "model.safetensors" });
        defer al.free(path);
        var f = try st.File.open(al, i, path);
        defer f.close(i);
        inline for (.{ "up_proj", "down_proj" }, 0..) |proj, p| {
            const w = std.mem.bytesAsSlice(u32, f.get(base ++ proj ++ ".weight").?.bytes);
            const s = std.mem.bytesAsSlice(u16, f.get(base ++ proj ++ ".scales").?.bytes);
            const b = std.mem.bytesAsSlice(u16, f.get(base ++ proj ++ ".biases").?.bytes);
            for (0..d) |r| {
                var ws: [8]u32 = undefined;
                const start = if (p == 0) (n + r) * d / 8 else r * (n + n) / 8 + n / 8;
                const g = if (p == 0) n + r else r * 2 + 1;
                @memcpy(&ws, w[start..][0..8]);
                var row: [d]f32 = undefined;
                living.affine.dequantize(living.families.nemotron_h.quant, &ws, &.{s[g]}, &.{b[g]}, &row);
                if (p == 0) {
                    @memcpy(up[r * d ..][0..d], &row);
                } else for (row, 0..) |x, k| {
                    down[k * d + r] = x;
                }
            }
        }
    }
};

pub const Host = living.restart_fixture.Fixture(Store);
pub const Adapter = @import("native/slide_adapter.zig").Adapter(struct {
    pub const Learner = Host.Learner;
    pub const Lesson = living.learner.Lesson;
    pub const Example = living.learner.Example;
});
