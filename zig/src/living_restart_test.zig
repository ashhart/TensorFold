//! A single saved fact, a fresh learner, then another fact through actual affine shards.
const std = @import("std");
const living = @import("living");
const st = @import("core/safetensors.zig");
const se = @import("core/shard_edit.zig");
const a = std.testing.allocator;
const io = std.testing.io;
const base = "backbone.layers.0.mixer.shared_experts.";
const d = 64;
const n = 64;
const Store = struct {
    pub const write = living.shards.Io(struct {
        pub const safetensors = st;
        pub const shard_edit = se;
    }).write;

    fn init(dir: []const u8) !void {
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

test "qualification: learn one fact, save, restart and teach a second without losing the first" {
    const H = living.restart_fixture.Fixture(Store);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    try Store.init(dir);
    var h: H = .{ .dir = dir };
    {
        var l = H.Learner.init(a, io, &h);
        defer l.deinit();
        try living.restart_fixture.teach(&l, 1, &.{});
        try living.restart_fixture.save(&l);
        try std.testing.expectEqual(@as(usize, 1), h.used);
    }
    try h.reopen();
    try std.testing.expectEqual(@as(usize, 1), h.used);
    const saved_up = h.up;
    var l = H.Learner.init(a, io, &h);
    defer l.deinit();
    const old_ids = [_]u32{ 1, 2, 3, 4 };
    const old = [_]living.learner.Example{.{ .ids = &old_ids, .start = 3, .decide = 2 }};
    try living.restart_fixture.teach(&l, 2, &old);
    try std.testing.expectEqual(@as(usize, 2), h.used);
    try std.testing.expectEqualSlices(f32, saved_up[0..d], h.up[0..d]);
    try std.testing.expect(try l.recalls(&old, false));
    try living.restart_fixture.save(&l);
    try h.reopen();
    try std.testing.expect(try l.recalls(&old, false));
    const before_up = h.up;
    const before_down = h.down;
    const before_file = try tmp.dir.readFileAlloc(io, "model.safetensors", a, .limited(1 << 20));
    defer a.free(before_file);
    try living.restart_fixture.teach(&l, 3, &old);
    try l.begin(.{ .undo = true });
    try std.testing.expect(l.step().done);
    try std.testing.expectEqualSlices(f32, &before_up, &h.up);
    try std.testing.expectEqualSlices(f32, &before_down, &h.down);
    try living.restart_fixture.save(&l);
    try h.reopen();
    try std.testing.expectEqualSlices(f32, &before_up, &h.up);
    try std.testing.expectEqualSlices(f32, &before_down, &h.down);
    const after_file = try tmp.dir.readFileAlloc(io, "model.safetensors", a, .limited(1 << 20));
    defer a.free(after_file);
    try std.testing.expectEqualSlices(u8, before_file, after_file);
}

test "multi-fact delete: saved disjoint rows equal never taught forward bytes after reopen" {
    const H = living.restart_fixture.Fixture(Store);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    try Store.init(dir);
    var h: H = .{ .dir = dir, .used = 3 };
    for (0..3) |k| {
        h.up[k * d + k] = 15;
        @memset(h.down[k * d ..][0..d], 15);
    }
    _ = try Store.write(a, io, dir, H.family, H.layout(&h), &.{0}, n, &.{&h.up}, &h.down);
    try h.reopen();
    const old_down = h.down;
    var reference = h;
    @memset(reference.up[0 .. 2 * d], 0);
    {
        var l = H.Learner.init(a, io, &h);
        defer l.deinit();
        try l.begin(.{ .forget = &.{ .{ 0, 0 }, .{ 0, 1 } } });
        const step = l.step();
        try std.testing.expect(step.done);
        try std.testing.expectEqual(@as(u32, 2), step.report.?.learned.steps);
        try living.restart_fixture.save(&l);
    }
    try h.reopen();
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&old_down), std.mem.sliceAsBytes(&h.down));
    try std.testing.expectEqualSlices(f32, &reference.up, &h.up);
    for (0..9) |case| {
        var x: [d]f32 = @splat(0.25);
        x[case] = 2;
        const want = forward(reference.up, reference.down, x);
        const got = forward(h.up, h.down, x);
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&want), std.mem.sliceAsBytes(&got));
    }
}

fn forward(up: [d * n]f32, down: [d * n]f32, x: [d]f32) [d]f32 {
    var y: [d]f32 = @splat(0.5);
    for (0..n) |k| {
        var dot: f32 = 0;
        for (x, up[k * d ..][0..d]) |v, w| dot += v * w;
        const z = @max(dot, 0);
        for (&y, down[k * d ..][0..d]) |*v, w| v.* += z * z * w;
    }
    return y;
}
