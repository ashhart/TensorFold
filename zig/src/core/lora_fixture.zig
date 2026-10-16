//! Synthetic dense and shipped checkpoint folders and PEFT adapters for the merge tests: no model, no GPU.
const std = @import("std");
const se = @import("shard_edit.zig");
const host4 = @import("affine4_host.zig");
const lm = @import("lora_quant.zig");
const Io = std.Io;

pub const io = std.testing.io;
pub const base_config = "{\"model_type\":\"x\",\"quantization\":{\"group_size\":64,\"bits\":4,\"mode\":\"affine\"}}";
pub const module = "backbone.layers.0.mixer.in_proj";

pub const Scratch = struct {
    arena: std.heap.ArenaAllocator,
    root: []const u8,

    pub fn init() !Scratch {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        const root = try std.fmt.allocPrint(arena.allocator(), ".tf-merge-test-{d}", .{Io.Clock.awake.now(io).toNanoseconds()});
        return .{ .arena = arena, .root = root };
    }

    pub fn deinit(s: *Scratch) void {
        Io.Dir.cwd().deleteTree(io, s.root) catch {};
        s.arena.deinit();
    }

    pub fn path(s: *Scratch, parts: []const []const u8) ![]const u8 {
        return std.fs.path.join(s.arena.allocator(), &.{ s.root, std.fs.path.join(s.arena.allocator(), parts) catch unreachable });
    }

    pub fn read(s: *Scratch, folder: []const u8, name: []const u8) ![]const u8 {
        return Io.Dir.cwd().readFileAlloc(io, try s.path(&.{ folder, name }), s.arena.allocator(), .limited(1 << 24));
    }

    pub fn write(s: *Scratch, folder: []const u8, name: []const u8, data: []const u8) !void {
        const p = try s.path(&.{ folder, name });
        try Io.Dir.cwd().createDirPath(io, std.fs.path.dirname(p).?);
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = data });
    }

    /// The names in `folder` (relative to the scratch root), sorted.
    pub fn list(s: *Scratch, folder: []const u8) ![]const []const u8 {
        var names: std.ArrayList([]const u8) = .empty;
        var d = Io.Dir.cwd().openDir(io, try s.path(&.{folder}), .{ .iterate = true }) catch return &.{};
        defer d.close(io);
        var it = d.iterate();
        while (try it.next(io)) |e| try names.append(s.arena.allocator(), try s.arena.allocator().dupe(u8, e.name));
        std.mem.sort([]const u8, names.items, {}, struct {
            fn lt(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.lt);
        return names.items;
    }
};

/// The three leaves and the dense weight of a projection whose values are `vals` (each exactly a bf16).
pub fn leaves(a: std.mem.Allocator, name: []const u8, n: usize, k: usize, vals: []const f32, dense: *std.ArrayList(se.New), quantized: *std.ArrayList(se.New)) !void {
    const bf = try a.alloc(u16, n * k);
    for (bf, vals) |*h, v| h.* = host4.bf16of(v);
    const q: lm.Quant = .{ .words = try a.alloc(u32, n * k / 8), .scales = try a.alloc(u16, n * k / 64), .biases = try a.alloc(u16, n * k / 64) };
    if ((try lm.quantize(64, vals, q)) != null) return error.NonFinite;
    const g = try a.dupe(usize, &.{ n, k / 64 });
    try dense.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.weight", .{name}), .dtype = .bf16, .shape = try a.dupe(usize, &.{ n, k }), .bytes = std.mem.sliceAsBytes(bf) });
    try quantized.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.weight", .{name}), .dtype = .u32, .shape = try a.dupe(usize, &.{ n, k / 8 }), .bytes = std.mem.sliceAsBytes(q.words) });
    try quantized.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.scales", .{name}), .dtype = .bf16, .shape = g, .bytes = std.mem.sliceAsBytes(q.scales) });
    try quantized.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.biases", .{name}), .dtype = .bf16, .shape = g, .bytes = std.mem.sliceAsBytes(q.biases) });
}

/// Pseudo-random bf16-exact values.
pub fn randomValues(a: std.mem.Allocator, count: usize, seed: u64) ![]f32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const v = try a.alloc(f32, count);
    for (v) |*x| x.* = host4.f32of(host4.bf16of((prng.random().float(f32) - 0.5) * 0.2));
    return v;
}

/// A one-projection pair of folders `dense` and `base` (one shard, no index) around `vals` (n x k, k a multiple of 64).
pub fn makeSingle(s: *Scratch, n: usize, k: usize, vals: []const f32) ![]const u8 {
    const a = s.arena.allocator();
    var dense: std.ArrayList(se.New) = .empty;
    var quantized: std.ArrayList(se.New) = .empty;
    try leaves(a, module, n, k, vals, &dense, &quantized);
    try Io.Dir.cwd().createDirPath(io, try s.path(&.{"dense"}));
    try Io.Dir.cwd().createDirPath(io, try s.path(&.{"base"}));
    try se.create(std.testing.allocator, try s.path(&.{ "dense", "model.safetensors" }), &.{}, dense.items);
    try se.create(std.testing.allocator, try s.path(&.{ "base", "model.safetensors" }), &.{.{ "format", "mlx" }}, quantized.items);
    try s.write("base", "config.json", base_config);
    return try s.path(&.{"base"});
}
