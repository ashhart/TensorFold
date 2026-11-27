//! Host replacements for engine allocation/device attributes and rope-table upload; CPU resolution is real.
const std = @import("std");
const cuda = @import("cuda");
const rope = @import("rope");
pub const supportsTarget = cuda.round_attribution.supportsTarget;
pub const Options = struct { rope_yarn: ?f64 = null, context: usize = 4096, kv8: bool = true };
pub const Context = struct {
    handle: cuda.abi.Context = null,
    pub fn attribute(_: @This(), tag: anytype) !c_int {
        return if (tag == .compute_capability_major) 12 else 1;
    }
};
pub fn fitOf(_: Options) void {}
const Widths = struct {
    head: u8 = 4,
    pub fn mixed(w: @This(), _: usize) bool {
        return w.head != 4;
    }
};
const Geometry = struct {
    hidden: usize = 5120,
    layers: usize = 64,
    experts: usize = 0,
    exl3: bool = false,
    tfq3: bool = false,
    kinds: struct { modelopt: bool = true } = .{},
    bits: Widths = .{},
    first: usize = 0,
    whole: usize = 0,
    context: usize,
};
pub const Engine = struct {
    a: std.mem.Allocator,
    g: Geometry,
    stream: struct { handle: cuda.abi.Stream = null } = .{},
    kv8: bool,
    uploaded: ?rope.Rope = null,
    uploads: usize = 0,
    pub fn init(a: std.mem.Allocator, io: std.Io, _: Context, dir: []const u8, kv8: bool, _: void, _: usize) !*Engine {
        const path = try std.fs.path.join(a, &.{ dir, "config.json" });
        defer a.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
        defer a.free(text);
        const parsed = try std.json.parseFromSlice(std.json.Value, a, text, .{});
        defer parsed.deinit();
        const e = try a.create(Engine);
        e.* = .{ .a = a, .g = .{ .context = @intCast(parsed.value.object.get("max_position_embeddings").?.integer) }, .kv8 = kv8 };
        e.g.kinds.modelopt = !parsed.value.object.contains("quantization");
        return e;
    }
    pub fn deinit(e: *Engine) void {
        e.a.destroy(e);
    }
};
pub fn setRope(e: *Engine, value: rope.Rope) !void {
    e.uploads += 1;
    e.uploaded = value;
}
