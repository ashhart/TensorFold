//! A learned Metal checkpoint is viewed at its original MLP width before either native packing layout is made.
const std = @import("std");
const metal = @import("metal");
const cfg = @import("config.zig");
const ckpt = @import("../../core/checkpoint_metal.zig");
const matrix = @import("living").dense_matrix;
pub const Proxy = struct {
    a: std.mem.Allocator,
    device: metal.Device,
    checkpoint: *const ckpt.Checkpoint,
    config: cfg.Config,
    owned: std.ArrayList(metal.Buffer) = .empty,
    cache: std.StringHashMapUnmanaged(ckpt.Tensor) = .empty,
    pub fn deinit(p: *Proxy) void {
        for (p.owned.items) |b| b.deinit();
        p.owned.deinit(p.a);
        var it = p.cache.keyIterator();
        while (it.next()) |name| p.a.free(name.*);
        p.cache.deinit(p.a);
    }
    pub fn get(p: *Proxy, name: []const u8) !ckpt.Tensor {
        if (p.cache.get(name)) |t| return t;
        const original = try p.checkpoint.get(name);
        const last = std.mem.lastIndexOfScalar(u8, name, '.') orelse return original;
        const role = matrix.role(name[0..last]) orelse return original;
        if (p.config.learned == 0) return original;
        const part = std.meta.stringToEnum(matrix.Part, name[last + 1 ..]) orelse return error.DensePocketShape;
        var arena: std.heap.ArenaAllocator = .init(p.a);
        defer arena.deinit();
        const view = matrix.View.of(original);
        const bytes = try matrix.base(arena.allocator(), p.config.geo.hidden, p.config.geo.intermediate, p.config.learned, role, part, view);
        var out = original;
        out.bytes = bytes.len;
        if (role == .down) {
            out.shape[1] = p.config.geo.intermediate / (if (part == .weight) @as(usize, 8) else 64);
            const buffer = try p.device.buffer(bytes.len, metal.ResourceOptions.shared | metal.ResourceOptions.untracked);
            errdefer buffer.deinit();
            @memcpy(buffer.contents()[0..bytes.len], bytes);
            try p.owned.append(p.a, buffer);
            out.buffer = buffer;
            out.offset = 0;
        } else out.shape[0] = p.config.geo.intermediate;
        const key = try p.a.dupe(u8, name);
        errdefer p.a.free(key);
        try p.cache.put(p.a, key, out);
        return out;
    }
};
