//! The verify probe's paged twin, using the serving paged partial source and reversed physical pages.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const rig = @import("rig.zig");
pub const page = 256;
pub const page_bytes: u64 = 2 * 2 * page * 128 * 2;
pub const State = struct {
    enabled: bool = false,
    lib: ?mtl.Library = null,
    partials: [16]?mtl.Pipeline = @splat(null),
    arena: rig.Ref = undefined,
    table: rig.Ref = undefined,
    pub fn deinit(s: *State) void {
        for (s.partials) |p| if (p) |x| x.deinit();
        if (s.lib) |l| l.deinit();
    }
    pub fn pipeline(s: *State, a: std.mem.Allocator, device: mtl.Device, tiles: usize) !mtl.Pipeline {
        if (s.lib == null) {
            const text = try tf.nemotron.simd_attention.rewrite(a, sources.nemotron_paged.partial);
            defer a.free(text);
            s.lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        }
        if (s.partials[tiles - 1] == null) s.partials[tiles - 1] = try mtl.Pipeline.init(device, s.lib.?, sources.nemotron_paged.partial_names[tiles - 1], false);
        return s.partials[tiles - 1].?;
    }
    pub fn prepare(s: *State, p: anytype) !void {
        const v = &p.inner;
        const pages = (v.cap + page - 1) / page;
        const ids = try p.a.alloc(u32, pages);
        defer p.a.free(ids);
        for (ids, 0..) |*id, i| id.* = @intCast(pages - 1 - i);
        s.table = try p.rig.input(std.mem.sliceAsBytes(ids));
        const data = try p.a.alloc(u8, pages * page_bytes);
        defer p.a.free(data);
        @memset(data, 0xff);
        for (0..v.cap) |key| for (0..2) |plane| for (0..2) |head| {
            const dst = (pages - 1 - key / page) * page_bytes + ((plane * 2 + head) * page + key % page) * 128 * 2;
            const src = v.at(plane == 1, head, key);
            @memcpy(data[dst..][0..256], std.mem.sliceAsBytes(v.host[src..][0..128]));
        };
        s.arena = try p.rig.input(data);
    }
};
