//! One owner for the checkpoint's unchanged BF16 tensors on HIP, admitted before device allocation.
const std = @import("std");
const hip = @import("hip");
const st = @import("core").safetensors;
const cfg = @import("config.zig");
const weights = @import("weights.zig");
const Allocator = std.mem.Allocator;

pub const Placement = struct { spec: weights.Spec, offset: usize };

pub const Layout = struct {
    arena: std.heap.ArenaAllocator,
    tensors: std.StringArrayHashMapUnmanaged(Placement),
    bytes: usize,

    /// `c` must have passed config admission; every tensor starts on a 256-byte boundary.
    pub fn init(gpa: Allocator, c: cfg.Config) !Layout {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const specs = try weights.inventory(a, c);
        var tensors: std.StringArrayHashMapUnmanaged(Placement) = .empty;
        var bytes: usize = 0;
        for (specs) |s| {
            const pad = (256 - bytes % 256) % 256;
            const offset = try std.math.add(usize, bytes, pad);
            try tensors.put(a, s.name, .{ .spec = s, .offset = offset });
            bytes = try std.math.add(usize, offset, s.bytes());
        }
        return .{ .arena = arena, .tensors = tensors, .bytes = bytes };
    }

    pub fn deinit(l: *Layout) void {
        l.arena.deinit();
        l.* = undefined;
    }
};

pub const Weights = struct {
    layout: Layout,
    buffer: hip.DeviceBuffer,

    /// `budget` is the caller's admitted weight budget after runtime and stream reservations.
    pub fn load(gpa: Allocator, r: *const hip.Runtime, io: std.Io, dir: []const u8, c: cfg.Config, budget: usize) !Weights {
        var layout = try Layout.init(gpa, c);
        errdefer layout.deinit();
        if (layout.bytes > budget) return error.DotsWeightsDoNotFit;
        _ = try weights.inspect(gpa, io, dir, c);
        var scratch: std.heap.ArenaAllocator = .init(gpa);
        defer scratch.deinit();
        const a = scratch.allocator();
        const text = try cfg.readFile(a, io, dir, "model.safetensors.index.json");
        const root = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        const map = try st.index.weightMap(root);
        var seen: std.StringHashMapUnmanaged(void) = .empty;
        var buffer = try hip.DeviceBuffer.alloc(r, layout.bytes);
        errdefer buffer.free();
        var uploaded: usize = 0;
        for (map.values()) |v| {
            if (seen.contains(v.string)) continue;
            try seen.put(a, v.string, {});
            const path = try std.fs.path.join(a, &.{ dir, v.string });
            var file = try st.File.open(gpa, io, path);
            defer file.close(io);
            for (file.names.keys(), file.names.values()) |name, entry| {
                const target = map.get(name) orelse return error.DotsIndexMismatch;
                if (!std.mem.eql(u8, target.string, v.string)) return error.DotsIndexMismatch;
                const place = layout.tensors.get(name) orelse return error.DotsMissingTensor;
                try place.spec.check(entry);
                const tensor = file.get(name) orelse return error.DotsMissingTensor;
                try buffer.upload(place.offset, tensor.bytes);
                uploaded += 1;
            }
        }
        if (uploaded != layout.tensors.count()) return error.DotsTensorCount;
        return .{ .layout = layout, .buffer = buffer };
    }

    /// The address stays valid until deinit, which must follow completion of all commands that read it.
    pub fn get(w: *const Weights, name: []const u8) !u64 {
        const place = w.layout.tensors.get(name) orelse return error.DotsMissingTensor;
        return w.buffer.address(place.offset);
    }

    pub fn deinit(w: *Weights) void {
        w.buffer.free();
        w.layout.deinit();
        w.* = undefined;
    }
};

test "dots HIP weights: layout is complete, disjoint, aligned and admitted before runtime calls" {
    const cases = @import("config_test.zig");
    const c = try cfg.Config.parse(std.testing.allocator, cases.sample, cases.generation);
    var layout = try Layout.init(std.testing.allocator, c);
    defer layout.deinit();
    try std.testing.expectEqual(@as(usize, 643), layout.tensors.count());
    try std.testing.expectEqual(@as(usize, 6078358528), layout.bytes);
    var end: usize = 0;
    for (layout.tensors.values()) |place| {
        try std.testing.expectEqual(@as(usize, 0), place.offset % 256);
        try std.testing.expect(place.offset >= end);
        end = place.offset + place.spec.bytes();
    }
    try std.testing.expectEqual(layout.bytes, end);
    const r: hip.Runtime = undefined;
    try std.testing.expectError(error.DotsWeightsDoNotFit, Weights.load(std.testing.allocator, &r, std.testing.io, "absent", c, layout.bytes - 1));
    std.testing.refAllDecls(Weights);
}
