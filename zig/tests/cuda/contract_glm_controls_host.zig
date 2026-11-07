//! GLM page-case registration and planted-control wiring, without opening or calling a device.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const Controls = @import("contract_glm_controls.zig").Controls;
const Set = @import("contract_glm_regions.zig").Set;
const pages = @import("contract_glm_pages.zig");
const g = @import("core").contract.gates;
const t = std.testing;

const regions = "two regions";

/// Sixteen controls: the eight paged cases on interleaved pages, then the same eight on two regions.
fn validate(kernels: []const g.Kernel) !void {
    if (kernels.len != 16) return error.MissingPageControl;
    for (kernels, 0..) |k, i| {
        if (k.control != .shapes or !std.mem.startsWith(u8, k.spec.name, "control:"))
            return error.MissingPageControl;
        if (k.spec.served.len == 0) return error.MissingPageShape;
        if ((std.mem.indexOf(u8, k.spec.name, regions) != null) != (i >= 8)) return error.MissingRegionControl;
    }
}

/// Stand-in functions, a driver and a context that are never called: only descriptors are read.
const Stubs = struct {
    d: cuda.Driver = undefined,
    ctx: cuda.Context = undefined,
    functions: glm.glue.Kernels = undefined,

    fn init(x: *Stubs) void {
        inline for (@typeInfo(glm.glue.Kernels).@"struct".field_names) |name|
            @field(x.functions, name) = .{ .d = &x.d, .handle = null };
    }
    fn gpu(x: *Stubs) check.Gpu {
        return .{ .d = &x.d, .ctx = &x.ctx, .gpa = t.allocator, .io = t.io };
    }
    fn stream(x: *Stubs) cuda.Stream {
        return .{ .d = &x.d, .handle = null };
    }
};

fn registered(fault: enum { none, pages, regions }) !void {
    var x: Stubs = .{};
    x.init();
    var controls = Controls.init(x.gpu(), x.stream(), x.functions, .{ .d = &x.d, .handle = null }, null);
    // Only descriptors are read: empty rigs own no allocations and no CUDA callback is invoked.
    switch (fault) {
        .none => {},
        .pages => controls.sets[0].lat.wrong_table = false,
        .regions => controls.sets[1].chunks[3].wrong_table = false,
    }
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(t.allocator);
    try controls.append(t.allocator, &kernels);
    try validate(kernels.items);
}

test "GLM page contract: all sixteen paged numerical cases have a swapped-table or physical-row control" {
    try registered(.none);
}

test "GLM page contract: a disabled planted control is rejected by host inventory" {
    try t.expectError(error.MissingPageControl, registered(.pages));
    try t.expectError(error.MissingPageControl, registered(.regions));
}

test "GLM page contract: the eight cases on two regions register apart from the interleaved ones, each with its control" {
    var x: Stubs = .{};
    x.init();
    const tokens: cuda.Function = .{ .d = &x.d, .handle = null };
    var on = Set.init(x.gpu(), x.stream(), x.functions, tokens, true, false);
    var off = Set.init(x.gpu(), x.stream(), x.functions, tokens, false, false);
    var controls = Controls.init(x.gpu(), x.stream(), x.functions, tokens, null);
    var lists: [3]std.ArrayList(g.Kernel) = .{ .empty, .empty, .empty };
    defer for (&lists) |*l| l.deinit(t.allocator);
    try on.append(t.allocator, &lists[0]);
    try off.append(t.allocator, &lists[1]);
    try controls.append(t.allocator, &lists[2]);
    try t.expectEqual(@as(usize, 8), lists[0].items.len);
    for (lists[0].items, lists[1].items, lists[2].items[8..]) |k, base, control| {
        try t.expect(k.control == null and std.mem.indexOf(u8, k.spec.name, regions) != null);
        try t.expect(!std.mem.eql(u8, k.spec.name, base.spec.name)); // its own entry beside the interleaved case
        try t.expectEqualDeep(base.spec.served, k.spec.served); // at the same served shapes
        try t.expectEqualDeep(k.spec.served, control.spec.served); // and so is its control
    }
}

test "GLM page contract: two regions fill the interleaved arena, pooled keys first; off, every place is as before" {
    for ([_]usize{ 4, 11 }) |n| {
        const off: pages.Geometry = .{ .regions = false, .n = n };
        try t.expectEqual(pages.part_bytes, off.latAt());
        try t.expectEqual(pages.part_bytes + pages.lat_part, off.pkAt());
        try t.expect(off.latStride() == pages.page_bytes and off.pkStride() == pages.page_bytes);
        const on: pages.Geometry = .{ .regions = true, .n = n };
        // the tested cache is the second: the last page's latent rows end the arena, its pooled keys the first region
        try t.expectEqual(n * pages.page_bytes, on.latAt() + (n - 1) * on.latStride() + pages.lat_part);
        try t.expectEqual(on.latAt() - pages.lat_part, on.pkAt() + (n - 1) * on.pkStride() + pages.pk_part);
        try t.expectEqual(@as(usize, 1024), on.latStride() / (pages.lw * 2)); // token rows a page: 1,024, not 1,088
        try t.expectEqual(@as(usize, 0), on.pkStride() % 16); // tf_glm_index_scores' fast path takes it
    }
}

test "GLM page contract: a regions control's table differs from the launch's own at every entry, on the same pages" {
    var prng = std.Random.DefaultPrng.init(0x1431);
    for (1..12) |n| for (0..64) |_| {
        const ids = try pages.shuffled(t.allocator, prng.random(), n);
        defer t.allocator.free(ids);
        const other = try pages.otherTable(t.allocator, prng.random(), ids);
        defer t.allocator.free(other);
        for (ids, other) |a, b| try t.expect(a != b and b >= 0 and b < n + 3);
    };
}
