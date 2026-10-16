//! `ling-ckpt-check DIR [INDEX]`: a Ling checkpoint against the loader's names, dtypes and shapes, from headers only.
const std = @import("std");
const core = @import("core");
const check = @import("check.zig");
const ling = @import("ling_cuda");

const names = ling.names;
const Config = ling.config.Config;
const Tensor = core.checkpoint.Tensor;

/// A tensor as the loader takes it: names.dims' shape, its format's dtype; side tensors as weights_proj reads them.
fn fits(c: *const Config, name: []const u8, t: names.Tensor, x: Tensor) bool {
    const tail = name[std.mem.lastIndexOfScalar(u8, name, '.').? + 1 ..];
    const small = x.dtype == .bf16 or x.dtype == .f32; // Reader.small and Reader.convs take either
    switch (names.dims(c, t.role)) {
        .values => |n| return small and x.is(x.dtype, &.{n}),
        .conv => |s| return small and x.is(x.dtype, &s),
        .linear => |nk| {
            const n = nk[0];
            const k = nk[1];
            if (t.role == .embed) return x.is(.bf16, &nk);
            if (t.role == .router) return c.format(name[0 .. name.len - tail.len - 1]) == .bf16 and small and x.is(x.dtype, &nk);
            const f = c.format(name[0 .. name.len - tail.len - 1]);
            if (!t.scale) return switch (f) {
                .bf16 => x.is(.bf16, &nk),
                .fp8_block128 => x.is(.f8_e4m3, &nk) and k % 128 == 0,
                .nvfp4 => x.is(.u8, &.{ n, k / 2 }) and k % 16 == 0,
            };
            return switch (f) {
                .bf16 => false,
                .fp8_block128 => x.is(.f32, &.{ (n + 127) / 128, k / 128 }),
                .nvfp4 => if (std.mem.eql(u8, tail, "weight_scale")) x.is(.f8_e4m3, &.{ n, k / 16 }) else x.dtype == .f32 and x.numel() == 1,
            };
        },
    }
}

const Tally = struct {
    ck: *core.Checkpoint,
    c: *const Config,
    expected: usize = 0,
    mtp: usize = 0,
    missing: usize = 0,
    wrong: usize = 0,
    bytes: u64 = 0,
    kinds: [3]usize = @splat(0), // weights, scale tensors, small tensors and convs

    pub fn put(k: *Tally, name: []const u8, t: names.Tensor) !void {
        k.expected += 1;
        if (t.layer != null and t.layer.? >= k.c.layers) k.mtp += 1;
        if (!k.ck.has(name)) {
            k.missing += 1;
            if (k.missing <= 10) std.debug.print("  missing {s}\n", .{name});
            return;
        }
        const x = try k.ck.get(name);
        k.bytes += x.bytes.len;
        k.kinds[if (names.dims(k.c, t.role) != .linear) 2 else if (t.scale) 1 else 0] += 1;
        if (fits(k.c, name, t, x)) return;
        k.wrong += 1;
        if (k.wrong <= 10) std.debug.print("  {s}: {t} {any}, not what the loader takes\n", .{ name, x.dtype, x.shape[0..x.rank] });
    }
};

pub fn run(gpu: check.Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 1) return error.MissingArgument;
    const dir = rest[0];
    var why: ling.config.Why = .{};
    var opened = ling.weights.open(gpu.gpa, gpu.io, dir, &why) catch |err| {
        std.debug.print("FAIL {s}: {s}\n", .{ dir, why.text() });
        return err;
    };
    defer opened.ck.close();
    const c = &opened.c;
    var stored: usize = 0;
    for (opened.ck.files.items) |f| stored += f.names.count();
    var k: Tally = .{ .ck = &opened.ck, .c = c };
    try names.all(c, &k);
    const unknown = opened.ck.unused();
    // the index's total_size, or without an index the files' data regions: every byte belongs to a checked tensor
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const index = if (rest.len > 1) rest[1] else try std.fs.path.join(arena.allocator(), &.{ dir, "model.safetensors.index.json" });
    var total: u64 = 0;
    if (std.Io.Dir.cwd().readFileAlloc(gpu.io, index, arena.allocator(), .limited(1 << 26))) |text| {
        const parsed = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), text, .{});
        total = @intCast(parsed.object.get("metadata").?.object.get("total_size").?.integer);
    } else |_| for (opened.ck.files.items) |f| {
        total += f.map.memory.len - f.data;
    }
    std.debug.print("RESULT ling-ckpt-check {t}: {d} shards, {d} tensors stored, {d} expected ({d} of the MTP layer): {d} weights, {d} scale tensors, {d} small tensors and convs\n", .{ c.quant, opened.ck.files.items.len, stored, k.expected, k.mtp, k.kinds[0], k.kinds[1], k.kinds[2] });
    std.debug.print("RESULT   {d} missing, {d} stored but not expected, {d} of another dtype or shape; {d} bytes against {d} (the index's total_size, else the files' data regions)\n", .{ k.missing, unknown, k.wrong, k.bytes, total });
    try check.expect(k.missing == 0 and unknown == 0 and k.wrong == 0 and stored == k.expected and k.bytes == total, "every stored tensor is one the loader reads, at its dtype and shape, and nothing is missing", .{});
    check.pass("ling-ckpt-check: all {d} tensors' names, dtypes and shapes as the loader takes them, {d} bytes (headers only)", .{ stored, k.bytes });
}
