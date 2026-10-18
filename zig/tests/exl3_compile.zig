//! Compile the real loader and device entry points without executing CUDA work.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
export fn exl3Compile(d: *const cuda.Driver, dense: *cuda.exl3.Dense, group: *cuda.exl3_grouped.Grouped, w: *const cuda.exl3.Weight, l: *const cuda.experts.Layer, s: *const cuda.Stream, p: *const cuda.grouped.Plan, io: *const std.Io, path: [*:0]const u8) void {
    var scan = core.exl3_load.scan(std.heap.page_allocator, io.*, std.mem.span(path)) catch return;
    defer scan.deinit();
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var upload: cuda.exl3_upload.Uploader = .{ .d = d, .gpa = arena.allocator(), .max_buffers = scan.groups.len * 4 };
    defer upload.deinit();
    for (scan.groups, 0..) |g, i| {
        const parts = scan.parts(arena.allocator(), i) catch return;
        _ = upload.dense(parts, g) catch return;
    }
    _ = cuda.exl3_upload.layer(&upload, arena.allocator(), 128, 128, .mul1, 0, 0, &.{}, &.{}, &.{}) catch {};
    const linear: cuda.qlinear.Linear = .{ .exl3 = dense };
    const weight: cuda.qlinear.Weight = .{ .exl3 = w.* };
    linear.decode(s.*, weight, .{ .x = 0 }, 0, 1) catch {};
    linear.prompt(s.*, weight, .{ .x = 0 }, 0, 1) catch {};
    linear.decodeF32(s.*, weight, .{ .x = 0 }, 0, 1) catch {};
    linear.promptF32(s.*, weight, .{ .x = 0 }, 0, 1) catch {};
    const grouped: cuda.experts.Grouped = .{ .exl3 = group };
    grouped.decode(s.*, .up, .{ .x = 0, .stride = l.dims }, l.*, p.*, 0, 1, -1) catch {};
    grouped.prompt(s.*, .down_f32, .{ .x = 0, .stride = l.width }, l.*, p.*, 0, 1, -1) catch {};
    grouped.prompt(s.*, .down_bf16, .{ .x = 0, .stride = l.width }, l.*, p.*, 0, 1, -1) catch {};
}

export fn candidateCompile(d: *const cuda.Driver, m: *const cuda.Module, s: *const cuda.Stream) void {
    var k = cuda.exl3_candidate.Kernels.resolve(d, m.*, 128) catch return;
    defer k.deinit();
    const w: cuda.exl3_candidate.Matrix = .{ .trellis = 0, .suh = 0, .svh = 0, .k = 128, .n = 128, .bits = 4, .codebook = .mul1 };
    k.rows(s.*, 0, 128, w, 0, 1, true) catch {};
    k.prompt(s.*, 0, 128, w, 0, 1, true) catch {};
}
