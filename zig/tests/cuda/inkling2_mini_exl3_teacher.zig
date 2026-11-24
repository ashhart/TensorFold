//! Test hook for scoring the Mini CUDA forward on a fixed, externally generated token trajectory.
const std = @import("std");
const cuda = @import("cuda");
const ink = @import("inkling2");
const check = @import("check.zig");

const Row = struct { id: []const u8, prompt_ids: []const u32, generated_ids: []const u32 };

pub fn run(gpu: check.Gpu, model_dir: []const u8, fixture: []const u8) !void {
    const a = gpu.gpa;
    const data = try std.Io.Dir.cwd().readFileAlloc(gpu.io, fixture, a, .limited(1 << 22));
    defer a.free(data);
    const rows = try std.json.parseFromSlice([]const Row, a, data, .{ .ignore_unknown_fields = true });
    defer rows.deinit();
    var context: usize = 0;
    for (rows.value) |r| {
        if (r.prompt_ids.len == 0 or r.generated_ids.len == 0) return error.EmptyTeacherFixture;
        context = @max(context, r.prompt_ids.len + r.generated_ids.len);
    }
    const loaded = try ink.native.open(a, gpu.io, gpu.ctx, model_dir, null, .{ .context = context, .drafts = false });
    defer loaded.deinit(loaded.ctx);
    const o = loaded.target;
    var best = try cuda.DeviceBuffer.alloc(gpu.d, 8);
    defer best.free();
    var matched: usize = 0;
    var total: usize = 0;
    for (rows.value) |r| {
        try o.model.reset(o.stream);
        var at: usize = 0;
        while (at < r.prompt_ids.len) {
            const end = @min(at + ink.native.prompt_rows, r.prompt_ids.len);
            try o.model.forward(&o.kernels, o.stream, r.prompt_ids[at..end], at, end - at - 1, @intFromBool(end == r.prompt_ids.len), false);
            at = end;
        }
        const vocab = o.weights.c.text.unpadded_vocab;
        var same: usize = 0;
        for (r.generated_ids, 0..) |token, i| {
            try ink.ops.top1(&o.kernels.glue, o.stream, o.model.s.logits, 1, vocab, vocab, 0, best.ptr, best.ptr + 4);
            var chosen: [1]u32 = undefined;
            try o.stream.synchronize();
            try best.download(4, std.mem.asBytes(&chosen));
            same += @intFromBool(chosen[0] == token);
            if (chosen[0] != token) std.debug.print("MISMATCH {s} {d} {d} {d}\n", .{ r.id, i, token, chosen[0] });
            if (i + 1 < r.generated_ids.len) try o.model.forward(&o.kernels, o.stream, &.{token}, at + i, 0, 1, false);
        }
        std.debug.print("TEACHER {s} {d}/{d}\n", .{ r.id, same, r.generated_ids.len });
        matched += same;
        total += r.generated_ids.len;
    }
    std.debug.print("TEACHER TOTAL {d}/{d}\n", .{ matched, total });
    try check.expect(total > 0 and matched * 1000 >= total * 967, "Mini EXL3 teacher top-1 below 96.7%", .{});
}
