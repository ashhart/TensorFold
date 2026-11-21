//! Real-checkpoint calibrated GPU stage timings, with profile-on/off logits checked before reporting.
const std = @import("std");
const tf = @import("tensorfold");
const counters = tf.deepseek_v41_profiler;
const Model = tf.deepseek_v41.Model;

pub fn run(gpa: std.mem.Allocator, io: std.Io, model: *Model, ids: []const u32, dir: []const u8) !void {
    if (ids.len == 0 or ids.len > model.workspace.plan.rows) return error.InvalidProfileWindow;
    const previous_grouping = model.grouped_prefill;
    defer model.grouped_prefill = previous_grouping;
    model.grouped_prefill = false;
    try model.reset();
    const baseline = try model.forward(ids, .all);
    const baseline_gpu_seconds = model.last_gpu_seconds;
    const expected = try gpa.dupe(f32, baseline);
    defer gpa.free(expected);
    model.grouped_prefill = true;
    try model.reset();
    const candidate = try model.forward(ids, .all);
    const candidate_gpu_seconds = model.last_gpu_seconds;
    try check(expected, candidate);
    var recorder = try counters.Recorder.init(gpa, model.device, .{ .enabled = true, .max_spans = 512 });
    defer recorder.deinit() catch unreachable;
    const previous_profile = model.profile;
    model.profile = &recorder;
    defer model.profile = previous_profile;
    try model.reset();
    const actual = try model.forward(ids, .last);
    try check(expected[expected.len - 129280 ..], actual);
    const totals = recorder.aggregates();
    var sampled_gpu_seconds: f64 = 0;
    for (totals) |value| sampled_gpu_seconds += value.gpu_seconds;
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(gpa);
    try text.print(gpa, "{{\"tokens\":{d},\"comparison_logits\":\"all\",\"candidate_grouping\":true,\"same_logits\":true,\"baseline_command_gpu_seconds\":{d},\"candidate_command_gpu_seconds\":{d},\"profiled_logits\":\"last\",\"sampled_span_gpu_seconds\":{d},\"stages\":[", .{ ids.len, baseline_gpu_seconds, candidate_gpu_seconds, sampled_gpu_seconds });
    for (std.enums.values(counters.Stage), totals, 0..) |stage, value, i| {
        try text.print(gpa, "{s}{{\"stage\":\"{s}\",\"calls\":{d},\"gpu_seconds\":{d}}}", .{ if (i == 0) "" else ",", @tagName(stage), value.calls, value.gpu_seconds });
    }
    try text.appendSlice(gpa, "]}\n");
    const path = try std.fs.path.join(gpa, &.{ dir, "gpu-profile.json" });
    defer gpa.free(path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = text.items });
    std.debug.print("{s}", .{text.items});
}

fn check(expected: []const f32, actual: []const f32) !void {
    if (expected.len != actual.len) return error.ProfileLogitShapeMismatch;
    for (actual, expected) |a, b| {
        if (!std.math.isFinite(a) or !std.math.isFinite(b)) return error.NonfiniteProfileLogits;
        if (@as(u32, @bitCast(a)) != @as(u32, @bitCast(b))) return error.ProfileChangedLogits;
    }
}
