//! Native prefill GPU classes through the shared counter recorder, checked against an uninstrumented pass.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.lfm3;
const counters = q.forward.counters;

fn now() i128 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;
}

const Result = struct {
    logits: [q.config.vocab]u16,
    wall_seconds: f64,
    command_gpu_seconds: f64,
    totals: [counters.class_count]counters.Aggregate,
};

fn run(a: std.mem.Allocator, model: *q.Model, ids: []const u32, chunk: usize, profile: ?*counters.Recorder) !*Result {
    const result = try a.create(Result);
    errdefer a.destroy(result);
    result.totals = @splat(.{});
    var scratch = try q.state.Scratch.init(a, model.device, &model.config.layer_types, chunk);
    defer scratch.deinit();
    var cache = try q.state.Cache.init(a, model.device, &model.config.layer_types, ids.len + 32);
    defer cache.deinit();
    model.profile = profile;
    defer model.profile = null;
    const t0 = now();
    const g0 = q.forward.gpu_seconds;
    var at: usize = 0;
    while (at < ids.len) {
        const rows = @min(chunk, ids.len - at);
        try q.forward.run(model, &scratch, &.{.{ .cache = &cache, .rows = rows }}, ids[at..][0..rows], false, .last);
        if (profile) |recorder| {
            for (&result.totals, recorder.aggregates()) |*total, part| {
                total.calls += part.calls;
                total.gpu_seconds += part.gpu_seconds;
            }
        }
        at += rows;
    }
    result.wall_seconds = @as(f64, @floatFromInt(now() - t0)) / 1e9;
    result.command_gpu_seconds = q.forward.gpu_seconds - g0;
    @memcpy(&result.logits, cache.logits.slice(u16, q.config.vocab));
    return result;
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 4 and args.len != 5) return error.ExpectedModelTokensOutputOptionalChunk;
    const chunk = if (args.len == 5) try std.fmt.parseInt(usize, args[4], 10) else 128;
    if (chunk == 0 or chunk > 128) return error.ProfileChunkOutsideSupportedSpanBudget;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], init.gpa, .limited(4 << 20));
    defer init.gpa.free(bytes);
    const input = try tf.npy.parse(bytes);
    if (!std.mem.eql(u8, input.descr, "<u4") or input.rank != 1 or input.count() == 0 or input.count() + 32 > q.state.max_context)
        return error.ExpectedBoundedU32Tokens;
    const ids = try init.gpa.alloc(u32, input.count());
    defer init.gpa.free(ids);
    @memcpy(std.mem.sliceAsBytes(ids), input.data);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const model = try q.Model.load(init.gpa, init.io, args[1]);
    defer model.deinit();
    var recorder = try counters.Recorder.init(init.gpa, model.device, .{ .enabled = true, .max_spans = 512 });
    defer recorder.deinit() catch unreachable;
    const plain = try run(init.gpa, model, ids, chunk, null);
    defer init.gpa.destroy(plain);
    const measured = try run(init.gpa, model, ids, chunk, &recorder);
    defer init.gpa.destroy(measured);
    if (!std.mem.eql(u16, &plain.logits, &measured.logits)) return error.ProfileChangedLogitBytes;
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(std.mem.asBytes(&measured.logits), &digest, .{});
    const json = try std.json.Stringify.valueAlloc(init.gpa, .{
        .tokens = ids.len,
        .chunk = chunk,
        .mode = recorder.capability().mode(),
        .instrumented_final_logit_bytes_equal = true,
        .logits_sha256 = std.fmt.bytesToHex(digest, .lower),
        .plain_wall_seconds = plain.wall_seconds,
        .plain_command_gpu_seconds = plain.command_gpu_seconds,
        .profiled_wall_seconds = measured.wall_seconds,
        .profiled_command_gpu_seconds = if (measured.command_gpu_seconds > 0 and measured.command_gpu_seconds <= measured.wall_seconds) @as(?f64, measured.command_gpu_seconds) else null,
        .profiled_command_gpu_timestamp_valid = measured.command_gpu_seconds > 0 and measured.command_gpu_seconds <= measured.wall_seconds,
        .timing_source = "calibrated_gpu_counter_samples",
        .aggregate_classes = std.meta.tags(counters.Class),
        .aggregates = measured.totals,
        .activation_includes_convolution = true,
    }, .{ .whitespace = .indent_2 });
    defer init.gpa.free(json);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[3], .data = json });
    std.debug.print("LFM3_PROFILE tokens={d} chunk={d} mode={t} final_logits_equal=true\n", .{ ids.len, chunk, recorder.capability().mode() });
}
