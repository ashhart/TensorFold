//! Full-checkpoint admission and execution receipts; no synthetic weights or replay substitute.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const ds = tf.deepseek_v41;
const config = tf.deepseek_v41_config;
const load = tf.deepseek_v41_load;
const weights = tf.deepseek_v41_weights;
const process_memory = @import("src/server/process_memory.zig");
const oracle = @import("tests/deepseek_v41_reference/qualify.zig");
const margins = @import("tests/deepseek_v41_reference/margins.zig");
const profile = @import("tests/deepseek_v41_reference/profile.zig");
const trace = @import("tests/deepseek_v41_reference/trace.zig");
const operators = @import("tests/deepseek_v41_reference/operators.zig");
const attention_operators = @import("tests/deepseek_v41_reference/operator_attention.zig");
const history_operators = @import("tests/deepseek_v41_reference/operator_history.zig");
const hc_operators = @import("tests/deepseek_v41_reference/operator_hc.zig");
const moe_operators = @import("tests/deepseek_v41_reference/operator_moe.zig");

fn usage(code: u8) noreturn {
    std.debug.print("usage: tf-dsv41-run plan MODEL TABLES\n" ++
        "       tf-dsv41-run teacher MODEL TABLES IDS_JSON OUT_DIR [CHUNK]\n" ++
        "       tf-dsv41-run seams MODEL TABLES IDS_JSON OUT_DIR\n" ++
        "       tf-dsv41-run oracle MODEL TABLES IDS_JSON OUT_DIR\n" ++
        "       tf-dsv41-run margins MODEL TABLES IDS_JSON OUT_DIR [DEEP]\n" ++
        "       tf-dsv41-run profile MODEL TABLES IDS_JSON OUT_DIR\n" ++
        "       tf-dsv41-run trace MODEL TABLES IDS_JSON OUT_DIR\n" ++
        "       tf-dsv41-run operators MODEL TABLES OUT_DIR\n" ++
        "       tf-dsv41-run attention-sameinput MODEL TABLES TRACE_CPU_BIN OUT_DIR [LAYER]\n" ++
        "       tf-dsv41-run attention-history-sameinput MODEL TABLES TRACE_DIR OUT_DIR TOKEN LAYER\n" ++
        "       tf-dsv41-run hc-sameinput MODEL TABLES TRACE_CPU_BIN OUT_DIR [LAYER]\n" ++
        "       tf-dsv41-run moe-sameinput MODEL TABLES TRACE_GPU_BIN OUT_DIR [LAYER]\n" ++
        "       tf-dsv41-run generate MODEL TABLES IDS_JSON OUT_DIR [TOKENS]\n" ++
        "IDS_JSON: nonempty JSON array of token IDs below 129280.\n" ++
        "Limits: oracle/trace 16 IDs; profile 256 IDs; teacher/seams 4096 IDs.\n" ++
        "History: TOKEN below 4096; retained normalized prefix at most 256 MiB.\n" ++
        "Local diagnostics do not qualify the model. Oracle retains the independent 1% NRMS gate.\n" ++
        "DSV41_ENGRAM_STREAM=1 reads the Engram tables from their shards (not with oracle or trace);\n" ++
        "the receipt then admits disk reads up to the counted Engram row bytes and no page-ins.\n", .{});
    std.process.exit(code);
}
fn streamEngram() bool {
    const v = std.c.getenv("DSV41_ENGRAM_STREAM") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}
fn number(s: []const u8) !u32 {
    return std.fmt.parseInt(u32, s, 10);
}
fn validArguments(args: []const [:0]const u8) bool {
    if (args.len < 2) return false;
    const mode = args[1];
    for ([_][]const u8{ "plan", "operators" }, [_]usize{ 4, 5 }) |name, count|
        if (std.mem.eql(u8, mode, name)) return args.len == count;
    if (std.mem.eql(u8, mode, "attention-history-sameinput")) return args.len == 8;
    for ([_][]const u8{ "teacher", "generate", "attention-sameinput", "hc-sameinput", "moe-sameinput", "margins" }) |name|
        if (std.mem.eql(u8, mode, name)) return args.len == 6 or args.len == 7;
    if (std.mem.eql(u8, mode, "lanes")) return args.len == 6 or args.len == 7 or args.len == 8;
    for ([_][]const u8{ "seams", "oracle", "profile", "trace" }) |name|
        if (std.mem.eql(u8, mode, name)) return args.len == 6;
    return false;
}
fn argmax(row: []const f32) !u32 {
    var best: u32 = 0;
    if (row.len == 0) return error.EmptyLogits;
    var score = -std.math.inf(f32);
    for (row, 0..) |v, i| {
        if (!std.math.isFinite(v)) return error.NonfiniteLogits;
        if (v > score) {
            best = @intCast(i);
            score = v;
        }
    }
    return best;
}
fn write(io: std.Io, a: std.mem.Allocator, dir: []const u8, name: []const u8, data: []const u8) !void {
    const path = try std.fs.path.join(a, &.{ dir, name });
    defer a.free(path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = data });
}
const Counters = struct { disk_read: u64, pageins: u64, footprint: u64 };
/// Darwin's full rusage_info_v4: the kernel writes all 336 bytes, so every field must be here.
const RUsage = extern struct {
    uuid: [16]u8,
    user_time: u64,
    system_time: u64,
    pkg_idle_wkups: u64,
    interrupt_wkups: u64,
    pageins: u64,
    wired_size: u64,
    resident_size: u64,
    phys_footprint: u64,
    proc_start_abstime: u64,
    proc_exit_abstime: u64,
    child_user_time: u64,
    child_system_time: u64,
    child_pkg_idle_wkups: u64,
    child_interrupt_wkups: u64,
    child_pageins: u64,
    child_elapsed_abstime: u64,
    diskio_bytesread: u64,
    diskio_byteswritten: u64,
    cpu_time_qos_default: u64,
    cpu_time_qos_maintenance: u64,
    cpu_time_qos_background: u64,
    cpu_time_qos_utility: u64,
    cpu_time_qos_legacy: u64,
    cpu_time_qos_user_initiated: u64,
    cpu_time_qos_user_interactive: u64,
    billed_system_time: u64,
    serviced_system_time: u64,
    logical_writes: u64,
    lifetime_max_phys_footprint: u64,
    instructions: u64,
    cycles: u64,
    billed_energy: u64,
    serviced_energy: u64,
    interval_max_phys_footprint: u64,
    runnable_time: u64,
};
fn counters() !Counters {
    const proc_pid_rusage = struct {
        extern "c" fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *anyopaque) c_int;
    }.proc_pid_rusage;
    const getpid = struct {
        extern "c" fn getpid() c_int;
    }.getpid;
    var r: RUsage = undefined;
    if (proc_pid_rusage(getpid(), 4, &r) != 0) return error.MemoryCountersUnavailable;
    return .{ .disk_read = r.diskio_bytesread, .pageins = r.pageins, .footprint = r.phys_footprint };
}
fn teacher(m: *ds.Model, ids: []const u32, chunk: u32, out: []f32) !f64 {
    if (chunk == 0 or chunk > m.workspace.plan.rows or ids.len > m.cache.plan.capacity)
        return error.InvalidTeacherWindow;
    if (out.len != ids.len * 129280) return error.InvalidTeacherOutputLength;
    try m.reset();
    var elapsed: f64 = 0;
    var start: usize = 0;
    while (start < ids.len) {
        const end = @min(start + chunk, ids.len);
        const row = try m.forward(ids[start..end], .all);
        @memcpy(out[start * 129280 .. end * 129280], row);
        for (row) |value| if (!std.math.isFinite(value)) return error.NonfiniteLogits;
        elapsed += m.last_gpu_seconds;
        start = end;
    }
    return elapsed;
}

fn teacherOn(m: *ds.Model, slot: u32, ids: []const u32, chunk: u32, out: []f32) !f64 {
    if (chunk == 0 or chunk > m.workspace.plan.rows or ids.len > m.caches[slot].plan.capacity)
        return error.InvalidTeacherWindow;
    if (out.len != ids.len * 129280) return error.InvalidTeacherOutputLength;
    try m.resetSlot(slot);
    var elapsed: f64 = 0;
    var start: usize = 0;
    while (start < ids.len) {
        const end = @min(start + chunk, ids.len);
        const row = try m.forwardOn(slot, ids[start..end], .all);
        @memcpy(out[start * 129280 .. end * 129280], row);
        for (row) |value| if (!std.math.isFinite(value)) return error.NonfiniteLogits;
        elapsed += m.last_gpu_seconds;
        start = end;
    }
    return elapsed;
}

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len == 2 and (std.mem.eql(u8, args[1], "--help") or std.mem.eql(u8, args[1], "-h"))) usage(0);
    if (!validArguments(args)) usage(2);
    const arena = init.arena.allocator();
    const mode = args[1];
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    if (std.mem.eql(u8, mode, "plan")) {
        const cfg = try config.Config.load(arena, io, args[2]);
        var plan = try load.Plan.openWith(a, io, args[2], args[3], if (streamEngram()) .{ .engram = .streamed, .bound_only = true } else .{});
        defer plan.deinit();
        try weights.validate(&plan);
        const device = try mtl.Device.init();
        defer device.deinit();
        std.debug.print("{{\"mode\":\"plan\",\"weight_bytes\":{d},\"shards\":{d},\"tensors\":{d},\"context\":{d},\"metal_working_set_bytes\":{d}}}\n", .{ plan.bytes, plan.shards.items.len, plan.tensors.count(), cfg.text_config.max_position_embeddings, device.maxWorkingSet() });
        return;
    }
    const is_operators = std.mem.eql(u8, mode, "operators");
    const is_attention = std.mem.eql(u8, mode, "attention-sameinput");
    const is_attention_history = std.mem.eql(u8, mode, "attention-history-sameinput");
    const is_hc = std.mem.eql(u8, mode, "hc-sameinput");
    const is_moe = std.mem.eql(u8, mode, "moe-sameinput");
    const is_lanes = std.mem.eql(u8, mode, "lanes");
    const max_ids: usize = if (std.mem.eql(u8, mode, "margins")) 1048576 else if (std.mem.eql(u8, mode, "oracle") or std.mem.eql(u8, mode, "trace")) 16 else if (std.mem.eql(u8, mode, "profile")) ds.prefill_step else if (std.mem.eql(u8, mode, "teacher") or std.mem.eql(u8, mode, "seams") or is_lanes) 4096 else 1048576;
    const input = if (is_operators or is_attention or is_attention_history or is_hc or is_moe) &.{} else try std.Io.Dir.cwd().readFileAlloc(io, args[4], arena, .limited(max_ids * 12 + 2));
    const ids: []const u32 = if (is_operators or is_attention or is_attention_history or is_hc or is_moe) &.{} else try std.json.parseFromSliceLeaky([]const u32, arena, input, .{});
    if (!is_operators and !is_attention and !is_attention_history and !is_hc and !is_moe and (ids.len == 0 or ids.len > 1048576)) return error.InvalidPrompt;
    if (ids.len > max_ids) return error.InvalidPrompt;
    for (ids) |id| if (id >= 129280) return error.InvalidToken;
    const is_oracle = std.mem.eql(u8, mode, "oracle");
    const is_trace = std.mem.eql(u8, mode, "trace");
    const dir = args[if (is_operators) @as(usize, 4) else 5];
    try std.Io.Dir.cwd().createDirPath(io, dir);
    const map = try std.fs.path.join(arena, &.{ args[2], "engram-token-map.bin" });
    const map_blob = if (is_oracle or is_trace)
        try std.Io.Dir.cwd().readFileAlloc(io, map, arena, .limited(2 << 20))
    else
        &.{};
    const model = try a.create(ds.Model);
    defer a.destroy(model);
    const begin = mtl.clock.seconds();
    model.* = try ds.Model.open(a, io, .{
        .dir = args[2],
        .table_dir = args[3],
        .token_map = map,
        .lanes = if (is_lanes) 2 else 1,
        // Two full-capacity caches sit at the machine's wired edge; the receipt needs only the prompt's window.
        .context = if (is_lanes and args.len > 7) try number(args[7]) else 0,
        .engram = if (streamEngram()) .streamed else .resident,
        .bound_only = streamEngram(),
    });
    defer model.deinit();
    var tp_transport: ?ds.tp_transport.Transport = null;
    defer if (tp_transport) |*transport| transport.deinit();
    var tp_context: ?ds.tp_moe.Context = null;
    defer if (tp_context) |*context| context.deinit();
    if (std.c.getenv("TF_DSV41_TP_SETTINGS")) |path| {
        const settings_json = try std.Io.Dir.cwd().readFileAlloc(io, std.mem.span(path), arena, .limited(65536));
        tp_transport = try ds.tp_transport.Transport.init(a, settings_json, model.workspace.plan.rows);
        const transport = &tp_transport.?;
        tp_context = try ds.tp_moe.Context.init(a, model.device, &model.weights, @intCast(transport.channel.myRank()), model.workspace.plan.rows, transport.exchange());
        for (tp_context.?.descriptors) |layer| for (layer) |buffer| model.residency.add(buffer);
        model.residency.commit();
        model.residency.requestResidency();
        model.tp = &tp_context.?;
    }
    const ready = mtl.clock.seconds();
    const before = try counters();
    const engram_before = model.engramStats();
    var execution_after: ?Counters = null;
    var execution_error: ?anyerror = null;
    execute(a, io, model, args, ids, map_blob, dir, before, &execution_after) catch |err| {
        execution_error = err;
    };
    const after = execution_after orelse try counters();
    if (after.disk_read < before.disk_read or after.pageins < before.pageins) return error.NonmonotonicCounters;
    const engram_after = model.engramStats();
    const engram_read_bytes = engram_after.read_bytes - engram_before.read_bytes;
    const streamed = model.engram_sources != null;
    // Streamed I/O may read only counted F_NOCACHE row bytes; page-fault reads still fail the gate.
    const resident_gate_passed = if (streamed)
        after.disk_read - before.disk_read <= engram_read_bytes and after.pageins == before.pageins
    else
        after.disk_read == before.disk_read;
    const receipt = try std.json.Stringify.valueAlloc(a, .{
        .mode = mode,
        .tensor_parallel_ranks = if (model.tp != null) @as(u32, 2) else 1,
        .tensor_parallel_rank = if (model.tp) |context| @as(u32, context.rank) else 0,
        .tensor_parallel_partition = if (model.tp != null) "moe_output_columns_replicated_weights" else "none",
        .attention_parallel = model.pipelines.attention.sparse.parallel_enabled,
        .tensor_parallel_exchanges = if (tp_transport) |transport| transport.channel.step else 0,
        .evidence_origin = "current_invocation_not_historical_qualification",
        .context_capacity = model.cache.plan.capacity,
        .consumed_tokens = model.cache.position,
        .weight_bytes = model.checkpoint.residentBytes(),
        .resident_allocation_bytes = model.residency.allocatedSize() + model.weight_residency.bytes(),
        .weight_residency_sets = model.weight_residency.count,
        .residency_refresh_ticks = if (model.residency_refresh) |refresh| refresh.ticks.load(.acquire) else 0,
        .ready_footprint_bytes = before.footprint,
        .final_footprint_bytes = after.footprint,
        .load_seconds = ready - begin,
        .measurement_scope = if (execution_after != null) "model_execution" else "execution_and_diagnostic_io",
        .execution_disk_read_bytes = after.disk_read - before.disk_read,
        .execution_pageins = after.pageins - before.pageins,
        .engram = if (streamed) "streamed" else "resident",
        .engram_read_bytes = engram_read_bytes,
        .engram_rows_requested = engram_after.requested - engram_before.requested,
        .engram_rows_unique = engram_after.unique - engram_before.unique,
        .engram_cache_hits = engram_after.hits - engram_before.hits,
        .engram_reads = engram_after.reads - engram_before.reads,
        .engram_wait_seconds = @as(f64, @floatFromInt(engram_after.wait_ns - engram_before.wait_ns)) / 1e9,
        .resident_gate_passed = resident_gate_passed,
        .execution_error = if (execution_error) |err| @as(?[]const u8, @errorName(err)) else null,
    }, .{});
    defer a.free(receipt);
    std.debug.print("{s}\n", .{receipt});
    try write(io, a, dir, "residency.json", receipt);
    if (!resident_gate_passed) std.debug.print("residency gate failed: ResidentExecutionReadFromDisk\n", .{});
    if (execution_error) |err| return err;
    if (!resident_gate_passed) return error.ResidentExecutionReadFromDisk;
}

fn execute(a: std.mem.Allocator, io: std.Io, model: *ds.Model, args: []const [:0]const u8, ids: []const u32, map_blob: []const u8, dir: []const u8, before: Counters, execution_after: *?Counters) !void {
    const mode = args[1];
    const is_operators = std.mem.eql(u8, mode, "operators");
    const is_attention = std.mem.eql(u8, mode, "attention-sameinput");
    const is_attention_history = std.mem.eql(u8, mode, "attention-history-sameinput");
    const is_hc = std.mem.eql(u8, mode, "hc-sameinput");
    const is_moe = std.mem.eql(u8, mode, "moe-sameinput");
    const is_oracle = std.mem.eql(u8, mode, "oracle");
    const is_profile = std.mem.eql(u8, mode, "profile");
    const is_trace = std.mem.eql(u8, mode, "trace");
    const is_teacher = std.mem.eql(u8, mode, "teacher");
    const is_seams = std.mem.eql(u8, mode, "seams");
    const is_lanes = std.mem.eql(u8, mode, "lanes");
    if (is_operators) {
        try operators.run(a, io, model, dir);
    } else if (is_attention) {
        try attention_operators.runAttention(a, io, model, args[4], dir, if (args.len > 6) try number(args[6]) else 0);
    } else if (is_attention_history) {
        try history_operators.runAttentionHistory(a, io, model, args[4], dir, try number(args[6]), try number(args[7]));
    } else if (is_hc) {
        try hc_operators.runHyperconnections(a, io, model, args[4], dir, if (args.len > 6) try number(args[6]) else 0);
    } else if (is_moe) {
        try moe_operators.runMoe(a, io, model, args[4], dir, if (args.len > 6) try number(args[6]) else 21);
    } else if (is_oracle) {
        try operators.run(a, io, model, dir);
        _ = try oracle.run(a, io, model, ids, map_blob, dir);
    } else if (std.mem.eql(u8, mode, "margins")) {
        try margins.run(a, io, model, ids, dir, if (args.len > 6) try number(args[6]) else 1536);
    } else if (is_profile) {
        try profile.run(a, io, model, ids, dir);
    } else if (is_lanes) {
        // Two lane slots, interleaved: each slot's rows and logits equal its own solo run's bits.
        if (ids.len < 4) return error.InvalidPrompt;
        const split = ids.len / 2;
        const chunk = if (args.len > 6) try number(args[6]) else 1;
        if (chunk == 0 or chunk > model.workspace.plan.rows) return error.InvalidPrefillChunk;
        const solo_a = try a.alloc(f32, split * 129280);
        defer a.free(solo_a);
        const solo_b = try a.alloc(f32, (ids.len - split) * 129280);
        defer a.free(solo_b);
        var elapsed: f64 = 0;
        elapsed += try teacherOn(model, 0, ids[0..split], chunk, solo_a);
        elapsed += try teacherOn(model, 1, ids[split..], chunk, solo_b);
        const position_a: u32 = model.caches[0].position;
        const position_b: u32 = model.caches[1].position;
        try model.reset();
        const mixed_a = try a.alloc(f32, solo_a.len);
        defer a.free(mixed_a);
        const mixed_b = try a.alloc(f32, solo_b.len);
        defer a.free(mixed_b);
        var at_a: usize = 0;
        var at_b: usize = 0;
        while (at_a < split or at_b < ids.len - split) {
            if (at_a < split) {
                const end = @min(at_a + chunk, split);
                const row = try model.forwardOn(0, ids[at_a..end], .all);
                @memcpy(mixed_a[at_a * 129280 .. end * 129280], row);
                at_a = end;
            }
            if (at_b < ids.len - split) {
                const end = @min(at_b + chunk, ids.len - split);
                const row = try model.forwardOn(1, ids[split + at_b .. split + end], .all);
                @memcpy(mixed_b[at_b * 129280 .. end * 129280], row);
                at_b = end;
            }
        }
        var error_a: f64 = 0;
        var error_b: f64 = 0;
        var bit_disagreements: usize = 0;
        for (solo_a, mixed_a) |x, y| {
            error_a = @max(error_a, @abs(@as(f64, x) - y));
            bit_disagreements += @intFromBool(@as(u32, @bitCast(x)) != @as(u32, @bitCast(y)));
        }
        for (solo_b, mixed_b) |x, y| {
            error_b = @max(error_b, @abs(@as(f64, x) - y));
            bit_disagreements += @intFromBool(@as(u32, @bitCast(x)) != @as(u32, @bitCast(y)));
        }
        execution_after.* = try counters();
        std.debug.print("{{\"mode\":\"lanes\",\"slots\":2,\"chunk\":{d},\"max_logit_error_slot0\":{d},\"max_logit_error_slot1\":{d},\"bit_disagreements\":{d},\"position_slot0\":{d},\"position_slot1\":{d},\"gpu_seconds\":{d:.3}}}\n", .{ chunk, error_a, error_b, bit_disagreements, position_a, position_b, elapsed });
        if (bit_disagreements != 0) return error.LaneInterference;
    } else if (is_trace) {
        try trace.run(a, io, model, ids, map_blob, dir);
    } else if (is_teacher or is_seams) {
        const chunk = if (args.len > 6) try number(args[6]) else model.workspace.plan.rows;
        if (chunk == 0 or chunk > model.workspace.plan.rows) return error.InvalidPrefillChunk;
        const outputs = try a.alloc(f32, ids.len * 129280);
        defer a.free(outputs);
        const teacher_start = mtl.clock.seconds();
        const elapsed = try teacher(model, ids, chunk, outputs);
        const wall_seconds = mtl.clock.seconds() - teacher_start;
        // Capture execution before writing the receipt, not filesystem metadata reads from the receipt itself.
        if (is_seams) {
            const alternative = try a.alloc(f32, outputs.len);
            defer a.free(alternative);
            for ([_]u32{ 1, 2, 3, 7, 31, 127, 128, 129, 255, 256 }) |split| {
                if (split > model.workspace.plan.rows) continue;
                _ = try teacher(model, ids, split, alternative);
                var error_max: f64 = 0;
                var disagreements: usize = 0;
                var bit_disagreements: usize = 0;
                for (outputs, alternative) |x, y| {
                    error_max = @max(error_max, @abs(@as(f64, x) - y));
                    bit_disagreements += @intFromBool(@as(u32, @bitCast(x)) != @as(u32, @bitCast(y)));
                }
                for (0..ids.len) |i| {
                    if (try argmax(outputs[i * 129280 ..][0..129280]) != try argmax(alternative[i * 129280 ..][0..129280]))
                        disagreements += 1;
                }
                std.debug.print("{{\"mode\":\"seam\",\"chunk\":{d},\"max_logit_error\":{d},\"bit_disagreements\":{d},\"greedy_disagreements\":{d}}}\n", .{ split, error_max, bit_disagreements, disagreements });
                if (disagreements != 0 or bit_disagreements != 0) {
                    var name: [48]u8 = undefined;
                    const file = try std.fmt.bufPrint(&name, "seam-{d}-logits.f32", .{split});
                    execution_after.* = try counters();
                    try write(io, a, dir, file, std.mem.sliceAsBytes(alternative));
                    return error.ChunkSeamMismatch;
                }
            }
        }
        std.debug.print("{{\"mode\":\"teacher\",\"tokens\":{d},\"elapsed_seconds\":{d},\"gpu_seconds\":{d}}}\n", .{ ids.len, wall_seconds, elapsed });
        execution_after.* = try counters();
        try write(io, a, dir, "logits.f32", std.mem.sliceAsBytes(outputs));
    } else if (std.mem.eql(u8, mode, "generate")) {
        const n = if (args.len > 6) try number(args[6]) else 64;
        if (ids.len > model.cache.plan.capacity or n == 0 or n > model.cache.plan.capacity - ids.len)
            return error.ContextFull;
        try model.reset();
        const prefill_start = mtl.clock.seconds();
        var row: []const f32 = &.{};
        var start: usize = 0;
        var progress_at = prefill_start;
        while (start < ids.len) {
            const end = @min(start + model.workspace.plan.rows, ids.len);
            row = try model.forward(ids[start..end], if (end == ids.len) .last else .none);
            start = end;
            const now = mtl.clock.seconds();
            if (now - progress_at >= 60 or start == ids.len) {
                const memory = try counters();
                std.debug.print("{{\"mode\":\"prefill_progress\",\"completed_tokens\":{d},\"prompt_tokens\":{d},\"elapsed_seconds\":{d},\"footprint_bytes\":{d},\"disk_read_bytes\":{d},\"pageins\":{d}}}\n", .{ start, ids.len, now - prefill_start, memory.footprint, memory.disk_read - before.disk_read, memory.pageins - before.pageins });
                progress_at = now;
            }
        }
        const prefill_end = mtl.clock.seconds();
        const generated = try a.alloc(u32, n);
        defer a.free(generated);
        var gpu_seconds: f64 = 0;
        const decode_start = mtl.clock.seconds();
        for (generated, 0..) |*token, i| {
            token.* = try argmax(row);
            if (i + 1 < generated.len) {
                row = try model.forward(&.{token.*}, .last);
                gpu_seconds += model.last_gpu_seconds;
            }
        }
        const decode_end = mtl.clock.seconds();
        execution_after.* = try counters();
        const json = try std.json.Stringify.valueAlloc(a, generated, .{});
        defer a.free(json);
        try write(io, a, dir, "tokens.json", json);
        std.debug.print("{{\"mode\":\"generate\",\"prompt_tokens\":{d},\"completion_tokens\":{d},\"prefill_seconds\":{d},\"decode_seconds\":{d},\"decode_gpu_seconds\":{d}}}\n", .{ ids.len, n, prefill_end - prefill_start, decode_end - decode_start, gpu_seconds });
    } else usage(2);
}
