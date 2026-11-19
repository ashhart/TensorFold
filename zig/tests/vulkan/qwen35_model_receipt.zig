// The Qwen3.5 probe's receipt: the result.json shape, its JSON writer and the execution-mode labels.
const std = @import("std");

// Receipt execution_mode labels: free-running greedy (default) and teacher-forced per-step; v1.1 keys on these strings.
pub const free_running_mode = "free-running-greedy";
pub const teacher_forced_mode = "teacher-forced";

pub const RepeatChecks = struct { runs: u64 = 2, generated_ids_equal: bool, logits_bytes_equal: bool };
pub const OracleMatch = struct { ids_run_a: bool, ids_run_b: bool };
pub const OracleEcho = struct {
    generated_ids_hash: []const u8,
    logits_bf16_hash: []const u8,
    logits_bytes: u64,
    checkpoint_note: []const u8,
};

pub const Receipt = struct {
    status: []const u8,
    backend: []const u8 = "vulkan-qwen35-first-core",
    steps: usize,
    execution_mode: []const u8,
    prefill_schedule: []const u8,
    context_cap: usize,
    prompt_chunks: usize,
    prefill_chunk_rows: usize,
    capacity: usize,
    model_context: usize,
    prompt_tokens: usize,
    input_npy_sha256: []const u8,
    prompt_u32le_sha256: []const u8,
    generated_ids: []const u32,
    generated_ids_hash: []const u8,
    logits_bytes: usize,
    logits_bf16_sha256: []const u8,
    repeat_checks: RepeatChecks,
    oracle_match: OracleMatch,
    oracle: OracleEcho,
    oracle_logits_element_mismatches: ?u64,
    elapsed_ms: i64,
};

// The receipt's JSON via the house stringify: strings escaped by std.json, no manual interpolation; pure, host-tested.
pub fn buildResult(a: std.mem.Allocator, r: Receipt) error{OutOfMemory}![]u8 {
    return std.json.Stringify.valueAlloc(a, r, .{});
}

pub fn writeResult(a: std.mem.Allocator, io: std.Io, out: std.Io.Dir, r: Receipt) !void {
    const j = try buildResult(a, r);
    defer a.free(j);
    try out.writeFile(io, .{ .sub_path = "result.json", .data = j });
}

const testing = std.testing;

test "buildResult escapes oracle strings and round-trips through the JSON parser" {
    const r: Receipt = .{
        .status = "mismatch",
        .steps = 2,
        .execution_mode = teacher_forced_mode,
        .prefill_schedule = "rowwise",
        .context_cap = 512,
        .prompt_chunks = 1,
        .prefill_chunk_rows = 32,
        .capacity = 256,
        .model_context = 4096,
        .prompt_tokens = 3,
        .input_npy_sha256 = &@as([64]u8, @splat('a')),
        .prompt_u32le_sha256 = &@as([64]u8, @splat('b')),
        .generated_ids = &.{ 262, 413 },
        .generated_ids_hash = &@as([64]u8, @splat('c')),
        .logits_bytes = 993280,
        .logits_bf16_sha256 = &@as([64]u8, @splat('d')),
        .repeat_checks = .{ .generated_ids_equal = true, .logits_bytes_equal = false },
        .oracle_match = .{ .ids_run_a = true, .ids_run_b = false },
        .oracle = .{
            .generated_ids_hash = &@as([64]u8, @splat('e')),
            .logits_bf16_hash = &@as([64]u8, @splat('f')),
            .logits_bytes = 993280,
            .checkpoint_note = "rev \"93760be4\" \\ mlx\nsecond line",
        },
        .oracle_logits_element_mismatches = null,
        .elapsed_ms = 12,
    };
    const j = try buildResult(std.testing.allocator, r);
    defer std.testing.allocator.free(j);
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, j, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    // the escaped note survives a parse round-trip byte for byte
    try testing.expectEqualStrings(r.oracle.checkpoint_note, root.get("oracle").?.object.get("checkpoint_note").?.string);
    // no raw control characters or unescaped quotes leaked into the bytes
    try testing.expect(std.mem.indexOf(u8, j, "rev \"937") == null);
    try testing.expectEqualStrings(&@as([64]u8, @splat('c')), root.get("generated_ids_hash").?.string);
    try testing.expectEqualStrings("teacher-forced", root.get("execution_mode").?.string);
    try testing.expectEqualStrings("rowwise", root.get("prefill_schedule").?.string);
    try testing.expect(root.get("repeat_checks").?.object.get("generated_ids_equal").? == .bool);
    try testing.expect(root.get("oracle_logits_element_mismatches").? == .null);
}
