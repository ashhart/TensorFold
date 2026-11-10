//! Developer probe: MODEL PROMPTS.json OUTPUT.json [BASELINE.json], with raw token IDs and 16 fixed-history positions.
const std = @import("std");
const cuda = @import("cuda");
const qwen = @import("qwen35");
const data = @import("qwen_prefill_fidelity_data.zig");
const Engine = qwen.Engine;

const Position = struct { index: usize, history_tokens: usize, history_sha256: []const u8, logits: data.Summary };
const CaseResult = struct {
    id: []const u8,
    prompt_tokens: usize,
    prompt_sha256: []const u8,
    continuation: []u32,
    positions: []Position,
    top1_mismatches: usize,
};
const Policy = struct {
    kv8: bool,
    chunk_rows: usize,
    native_nvfp4: bool,
    native_tma: bool,
    fast_prefill: bool,
    affine_prepared: bool,
    compiled_max_rows: usize,
    prefill_graph: bool,
    expanded_kv: bool,
    alternate_scope: []const u8,
    alternate_source: []const u8,
    cublas_prefill: bool,
    gdn_chunked: bool,
    fp8_attention: []const u8,
    exact_attention: []const u8,
};
const Report = struct { version: u32 = 1, mode: []const u8, model: []const u8, steps: usize, policy: Policy, cases: []CaseResult };

fn evaluate(a: std.mem.Allocator, e: *Engine, fixture: data.Fixture, case: data.Case, forced: ?[]const u32) !CaseResult {
    const capacity = @max(e.window_rows, case.tokens.len + fixture.steps);
    if (capacity > e.g.context) return error.FidelityContextTooLong;
    var cache = try qwen.cache.Cache.init(a, e, capacity);
    defer {
        cuda.completion.drain(e.d, e.stream, "fidelity cache release");
        cache.deinit(a);
    }
    var at: usize = 0;
    while (at < case.tokens.len) {
        const end = @min(case.tokens.len, at + fixture.chunk_rows);
        try e.forward(&.{.{ .cache = &cache, .rows = end - at }}, case.tokens[at..end], .last);
        at = end;
    }
    const continuation = try a.alloc(u32, fixture.steps);
    const positions = try a.alloc(Position, fixture.steps);
    var mismatches: usize = 0;
    for (positions, 0..) |*position, index| {
        const logits = if (index == 0) cache.logits else e.rowLogits(0);
        const summary = try data.summarize(logits, if (forced) |ids| ids[index] else null);
        continuation[index] = summary.chosen;
        mismatches += @intFromBool(summary.top[0].token != summary.chosen);
        position.* = .{ .index = index, .history_tokens = case.tokens.len + index, .history_sha256 = try data.digest(a, case.tokens, continuation[0..index]), .logits = summary };
        if (index + 1 < positions.len) {
            try e.forward(&.{.{ .cache = &cache, .rows = 1 }}, continuation[index..][0..1], .logits);
            try cache.keep(e, &.{0});
        }
    }
    return .{ .id = case.id, .prompt_tokens = case.tokens.len, .prompt_sha256 = try data.digest(a, case.tokens, &.{}), .continuation = continuation, .positions = positions, .top1_mismatches = mismatches };
}

pub fn main(init: std.process.Init) !void {
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 4 and args.len != 5) {
        std.debug.print("usage: tf-qwen-prefill-fidelity MODEL PROMPTS.json OUTPUT.json [BASELINE.json]\n", .{});
        return error.Usage;
    }
    const source = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], a, .limited(16 << 20));
    const fixture = try data.parse(a, source);
    if (fixture.chunk_rows > qwen.engine.max_rows) return error.FidelityChunkExceedsBuild;
    const fixed = if (args.len == 5) try data.continuations(a, try std.Io.Dir.cwd().readFileAlloc(init.io, args[4], a, .limited(32 << 20)), fixture) else null;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var context = try cuda.Context.init(&driver, 0);
    defer context.deinit();
    const e = try Engine.init(init.gpa, init.io, &context, args[1], fixture.kv8, null);
    defer {
        cuda.completion.drain(e.d, e.stream, "fidelity engine release");
        e.deinit();
    }
    for (fixture.cases, 0..) |case, index| {
        for (case.tokens) |token| if (token >= e.g.vocab) return error.InvalidTokenFixture;
        if (fixed) |sequences| for (sequences[index]) |token| {
            if (token >= e.g.vocab) return error.InvalidContinuationFixture;
        };
    }
    const results = try a.alloc(CaseResult, fixture.cases.len);
    for (fixture.cases, results, 0..) |case, *out, index| {
        out.* = try evaluate(a, e, fixture, case, if (fixed) |sequences| sequences[index] else null);
        std.debug.print("fidelity {s}: {d} fixed positions, {d} top-token differences\n", .{ case.id, fixture.steps, out.top1_mismatches });
    }
    const policy: Policy = .{ .kv8 = fixture.kv8, .chunk_rows = fixture.chunk_rows, .native_nvfp4 = if (e.mo) |m| m.native_requested else false, .native_tma = if (e.mo) |m| if (m.native) |n| n.tma != null else false else false, .fast_prefill = if (e.mo) |m| m.fast8 != null else false, .affine_prepared = if (e.mo) |m| if (m.fast8) |n| n.affine_prepared != null else false else false, .compiled_max_rows = qwen.engine.max_rows, .prefill_graph = try @import("sys").environment.read(bool, "TF_QWEN_PREFILL_GRAPH", false), .expanded_kv = e.prefill_paths.expand != null, .alternate_scope = if (e.alternate) |alt| @tagName(alt.scope) else "none", .alternate_source = if (e.alternate) |alt| alt.dir else "", .cublas_prefill = if (e.mo) |m| m.vendor != null else false, .gdn_chunked = e.prefill_paths.config.gdn, .fp8_attention = @tagName(e.prefill_paths.config.attention), .exact_attention = @tagName(e.prefill_paths.config.exact) };
    const report: Report = .{ .mode = if (fixed != null) "replay" else "greedy-baseline", .model = args[1], .steps = fixture.steps, .policy = policy, .cases = results };
    const json = try std.json.Stringify.valueAlloc(a, report, .{ .whitespace = .indent_2 });
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[3], .data = json });
}
