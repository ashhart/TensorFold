//! Inkling 2 Mini on the host: config.json, loop visits and slots, weight map vs published headers, refusals.
const std = @import("std");
const config = @import("config.zig");
const names = @import("names.zig");
const plan = @import("plan.zig");
const st = @import("safetensors");
const unfold = @import("names_test.zig").unfold;

const t = std.testing;
pub const mini = @embedFile("fixtures/mini_config.json");
const model_fixture = @embedFile("fixtures/mini_model_header.txt");
const mtp_fixture = @embedFile("fixtures/mini_mtp_header.txt");

fn parse(a: std.mem.Allocator) !config.Config {
    var why: config.Why = .{};
    return config.parse(a, mini, &why);
}

fn refused(from: []const u8, to: []const u8, want: anyerror, says: []const u8) !void {
    if (std.mem.count(u8, mini, from) != 1) return error.EditAnchor;
    const edited = try std.mem.replaceOwned(u8, t.allocator, mini, from, to);
    defer t.allocator.free(edited);
    var why: config.Why = .{};
    try t.expectError(want, config.parse(t.allocator, edited, &why));
    if (std.mem.indexOf(u8, why.text(), says) == null) {
        std.debug.print("refusal \"{s}\" does not name \"{s}\"\n", .{ why.text(), says });
        return error.TestUnexpectedResult;
    }
}

test "Mini's config.json parses: 24 layers, a core of 6..17 three times (48 visits), rank 64, taps by visit" {
    const c = try parse(t.allocator);
    try t.expectEqual(@as(u32, 2304), c.text.hidden);
    try t.expectEqual(@as(u32, 24), c.text.layers);
    try t.expectEqual(config.Heads{ .q = 24, .kv = 8, .dim = 128 }, c.text.global);
    try t.expectEqual(config.Heads{ .q = 24, .kv = 8, .dim = 128 }, c.text.local);
    try t.expectEqual(@as(u32, 128), c.text.experts);
    try t.expectEqual(@as(u32, 1152), c.text.expert_inter);
    try t.expectEqual(@as(u32, 8192), c.text.log_floor);
    try t.expectEqual(config.Loop{ .start = 6, .end = 18, .passes = 3, .rank = 64 }, c.text.loop.?);
    try t.expectEqual(@as(u32, 48), c.text.visits());
    const m = c.mtp.?;
    try t.expectEqual(@as(u32, 3), m.layers);
    try t.expectEqualSlices(u32, &.{ 5, 23, 41, 48 }, m.taps[0..m.taps_n]);
}

test "Mini's visits: prelude 0..5, the core three passes, coda 18..23; 40 sliding, 8 full and 48 conv slots" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try parse(a);
    var prec: [24]plan.Precision = undefined;
    try names.precisions(try unfold(a, model_fixture, "inkling2-mini-model"), 24, &prec);
    const p = try plan.build(&c.text, &prec);
    const v = &p.visits;
    try t.expectEqual(@as(u32, 48), v.len);
    try t.expectEqual(@as(u32, 40), v.count(.sliding));
    try t.expectEqual(@as(u32, 8), v.count(.full));
    try t.expectEqual(@as(u32, 48), v.count(.conv));
    try t.expectEqual(@as(u32, 1), p.pre_blocks);
    try t.expectEqual(@as(u32, 1), p.bank_blocks);
    for (v.slice(), 0..) |s, k| {
        const want: u32 = if (k < 6) @intCast(k) else if (k < 42) 6 + @as(u32, @intCast(k - 6)) % 12 else @intCast(k - 24);
        try t.expectEqual(want, s.layer);
        try t.expectEqual(@as(u8, if (k < 6 or k >= 42) 0 else @intCast((k - 6) / 12)), s.pass);
        try t.expectEqual(k >= 18 and k < 42, p.messages(@intCast(k)));
        try t.expectEqual(k == 18 or k == 30, p.restarts(@intCast(k)));
        try t.expectEqual(@as(?u16, @intCast(k)), s.of(.conv));
    }
    // the taps' visits: 5 is L5 pass 0, 23 is L11 pass 1, 41 is L17 pass 2 (global slots 0, 3 and 6); 48 the output
    for ([_]u32{ 5, 23, 41 }, [_]u32{ 5, 11, 17 }, [_]u16{ 0, 3, 6 }) |tap, layer, full| {
        const s = v.steps[(try v.tap(tap)).input];
        try t.expectEqual(layer, s.layer);
        try t.expectEqual(@as(?u16, full), s.of(.full));
    }
    try t.expectEqual(@as(@TypeOf(try v.tap(48)), .output), try v.tap(48));
    try t.expectEqual(@as(u32, 5), p.coreIndex(23));
    try t.expectEqual(@as(u32, 0), p.coreIndex(3));
}

test "Mini's folded headers unfold to its 1244 and 54 tensors, digests equal" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    try t.expectEqual(@as(usize, 1244), (try unfold(arena.allocator(), model_fixture, "inkling2-mini-model")).count());
    try t.expectEqual(@as(usize, 54), (try unfold(arena.allocator(), mtp_fixture, "inkling2-mini-mtp")).count());
}

test "the map names exactly Mini's tensors: controllers, cross-pass queries, NVFP4 L0-19 and MXFP8 L20-23" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try parse(a);
    const entries = try unfold(a, model_fixture, "inkling2-mini-model");
    const r = try names.check(a, &c, entries, .model);
    try t.expectEqual(@as(usize, 1244), r.expected);
    try t.expect(r.clean());
    var prec: [24]plan.Precision = undefined;
    try names.precisions(entries, 24, &prec);
    for (prec, 0..) |p, i| try t.expectEqual(if (i < 20) plan.Precision.nvfp4 else plan.Precision.mxfp8, p);
    const mtp = try names.check(a, &c, try unfold(a, mtp_fixture, "inkling2-mini-mtp"), .mtp);
    try t.expectEqual(@as(usize, 54), mtp.expected);
    try t.expect(mtp.clean());
}

test "a missing controller and a wrong cross-pass shape are reported by name" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try parse(a);
    const all = try unfold(a, model_fixture, "inkling2-mini-model");
    var kept: st.Header = .empty;
    for (all.keys(), all.values()) |name, e| {
        var x = e;
        if (std.mem.eql(u8, name, "model.llm.layers.9.controllers.v.gain_weight")) continue;
        if (std.mem.eql(u8, name, "model.llm.middle_cycle_cross_pass_kv_queries")) x.shape[0] = 12;
        try kept.put(a, name, x);
    }
    const r = try names.check(a, &c, kept, .model);
    try t.expectEqual(@as(usize, 1), r.missing.items.len);
    try t.expectEqualStrings("model.llm.layers.9.controllers.v.gain_weight", r.missing.items[0]);
    try t.expectEqual(@as(usize, 1), r.wrong.items.len);
    try t.expectEqualStrings("model.llm.middle_cycle_cross_pass_kv_queries", r.wrong.items[0]);
}

test "the loop parse refuses is_varloop, a core off the blocks, one pass, a wide rank and a loop on Small by name" {
    try refused("\"is_varloop\": false", "\"is_varloop\": true", error.Unsupported, "is_varloop must be false");
    try refused("\"core_end\": 18", "\"core_end\": 17", error.Layout, "not whole blocks of 6");
    try refused("\"core_end\": 18", "\"core_end\": 30", error.Layout, "not whole blocks of 6");
    try refused("\"num_passes\": 3", "\"num_passes\": 1", error.Layout, "at least 2");
    try refused("\"gate_rank\": 64", "\"gate_rank\": 129", error.Layout, "gate_rank 129 at most 128");
    try refused("\"num_passes\": 3", "\"num_passes\": 40", error.Layout, "the plan tables hold 256");
    try refused("\"loop_config\"", "\"loop_cfg\"", error.BadConfig, "no loop_config");
    try refused("\"attn_res_normalize_values\": true", "\"attn_res_normalize_values\": false", error.Unsupported, "attn_res_normalize_values");
    try refused("\"model_type\": \"inkling2_loop\"", "\"model_type\": \"inkling2\"", error.NotInkling2, "inkling2_loop");
    try refused("23,\n      41,", "41,\n      23,", error.Layout, "must rise");
    try refused("23,\n      41,", "23,\n      49,", error.Layout, "a layer the backbone does not have");
}

test "Small with a loop_config is refused: the loop needs model_type inkling2_loop" {
    const tiny = @import("config_test.zig").tiny;
    const from = "\"n_layers_per_attn_res_block\": 6,";
    const to = from ++ " \"loop_config\": {\"core_start\": 6, \"core_end\": 12, \"num_passes\": 2, \"gate_rank\": 8},";
    const edited = try std.mem.replaceOwned(u8, t.allocator, tiny, from, to);
    defer t.allocator.free(edited);
    var why: config.Why = .{};
    try t.expectError(error.Unsupported, config.parse(t.allocator, edited, &why));
    try t.expectEqualStrings("loop_config needs model_type inkling2_loop", why.text());
}

test "Mini's quantization_config goes through core.modelopt: NVFP4 without config_groups, FP8 refused" {
    try refused("\"quant_algo\": \"NVFP4\"", "\"quant_algo\": \"FP8\"", error.Quant, "ModelOpt FP8");
    try refused("\"quant_method\": \"modelopt\"", "\"quant_method\": \"awq\"", error.Quant, "no ModelOpt quantization_config");
}

/// A file's header read from disk (the shard holds data too; only the header is read).
fn headerOf(a: std.mem.Allocator, dir: []const u8, file: []const u8) !st.Header {
    const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, file });
    const f = try std.Io.Dir.cwd().openFile(t.io, path, .{});
    defer f.close(t.io);
    var prefix: [8]u8 = undefined;
    if (try f.readPositionalAll(t.io, &prefix, 0) != 8) return error.BadSafetensors;
    const size = std.math.cast(usize, try f.length(t.io)) orelse return error.BadSafetensors;
    const json = try a.alloc(u8, try st.headerSize(std.mem.readInt(u64, &prefix, .little), size));
    if (try f.readPositionalAll(t.io, json, 8) != json.len) return error.BadSafetensors;
    return st.parseHeader(a, json, std.math.maxInt(usize));
}

test "the released Mini's model and mtp headers and config match the map (TF_INKLING2_MINI=<dir>)" {
    const dir = std.mem.span(std.c.getenv("TF_INKLING2_MINI") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try parse(a);
    const model = try names.check(a, &c, try headerOf(a, dir, "model-00001-of-00001.safetensors"), .model);
    try t.expectEqual(@as(usize, 1244), model.expected);
    try t.expect(model.clean());
    const mtp = try names.check(a, &c, try headerOf(a, dir, "mtp.safetensors"), .mtp);
    try t.expect(mtp.clean());
}
