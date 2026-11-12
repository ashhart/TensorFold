//! The real startup entry rejects bad settings before allocation, assets or a CUDA driver.
const std = @import("std");
const sys = @import("sys");
const env = sys.env;
const v = @import("env_validate.zig");
const rules = sys.environment.rules;
const model = @import("cuda_model.zig");
const t = std.testing;

fn refused(name: [:0]const u8) !void {
    try t.expectEqual(@as(c_int, 0), env.setenv(name, "mistyped value", 1));
    defer _ = env.unsetenv(name);
    try t.expectError(error.InvalidEnvironment, model.Model.open(t.failing_allocator, t.io, "missing-checkpoint", "missing-assets", .{}));
    try t.expectError(error.InvalidEnvironment, @import("aot_needs.zig").Served.fromEnv());
}

fn blank(name: [:0]const u8) !void {
    try t.expectEqual(@as(c_int, 0), env.setenv(name, " ", 1));
    defer _ = env.unsetenv(name);
    try t.expectError(error.ExperimentalDeepSeekRequired, model.Model.open(t.failing_allocator, t.io, "missing-checkpoint", "missing-assets", .{}));
    _ = try @import("aot_needs.zig").Served.fromEnv();
}

test "DeepSeek strict env: every malformed setting stops the real loader before resources" {
    _ = env.unsetenv("TENSORFOLD_TOOL_GRAMMAR");
    for (rules) |r| if (r.scope == .dsv41) try refused(r.name);
}

test "DeepSeek strict env: a blank setting is unset, so the loader and planner pass the check" {
    _ = env.unsetenv("TENSORFOLD_TOOL_GRAMMAR");
    _ = env.unsetenv("TF_EXPERIMENTAL");
    for (rules) |r| if (r.scope == .dsv41) try blank(r.name);
}

test "DeepSeek strict env: empty served-knob reads retain their local defaults" {
    const knobs = @import("serve_knobs.zig");
    for ([_][:0]const u8{ "TF_DSV41_PF_DENSE", "TF_DSV41_PF_DENSE_CFG", "TF_DSV41_PF_DENSE_TABLE", "TF_DSV41_PF_4K", "TF_DSV41_PF_OVERLAP", "TF_DSV41_MHC_SITE_ROWS", "TF_DSV41_GM_GU2", "TF_DSV41_PF_OVERLAP_SITE", "TF_DSV41_PF_TBO", "TF_DSV41_PF_TBO_GM_SMS", "TF_DSV41_STREAM_RB", "TF_DSV41_INDEX_BOUND", "TF_DSV41_X3LD_EPI", "TF_DSV41_MHC_PFDEC", "TF_DSV41_INDEX_BUDGET_MIB", "TF_DSV41_PREFILL", "TF_DSV41_PROMPT_TAIL" }) |name| {
        _ = env.setenv(name, "", 1);
        defer _ = env.unsetenv(name);
        try t.expect(knobs.env(name) == null);
        if (std.mem.eql(u8, name, "TF_DSV41_PF_TBO")) try t.expect(!(try knobs.pfTbo(knobs.env)));
        if (std.mem.eql(u8, name, "TF_DSV41_PREFILL")) try t.expectEqual(knobs.PrefillMode.full, try knobs.prefillMode(knobs.env));
        if (std.mem.eql(u8, name, "TF_DSV41_PROMPT_TAIL")) try t.expectEqual(knobs.PromptTail.verify, try knobs.promptTail(knobs.env));
    }
}

test "DeepSeek strict env: actual flag readers keep defaults, zero, one and named refusals" {
    const Case = struct { name: [:0]const u8, default: bool, read: *const fn () anyerror!bool };
    const cases = [_]Case{
        .{ .name = "TF_DSV41_OWN_PREFILL", .default = true, .read = model.ownPrefillEnv },
        .{ .name = "TF_DSV41_PREFILL_PIECES", .default = false, .read = model.Model.piecesEnv },
        .{ .name = "TF_DSV41_SLOT_DRAFTS", .default = false, .read = @import("dspark_rows.zig").enabled },
        .{ .name = "TF_DSV41_SAMPLING", .default = false, .read = @import("sampling_gpu.zig").enabled },
        .{ .name = "TF_DSV41_FORMS_INPLACE", .default = false, .read = @import("forms_inplace.zig").enabled },
        .{ .name = "TF_DSV41_EX_STACKS", .default = false, .read = @import("ex_stacks.zig").enabled },
        .{ .name = "TF_DSV41_STREAM_LAZY", .default = false, .read = @import("longpf.zig").lazyEnv },
        .{ .name = "TF_DSV41_GRAPH_LOG", .default = false, .read = @import("batch_graph.zig").graphLog },
        .{ .name = "TF_DSV41_MHC_DEFER", .default = false, .read = @import("branches.zig").mhcDefer },
        .{ .name = "TF_DSV41_COEF_LATE", .default = false, .read = @import("branches.zig").coefLate },
        .{ .name = "TF_DSV41_ENGRAM_POLL", .default = false, .read = @import("engram_io.zig").pollEnv },
        .{ .name = "TF_DSV41_ENGRAM_AIO", .default = false, .read = @import("engram_io.zig").aioEnv },
        .{ .name = "TF_DSV41_KV_SPLIT_DEDUP", .default = false, .read = @import("kv").split.dedupOn },
        .{ .name = "TF_DSV41_KV_SPLIT_EARLY", .default = false, .read = @import("kv").split.earlyOn },
        .{ .name = "TF_DSV41_KV_SPLIT_HISTORY", .default = false, .read = @import("kv").split.historyOn },
    };
    for (cases) |c| {
        _ = env.unsetenv(c.name);
        defer _ = env.unsetenv(c.name);
        try t.expectEqual(c.default, try c.read());
        _ = env.setenv(c.name, "0", 1);
        try t.expect(!try c.read());
        _ = env.setenv(c.name, "1", 1);
        try t.expect(try c.read());
        _ = env.setenv(c.name, "\t", 1);
        try t.expectEqual(c.default, try c.read());
        _ = env.setenv(c.name, "garbage", 1);
        try t.expectError(error.InvalidEnvironment, c.read());
        _ = env.unsetenv(c.name);
    }
}

test "DeepSeek strict env: shared floor, graph room and reader counts retain their valid values" {
    const b = @import("kv").budget;
    const pf = @import("forward_prefill_k4.zig");
    for ([_][:0]const u8{ "TF_DSV41_FLOOR_HARD_GIB", "TF_DSV41_PF_WS_FLOOR_GIB", "TF_DSV41_GRAPH_COST_MIB", "TF_DSV41_ENGRAM_QD", "TF_DSV41_ENGRAM_READERS", "TF_DSV41_BIAS_VL" }) |name| _ = env.unsetenv(name);
    defer for ([_][:0]const u8{ "TF_DSV41_FLOOR_HARD_GIB", "TF_DSV41_PF_WS_FLOOR_GIB", "TF_DSV41_GRAPH_COST_MIB", "TF_DSV41_ENGRAM_QD", "TF_DSV41_ENGRAM_READERS", "TF_DSV41_BIAS_VL" }) |name| {
        _ = env.unsetenv(name);
    };
    try t.expectEqual(@as(f64, 4), try b.hardFloorGiB());
    try t.expectEqual(@as(f64, 4), try pf.wsFloorGiB());
    _ = env.setenv("TF_DSV41_FLOOR_HARD_GIB", "3.25", 1);
    try t.expectEqual(try b.hardFloorGiB(), try pf.hardFloorGiB());
    try t.expectEqual(@as(f64, 3.25), try pf.wsFloorGiB());
    _ = env.setenv("TF_DSV41_PF_WS_FLOOR_GIB", "6.5", 1);
    try t.expectEqual(@as(f64, 6.5), try pf.wsFloorGiB());
    try t.expectEqual(@as(f64, 64.0 / 1024.0), (try @import("graphs.zig").Room.fromEnv()).cost_gib);
    _ = env.setenv("TF_DSV41_GRAPH_COST_MIB", "128", 1);
    try t.expectEqual(@as(f64, 0.125), (try @import("graphs.zig").Room.fromEnv()).cost_gib);
    try t.expectEqual(@as(u32, 64), try @import("engram_io.zig").poolThreads());
    _ = env.setenv("TF_DSV41_ENGRAM_QD", "0", 1);
    try t.expectEqual(@as(u32, 0), try @import("engram_io.zig").poolThreads());
    try t.expectEqual(@as(u32, 16), try @import("engram_rows_darwin.zig").poolThreads());
    _ = env.setenv("TF_DSV41_ENGRAM_READERS", "32", 1);
    try t.expectEqual(@as(u32, 32), try @import("engram_rows_darwin.zig").poolThreads());
    _ = env.setenv("TF_DSV41_ENGRAM_READERS", "33", 1);
    try t.expectError(error.InvalidEnvironment, @import("engram_rows_darwin.zig").poolThreads());
    _ = env.setenv("TF_DSV41_FLOOR_HARD_GIB", "", 1);
    try t.expectEqual(@as(f64, 4), try b.hardFloorGiB());
    try t.expect(!try v.imageBiasEnv());
    for ([_][:0]const u8{ "0", "1", " 0 " }) |raw| {
        _ = env.setenv("TF_DSV41_BIAS_VL", raw, 1);
        try t.expect(try v.imageBiasEnv());
    }
    for ([_][:0]const u8{ "", " \t" }) |raw| {
        _ = env.setenv("TF_DSV41_BIAS_VL", raw, 1);
        try t.expect(!try v.imageBiasEnv());
    }
    _ = env.setenv("TF_DSV41_BIAS_VL", "on", 1);
    try t.expectError(error.InvalidEnvironment, v.imageBiasEnv());
}

test "DeepSeek strict env: default-on graphs and lookup, nucleus and layer values are unchanged" {
    const graphs = @import("graphs.zig");
    const lookup = @import("draft/lookup.zig");
    const pipe = @import("sampling_pipe.zig");
    for ([_][:0]const u8{ "TF_DSV41_GRAPHS", "TF_DSV41_LOOKUP", "TF_DSV41_LOOKUP_PLAN", "TF_DSV41_NUCLEUS", "TF_DSV41_LAYERS" }) |name| _ = env.unsetenv(name);
    defer for ([_][:0]const u8{ "TF_DSV41_GRAPHS", "TF_DSV41_LOOKUP", "TF_DSV41_LOOKUP_PLAN", "TF_DSV41_NUCLEUS", "TF_DSV41_LAYERS" }) |name| {
        _ = env.unsetenv(name);
    };
    try t.expect((try graphs.Settings.fromEnvOr(true)).on);
    try t.expect(!(try graphs.Settings.fromEnvOr(false)).on);
    try t.expect((try lookup.Settings.fromEnv()).lookup);
    try t.expect(!(try lookup.Settings.fromEnv()).plan);
    try t.expectEqual(@as(u32, 512), try pipe.nucleusEnv());
    try t.expect(try model.layersEnv(t.allocator) == null);
    _ = env.setenv("TF_DSV41_LAYERS", "2-4", 1);
    const layers = (try model.layersEnv(t.allocator)).?;
    defer t.allocator.free(layers);
    try t.expectEqualSlices(u32, &.{ 2, 3, 4 }, layers);
    _ = env.setenv("TF_DSV41_NUCLEUS", "0", 1);
    try t.expectEqual(@as(u32, 0), try pipe.nucleusEnv());
    _ = env.setenv("TF_DSV41_GRAPHS", "0", 1);
    try t.expect(!(try graphs.Settings.fromEnvOr(true)).on);
    for ([_][:0]const u8{ "false", "OFF", "no", "", " \t" }) |raw| {
        _ = env.setenv("TF_DSV41_LOOKUP", raw, 1);
        try t.expect(!(try lookup.Settings.fromEnv()).lookup);
    }
    for ([_][:0]const u8{ "TF_DSV41_GRAPHS", "TF_DSV41_LOOKUP_PLAN" }) |name| _ = env.setenv(name, "", 1);
    _ = env.setenv("TF_DSV41_LOOKUP", "yes", 1);
    try t.expect((try graphs.Settings.fromEnvOr(true)).on);
    try t.expect((try lookup.Settings.fromEnv()).lookup and !(try lookup.Settings.fromEnv()).plan);
}

test "DeepSeek strict env: context, slots, pool and layers read blank as unset and refuse bad text by name" {
    const names = [_][:0]const u8{ "TF_DSV41_CONTEXT", "TF_DSV41_SLOTS", "TF_DSV41_POOL_TOKENS", "TF_DSV41_LAYERS" };
    defer for (names) |name| {
        _ = env.unsetenv(name);
    };
    for (names) |name| _ = env.setenv(name, "", 1);
    try t.expect(try model.contextEnv() == null);
    try t.expectEqual(@as(u32, 1), try @import("slots.zig").countFromEnv());
    try t.expect(try @import("cuda_kv.zig").optionsFromEnv(4096, 2) == null);
    try t.expect(try model.layersEnv(t.allocator) == null);
    for (names, [_][:0]const u8{ "0", "17", "1x", "3" }) |name, raw| _ = env.setenv(name, raw, 1);
    try t.expectError(error.InvalidEnvironment, model.contextEnv());
    try t.expectError(error.InvalidEnvironment, @import("slots.zig").countFromEnv());
    try t.expectError(error.InvalidEnvironment, @import("cuda_kv.zig").optionsFromEnv(4096, 2));
    try t.expectError(error.InvalidEnvironment, model.layersEnv(t.allocator));
}

test "DeepSeek strict env: actual rows-cap reader accepts all existing buckets" {
    const batch = @import("batch.zig");
    const name = "TF_DSV41_ROWS_CAP";
    defer _ = env.unsetenv(name);
    _ = env.unsetenv(name);
    try t.expectEqual(@as(u32, 16), try batch.capFromEnv());
    for ([_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64 }) |rows| {
        var buf: [16]u8 = undefined;
        const raw = try std.fmt.bufPrint(&buf, "{d}", .{rows});
        buf[raw.len] = 0;
        _ = env.setenv(name, buf[0..raw.len :0], 1);
        try t.expectEqual(rows, try batch.capFromEnv());
    }
    for ([_][:0]const u8{ "0", "9", "63", "65", "bad" }) |raw| {
        _ = env.setenv(name, raw, 1);
        try t.expectError(error.InvalidEnvironment, batch.capFromEnv());
    }
    _ = env.setenv(name, " ", 1);
    try t.expectEqual(@as(u32, 16), try batch.capFromEnv());
}

test "DeepSeek strict env: pool boot target uses the same floor trim and named parser" {
    const budget = @import("kv").budget;
    const name = "TF_DSV41_FLOOR_GIB";
    defer _ = env.unsetenv(name);
    _ = env.unsetenv(name);
    try t.expectEqual(@as(f64, 5), try budget.targetFloorGiB(4));
    try t.expectEqual(@as(f64, 7), try budget.targetFloorGiB(7));
    _ = env.setenv(name, " 6.5 ", 1);
    try t.expectEqual(@as(f64, 6.5), try budget.targetFloorGiB(4));
    for ([_][:0]const u8{ "bad", "nan", "0", "-1" }) |raw| {
        _ = env.setenv(name, raw, 1);
        try t.expectError(error.InvalidEnvironment, budget.targetFloorGiB(4));
    }
    _ = env.setenv(name, "", 1);
    try t.expectEqual(@as(f64, 5), try budget.targetFloorGiB(4));
}
