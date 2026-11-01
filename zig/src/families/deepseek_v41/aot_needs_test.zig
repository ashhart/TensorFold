//! aot-needs host tests: the served pool, a launch reduced to its Triton specialization key, the AOT guard's key.
const std = @import("std");
const calls = @import("block_calls.zig");
const aot_needs = @import("aot_needs.zig");
const Param = aot_needs.Param;
const Fn = aot_needs.Fn;
const Sigs = aot_needs.Sigs;
const Needs = aot_needs.Needs;
const kvs = @import("cuda_kv.zig");
const block = @import("block.zig");
const block_prefill = @import("block_prefill.zig");
const ced = @import("ced.zig");
const check = @import("calls_check.zig");
const Config = @import("config.zig").Config;

const testing = std.testing;

const env = @import("sys").env;

test "aot-needs: the pool is the serve path's (kv_state.optionsFromEnv, model.zig's rule), TF_DSV41_NEEDS_POOL aside" {
    const Case = struct { tokens: ?[:0]const u8, split: ?[:0]const u8, force: ?[:0]const u8, pool: bool };
    const cases = [_]Case{
        .{ .tokens = null, .split = null, .force = null, .pool = false }, // the decode TP check, `generate` without the pool
        .{ .tokens = null, .split = "1", .force = null, .pool = false },
        .{ .tokens = "3000000", .split = null, .force = null, .pool = true },
        .{ .tokens = "3000000", .split = "1", .force = null, .pool = true }, // the served settings (pool + split KV)
        .{ .tokens = null, .split = null, .force = "1", .pool = true },
        .{ .tokens = "3000000", .split = "1", .force = "0", .pool = false },
    };
    defer for ([_][:0]const u8{ "TF_DSV41_POOL_TOKENS", "TF_DSV41_KV_SPLIT", "TF_DSV41_NEEDS_POOL" }) |k| {
        _ = env.unsetenv(k);
    };
    const limit: i64 = 1 << 20;
    for (cases) |c| {
        inline for (.{ .{ "TF_DSV41_POOL_TOKENS", "tokens" }, .{ "TF_DSV41_KV_SPLIT", "split" }, .{ "TF_DSV41_NEEDS_POOL", "force" } }) |kv| {
            if (@field(c, kv[1])) |v| _ = env.setenv(kv[0], v, 1) else _ = env.unsetenv(kv[0]);
        }
        const got = try aot_needs.poolFromEnv(limit, 2);
        try testing.expectEqual(c.pool, got != null);
        // unforced, the listing and the served engine agree: a pool exactly when model.zig opens one, split alike
        const served = try kvs.optionsFromEnv(@intCast(limit), 2);
        if (c.force == null) {
            try testing.expectEqual(served != null, got != null);
            if (served) |k| try testing.expectEqual(k.split, got.?.split);
        }
        if (got) |p| try testing.expectEqual(limit >> 8, p.pts);
    }
}

test "aot-needs: a launch reduces to Triton's specialization (constexprs, == 1, % 16, width, alignment)" {
    var params = [_]Param{
        .{ .name = "X", .constexpr = false, .nospec = false, .noalign = false },
        .{ .name = "Y", .constexpr = false, .nospec = false, .noalign = true },
        .{ .name = "N", .constexpr = false, .nospec = false, .noalign = false },
        .{ .name = "M", .constexpr = false, .nospec = true, .noalign = false },
        .{ .name = "S", .constexpr = false, .nospec = false, .noalign = false },
        .{ .name = "F", .constexpr = false, .nospec = false, .noalign = false },
        .{ .name = "P", .constexpr = false, .nospec = false, .noalign = false },
        .{ .name = "K", .constexpr = true, .nospec = false, .noalign = false },
        .{ .name = "E", .constexpr = true, .nospec = false, .noalign = false },
    };
    var fns = [_]Fn{.{ .function = "m._k", .params = &params }};
    var sigs: Sigs = .{};
    defer sigs.map.deinit(testing.allocator);
    try sigs.map.put(testing.allocator, "_k", &fns);
    var n: Needs = .{ .gpa = testing.allocator, .sigs = &sigs };
    defer n.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const t = calls.Tensor{ .role = .{ .buf = "w.x" }, .dt = .bf16, .shape = &.{4}, .stride = &.{1}, .offset = 8 };
    const c: calls.Call = .{ .triton = true, .name = "_k", .args = &.{
        .{ .name = "X", .arg = .{ .t = t } },       .{ .name = "Y", .arg = .{ .t = .{ .role = .{ .buf = "w.y" }, .dt = .f32, .shape = &.{4}, .stride = &.{1}, .offset = 0 } } },
        .{ .name = "N", .arg = .{ .i = 1 } },       .{ .name = "M", .arg = .{ .i = 32 } },
        .{ .name = "S", .arg = .{ .i = 1 << 33 } }, .{ .name = "F", .arg = .{ .f = 0.5 } },
        .{ .name = "P", .arg = .none },             .{ .name = "K", .arg = .{ .i = 7 } },
        .{ .name = "E", .arg = .{ .f = 0.125 } },
    } };
    const key = try n.reduce(a, &c);
    try testing.expectEqualStrings(
        "{\"fn\":\"_k\",\"function\":\"m._k\",\"consts\":{\"N\":1,\"P\":null,\"K\":7,\"E\":{\"fp64_bits\":\"0x3fc0000000000000\"}}," ++
            "\"runtime\":{\"X\":{\"type\":\"*bf16\",\"div16\":false},\"Y\":{\"type\":\"*fp32\",\"div16\":false},\"M\":{\"type\":\"i32\",\"div16\":false}," ++
            "\"S\":{\"type\":\"i64\",\"div16\":true},\"F\":{\"type\":\"fp32\"}}}",
        key,
    );
    // a call naming a parameter no kernel of that name has is refused
    const bad: calls.Call = .{ .triton = true, .name = "_k", .args = &.{.{ .name = "Z", .arg = .{ .i = 3 } }} };
    try testing.expectError(error.NoKernel, n.reduce(a, &bad));
}

test "aot guard: a program's key is its lookups' as `find` tells them apart, the trace's the same" {
    const runner = @import("run.zig");
    const aot = @import("cuda").aot;
    const PJ = @typeInfo(@FieldType(aot.Spec, "params")).pointer.child;
    var pj = [_]PJ{ .{ .name = "X", .type = "*bf16", .div16 = true, .nospec = false }, .{ .name = "N", .type = "i32", .div16 = true, .nospec = false } };
    const specs = [_]aot.Spec{.{ .@"fn" = "_k", .hash = "h", .name = "_k", .num_warps = 4, .params = &pj, .consts = .{} }};
    var ps = try runner.paramsOfSpecs(testing.allocator, &specs);
    defer runner.freeParams(testing.allocator, &ps);
    const t = calls.Tensor{ .role = .{ .buf = "w.x" }, .dt = .bf16, .shape = &.{4}, .stride = &.{1}, .offset = 0 };
    var u = t;
    u.role = .{ .buf = "w.y" };
    u.offset = 4096;
    const k = struct {
        fn of(p: *const runner.Params, x: calls.Tensor, n: i64, b: i64) u64 {
            const args = [_]calls.Named{ .{ .name = "X", .arg = .{ .t = x } }, .{ .name = "N", .arg = .{ .i = n } }, .{ .name = "B", .arg = .{ .i = b } } };
            return runner.programKey(&.{ .{ .triton = false, .name = "ext.f", .args = &.{} }, .{ .triton = true, .name = "_k", .args = &args } }, p);
        }
    };
    // not where a tensor lives (its role's base is taken 16-aligned); a launch int's class, a constexpr's value
    try testing.expectEqual(k.of(&ps, t, 32, 64), k.of(&ps, u, 48, 64));
    try testing.expect(k.of(&ps, t, 32, 64) != k.of(&ps, t, 33, 64));
    try testing.expect(k.of(&ps, t, 32, 64) != k.of(&ps, t, 1, 64));
    try testing.expect(k.of(&ps, t, 32, 64) != k.of(&ps, t, 32, 128));
    try testing.expect(k.of(&ps, t, 32, 64) != k.of(&ps, t, 1 << 40, 64));
    try testing.expect(k.of(&ps, t, 32, 64) != runner.programKey(&.{}, &ps));
    // the trace's key: an int's class, a pointer's dtype and alignment, the constexprs
    const tk = struct {
        fn of(n: i32, addr: u64) u64 {
            return aot.Trace.key("_k", &.{ aot.ptr("X", "*bf16", addr), aot.int("N", n) }, &.{aot.ci("B", 64)});
        }
    };
    try testing.expectEqual(tk.of(32, 0x1000), tk.of(48, 0x2000));
    try testing.expect(tk.of(32, 0x1000) != tk.of(33, 0x1000));
    try testing.expect(tk.of(32, 0x1000) != tk.of(1, 0x1000));
    try testing.expect(tk.of(32, 0x1000) != tk.of(32, 0x1008));
}

/// The served PF_4K knobs (TF_DSV41_PIECE_RUNS, replay and images on), at the default 4,096-position context.
const pf4k_env = [_][2][:0]const u8{
    .{ "TF_DSV41_PF_4K", "1" },        .{ "TF_DSV41_PF_4K_MEMORY_OK", "1" }, .{ "TF_DSV41_PREFILL_ROWS", "4096" },
    .{ "TF_DSV41_PREFILL", "replay" }, .{ "TF_DSV41_PIECE_RUNS", "1" },      .{ "TF_DSV41_IMAGES", "native" },
    .{ "TF_DSV41_SLOTS", "2" },        .{ "TF_DSV41_NEEDS_POOL", "1" },      .{ "TF_DSV41_DRAFTS", "0" },
    .{ "TF_DSV41_CONTEXT", "4096" },
};

/// main's widths: the K2 fixtures, every layer's x3gm sets, DSpark's main_proj.
fn mainWidths(a: std.mem.Allocator, cfg: *const Config) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    for (0..cfg.layers) |L| try w.gm.put(a, @intCast(L), .{ (1 << 4) | (1 << 6) | (1 << 8), (1 << 4) | (1 << 10) });
    try w.dense.put(a, "dspark.main_proj", try w.k2("L40.attn.wkv"));
    return w;
}

/// How many of a program's variants `n` does not list.
fn unlisted(a: std.mem.Allocator, n: *const Needs, cs: []const calls.Call) !usize {
    var miss: usize = 0;
    for (cs) |*c| if (c.triton) {
        if (!n.keys.contains(try n.reduce(a, c))) miss += 1;
    };
    return miss;
}

test "aot-needs: TF_DSV41_PF_4K enumerates the forward's 2,048-row pass too (its staging ring, runs, images, CED decoder)" {
    defer for (pf4k_env) |kv| {
        _ = env.unsetenv(kv[0]);
    };
    for (pf4k_env) |kv| _ = env.setenv(kv[0], kv[1], 1);
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(struct { kernels: Sigs }, gpa, @embedFile("fixtures/aot-sigs.json"), .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
    defer parsed.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg: Config = .{};
    const w = try mainWidths(a, &cfg);
    const sv = try aot_needs.Served.fromEnv();
    try testing.expect(sv.o.pf4k and sv.o.prefill_rows == 4096 and sv.runs and sv.images and sv.replay);
    var n: Needs = .{ .gpa = gpa, .sigs = &parsed.value.kernels };
    defer n.deinit();
    var log: std.Io.Writer.Allocating = .init(gpa);
    defer log.deinit();
    try aot_needs.enumerate(gpa, &n, &cfg, &w, sv, &log.writer);
    // the staging ring of each pass: ceilPow2(rows + window - 1)
    var rings: [2]bool = undefined;
    for (&rings, [_][]const u8{ "\"RING\":4096,", "\"RING\":8192," }) |*got, ring| {
        got.* = for (n.keys.keys()) |k| {
            if (std.mem.startsWith(u8, k, "{\"fn\":\"_kv_store\"") and std.mem.indexOf(u8, k, ring) != null) break true;
        } else false;
    }
    // the 2,048-row pass's own programs (not promptPasses'): a staged piece run, an image segment, CED's decoder
    var two = sv.o;
    two.pf4k = false;
    two.prefill_rows = 2048;
    var img = two;
    img.image_keep = true;
    var backbone: [64]u32 = undefined;
    for (backbone[0..cfg.layers], 0..) |*l, i| l.* = @intCast(i);
    const layers = backbone[0..cfg.layers];
    const d0 = try ced.decoderStart(&cfg);
    const runs = try block_prefill.emitMulti(a, &cfg, &w, two, layers[0 .. d0 + 1], &.{ .{ .slot = 0, .start = 0, .n = 17 }, .{ .slot = 1, .start = 0, .n = 1517 } }, .encoder);
    const image = try block_prefill.emitPrefill(a, &cfg, &w, img, layers, 1517, 0, true);
    const dec = try block_prefill.emitReplay(a, &cfg, &w, two, layers[d0..], ced.keep, 0, .decoder);
    var miss: [3]usize = undefined;
    for (&miss, [_][]const calls.Call{ runs, image, dec }) |*m, cs| m.* = try unlisted(a, &n, cs);
    if (!rings[0] or !rings[1] or !std.mem.eql(usize, &miss, &.{ 0, 0, 0 })) {
        std.debug.print("PF_4K: _kv_store RING 4096 / 8192 listed {any}; unlisted variants (run, image, decoder) {any}\n", .{ rings, miss });
        return error.TestUnexpectedResult;
    }
}
