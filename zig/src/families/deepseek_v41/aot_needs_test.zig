//! aot-needs host tests: the served pool, a launch reduced to its Triton specialization key, the AOT guard's key.
const std = @import("std");
const calls = @import("block_calls.zig");
const aot_needs = @import("aot_needs.zig");
const Param = aot_needs.Param;
const Fn = aot_needs.Fn;
const Sigs = aot_needs.Sigs;
const Needs = aot_needs.Needs;
const kvs = @import("cuda_kv.zig");

const testing = std.testing;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

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
        _ = unsetenv(k);
    };
    const limit: i64 = 1 << 20;
    for (cases) |c| {
        inline for (.{ .{ "TF_DSV41_POOL_TOKENS", "tokens" }, .{ "TF_DSV41_KV_SPLIT", "split" }, .{ "TF_DSV41_NEEDS_POOL", "force" } }) |kv| {
            if (@field(c, kv[1])) |v| _ = setenv(kv[0], v, 1) else _ = unsetenv(kv[0]);
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
