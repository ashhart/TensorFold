//! Deterministic host tests for runtime loading, immutable identities, exact overlay arithmetic and refusal.
const std = @import("std");
const rl = @import("runtime_lora.zig");
const st = @import("safetensors.zig");
const fx = @import("lora_fixture.zig");

const gpa = std.testing.allocator;
const module = "model.layers.0.self_attn.q_proj";

fn source(byte: u8) [32]u8 {
    return @splat(byte);
}

fn config(rank: usize, alpha: f64) rl.Config {
    return .{ .rank = rank, .alpha = alpha, .scale = alpha / @as(f64, @floatFromInt(rank)) };
}

fn namedTensor(dtype: st.DType, dims: []const usize, bytes: []const u8) st.Tensor {
    var shape: [st.max_rank]usize = @splat(1);
    @memcpy(shape[0..dims.len], dims);
    return .{ .dtype = dtype, .rank = @intCast(dims.len), .shape = shape, .bytes = bytes };
}

fn pair(a: []const f32, b: []const f32, k: usize, n: usize, r: usize) [2]rl.NamedTensor {
    return .{
        .{ .name = "base_model.model." ++ module ++ ".lora_A.weight", .tensor = namedTensor(.f32, &.{ r, k }, std.mem.sliceAsBytes(a)) },
        .{ .name = "base_model.model." ++ module ++ ".lora_B.weight", .tensor = namedTensor(.f32, &.{ n, r }, std.mem.sliceAsBytes(b)) },
    };
}

fn target(n: usize, k: usize) rl.Target {
    return .{ .module = module, .n = n, .k = k, .storage = .{ .dense = .f32 } };
}

fn oracle(x: []const f64, xs: usize, y: []f64, ys: usize, rows: usize, a: []const f32, b: []const f32, n: usize, k: usize, r: usize, scale: f64, latent: []f64) void {
    for (0..rows) |row| {
        for (0..r) |j| {
            latent[j] = 0;
            for (0..k) |c| latent[j] += x[row * xs + c] * @as(f64, a[j * k + c]);
        }
        for (0..n) |column| {
            var dot: f64 = 0;
            for (0..r) |j| dot += latent[j] * @as(f64, b[column * r + j]);
            if (dot * scale != 0) y[row * ys + column] += dot * scale;
        }
    }
}

fn sameBits(a: []const f64, b: []const f64) bool {
    return std.mem.eql(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
}

test "runtime overlays match an independent scalar oracle over ragged rows" {
    const a = [_]f32{ 1, -2, 0.5, 3, -1, 4, 2, -3 };
    const b = [_]f32{ 2, 1, -1, 0.5, 3, -2 };
    var source_a = a;
    var source_b = b;
    const factors = pair(&source_a, &source_b, 4, 3, 2);
    var prepared = try rl.prepare(gpa, .fromDigest(source(1)), .fromBytes("cfg", "weights"), config(2, 3), &factors, &.{target(3, 4)});
    defer prepared.deinit();
    @memset(&source_a, 99);
    @memset(&source_b, 99);
    const x = [_]f64{ 1, 2, -1, 0.5, 91, 92, -2, 3, 4, -3, 93, 94 };
    var actual = [_]f64{ 5, -7, 11, 81, 82, -1, 2, 13 };
    var expected = actual;
    var oracle_work: [2]f64 = undefined;
    oracle(&x, 6, &expected, 5, 2, &a, &b, 3, 4, 2, 1.5, &oracle_work);
    var work: [4]f64 = undefined;
    try prepared.apply(module, &x, 6, &actual, 5, 2, &work);
    try std.testing.expect(sameBits(&expected, &actual));
    try std.testing.expectEqual(@as(f64, 81), actual[3]);
    try std.testing.expectEqual(@as(f64, 82), actual[4]);
    try std.testing.expectError(error.InvalidBatchStride, prepared.apply(module, &x, 3, &actual, 5, 2, &work));
    try std.testing.expectError(error.InvalidBatchStorage, prepared.apply(module, x[0..9], 6, &actual, 5, 2, &work));
    try std.testing.expectError(error.MissingOverlay, prepared.apply("model.layers.0.v_proj", &x, 6, &actual, 5, 2, &work));
    var none: [0]f64 = .{};
    try prepared.apply(module, &.{}, 0, &none, 0, 0, &none);
}

test "zero and negative-zero adapters leave every output bit unchanged" {
    const a = [_]f32{ 1, 2, 3 };
    const zeros = [_]f32{ -0.0, 0.0 };
    const factors = pair(&a, &zeros, 3, 2, 1);
    var negative = try rl.prepare(gpa, .fromDigest(source(2)), .fromBytes("cfg", "negative-zero"), config(1, 2), &factors, &.{target(2, 3)});
    defer negative.deinit();
    var y = [_]f64{ 0.0, -0.0 };
    const original = y;
    var work: [1]f64 = undefined;
    try negative.apply(module, &.{ 1, 2, 3 }, 3, &y, 2, 1, &work);
    try std.testing.expect(sameBits(&original, &y));
    var zero_alpha = try rl.prepare(gpa, .fromDigest(source(2)), .fromBytes("cfg", "zero-alpha"), config(1, 0), &factors, &.{target(2, 3)});
    defer zero_alpha.deinit();
    try zero_alpha.apply(module, &.{ 1, 2, 3 }, 3, &y, 2, 1, &work);
    try std.testing.expect(sameBits(&original, &y));
    var negative_alpha = try rl.prepare(gpa, .fromDigest(source(2)), .fromBytes("cfg", "negative-zero-alpha"), config(1, -0.0), &factors, &.{target(2, 3)});
    defer negative_alpha.deinit();
    try std.testing.expect(std.math.signbit(negative_alpha.scale));
    try negative_alpha.apply(module, &.{ 1, 2, 3 }, 3, &y, 2, 1, &work);
    try std.testing.expect(sameBits(&original, &y));
}

test "f16 and BF16 adapter factors retain exact converted values" {
    const half = [_]f16{ 1, -2 };
    const bf16_bytes = [_]u8{ 0, 0x40 };
    const tensors = [_]rl.NamedTensor{
        .{ .name = "base_model.model." ++ module ++ ".lora_A.weight", .tensor = namedTensor(.f16, &.{ 1, 2 }, std.mem.sliceAsBytes(&half)) },
        .{ .name = "base_model.model." ++ module ++ ".lora_B.weight", .tensor = namedTensor(.bf16, &.{ 1, 1 }, &bf16_bytes) },
    };
    var prepared = try rl.prepare(gpa, .fromDigest(source(11)), .fromBytes("cfg", "f16-bf16"), config(1, 1), &tensors, &.{target(1, 2)});
    defer prepared.deinit();
    try std.testing.expectEqual(@as(f32, -2), prepared.overlays[0].a[1]);
    try std.testing.expectEqual(@as(f32, 2), prepared.overlays[0].b[0]);
    var y = [_]f64{5};
    var work: [1]f64 = undefined;
    try prepared.apply(module, &.{ 1, 2 }, 2, &y, 1, 1, &work);
    try std.testing.expectEqual(@as(f64, -1), y[0]);
}

test "rank one and rank three prepare with exact factor shapes" {
    inline for (.{ 1, 3 }) |rank| {
        const a = [_]f32{ 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6 };
        const b = [_]f32{ 1, 2, 3, 4, 5, 6 };
        const factors = pair(a[0 .. rank * 2], b[0 .. 2 * rank], 2, 2, rank);
        var prepared = try rl.prepare(gpa, .fromDigest(source(3)), .fromBytes("cfg", "ranks"), config(rank, rank), &factors, &.{target(2, 2)});
        defer prepared.deinit();
        try std.testing.expectEqual(@as(usize, rank), prepared.rank);
        try std.testing.expectEqual(rank * 2, prepared.overlays[0].a.len);
        try std.testing.expectEqual(2 * rank, prepared.overlays[0].b.len);
    }
}

fn expectError(expected: anyerror, cfg: rl.Config, tensors: []const rl.NamedTensor, targets: []const rl.Target) !void {
    try std.testing.expectError(expected, rl.prepare(gpa, .fromDigest(source(4)), .fromBytes("bad", "adapter"), cfg, tensors, targets));
}

test "malformed names, missing pairs, duplicate factors and target rules are refused" {
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 3, 4 };
    const factors = pair(&a, b[0..2], 2, 1, 2);
    var malformed = factors;
    malformed[0].name = "model.q_proj.lora_A.weight";
    try expectError(error.MalformedAdapterName, config(2, 2), &malformed, &.{target(1, 2)});
    try expectError(error.MissingFactor, config(2, 2), factors[0..1], &.{target(1, 2)});
    var duplicate: [4]rl.NamedTensor = .{ factors[0], factors[1], factors[0], factors[1] };
    duplicate[3].name = "base_model.model." ++ module ++ ".lora_B.default.weight";
    try expectError(error.DuplicateFactor, config(2, 2), &duplicate, &.{target(1, 2)});
    var other = factors;
    other[0].name = "base_model.model.model.layers.0.v_proj.lora_A.weight";
    other[1].name = "base_model.model.model.layers.0.v_proj.lora_B.weight";
    try expectError(error.UntargetedModule, config(2, 2), &other, &.{target(1, 2)});
    var denied = config(2, 2);
    denied.target_modules = &.{"v_proj"};
    try expectError(error.TargetModuleNotAllowed, denied, &factors, &.{target(1, 2)});
    var suffix = config(2, 2);
    suffix.target_modules = &.{"q_proj"};
    var accepted = try rl.prepare(gpa, .fromDigest(source(4)), .fromBytes("cfg", "suffix"), suffix, &factors, &.{target(1, 2)});
    accepted.deinit();
    suffix.target_modules = &.{ "q_proj", "v_proj" };
    var second = target(1, 2);
    second.module = "model.layers.0.self_attn.v_proj";
    try expectError(error.MissingTargetFactors, suffix, &factors, &.{ target(1, 2), second });
}

test "mismatched rank, ragged tensors, bad payloads and unsupported factors are refused" {
    const values = [_]f32{ 1, 2, 3, 4 };
    var factors = pair(&values, values[0..2], 2, 1, 2);
    factors[0].tensor = namedTensor(.f32, &.{ 1, 2 }, std.mem.sliceAsBytes(&values));
    try expectError(error.MismatchedTensorShape, config(2, 2), &factors, &.{target(1, 2)});
    factors = pair(&values, values[0..2], 2, 1, 2);
    factors[0].tensor = namedTensor(.f32, &.{ 2, 1, 2 }, std.mem.sliceAsBytes(&values));
    try expectError(error.MismatchedTensorShape, config(2, 2), &factors, &.{target(1, 2)});
    factors = pair(&values, values[0..2], 2, 1, 2);
    factors[1].tensor.shape[1] = 1;
    try expectError(error.MismatchedTensorShape, config(2, 2), &factors, &.{target(1, 2)});
    factors = pair(&values, values[0..2], 2, 1, 2);
    factors[0].tensor.bytes = std.mem.sliceAsBytes(&values)[0..12];
    try expectError(error.MismatchedTensorShape, config(2, 2), &factors, &.{target(1, 2)});
    factors = pair(&values, values[0..2], 2, 1, 2);
    const wide = [_]f64{ 1, 2, 3, 4 };
    factors[0].tensor = namedTensor(.f64, &.{ 2, 2 }, std.mem.sliceAsBytes(&wide));
    try expectError(error.UnsupportedAdapterDType, config(2, 2), &factors, &.{target(1, 2)});
    const nonfinite = [_]f32{ std.math.nan(f32), 2, 3, 4 };
    factors = pair(&nonfinite, values[0..2], 2, 1, 2);
    try expectError(error.NonFiniteAdapter, config(2, 2), &factors, &.{target(1, 2)});
}

test "unsupported base quantization and geometry never prepare an overlay" {
    const values = [_]f32{ 1, 2, 3, 4 };
    const factors = pair(&values, &values, 2, 1, 2);
    var bad = target(1, 2);
    bad.storage = .unsupported;
    try expectError(error.UnsupportedQuantization, config(2, 2), &factors, &.{bad});
    bad.storage = .{ .affine = .{ .bits = 8, .group = 32 } };
    bad.k = 32;
    try expectError(error.UnsupportedQuantization, config(2, 2), &factors, &.{bad});
    bad.storage = .{ .affine = .{ .bits = 4, .group = 64 } };
    try expectError(error.UnsupportedGeometry, config(2, 2), &factors, &.{bad});
    bad = target(1, 2);
    bad.rank = 3;
    try expectError(error.UnsupportedGeometry, config(2, 2), &factors, &.{bad});
    bad = target(1, 2);
    bad.storage = .{ .dense = .f64 };
    try expectError(error.UnsupportedQuantization, config(2, 2), &factors, &.{bad});
}

fn allocationTrial(a: std.mem.Allocator, tensors: []const rl.NamedTensor, targets: []const rl.Target) !void {
    var prepared = try rl.prepare(a, .fromDigest(source(5)), .fromBytes("cfg", "rollback"), config(2, 4), tensors, targets);
    defer prepared.deinit();
    try std.testing.expectEqual(targets.len, prepared.overlays.len);
}

test "every preparation allocation failure rolls back all ownership" {
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 5, 6, 7, 8 };
    const factors = pair(&a, &b, 2, 2, 2);
    var at: usize = 0;
    while (at < 100) : (at += 1) {
        var failing = std.testing.FailingAllocator.init(gpa, .{ .fail_index = at });
        allocationTrial(failing.allocator(), &factors, &.{target(2, 2)}) catch |err| {
            try std.testing.expect(failing.has_induced_failure);
            try std.testing.expectEqual(error.OutOfMemory, err);
            try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
            continue;
        };
        try std.testing.expect(!failing.has_induced_failure);
        try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
        break;
    }
    try std.testing.expect(at > 0 and at < 100);
}

test "cache identity binds base, adapter, canonical overlays and a separate namespace" {
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 3, 4 };
    const factors = pair(&a, b[0..2], 2, 1, 2);
    const base = rl.BaseIdentity.fromDigest(source(6));
    const adapter = rl.AdapterIdentity.fromBytes("cfg", "weights-a");
    var first = try rl.prepare(gpa, base, adapter, config(2, 2), &factors, &.{target(1, 2)});
    defer first.deinit();
    var same = try rl.prepare(gpa, base, adapter, config(2, 2), &factors, &.{target(1, 2)});
    defer same.deinit();
    try std.testing.expectEqual(first.cache_key.digest, same.cache_key.digest);
    try std.testing.expectEqual(base.digest, first.cache_key.base.digest);
    try std.testing.expectEqual(adapter.digest, first.cache_key.adapter.digest);
    var other_base = try rl.prepare(gpa, .fromDigest(source(7)), adapter, config(2, 2), &factors, &.{target(1, 2)});
    defer other_base.deinit();
    try std.testing.expect(!std.mem.eql(u8, &first.cache_key.digest, &other_base.cache_key.digest));
    var other_adapter = try rl.prepare(gpa, base, .fromBytes("cfg", "weights-b"), config(2, 2), &factors, &.{target(1, 2)});
    defer other_adapter.deinit();
    try std.testing.expect(!std.mem.eql(u8, &first.cache_key.digest, &other_adapter.cache_key.digest));
    var affine_target = target(1, 2);
    affine_target.storage = .{ .affine = .{ .bits = 4, .group = 32 } };
    affine_target.k = 32;
    const a32: [64]f32 = @splat(1);
    const affine_factors = pair(&a32, &b, 32, 1, 2);
    var other_layout = try rl.prepare(gpa, base, adapter, config(2, 2), &affine_factors, &.{affine_target});
    defer other_layout.deinit();
    try std.testing.expect(!std.mem.eql(u8, &first.cache_key.digest, &other_layout.cache_key.digest));
    const base_key = rl.baseProjectionKey(base);
    try std.testing.expect(!std.mem.eql(u8, &first.cache_key.digest, &base_key));
    try std.testing.expect(!std.mem.eql(u8, rl.cache_namespace, rl.base_projection_namespace));
    const changed_config = rl.AdapterIdentity.fromBytes("changed", "weights-a");
    const changed_weights = rl.AdapterIdentity.fromBytes("cfg", "weights-b");
    try std.testing.expect(!std.mem.eql(u8, &adapter.digest, &changed_config.digest));
    try std.testing.expect(!std.mem.eql(u8, &adapter.digest, &changed_weights.digest));
}

test "adapter config validates rank, alpha, variants and target-module declarations" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const good = try rl.parseConfig(a, "{\"peft_type\":\"LORA\",\"r\":4,\"lora_alpha\":8,\"target_modules\":[\"q_proj\",\"v_proj\"]}");
    try std.testing.expectEqual(@as(usize, 4), good.rank);
    try std.testing.expectEqual(@as(f64, 2), good.scale);
    try std.testing.expect(good.targets("model.layers.0.q_proj"));
    try std.testing.expect(!good.targets("model.layers.0.notq_proj"));
    for ([_][]const u8{ "{}", "{\"r\":0,\"lora_alpha\":1}", "{\"r\":1.5,\"lora_alpha\":1}" }) |json| try std.testing.expectError(error.InvalidRank, rl.parseConfig(a, json));
    try std.testing.expectError(error.InvalidAlpha, rl.parseConfig(a, "{\"r\":2,\"lora_alpha\":-1}"));
    try std.testing.expectError(error.InvalidAlpha, rl.parseConfig(a, "{\"r\":2}"));
    try std.testing.expectError(error.UnsupportedAdapterType, rl.parseConfig(a, "{\"peft_type\":\"IA3\",\"r\":2,\"lora_alpha\":2}"));
    try std.testing.expectError(error.UnsupportedLoRAVariant, rl.parseConfig(a, "{\"r\":2,\"lora_alpha\":2,\"use_dora\":true}"));
    try std.testing.expectError(error.InvalidTargetModules, rl.parseConfig(a, "{\"r\":2,\"lora_alpha\":2,\"target_modules\":[]}"));
    try std.testing.expectError(error.UnsupportedTargetModules, rl.parseConfig(a, "{\"r\":2,\"lora_alpha\":2,\"target_modules\":[\"all-linear\"]}"));
}

test "target enumeration order cannot change overlay cache identity" {
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 3, 4 };
    const q = pair(&a, b[0..2], 2, 1, 2);
    var v = q;
    v[0].name = "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight";
    v[1].name = "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight";
    const q_target = target(1, 2);
    var v_target = q_target;
    v_target.module = "model.layers.0.self_attn.v_proj";
    const tensors = [_]rl.NamedTensor{ q[0], q[1], v[0], v[1] };
    var forward = try rl.prepare(gpa, .fromDigest(source(10)), .fromBytes("cfg", "ordered"), config(2, 2), &tensors, &.{ q_target, v_target });
    defer forward.deinit();
    var backward = try rl.prepare(gpa, .fromDigest(source(10)), .fromBytes("cfg", "ordered"), config(2, 2), &tensors, &.{ v_target, q_target });
    defer backward.deinit();
    try std.testing.expectEqual(forward.cache_key.digest, backward.cache_key.digest);
    try std.testing.expectEqualStrings(module, forward.overlays[0].module);
}
test "a real PEFT directory loads into detached owned factors" {
    var scratch = try fx.Scratch.init();
    defer scratch.deinit();
    const adapter_dir = try fx.makeAdapter(&scratch, .{ .specs = &.{.{ .module = module, .n = 2, .k = 64, .r = 2 }} });
    var prepared = try rl.load(gpa, fx.io, .fromDigest(source(8)), adapter_dir, &.{target(2, 64)});
    defer prepared.deinit();
    try std.testing.expectEqual(@as(usize, 1), prepared.overlays.len);
    try std.testing.expectEqual(@as(usize, 2), prepared.rank);
    const cfg = try scratch.read("adapter", "adapter_config.json");
    const weights = try scratch.read("adapter", "adapter_model.safetensors");
    try std.testing.expectEqual(rl.AdapterIdentity.fromBytes(cfg, weights).digest, prepared.adapter.digest);
    var y = [_]f64{ 1, 2 };
    var work: [2]f64 = undefined;
    const zeros: [64]f64 = @splat(0);
    try prepared.apply(module, &zeros, 64, &y, 2, 1, &work);
    try std.testing.expectEqual(@as(f64, 1), y[0]);
    try std.testing.expectEqual(@as(f64, 2), y[1]);
}

test "real file loading balances every induced allocation failure" {
    var scratch = try fx.Scratch.init();
    defer scratch.deinit();
    const adapter_dir = try fx.makeAdapter(&scratch, .{ .specs = &.{.{ .module = module, .n = 2, .k = 64, .r = 2 }} });
    var at: usize = 0;
    while (at < 200) : (at += 1) {
        var failing = std.testing.FailingAllocator.init(gpa, .{ .fail_index = at });
        var prepared = rl.load(failing.allocator(), fx.io, .fromDigest(source(9)), adapter_dir, &.{target(2, 64)}) catch |err| {
            try std.testing.expect(failing.has_induced_failure);
            try std.testing.expectEqual(error.OutOfMemory, err);
            try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
            continue;
        };
        prepared.deinit();
        try std.testing.expect(!failing.has_induced_failure);
        try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
        break;
    }
    try std.testing.expect(at > 0 and at < 200);
}
