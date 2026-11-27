//! One host-checkable source-planted arithmetic fault for every Inkling 2 Mini glue kernel.
const std = @import("std");
const ct = @import("contract");
const spec = ct.spec;
const g = ct.gates;
const p = @import("inkling2_glue_plan");

/// Allocate, initialize, optionally load baseline pipelines, and transactionally register a probe.
pub fn addAcquired(
    comptime T: type,
    comptime Ctx: type,
    a: std.mem.Allocator,
    probes: *std.ArrayList(*T),
    kernels: *std.ArrayList(g.Kernel),
    ctx: Ctx,
    initFn: *const fn (Ctx) anyerror!T,
    loadBaseFn: ?*const fn (Ctx, *T) anyerror!void,
    control_name: ?[]const u8,
) !void {
    const probe = try a.create(T);
    errdefer a.destroy(probe);
    probe.* = try initFn(ctx);
    errdefer probe.deinit();
    if (loadBaseFn) |load| try load(ctx, probe);
    var k = probe.kernel();
    if (control_name) |label| {
        k.spec.name = label;
        k.control = .shapes;
    }
    try ct.register(a, probes, kernels, probe, k);
}

const Edit = struct { from: []const u8, to: []const u8 };
fn edit(kind: p.Kind) Edit {
    return switch (kind) {
        .embed_norm => .{ .from = "float(weight[d]) * (float(x[d]) * inverse)", .to = "float(weight[0]) * (float(x[d]) * inverse)" },
        .ar => .{ .from = "weights[s] = dot * INKLING_AR_SCALE;", .to = "weights[s] = dot;" },
        .cached_ar => .{ .from = "if (t + 1 < a.sources) means[t] = block_inv_rms[t];", .to = "if (t + 1 < a.sources) means[t] = block_inv_rms[(t + 1) & 3];" },
        .block_inv_rms => .{ .from = "if (t == 0) inv_rms[size_t(row) * 4 + a.slot] = inv;", .to = "if (t == 0) inv_rms[size_t(row) * 4 + a.slot] = inv * 2.0f;" },
        .attn_partial_ar => .{ .from = "means[a.sources - 1] = part_inv;", .to = "means[a.sources - 1] = part_inv * 2.0f;" },
        .mlp_partial_ar => .{ .from = "const bfloat next_p = bfloat(pin_row[d] + float(conv_row[d]));", .to = "const bfloat next_p = bfloat(pin_row[d] - float(conv_row[d]));" },
        .norm_bf16 => .{ .from = "float(norm_w[d]) * (norm_vals[k] * norm_inv)", .to = "float(norm_w[0]) * (norm_vals[k] * norm_inv)" },
        .attn_partial_ar_norm => .{ .from = "means[a.sources - 1] = attn_part_inv;", .to = "means[a.sources - 1] = attn_part_inv * 2.0f;" },
        .mlp_partial_ar_norm => .{ .from = "const bfloat next_p_n = bfloat(pin[d] + float(conv[d]));", .to = "const bfloat next_p_n = bfloat(pin[d] - float(conv[d]));" },
        .conv_static => .{ .from = "weight[c * 4 + tap]", .to = "weight[c * 4 + 3 - tap]" },
        .conv_dynamic => .{ .from = "static_w[lag * a.channels + c]", .to = "static_w[(3 - lag) * a.channels + c]" },
        .kv_conv_static => .{ .from = "is_v ? v_weight : k_weight", .to = "k_weight" },
        .kv_conv_dynamic => .{ .from = "is_v ? v_static_w : k_static_w", .to = "k_static_w" },
        .router => .{ .from = "key[t] = scores[t] + bias[t];", .to = "key[t] = scores[t];" },
        .swiglu => .{ .from = "ink_cap(u, 25.0f)", .to = "ink_cap(g, 25.0f)" },
        .logits => .{ .from = "ink_cap(float(bfloat(dot)), 30.0f)", .to = "ink_cap(dot, 30.0f)" },
        .expert_pick => .{ .from = "ids[r * 8 + j]", .to = "ids[r * 8 + 7 - j]" },
        .expert_combine => .{ .from = "acc += weights[r * 8 + j] * float(picked[(size_t(r) * 8 + j) * a.channels + d]);", .to = "acc -= weights[r * 8 + j] * float(picked[(size_t(r) * 8 + j) * a.channels + d]);" },
        .argmax => .{ .from = "return va > vb || (va == vb && ia < ib);", .to = "return va < vb || (va == vb && ia < ib);" },
    };
}

pub fn source(a: std.mem.Allocator, production: []const u8, kind: p.Kind) ![]u8 {
    const e = edit(kind);
    if (std.mem.count(u8, production, e.from) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, production, e.from, e.to);
}

pub fn name(kind: p.Kind) []const u8 {
    return switch (kind) {
        .embed_norm => "control: metal inkling2 glue embed_norm tf_inkling2_embed_norm",
        .ar => "control: metal inkling2 glue ar tf_inkling2_ar",
        .cached_ar => "control: metal inkling2 glue cached_ar tf_inkling2_cached_ar",
        .block_inv_rms => "control: metal inkling2 glue block_inv_rms tf_inkling2_block_inv_rms",
        .attn_partial_ar => "control: metal inkling2 glue attn_partial_ar tf_inkling2_attn_partial_ar",
        .mlp_partial_ar => "control: metal inkling2 glue mlp_partial_ar tf_inkling2_mlp_partial_ar",
        .norm_bf16 => "control: metal inkling2 glue norm_bf16 tf_inkling2_norm_bf16",
        .attn_partial_ar_norm => "control: metal inkling2 glue attn_partial_ar_norm tf_inkling2_attn_partial_ar_norm",
        .mlp_partial_ar_norm => "control: metal inkling2 glue mlp_partial_ar_norm tf_inkling2_mlp_partial_ar_norm",
        .conv_static => "control: metal inkling2 glue conv_static tf_inkling2_conv_static",
        .conv_dynamic => "control: metal inkling2 glue conv_dynamic tf_inkling2_conv_dynamic",
        .kv_conv_static => "control: metal inkling2 glue kv_conv_static tf_inkling2_kv_conv_static",
        .kv_conv_dynamic => "control: metal inkling2 glue kv_conv_dynamic tf_inkling2_kv_conv_dynamic",
        .router => "control: metal inkling2 glue router tf_inkling2_router",
        .swiglu => "control: metal inkling2 glue swiglu tf_inkling2_swiglu",
        .logits => "control: metal inkling2 glue logits tf_inkling2_logits",
        .expert_pick => "control: metal inkling2 glue expert_pick tf_inkling2_expert_pick",
        .expert_combine => "control: metal inkling2 glue expert_combine tf_inkling2_expert_combine",
        .argmax => "control: metal inkling2 glue argmax tf_ink2_argmax",
    };
}

/// The router's NaN-key fallback planted out: the rank picks run on every row, or the flag reads expert 0's key alone.
pub const NanControl = enum { no_fallback, key0_flag };

fn nanEdit(c: NanControl) Edit {
    return switch (c) {
        .no_fallback => .{ .from = "  if (any_nan != 0) {", .to = "  if (false) {" },
        .key0_flag => .{ .from = "if (t == 0) any_nan = nan ? 1u : 0u;", .to = "if (t == 0) any_nan = isnan(key[0]) ? 1u : 0u;" },
    };
}

pub fn nanSource(a: std.mem.Allocator, production: []const u8, c: NanControl) ![]u8 {
    const e = nanEdit(c);
    if (std.mem.count(u8, production, e.from) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, production, e.from, e.to);
}

pub fn nanName(c: NanControl) []const u8 {
    return switch (c) {
        .no_fallback => "control: metal inkling2 glue router tf_inkling2_router, NaN keys without the serial fallback",
        .key0_flag => "control: metal inkling2 glue router tf_inkling2_router, the NaN flag from expert 0's key only",
    };
}

test "every router NaN-key control applies once" {
    const production = @import("inkling2_metal").glue;
    const a = std.testing.allocator;
    for (std.enums.values(NanControl)) |c| {
        const changed = try nanSource(a, production, c);
        defer a.free(changed);
        try std.testing.expectEqual(production.len - nanEdit(c).from.len + nanEdit(c).to.len, changed.len);
        try std.testing.expectError(error.ControlEditMissing, nanSource(a, changed, c));
    }
}

test "every glue control applies once and changes only its arithmetic anchor" {
    const production = @import("inkling2_metal").glue;
    const a = std.testing.allocator;
    for (std.enums.values(p.Kind)) |kind| {
        const e = edit(kind);
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, production, e.from));
        const changed = try source(a, production, kind);
        defer a.free(changed);
        const at = std.mem.indexOf(u8, production, e.from).?;
        try std.testing.expectEqualStrings(production[0..at], changed[0..at]);
        try std.testing.expectEqualStrings(e.to, changed[at..][0..e.to.len]);
        try std.testing.expectEqualStrings(production[at + e.from.len ..], changed[at + e.to.len ..]);
        try std.testing.expectError(error.ControlEditMissing, source(a, changed, kind));
        try std.testing.expectError(error.ControlEditMissing, source(a, "", kind));
    }
}

test "a repeated glue control anchor is ambiguous and refuses to prove a fault" {
    const a = std.testing.allocator;
    const production = @import("inkling2_metal").glue;
    for (std.enums.values(p.Kind)) |kind| {
        const twice = try std.mem.concat(a, u8, &.{ production, production });
        defer a.free(twice);
        try std.testing.expectError(error.ControlEditMissing, source(a, twice, kind));
    }
}

const MockProbe = struct {
    a: std.mem.Allocator,
    payload: []u8,
    base_payload: ?[]u8 = null,
    deinit_count: *usize,

    fn deinit(self: *MockProbe) void {
        if (self.base_payload) |bp| self.a.free(bp);
        self.a.free(self.payload);
        self.deinit_count.* += 1;
    }

    fn kernel(self: *MockProbe) g.Kernel {
        const Mock = struct {
            fn depth(_: spec.Shape) usize {
                return 1;
            }
            fn prepare(_: *anyopaque, _: spec.Shape, _: g.Fill, _: u64) anyerror!bool {
                return false;
            }
            fn run(_: *anyopaque) anyerror!void {}
            fn bytes(_: *anyopaque) anyerror![]const u8 {
                return &.{};
            }
            fn intact(_: *anyopaque) anyerror!bool {
                return true;
            }
            fn overlap(_: *anyopaque) bool {
                return false;
            }
            fn count(_: *anyopaque) usize {
                return 0;
            }
            fn reference(_: *anyopaque, _: []f64, _: []f64) anyerror!void {}
            const vt: g.VTable = .{
                .prepare = prepare,
                .run = run,
                .output = bytes,
                .inputs = bytes,
                .intact = intact,
                .overlap = overlap,
                .count = count,
                .reference = reference,
            };
            const served = [_]spec.Shape{.{ .rows = 1, .n = 1, .k = 1 }};
        };
        return .{
            .ctx = self,
            .vt = &Mock.vt,
            .spec = .{
                .name = "mock",
                .accum = .none,
                .depth = Mock.depth,
                .out = .bf16,
                .served = &Mock.served,
            },
        };
    }
};

const MockCtx = struct {
    a: std.mem.Allocator,
    deinit_count: *usize,
    fail_base: bool = false,

    fn initProbe(ctx: MockCtx) !MockProbe {
        return .{
            .a = ctx.a,
            .payload = try ctx.a.alloc(u8, 16),
            .deinit_count = ctx.deinit_count,
        };
    }

    fn loadBase(ctx: MockCtx, probe: *MockProbe) !void {
        if (ctx.fail_base) return error.BaselineLoadFailed;
        probe.base_payload = try ctx.a.alloc(u8, 12);
    }
};

fn deinitMockSuite(a: std.mem.Allocator, probes: *std.ArrayList(*MockProbe), kernels: *std.ArrayList(g.Kernel)) void {
    for (probes.items) |probe| {
        probe.deinit();
        a.destroy(probe);
    }
    probes.deinit(a);
    kernels.deinit(a);
}

test "addAcquired cleans up candidate probe when baseline pipeline load fails" {
    const a = std.testing.allocator;
    var probes: std.ArrayList(*MockProbe) = .empty;
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer deinitMockSuite(a, &probes, &kernels);

    var new_deinits: usize = 0;
    const ctx: MockCtx = .{ .a = a, .deinit_count = &new_deinits, .fail_base = true };
    try std.testing.expectError(
        error.BaselineLoadFailed,
        addAcquired(MockProbe, MockCtx, a, &probes, &kernels, ctx, MockCtx.initProbe, MockCtx.loadBase, "control: mock"),
    );
    try std.testing.expectEqual(@as(usize, 1), new_deinits);
    try std.testing.expectEqual(@as(usize, 0), probes.items.len);
    try std.testing.expectEqual(@as(usize, 0), kernels.items.len);
}

test "addAcquired rolls back new probe when kernel-list append hits OutOfMemory" {
    var existing_deinits: usize = 0;
    var new_deinits: usize = 0;
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .resize_fail_index = 0 });
    const a = failing.allocator();
    {
        var probes: std.ArrayList(*MockProbe) = .empty;
        var kernels: std.ArrayList(g.Kernel) = .empty;
        defer deinitMockSuite(a, &probes, &kernels);

        try probes.ensureTotalCapacityPrecise(a, 1);
        try kernels.ensureTotalCapacityPrecise(a, 1);
        const existing = blk: {
            const probe = try a.create(MockProbe);
            errdefer a.destroy(probe);
            probe.* = try MockCtx.initProbe(.{ .a = a, .deinit_count = &existing_deinits });
            errdefer probe.deinit();
            try ct.register(a, &probes, &kernels, probe, probe.kernel());
            break :blk probe;
        };

        // Succeed through create (0), initProbe (1), loadBase (2), and probes growth (3); fail kernels growth (4).
        failing.fail_index = failing.alloc_index + 4;
        const ctx: MockCtx = .{ .a = a, .deinit_count = &new_deinits };
        try std.testing.expectError(
            error.OutOfMemory,
            addAcquired(MockProbe, MockCtx, a, &probes, &kernels, ctx, MockCtx.initProbe, MockCtx.loadBase, "control: mock"),
        );
        try std.testing.expect(failing.has_induced_failure);
        try std.testing.expectEqual(@as(usize, 1), new_deinits);
        try std.testing.expectEqual(@as(usize, 0), existing_deinits);
        try std.testing.expectEqual(@as(usize, 1), probes.items.len);
        try std.testing.expectEqual(existing, probes.items[0]);
        try std.testing.expectEqual(@as(usize, 1), kernels.items.len);
    }
    try std.testing.expectEqual(@as(usize, 1), existing_deinits);
    try std.testing.expectEqual(@as(usize, 1), new_deinits);
    try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
    try std.testing.expectEqual(failing.allocations, failing.deallocations);
}
