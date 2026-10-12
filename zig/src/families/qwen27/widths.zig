//! Affine widths of the modules the engine loads, admitted against the projection kernels this chip runs.
const std = @import("std");
const core = @import("core");
const cfg = @import("config.zig");
const affine = @import("affine.zig");
const glue = @import("glue.zig");
const row = core.row_projection;
const lane = core.lane_projection;

/// One affine width, as core.quant.mlx.Width: `bits` per weight, a scale and bias a `group` of K.
pub const Width = struct { bits: u8, group: u16 };
/// The qualified format: today's kernels and launches, unchanged.
pub const uniform = Width{ .bits = 4, .group = 64 };

pub fn isUniform(w: Width) bool {
    return w.bits == uniform.bits and w.group == uniform.group;
}

/// A layer's projections; a layer stores entries for both mixers and uses those of its kind.
pub const Member = enum { gate, up, down, qkv, z, b, a, out, q, k, v, o };
const members = @backingInt(Member.o) + 1;

pub fn leaf(m: Member) []const u8 {
    return switch (m) {
        .gate => "mlp.gate_proj",
        .up => "mlp.up_proj",
        .down => "mlp.down_proj",
        .qkv => "linear_attn.in_proj_qkv",
        .z => "linear_attn.in_proj_z",
        .b => "linear_attn.in_proj_b",
        .a => "linear_attn.in_proj_a",
        .out => "linear_attn.out_proj",
        .q => "self_attn.q_proj",
        .k => "self_attn.k_proj",
        .v => "self_attn.v_proj",
        .o => "self_attn.o_proj",
    };
}

/// Output rows and input width of a member.
pub fn shape(c: cfg.Config, m: Member) [2]usize {
    return switch (m) {
        .gate, .up => .{ c.intermediate, c.hidden },
        .down => .{ c.hidden, c.intermediate },
        .qkv => .{ c.gdnQkvDim(), c.hidden },
        .z => .{ c.gdnValueDim(), c.hidden },
        .b, .a => .{ c.v_heads, c.hidden },
        .out => .{ c.hidden, c.gdnValueDim() },
        .q => .{ c.qDim(), c.hidden },
        .k, .v => .{ c.kvDim(), c.hidden },
        .o => .{ c.hidden, c.heads * c.head_dim },
    };
}

/// The device matrices of a layer: members fused into one output, in output-row order.
pub fn stacks(kind: cfg.Kind) []const []const Member {
    return switch (kind) {
        .linear => &.{ &.{ .gate, .up }, &.{.down}, &.{.qkv}, &.{ .z, .b, .a }, &.{.out} },
        .attention => &.{ &.{ .gate, .up }, &.{.down}, &.{.q}, &.{ .k, .v }, &.{.o} },
    };
}

/// Consecutive members of one width as [first, count]; a stack of one width is one run.
pub fn runs(ws: []const Width, out: [][2]usize) [][2]usize {
    var n: usize = 0;
    for (ws, 0..) |w, i| {
        if (n > 0 and std.meta.eql(ws[out[n - 1][0]], w)) {
            out[n - 1][1] += 1;
        } else {
            out[n] = .{ i, 1 };
            n += 1;
        }
    }
    return out[0..n];
}

pub const Widths = struct {
    allocator: std.mem.Allocator,
    layers: [][members]Width,
    embed: Width,
    head: Width,

    /// Resolves only loaded modules, so draft-head (MTP) and vision overrides never decide admission.
    pub fn resolve(a: std.mem.Allocator, f: *const affine.Formats, c: cfg.Config) !Widths {
        if ((try f.resolve(null)) == null) return error.MissingAffineFormat;
        const layers = try a.alloc([members]Width, c.layers);
        errdefer a.free(layers);
        var name: [96]u8 = undefined;
        for (layers, 0..) |*l, i| {
            l.* = @splat(uniform);
            for (stacks(c.kind(i))) |stack| for (stack) |m| {
                l[@backingInt(m)] = try one(f, try std.fmt.bufPrint(&name, "model.layers.{d}.{s}", .{ i, leaf(m) }));
            };
        }
        return .{ .allocator = a, .layers = layers, .embed = try one(f, "model.embed_tokens"), .head = try one(f, "lm_head") };
    }

    /// Every module 4-bit g64, for synthetic graphs.
    pub fn uniformAll(a: std.mem.Allocator, layers: usize) !Widths {
        const all = try a.alloc([members]Width, layers);
        for (all) |*l| l.* = @splat(uniform);
        return .{ .allocator = a, .layers = all, .embed = uniform, .head = uniform };
    }

    pub fn deinit(w: *Widths) void {
        w.allocator.free(w.layers);
    }

    pub fn of(w: *const Widths, layer: usize, m: Member) Width {
        return w.layers[layer][@backingInt(m)];
    }
};

fn one(f: *const affine.Formats, path: []const u8) !Width {
    const s = (try f.resolve(path)) orelse return error.UnqualifiedQwenFormat;
    return .{ .bits = s.bits, .group = @intCast(s.group_size) };
}

/// What the chip's projection kernels run: the lane kernels on tensor units, the row kernels before them.
pub const Chip = struct { tensor_units: bool, row_bits: []const u8 = &row.bit_widths };

/// The first module admission refused, for the load's error message.
pub const Why = struct { layer: ?usize = null, module: []const u8 = "", width: Width = uniform, reason: anyerror = error.None };

pub fn admit(w: *const Widths, c: cfg.Config, chip: Chip, why: *Why) !void {
    var buf: [3][2]usize = undefined;
    for (0..c.layers) |i| for (stacks(c.kind(i))) |stack| {
        var ws: [3]Width = undefined;
        for (stack, 0..) |m, j| ws[j] = w.of(i, m);
        const list = runs(ws[0..stack.len], &buf);
        if (list.len == 1 and isUniform(ws[0])) continue;
        for (list) |r| {
            var n: usize = 0;
            for (stack[r[0]..][0..r[1]]) |m| n += shape(c, m)[0];
            check(chip, n, shape(c, stack[r[0]])[1], ws[r[0]]) catch |err| {
                why.* = .{ .layer = i, .module = leaf(stack[r[0]]), .width = ws[r[0]], .reason = err };
                return error.UnqualifiedQwenFormat;
            };
        }
    };
    glue.checkEmbed(.{ .width = @intCast(c.hidden), .rows = 1, .vocab = @intCast(c.vocab), .bits = w.embed.bits, .group = w.embed.group, .metadata = .bf16 }) catch |err| {
        why.* = .{ .module = "embed_tokens", .width = w.embed, .reason = err };
        return error.UnqualifiedQwenFormat;
    };
    if (!isUniform(w.head)) check(chip, c.vocab, c.hidden, w.head) catch |err| {
        why.* = .{ .module = "lm_head", .width = w.head, .reason = err };
        return error.UnqualifiedQwenFormat;
    };
}

/// One run's kernel: 4/64 keeps today's choice; other widths take the non-reg lane kernel or the row kernel.
fn check(chip: Chip, n: usize, k: usize, w: Width) !void {
    _ = try @import("projection.zig").split(n, k);
    // The lane sums and odd word counts assume group 64 off the 4-bit kernel.
    if (chip.tensor_units and w.bits != 4 and w.group != 64) return error.UnsupportedLaneGroup;
    if (chip.tensor_units) return (lane.Layout{ .n = n, .k = k, .format = .{ .bits = w.bits, .group = w.group }, .precompute_sums = isUniform(w), .reg = isUniform(w) }).validate();
    if (isUniform(w) and n % 8 == 0) return;
    if (std.mem.indexOfScalar(u8, chip.row_bits, w.bits) == null) return error.UnsupportedRowBits;
    return (row.Weights{ .w = undefined, .scales = undefined, .biases = undefined, .n = n, .k = k, .group = w.group }).validate();
}

const all_rows = [_]u8{ 2, 3, 4, 5, 6, 8 };

fn testConfig(a: std.mem.Allocator, quantization: []const u8) ![]u8 {
    return std.fmt.allocPrint(a,
        \\{{"model_type":"qwen3_5","text_config":{{"model_type":"qwen3_5_text","hidden_size":5120,"intermediate_size":17408,
        \\"num_hidden_layers":64,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"vocab_size":248320,
        \\"linear_num_key_heads":16,"linear_num_value_heads":48,"linear_key_head_dim":128,"linear_value_head_dim":128,
        \\"linear_conv_kernel_dim":4,"partial_rotary_factor":0.25,"eos_token_id":248044}},"quantization":{s}}}
    , .{quantization});
}

fn testAdmit(quantization: []const u8, chip: Chip, why: *Why) !void {
    const a = std.testing.allocator;
    const text = try testConfig(a, quantization);
    defer a.free(text);
    const c = try cfg.parse(a, text, null);
    var f = try affine.Formats.init(a, text);
    defer f.deinit();
    var w = try Widths.resolve(a, &f, c);
    defer w.deinit();
    return admit(&w, c, chip, why);
}

/// TensorFold/Qwen3.8-27B-oQ4 override shapes: 4-bit g64, 5-bit GDN, MLP and attention members, one 6-bit v.
const oq4 =
    \\{"group_size":64,"bits":4,"mode":"affine",
    \\"language_model.model.layers.37.linear_attn.in_proj_a":{"group_size":64,"bits":5},
    \\"language_model.model.layers.36.linear_attn.in_proj_b":{"group_size":64,"bits":5},
    \\"language_model.model.layers.26.linear_attn.in_proj_z":{"group_size":64,"bits":5},
    \\"language_model.model.layers.62.linear_attn.out_proj":{"group_size":64,"bits":5},
    \\"language_model.model.layers.25.mlp.down_proj":{"group_size":64,"bits":5},
    \\"language_model.model.layers.27.self_attn.k_proj":{"group_size":64,"bits":5},
    \\"language_model.model.layers.27.self_attn.o_proj":{"group_size":64,"bits":5},
    \\"language_model.model.layers.27.self_attn.v_proj":{"group_size":64,"bits":6}}
;

/// The override shapes of TensorFold/Qwen3.8-27B-oQ2: 2-bit g64 with an 8-bit embedding and 3/5/6-bit members.
const oq2 =
    \\{"group_size":64,"bits":2,"mode":"affine",
    \\"language_model.model.embed_tokens":{"group_size":64,"bits":8},
    \\"language_model.model.layers.50.linear_attn.in_proj_a":{"group_size":64,"bits":3},
    \\"language_model.model.layers.44.linear_attn.in_proj_a":{"group_size":64,"bits":5},
    \\"language_model.model.layers.42.linear_attn.in_proj_b":{"group_size":64,"bits":5},
    \\"language_model.model.layers.12.linear_attn.in_proj_z":{"group_size":64,"bits":5},
    \\"language_model.model.layers.10.mlp.down_proj":{"group_size":64,"bits":5},
    \\"language_model.model.layers.35.self_attn.k_proj":{"group_size":64,"bits":5},
    \\"language_model.model.layers.35.self_attn.v_proj":{"group_size":64,"bits":6}}
;

test "uniform 4-bit g64 admits on both chip kinds and ignores draft-head and vision overrides" {
    var why: Why = .{};
    const q = "{\"bits\":4,\"group_size\":64,\"mtp.layers.0.mlp.gate_proj\":{\"bits\":8,\"group_size\":64},\"vision_tower.blocks.0.attn.qkv\":false}";
    try testAdmit(q, .{ .tensor_units = false }, &why);
    try testAdmit(q, .{ .tensor_units = true }, &why);
}

test "oQ4 and oQ2 formats admit once the row kernels unpack every width they declare" {
    var why: Why = .{};
    try testAdmit(oq4, .{ .tensor_units = false, .row_bits = &all_rows }, &why);
    try testAdmit(oq2, .{ .tensor_units = false, .row_bits = &all_rows }, &why);
}

test "a width the row kernels do not unpack refuses naming the module, never guessing four bits" {
    var why: Why = .{};
    try std.testing.expectError(error.UnqualifiedQwenFormat, testAdmit(oq4, .{ .tensor_units = false, .row_bits = &.{ 2, 4, 6, 8 } }, &why));
    try std.testing.expectEqual(@as(?usize, 25), why.layer);
    try std.testing.expectEqualStrings("mlp.down_proj", why.module);
    try std.testing.expectEqual(Width{ .bits = 5, .group = 64 }, why.width);
    try std.testing.expectError(error.UnqualifiedQwenFormat, testAdmit(oq2, .{ .tensor_units = false, .row_bits = &.{ 2, 4, 6, 8 } }, &why));
    try std.testing.expectEqual(@as(u8, 5), why.width.bits);
}

test "a dense projection and an unsupported global group refuse" {
    var why: Why = .{};
    try std.testing.expectError(error.UnqualifiedQwenFormat, testAdmit("{\"bits\":4,\"group_size\":64,\"model.layers.3.self_attn.q_proj\":false}", .{ .tensor_units = false }, &why));
    try std.testing.expectError(error.BadAffine, testAdmit("{\"bits\":4,\"group_size\":16}", .{ .tensor_units = false }, &why));
}

test "fused members split into runs of one width" {
    var buf: [3][2]usize = undefined;
    const four = uniform;
    const five = Width{ .bits = 5, .group = 64 };
    try std.testing.expectEqualSlices([2]usize, &.{.{ 0, 3 }}, runs(&.{ four, four, four }, &buf));
    try std.testing.expectEqualSlices([2]usize, &.{ .{ 0, 1 }, .{ 1, 2 } }, runs(&.{ five, four, four }, &buf));
    try std.testing.expectEqualSlices([2]usize, &.{ .{ 0, 1 }, .{ 1, 1 }, .{ 2, 1 } }, runs(&.{ four, five, four }, &buf));
}

test "resolved widths follow overrides under any checkpoint prefix" {
    const a = std.testing.allocator;
    const text = try testConfig(a, oq4);
    defer a.free(text);
    const c = try cfg.parse(a, text, null);
    var f = try affine.Formats.init(a, text);
    defer f.deinit();
    var w = try Widths.resolve(a, &f, c);
    defer w.deinit();
    try std.testing.expectEqual(Width{ .bits = 6, .group = 64 }, w.of(27, .v));
    try std.testing.expectEqual(Width{ .bits = 5, .group = 64 }, w.of(27, .k));
    try std.testing.expectEqual(uniform, w.of(27, .q));
    try std.testing.expectEqual(uniform, w.head);
}
