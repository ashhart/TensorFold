//! Affine widths of the modules the engine loads, admitted against the projection kernels this chip runs.
const std = @import("std");
const core = @import("core");
const cfg = @import("config.zig");
const affine = @import("affine.zig");
const glue = @import("glue.zig");
const row = core.row_projection;
const lane = core.lane_projection;

/// Affine or native/AWQ width of one module.
pub const Kind = enum(u8) { affine, bf16, awq };
pub const Width = struct { bits: u8, group: u16, kind: Kind = .affine };
/// The qualified format: today's kernels and launches, unchanged.
pub const uniform = Width{ .bits = 4, .group = 64 };
/// Unquantized BF16 residual tensor (admitted only alongside a TFQ3 sidecar).
pub const bf16 = Width{ .bits = 16, .group = 0 };
/// Group tag marking TFQ3 (3/4/5-bit 2D-AffineCoset/half4) tensors stored in `.tffpw` shards.
pub const tfq3_group: u16 = 512;

pub fn isUniform(w: Width) bool {
    return w.kind == .affine and w.bits == uniform.bits and w.group == uniform.group;
}

pub fn isBf16(w: Width) bool {
    return w.bits == bf16.bits and w.group == bf16.group;
}

pub fn isTfq3(w: Width) bool {
    return w.group == tfq3_group and w.bits >= 3 and w.bits <= 5;
}

/// Callback interface to query TFQ3 sidecar bit widths without coupling `widths.zig` to Metal.
pub const Tfq3Lookup = struct {
    ctx: *const anyopaque,
    bitsFn: *const fn (*const anyopaque, []const u8, usize, usize) ?u8,

    pub fn bits(l: Tfq3Lookup, module: []const u8, n: usize, k: usize) ?u8 {
        return l.bitsFn(l.ctx, module, n, k);
    }
};

/// A layer's projections; a layer stores entries for both mixers and uses those of its kind.
pub const Member = enum { gate, up, down, qkv, z, b, a, out, q, k, v, o, router, shared_gate, egate, eup, edown };
pub const members = @backingInt(Member.edown) + 1;
/// The routed block's members: the router, the shared expert's gate and the stacked experts.
pub const routed = [_]Member{ .router, .shared_gate, .egate, .eup, .edown };
/// The widths the kernel contract checks the routed block's kernels at, in groups of `routed_group`.
pub const routed_bits = [_]u8{ 4, 8 };
pub const routed_group = 64;

/// A member's module under its layer; with experts, `gate`, `up` and `down` are the shared expert's.
pub fn leaf(c: cfg.Config, m: Member) []const u8 {
    return switch (m) {
        .gate => if (c.moe()) "mlp.shared_expert.gate_proj" else "mlp.gate_proj",
        .up => if (c.moe()) "mlp.shared_expert.up_proj" else "mlp.up_proj",
        .down => if (c.moe()) "mlp.shared_expert.down_proj" else "mlp.down_proj",
        .router => "mlp.gate",
        .shared_gate => "mlp.shared_expert_gate",
        .egate => "mlp.switch_mlp.gate_proj",
        .eup => "mlp.switch_mlp.up_proj",
        .edown => "mlp.switch_mlp.down_proj",
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

/// Output rows and input width of a member (one expert's, for the stacked experts).
pub fn shape(c: cfg.Config, m: Member) [2]usize {
    return switch (m) {
        .gate, .up => .{ c.intermediate, c.hidden },
        .down => .{ c.hidden, c.intermediate },
        .router => .{ c.experts, c.hidden },
        .shared_gate => .{ 1, c.hidden },
        .egate, .eup => .{ c.moe_intermediate, c.hidden },
        .edown => .{ c.hidden, c.moe_intermediate },
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

/// A stack's products as [first, count]: one for a stack of one width, else one a member, as the reference runs them.
pub fn runs(ws: []const Width, out: [][2]usize) [][2]usize {
    for (ws[1..]) |w| if (!std.meta.eql(ws[0], w)) {
        for (out[0..ws.len], 0..) |*o, i| o.* = .{ i, 1 };
        return out[0..ws.len];
    };
    out[0] = .{ 0, ws.len };
    return out[0..1];
}

pub const Widths = struct {
    allocator: std.mem.Allocator,
    layers: [][members]Width,
    embed: Width,
    head: Width,

    /// Resolves only loaded modules, so draft-head (MTP) and vision overrides never decide admission.
    pub fn resolve(a: std.mem.Allocator, f: *const affine.Formats, c: cfg.Config) !Widths {
        if ((try f.resolve(null)) == null) return error.MissingAffineFormat;
        const n = c.physicalLayers();
        const layers = try a.alloc([members]Width, n);
        errdefer a.free(layers);
        var name: [96]u8 = undefined;
        for (layers, 0..) |*l, i| {
            l.* = @splat(uniform);
            const k = if (c.looped) c.physKind(i) else c.kind(i);
            for (stacks(k)) |stack| for (stack) |m| {
                l[@backingInt(m)] = try one(f, try std.fmt.bufPrint(&name, "model.layers.{d}.{s}", .{ i, leaf(c, m) }));
            };
            if (c.moe()) for (routed) |m| {
                l[@backingInt(m)] = try one(f, try std.fmt.bufPrint(&name, "model.layers.{d}.{s}", .{ i, leaf(c, m) }));
            };
        }
        const embed: Width = if (try f.resolve("model.embed_tokens")) |s|
            .{ .bits = s.bits, .group = @intCast(s.group_size) }
        else if (c.looped)
            .{ .bits = 16, .group = 0, .kind = .bf16 }
        else
            return error.UnqualifiedQwenFormat;
        return .{ .allocator = a, .layers = layers, .embed = embed, .head = if (c.tied_head) embed else try one(f, "lm_head") };
    }

    /// Resolve native BF16 or AWQ w4g128 on looped models, or MLX affine everywhere else.
    pub fn resolveModel(a: std.mem.Allocator, json: []const u8, c: cfg.Config) !Widths {
        if (!c.looped) {
            var formats = try affine.Formats.init(a, json);
            defer formats.deinit();
            return Widths.resolve(a, &formats, c);
        }
        if (comptime !@hasDecl(@import("core"), "sensenova_loop")) return error.UnsupportedLoopedWidths;
        const parsed = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer parsed.deinit();
        if (parsed.value != .object) return error.BadConfig;
        const awq_mod = @import("core").sensenova_loop.awq;
        if (try awq_mod.parseConfig(a, parsed.value.object)) |ac_const| {
            var ac = ac_const;
            defer ac.deinit(a);
            const layers = try a.alloc([members]Width, c.physicalLayers());
            errdefer a.free(layers);
            var name: [128]u8 = undefined;
            const native_bf16: Width = .{ .bits = 16, .group = 0, .kind = .bf16 };
            const w4: Width = .{ .bits = ac.bits, .group = @intCast(ac.group_size), .kind = .awq };
            for (layers, 0..) |*l, i| {
                l.* = @splat(native_bf16);
                for (stacks(c.physKind(i))) |stack| for (stack) |m| {
                    const p = try std.fmt.bufPrint(&name, "model.language_model.layers.{d}.{s}", .{ i, leaf(c, m) });
                    l[@backingInt(m)] = if (ac.isQuantized(p)) w4 else native_bf16;
                };
            }
            if (ac.isQuantized("model.language_model.embed_tokens")) return error.UnqualifiedQwenFormat;
            return .{ .allocator = a, .layers = layers, .embed = native_bf16, .head = if (ac.isQuantized("lm_head")) w4 else native_bf16 };
        }
        var formats = affine.Formats.init(a, json) catch |err| switch (err) {
            error.NotAffine => return error.UnqualifiedQwenFormat,
            else => return err,
        };
        defer formats.deinit();
        if (formats.quant != null) return Widths.resolve(a, &formats, c);
        const native_bf16: Width = .{ .bits = 16, .group = 0, .kind = .bf16 };
        const layers = try a.alloc([members]Width, c.physicalLayers());
        for (layers) |*l| l.* = @splat(native_bf16);
        return .{ .allocator = a, .layers = layers, .embed = native_bf16, .head = native_bf16 };
    }
    /// Resolves a dense checkpoint with a TFQ3 sidecar: TFQ3 tensors first, then MLX affine, then BF16 residuals.
    pub fn resolveWithTfq3(a: std.mem.Allocator, f: *const affine.Formats, c: cfg.Config, tfq3: Tfq3Lookup) !Widths {
        if (c.moe()) return error.UnsupportedQuantization;
        const layers = try a.alloc([members]Width, c.layers);
        errdefer a.free(layers);
        var name: [96]u8 = undefined;
        for (layers, 0..) |*l, i| {
            l.* = @splat(uniform);
            for (stacks(c.kind(i))) |stack| for (stack) |m| {
                const path = try std.fmt.bufPrint(&name, "model.layers.{d}.{s}", .{ i, leaf(c, m) });
                const sh = shape(c, m);
                l[@backingInt(m)] = try oneTfq3(f, tfq3, path, sh[0], sh[1]);
            };
        }
        const embed = try oneResidual(f, "model.embed_tokens");
        const head = if (c.tied_head) embed else try oneTfq3(f, tfq3, "lm_head", c.vocab, c.hidden);
        return .{ .allocator = a, .layers = layers, .embed = embed, .head = head };
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

    /// 0 when every module is 4-bit g64, else a nonzero hash of every module's width.
    pub fn tag(w: *const Widths) u64 {
        var h = std.hash.Wyhash.init(0x27);
        var mixed = false;
        for (w.layers) |l| for (l) |x| {
            mixed = mixed or !isUniform(x);
            if (x.kind != .affine) h.update(&.{@backingInt(x.kind)});
            h.update(&.{ x.bits, @as(u8, @truncate(x.group)), @as(u8, @truncate(x.group >> 8)) });
        };
        for ([_]Width{ w.embed, w.head }) |x| {
            mixed = mixed or !isUniform(x);
            if (x.kind != .affine) h.update(&.{@backingInt(x.kind)});
            h.update(&.{ x.bits, @as(u8, @truncate(x.group)), @as(u8, @truncate(x.group >> 8)) });
        }
        return if (mixed) h.final() | 1 else 0;
    }
};

fn one(f: *const affine.Formats, path: []const u8) !Width {
    const s = (try f.resolve(path)) orelse return error.UnqualifiedQwenFormat;
    return .{ .bits = s.bits, .group = @intCast(s.group_size) };
}

fn oneResidual(f: *const affine.Formats, path: []const u8) !Width {
    if (try f.resolve(path)) |s| return .{ .bits = s.bits, .group = @intCast(s.group_size) };
    return bf16;
}

fn oneTfq3(f: *const affine.Formats, tfq3: Tfq3Lookup, path: []const u8, n: usize, k: usize) !Width {
    if (tfq3.bits(path, n, k)) |b| return .{ .bits = b, .group = tfq3_group };
    return oneResidual(f, path);
}

/// The timed window costs' model shape; mixed widths add their tag, so checkpoints of one shape keep their own costs.
pub fn costShape(c: cfg.Config, tag: u64, buf: []u8, rows: usize, drafted: bool) ![]const u8 {
    const s = try std.fmt.bufPrint(buf, "{d}/{d}/{d}/{d}/{d}/{d}", .{ c.layers, c.hidden, c.intermediate, c.vocab, rows, @intFromBool(drafted) });
    if (tag == 0 and !c.looped) return s;
    const t = if (tag != 0) s.len + (try std.fmt.bufPrint(buf[s.len..], "/w{x:0>16}", .{tag})).len else s.len;
    if (!c.looped) return buf[0..t];
    return buf[0 .. t + (try std.fmt.bufPrint(buf[t..], "/sn2", .{})).len];
}

/// What the chip's projection kernels run: lane kernels on tensor units for whole 32-row tiles, else the row kernels.
pub const Chip = struct { tensor_units: bool, row_bits: []const u8 = &row.bit_widths };

/// The first module admission refused, for the load's error message.
pub const Why = struct { layer: ?usize = null, module: []const u8 = "", width: Width = uniform, reason: anyerror = error.None };

pub fn admit(w: *const Widths, c: cfg.Config, chip: Chip, why: *Why) !void {
    return admitImpl(w, c, chip, why, false);
}

/// Admission for a dense checkpoint backed by a validated TFQ3 sidecar (BF16 or MLX affine residuals).
pub fn admitTfq3(w: *const Widths, c: cfg.Config, chip: Chip, why: *Why) !void {
    if (c.moe()) return error.UnsupportedQuantization;
    return admitImpl(w, c, chip, why, true);
}

fn admitImpl(w: *const Widths, c: cfg.Config, chip: Chip, why: *Why, allow_tfq3: bool) !void {
    var buf: [3][2]usize = undefined;
    for (0..w.layers.len) |i| for (stacks(if (c.looped) c.physKind(i) else c.kind(i))) |stack| {
        var ws: [3]Width = undefined;
        for (stack, 0..) |m, j| ws[j] = w.of(i, m);
        const list = runs(ws[0..stack.len], &buf);
        if (!allow_tfq3 and list.len == 1 and isUniform(ws[0])) continue;
        for (list) |r| {
            var n: usize = 0;
            for (stack[r[0]..][0..r[1]]) |m| n += shape(c, m)[0];
            const k = shape(c, stack[r[0]])[1];
            const check_res = if (allow_tfq3) checkTfq3OrResidual(chip, n, k, ws[r[0]]) else check(chip, n, k, ws[r[0]]);
            check_res catch |err| {
                why.* = .{ .layer = i, .module = leaf(c, stack[r[0]]), .width = ws[r[0]], .reason = err };
                return error.UnqualifiedQwenFormat;
            };
        }
    };
    if (c.moe()) for (0..c.layers) |i| {
        for (routed) |m| {
            const x = w.of(i, m);
            checkRouted(c, chip, m, x) catch |err| {
                why.* = .{ .layer = i, .module = leaf(c, m), .width = x, .reason = err };
                return error.UnqualifiedQwenFormat;
            };
        }
        // the router matrix stacks the shared gate's row under the experts': one width for both
        const r = w.of(i, .router);
        const g = w.of(i, .shared_gate);
        if (r.bits != g.bits or r.group != g.group) {
            why.* = .{ .layer = i, .module = leaf(c, .shared_gate), .width = g, .reason = error.SharedGateWidth };
            return error.UnqualifiedQwenFormat;
        }
        // decode runs the shared expert as the experts' extra slot: one width for each pair
        for ([_][2]Member{ .{ .gate, .egate }, .{ .up, .eup }, .{ .down, .edown } }) |pair| {
            const x = w.of(i, pair[0]);
            const e = w.of(i, pair[1]);
            if (x.bits != e.bits or x.group != e.group) {
                why.* = .{ .layer = i, .module = leaf(c, pair[0]), .width = x, .reason = error.SharedExpertWidth };
                return error.UnqualifiedQwenFormat;
            }
        }
    };
    if (w.embed.kind == .bf16 or (allow_tfq3 and isBf16(w.embed))) {
        if (c.hidden == 0 or c.vocab == 0) {
            why.* = .{ .module = "embed_tokens", .width = w.embed, .reason = error.BadEmbedding };
            return error.UnqualifiedQwenFormat;
        }
    } else glue.checkEmbed(.{ .width = @intCast(c.hidden), .rows = 1, .vocab = @intCast(c.vocab), .bits = w.embed.bits, .group = w.embed.group, .metadata = .bf16 }) catch |err| {
        why.* = .{ .module = "embed_tokens", .width = w.embed, .reason = err };
        return error.UnqualifiedQwenFormat;
    };
    if (allow_tfq3 or !isUniform(w.head)) {
        const head_res = if (allow_tfq3) checkTfq3OrResidual(chip, c.vocab, c.hidden, w.head) else check(chip, c.vocab, c.hidden, w.head);
        head_res catch |err| {
            why.* = .{ .module = "lm_head", .width = w.head, .reason = err };
            return error.UnqualifiedQwenFormat;
        };
    }
}

fn checkTfq3OrResidual(chip: Chip, n: usize, k: usize, w: Width) !void {
    if (isTfq3(w)) return if (n == 0 or k == 0 or n % 128 != 0 or k % 128 != 0) error.BadProjectionShape else {};
    if (isBf16(w)) return if (n == 0 or k == 0 or k % 16 != 0) error.BadProjectionShape else {};
    return check(chip, n, k, w);
}

/// One run's kernel: 4-, 5-, and 6-bit g64 take reg lanes; other widths take non-reg lanes or row kernels.
fn check(chip: Chip, n: usize, k: usize, w: Width) !void {
    if (w.kind == .bf16) {
        if (n == 0 or k == 0 or k % 8 != 0) return error.BadProjectionShape;
        return;
    }
    if (w.kind == .awq) {
        if (w.bits != 4 or w.group != 128 or n == 0 or k == 0 or n % 8 != 0 or k % 128 != 0) return error.UnsupportedLaneGroup;
        return;
    }
    _ = try @import("projection.zig").split(n, k);
    // The lane sums and odd word counts assume group 64 off the 4-bit kernel.
    if (chip.tensor_units and w.bits != 4 and w.group != 64) return error.UnsupportedLaneGroup;
    const reg = (w.bits == 4 or w.bits == 5 or w.bits == 6) and w.group == 64;
    if (chip.tensor_units and n % 32 == 0) return (lane.Layout{ .n = n, .k = k, .format = .{ .bits = w.bits, .group = w.group }, .precompute_sums = reg or (w.bits == 8 and w.group == 64), .reg = reg }).validate();
    if (isUniform(w) and n % 8 == 0) return;
    if (std.mem.indexOfScalar(u8, chip.row_bits, w.bits) == null) return error.UnsupportedRowBits;
    return (row.Weights{ .w = undefined, .scales = undefined, .biases = undefined, .n = n, .k = k, .group = w.group }).validate();
}

/// The routed block's kernels: the router's row matrix; the experts' row kernels and (tensor units) affine_mm gathers.
fn checkRouted(c: cfg.Config, chip: Chip, m: Member, w: Width) !void {
    const s = shape(c, m);
    if (std.mem.indexOfScalar(u8, &routed_bits, w.bits) == null or w.group != routed_group) return error.UncheckedRoutedWidth;
    if (std.mem.indexOfScalar(u8, chip.row_bits, w.bits) == null) return error.UnsupportedRowBits;
    const n = if (m == .router or m == .shared_gate) routerRows(c.experts) else s[0];
    try (row.Weights{ .w = undefined, .scales = undefined, .biases = undefined, .n = n, .k = s[1], .group = w.group, .bits = w.bits }).validate();
    if (m == .egate or m == .eup or m == .edown) {
        if (chip.tensor_units and (s[0] % 64 != 0 or s[1] % 64 != 0)) return error.UnsupportedExpertShape;
    }
}

/// The router matrix's rows: the experts, the shared gate, then zero rows to a multiple of 64 (whole weight tiles).
pub fn routerRows(experts: usize) usize {
    return (experts + 1 + 63) / 64 * 64;
}

test "a 4/64 run off the 32-row lanes, such as a lone 48-row in_proj_b, takes the row kernels on tensor units" {
    try check(.{ .tensor_units = true }, 48, 5120, uniform);
    try check(.{ .tensor_units = true }, 64, 5120, uniform);
}

test {
    _ = @import("widths_test.zig");
}
