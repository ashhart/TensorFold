//! `ling-synth DIR [--format bf16|fp8|nvfp4] [--layers N]`: a tiny Ling-3.1 checkpoint, every name from the weight map.
const std = @import("std");
const core = @import("core");
const ling = @import("ling_cuda");
const check = @import("check.zig");

const st = core.safetensors;
const New = core.shard_edit.New;
const Config = ling.config.Config;
const Quant = ling.config.Quant;
const Role = ling.names.Role;

/// A linear role's stored [n, k], from the config: the writer's own table, apart from the loader's.
fn linear(c: *const Config, role: Role) ?[2]usize {
    const d: usize = c.hidden;
    const n: usize = c.kda_heads * c.kda_dim;
    const h: usize = c.mla_heads;
    return switch (role) {
        .kda_q, .kda_k, .kda_v, .kda_f, .kda_g => .{ n, d },
        .kda_b => .{ c.kda_heads, d },
        .kda_o => .{ d, n },
        .mla_q => .{ h * c.qkDim(), d },
        .mla_q_a => .{ c.q_lora, d },
        .mla_q_b => .{ h * c.qkDim(), c.q_lora },
        .mla_kv_a => .{ c.kv_lora + c.rope_dim, d },
        .mla_kv_b => .{ h * (c.nope + c.v_dim), c.kv_lora },
        .mla_gate => .{ h, d },
        .mla_o => .{ d, h * c.v_dim },
        .dense_gate, .dense_up => .{ c.dense_inter, d },
        .dense_down => .{ d, c.dense_inter },
        .router => .{ c.experts, d },
        .expert_gate, .expert_up => .{ c.moe_inter, d },
        .expert_down => .{ d, c.moe_inter },
        .shared_gate, .shared_up => .{ c.shared_inter, d },
        .shared_down => .{ d, c.shared_inter },
        .mtp_eh => .{ d, 2 * d },
        .lm_head, .embed => .{ c.vocab, d },
        else => null,
    };
}

const Out = struct {
    a: std.mem.Allocator,
    rng: std.Random,
    c: *const Config,
    list: std.ArrayList(New) = .empty,
    bytes: usize = 0,

    fn add(o: *Out, name: []const u8, dtype: st.DType, shape: []const usize, bytes: []const u8) !void {
        try o.list.append(o.a, .{ .name = try o.a.dupe(u8, name), .dtype = dtype, .shape = try o.a.dupe(usize, shape), .bytes = bytes });
        o.bytes += bytes.len;
    }

    fn bf16s(o: *Out, count: usize, lo: f32, hi: f32) ![]u8 {
        const v = try o.a.alloc(u16, count);
        for (v) |*x| x.* = ling.weights.proj.toBf16(lo + o.rng.float(f32) * (hi - lo));
        return std.mem.sliceAsBytes(v);
    }

    fn f32s(o: *Out, count: usize, lo: f32, hi: f32) ![]u8 {
        const v = try o.a.alloc(f32, count);
        for (v) |*x| x.* = lo + o.rng.float(f32) * (hi - lo);
        return std.mem.sliceAsBytes(v);
    }

    /// One name the map emits: a weight in its module's format, a side tensor by its suffix, or a small tensor by role.
    pub fn put(o: *Out, name: []const u8, t: ling.names.Tensor) !void {
        const c = o.c;
        if (linear(c, t.role)) |nk| {
            const n, const k = nk;
            const module = name[0 .. std.mem.lastIndexOfScalar(u8, name, '.') orelse return error.BadName];
            const f = if (t.role == .embed) .bf16 else c.format(module);
            const s = 1.0 / @sqrt(@as(f32, @floatFromInt(k)));
            if (!t.scale) return switch (f) {
                .bf16 => o.add(name, .bf16, &.{ n, k }, try o.bf16s(n * k, -s, s)),
                .fp8_block128 => o.add(name, .f8_e4m3, &.{ n, k }, try o.codes(n * k, .e4m3)),
                .nvfp4 => o.add(name, .u8, &.{ n, k / 2 }, try o.codes(n * k / 2, .any)),
            };
            const tail = name[module.len + 1 ..];
            if (std.mem.eql(u8, tail, "weight_scale_inv")) return o.add(name, .f32, &.{ (n + 127) / 128, k / 128 }, try o.f32s((n + 127) / 128 * (k / 128), s / 4, s / 2));
            if (std.mem.eql(u8, tail, "weight_scale")) return o.add(name, .f8_e4m3, &.{ n, k / 16 }, try o.codes(n * k / 16, .scale));
            if (std.mem.eql(u8, tail, "weight_scale_2")) return o.add(name, .f32, &.{}, try o.f32s(1, s / 8, s / 4));
            if (std.mem.eql(u8, tail, "input_scale")) return o.add(name, .f32, &.{}, try o.f32s(1, 1, 1));
            return error.BadName;
        }
        const d: usize = c.hidden;
        const n: usize = c.kda_heads * c.kda_dim;
        return switch (t.role) {
            .attn_norm, .mlp_norm, .final_norm, .mtp_enorm, .mtp_hnorm, .mtp_final_norm => o.add(name, .bf16, &.{d}, try o.bf16s(d, 0.5, 1.5)),
            .kda_q_conv, .kda_k_conv, .kda_v_conv => o.add(name, .bf16, &.{ n, 1, c.conv }, try o.bf16s(n * c.conv, -0.5, 0.5)),
            .kda_a_log => o.add(name, .f32, &.{c.kda_heads}, try o.f32s(c.kda_heads, 0, 2.5)),
            .kda_dt_bias => o.add(name, .f32, &.{n}, try o.f32s(n, -0.5, 0.5)),
            .kda_o_norm => o.add(name, .bf16, &.{c.kda_dim}, try o.bf16s(c.kda_dim, 0.5, 1.5)),
            .mla_q_a_norm => o.add(name, .bf16, &.{c.q_lora}, try o.bf16s(c.q_lora, 0.5, 1.5)),
            .mla_kv_a_norm => o.add(name, .bf16, &.{c.kv_lora}, try o.bf16s(c.kv_lora, 0.5, 1.5)),
            .router_bias => o.add(name, .f32, &.{c.experts}, try o.f32s(c.experts, -0.1, 0.1)),
            else => error.BadName,
        };
    }

    /// Random bytes: any; e4m3 codes of magnitude 2^-3 to 3.75 (a forward stays in range); e4m3 scales 2^-3 to 2^1.
    fn codes(o: *Out, count: usize, what: enum { any, e4m3, scale }) ![]u8 {
        const v = try o.a.alloc(u8, count);
        for (v) |*b| b.* = switch (what) {
            .any => o.rng.int(u8),
            .e4m3 => (o.rng.int(u8) & 0x87) | (@as(u8, 4) + o.rng.uintLessThan(u8, 5)) << 3,
            .scale => 0x20 + o.rng.uintLessThan(u8, 0x28),
        };
        return v;
    }
};

fn formatOf(text: []const u8) !Quant {
    if (std.mem.eql(u8, text, "bf16")) return .bf16;
    if (std.mem.eql(u8, text, "fp8")) return .fp8_block128;
    if (std.mem.eql(u8, text, "nvfp4")) return .nvfp4;
    std.debug.print("--format takes bf16, fp8 or nvfp4, not {s}\n", .{text});
    return error.BadArgument;
}

pub fn run(gpu: check.Gpu, rest: []const [:0]const u8) !void {
    if (rest.len == 0) return error.MissingArgument;
    var format: Quant = .fp8_block128;
    var layers: u32 = 4;
    var i: usize = 1;
    while (i + 1 < rest.len) : (i += 2) {
        if (std.mem.eql(u8, rest[i], "--format")) format = try formatOf(rest[i + 1]) else if (std.mem.eql(u8, rest[i], "--layers")) layers = try std.fmt.parseInt(u32, rest[i + 1], 10) else return error.BadArgument;
    }
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const json = try ling.synth.configJson(a, format, layers);
    var why: ling.config.Why = .{};
    const c = ling.config.parse(a, json, &why) catch |e| {
        std.debug.print("the tiny config does not parse: {s}\n", .{why.text()});
        return e;
    };
    var prng = std.Random.DefaultPrng.init(0x11a6_3100 + @as(u64, @backingInt(format)));
    var o: Out = .{ .a = a, .rng = prng.random(), .c = &c };
    try ling.names.all(&c, &o);
    const dir = rest[0];
    try std.Io.Dir.cwd().createDirPath(gpu.io, dir);
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, "config.json" }), .data = json });
    try core.shard_edit.create(gpu.gpa, try std.fs.path.join(a, &.{ dir, "model.safetensors" }), &.{.{ "format", "pt" }}, o.list.items);
    // the file's names against the map: every name expected, nothing extra
    const all = try a.alloc([]const u8, o.list.items.len);
    for (o.list.items, all) |t, *n| n.* = t.name;
    const r = try ling.names.check(a, &c, all);
    try check.expect(r.missing.items.len == 0 and r.unknown.items.len == 0, "ling-synth names: {d} missing, {d} unaccounted", .{ r.missing.items.len, r.unknown.items.len });
    check.pass("ling-synth {t}: {d} layers + MTP, {d} tensors, {d} KiB in {s}", .{ format, layers, o.list.items.len, o.bytes >> 10, dir });
}
