//! A Qwen3.5 / Qwen3.6 checkpoint on the host: the Python ROCm loader's tensor names, fallbacks and conversions.

const std = @import("std");
const config = @import("config.zig");
const table = @import("table.zig");
const host = @import("host.zig");
const projection = @import("projection.zig");
const experts = @import("experts.zig");

const Tensor = table.Tensor;
const Table = table.Table;
/// The text tower's key prefix, as the MLX conversions name it.
pub const prefix = "language_model.model.";

pub const Checkpoint = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    config: config.Config,
    files: table.Files,
    main: Table,

    /// Reads config.json and maps the model's own safetensors files.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Checkpoint {
        var cfg = try config.Config.read(gpa, io, dir);
        errdefer cfg.deinit();
        var files = try table.Files.list(gpa, io, dir);
        errdefer files.deinit();
        if (files.model.len == 0) return error.NoSafetensors;
        const main = try Table.open(gpa, io, files.model, "", cfg.quant);
        return .{ .gpa = gpa, .io = io, .config = cfg, .files = files, .main = main };
    }

    pub fn close(c: *Checkpoint) void {
        c.main.close(c.io);
        c.files.deinit();
        c.config.deinit();
        c.* = undefined;
    }

    pub fn spec(c: *const Checkpoint) config.Spec {
        return c.config.spec;
    }

    /// The embedding, MLX affine words or a float table the row gather reads; `a` holds any converted tables.
    pub fn embed(c: *const Checkpoint, a: std.mem.Allocator) !host.Projection {
        const p = try projection.projection(a, &c.main, prefix ++ "embed_tokens");
        if (p.rows() != c.spec().vocab) return error.UnexpectedTensor;
        return p;
    }

    pub fn finalNorm(c: *const Checkpoint, a: std.mem.Allocator) !Tensor {
        return projection.float(a, &c.main, prefix ++ "norm.weight");
    }

    /// The output head, or null when it is the embedding (`tie_word_embeddings`).
    pub fn head(c: *const Checkpoint, a: std.mem.Allocator) !?host.Projection {
        if (c.config.tied) return null;
        return try projection.projection(a, &c.main, "language_model.lm_head");
    }

    /// Layer `index`, read and converted; free it with `deinit`.
    pub fn layer(c: *const Checkpoint, index: usize) !host.Layer {
        const s = c.spec();
        if (index >= s.n_layers) return error.InvalidModel;
        var arena: std.heap.ArenaAllocator = .init(c.gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const t = &c.main;
        var name: [128]u8 = undefined;
        const base = std.fmt.bufPrint(&name, prefix ++ "layers.{d}.", .{index}) catch unreachable;
        var scratch: [160]u8 = undefined;
        const input_norm = try projection.float(a, t, projection.join(&scratch, &.{ base, "input_layernorm.weight" }));
        const post_norm = try projection.float(a, t, projection.join(&scratch, &.{ base, "post_attention_layernorm.weight" }));
        const mlp = try readMlp(a, t, projection.join(&scratch, &.{ base, "mlp." }), s);
        if (s.full(index)) {
            const attn = projection.join(&scratch, &.{ base, "self_attn." });
            const body: host.Body = .{ .full = .{
                .input_norm = input_norm,
                .post_norm = post_norm,
                .q = try sub(a, t, attn, "q_proj"),
                .k = try sub(a, t, attn, "k_proj"),
                .v = try sub(a, t, attn, "v_proj"),
                .o = try sub(a, t, attn, "o_proj"),
                .q_norm = try subFloat(a, t, attn, "q_norm.weight"),
                .k_norm = try subFloat(a, t, attn, "k_norm.weight"),
                .mlp = mlp,
            } };
            return .{ .arena = arena, .body = body };
        }
        const lin = projection.join(&scratch, &.{ base, "linear_attn." });
        // the arena is copied into the result last: allocations made after the copy would be lost
        const body: host.Body = .{ .linear = .{
            .input_norm = input_norm,
            .post_norm = post_norm,
            .qkv = try sub(a, t, lin, "in_proj_qkv"),
            .z = try sub(a, t, lin, "in_proj_z"),
            .a = try sub(a, t, lin, "in_proj_a"),
            .b = try sub(a, t, lin, "in_proj_b"),
            .conv = try subConv(a, t, lin),
            .a_log = try subFloat(a, t, lin, "A_log"),
            .dt_bias = try subFloat(a, t, lin, "dt_bias"),
            .gnorm = try subFloat(a, t, lin, "norm.weight"),
            .out = try sub(a, t, lin, "out_proj"),
            .mlp = mlp,
        } };
        return .{ .arena = arena, .body = body };
    }

    /// The MTP layer from `mtp*.safetensors` or the model's own shards; null when absent.
    pub fn mtp(c: *const Checkpoint) !?host.Mtp {
        const side = c.files.side;
        const shards = if (side.len > 0) side else c.files.model;
        if (shards.len == 0) return null;
        const own = c.main.has("mtp.norm_e.weight") or c.main.has("mtp.pre_fc_norm_embedding.weight");
        var opened: ?Table = null;
        errdefer if (opened) |*o| o.close(c.io);
        var t = &c.main;
        if (side.len > 0 or !own) {
            opened = try Table.open(c.gpa, c.io, shards, "language_model.", c.config.quant);
            t = &opened.?;
        }
        if (side.len == 0 and !t.has("mtp.norm_e.weight") and !t.has("mtp.pre_fc_norm_embedding.weight")) {
            if (opened) |*o| o.close(c.io);
            return null;
        }
        var arena: std.heap.ArenaAllocator = .init(c.gpa);
        errdefer arena.deinit();
        var m = try readMtp(arena.allocator(), t, c.spec());
        m.arena = arena;
        m.owned = opened;
        m.io = c.io;
        return m;
    }
};

fn sub(a: std.mem.Allocator, t: *const Table, base: []const u8, name: []const u8) !host.Projection {
    var buf: [192]u8 = undefined;
    return projection.projection(a, t, projection.join(&buf, &.{ base, name }));
}

fn subFloat(a: std.mem.Allocator, t: *const Table, base: []const u8, name: []const u8) !Tensor {
    var buf: [192]u8 = undefined;
    return projection.float(a, t, projection.join(&buf, &.{ base, name }));
}

fn subConv(a: std.mem.Allocator, t: *const Table, base: []const u8) !Tensor {
    var buf: [192]u8 = undefined;
    return projection.conv(a, t, projection.join(&buf, &.{ base, "conv1d.weight" }));
}

/// The routed MLP when the config has experts, else gate, up and down.
fn readMlp(a: std.mem.Allocator, t: *const Table, mlp: []const u8, s: config.Spec) !host.Mlp {
    if (s.experts != 0) return .{ .routed = try experts.routed(a, t, mlp, s) };
    return .{ .dense = .{
        .gate = try sub(a, t, mlp, "gate_proj"),
        .up = try sub(a, t, mlp, "up_proj"),
        .down = try sub(a, t, mlp, "down_proj"),
    } };
}

/// The head's tensors under `mtp.`: the Qwen3 shape (`pre_fc_norm_embedding`) or the Flash Next one (`norm_e`).
fn readMtp(a: std.mem.Allocator, t: *const Table, s: config.Spec) !host.Mtp {
    const head: ?host.Projection = if (t.has("mtp.head_proj.weight")) try projection.projection(a, t, "mtp.head_proj") else null;
    if (!t.has("mtp.pre_fc_norm_embedding.weight")) {
        return .{
            .arena = undefined,
            .io = undefined,
            .fc_e_norm = try projection.float(a, t, "mtp.norm_e.weight"),
            .fc_h_norm = try projection.float(a, t, "mtp.norm_h.weight"),
            .fc_e = try projection.projection(a, t, "mtp.fc_e"),
            .fc_h = try projection.projection(a, t, "mtp.fc_h"),
            .q_norm = try projection.float(a, t, "mtp.q_norm.weight"),
            .k_norm = try projection.float(a, t, "mtp.k_norm.weight"),
            .q = try projection.projection(a, t, "mtp.q_proj"),
            .k = try projection.projection(a, t, "mtp.k_proj"),
            .v = try projection.projection(a, t, "mtp.v_proj"),
            .o = try projection.projection(a, t, "mtp.o_proj"),
            .final_norm = try projection.float(a, t, "mtp.final_norm.weight"),
            .head = head,
            .input_norm = null,
            .post_norm = null,
            .mlp = null,
            .gated = false,
        };
    }
    const layer = "mtp.layers.0.";
    const attn = layer ++ "self_attn.";
    const mlp = layer ++ "mlp.";
    const halves = try projection.halves(a, try projection.projection(a, t, "mtp.fc"));
    const routed = t.has(mlp ++ "switch_mlp.up_proj.weight") or t.has(mlp ++ "switch_mlp.up_proj.qweight");
    return .{
        .arena = undefined,
        .io = undefined,
        .fc_e_norm = try projection.float(a, t, "mtp.pre_fc_norm_embedding.weight"),
        .fc_h_norm = try projection.float(a, t, "mtp.pre_fc_norm_hidden.weight"),
        .fc_e = halves[0],
        .fc_h = halves[1],
        .q_norm = try projection.float(a, t, attn ++ "q_norm.weight"),
        .k_norm = try projection.float(a, t, attn ++ "k_norm.weight"),
        .q = try projection.projection(a, t, attn ++ "q_proj"),
        .k = try projection.projection(a, t, attn ++ "k_proj"),
        .v = try projection.projection(a, t, attn ++ "v_proj"),
        .o = try projection.projection(a, t, attn ++ "o_proj"),
        .final_norm = try projection.float(a, t, "mtp.norm.weight"),
        .head = head,
        .input_norm = try projection.float(a, t, layer ++ "input_layernorm.weight"),
        .post_norm = try projection.float(a, t, layer ++ "post_attention_layernorm.weight"),
        .mlp = if (routed) .{ .routed = try experts.routed(a, t, mlp, s) } else .{ .dense = .{
            .gate = try projection.projection(a, t, mlp ++ "gate_proj"),
            .up = try projection.projection(a, t, mlp ++ "up_proj"),
            .down = try projection.projection(a, t, mlp ++ "down_proj"),
        } },
        .gated = true,
    };
}
