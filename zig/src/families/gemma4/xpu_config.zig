//! The XPU runner's view of the Gemma 4 E4B: the family's admitted geometry, and the widths its kernels read.
const std = @import("std");
const family = @import("cuda_config.zig");

const Width = family.widths.Width;

pub const geo = family.e4b;
pub const hidden: comptime_int = geo.hidden;
pub const layers: comptime_int = geo.layers;
pub const ff: comptime_int = geo.intermediate;
pub const heads: comptime_int = geo.query_heads;
pub const kv_heads: comptime_int = geo.sliding_kv_heads; // the global layers take the same count
pub const dim_s: comptime_int = family.sliding_dim; // sliding layers' head
pub const dim_g: comptime_int = family.global_dim; // global layers' head
pub const window: comptime_int = geo.sliding_window;
pub const shared_from: comptime_int = geo.cacheLayers(); // the layers from here on read the K/V of an earlier one
pub const ple: comptime_int = geo.layer_input;
pub const ple_all: comptime_int = layers * ple;
pub const vocab: comptime_int = family.vocab;
pub const softcap = family.softcap;
pub const eps = family.eps;
pub const ring = 1536; // sliding K/V slots: the window plus a prefill block of 1024

pub const isGlobal = family.global;

/// The physical K/V cache a layer reads: its own below `shared_from`, else the last layer of its type before it.
pub fn kvSource(layer: usize) usize {
    return family.cacheOwner(layer, shared_from).?;
}

pub const Error = error{ NotGemma4E4B, XpuWidthNotRead } || family.Error;

/// The family's admission of `config.json`, then the widths the kernels read; any other model is refused.
pub fn admit(gpa: std.mem.Allocator, bytes: []const u8) Error!void {
    const c = try family.parse(gpa, bytes);
    if (!std.mem.eql(u8, c.geo.name, geo.name)) return error.NotGemma4E4B;
    const w = &c.widths;
    if (!four(w.embed) or !four(w.layer_table orelse return error.XpuWidthNotRead)) return error.XpuWidthNotRead;
    if (w.model_project) |p| if (!four(p)) return error.XpuWidthNotRead; // null: BF16, read by the dense kernel
    for (0..layers) |i| {
        if (w.layer_gate[i]) |g| if (!four(g)) return error.XpuWidthNotRead;
        if (w.layer_project[i]) |p| if (!four(p)) return error.XpuWidthNotRead;
        for ([_]family.widths.Module{ .gate, .up, .down }) |m| if (!four(w.of(i, m))) return error.XpuWidthNotRead;
        // attention at 4 or 8 bits (an entry of the quantization table); q, k and v are one fused projection
        const q = w.of(i, .q);
        if (!attention(q) or !attention(w.of(i, .o))) return error.XpuWidthNotRead;
        if (i < shared_from) for ([_]family.widths.Module{ .k, .v }) |m| if (!std.meta.eql(w.of(i, m), q)) return error.XpuWidthNotRead;
    }
}

fn four(w: Width) bool {
    return w.bits == 4 and w.group == family.group;
}

fn attention(w: Width) bool {
    return (w.bits == 4 or w.bits == 8) and w.group == family.group;
}

const fixture = @embedFile("fixtures/config_e4b.json");
const fixture_31b = @embedFile("fixtures/config.json");
const eight = "{\"bits\": 8, \"group_size\": 64, \"mode\": \"affine\"}";

/// The fixture with `entries` added to both of its quantization tables.
fn withEntries(gpa: std.mem.Allocator, entries: []const u8) ![]u8 {
    const to = try std.fmt.allocPrint(gpa, "\"mode\": \"affine\", {s}", .{entries});
    defer gpa.free(to);
    return std.mem.replaceOwned(u8, gpa, fixture, "\"mode\": \"affine\"", to);
}

fn expectAdmitted(entries: []const u8) !void {
    const bytes = try withEntries(std.testing.allocator, entries);
    defer std.testing.allocator.free(bytes);
    try admit(std.testing.allocator, bytes);
}

fn expectRefused(entries: []const u8, want: anyerror) !void {
    const bytes = try withEntries(std.testing.allocator, entries);
    defer std.testing.allocator.free(bytes);
    try std.testing.expectError(want, admit(std.testing.allocator, bytes));
}

const lm = "\"language_model.model.";

test "the E4B's config is admitted, with attention at 8 bits and the per-layer linears in BF16" {
    try admit(std.testing.allocator, fixture);
    try expectAdmitted(lm ++ "layers.3.self_attn.q_proj\": " ++ eight ++ ", " ++ lm ++ "layers.3.self_attn.k_proj\": " ++ eight ++ ", " ++
        lm ++ "layers.3.self_attn.v_proj\": " ++ eight ++ ", " ++ lm ++ "layers.3.self_attn.o_proj\": " ++ eight);
    try expectAdmitted(lm ++ "layers.30.self_attn.q_proj\": " ++ eight); // a layer that reads an earlier cache has no k or v
    try expectAdmitted(lm ++ "per_layer_model_projection\": false, " ++ lm ++ "layers.2.per_layer_input_gate\": false");
}

test "widths the kernels do not read are refused by name" {
    try expectRefused(lm ++ "embed_tokens\": " ++ eight, error.XpuWidthNotRead);
    try expectRefused(lm ++ "embed_tokens_per_layer\": " ++ eight, error.XpuWidthNotRead);
    try expectRefused(lm ++ "embed_tokens_per_layer\": false", error.XpuWidthNotRead);
    try expectRefused(lm ++ "layers.0.mlp.up_proj\": " ++ eight, error.XpuWidthNotRead);
    try expectRefused(lm ++ "layers.4.per_layer_projection\": " ++ eight, error.XpuWidthNotRead);
    try expectRefused(lm ++ "layers.3.self_attn.q_proj\": " ++ eight, error.XpuWidthNotRead); // fused with k and v
    try expectRefused(lm ++ "layers.3.self_attn.o_proj\": {\"bits\": 6, \"group_size\": 64, \"mode\": \"affine\"}", error.XpuWidthNotRead);
}

test "another Gemma 4 size is not the E4B" {
    try std.testing.expectError(error.NotGemma4E4B, admit(std.testing.allocator, fixture_31b));
}
