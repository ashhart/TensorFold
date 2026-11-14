//! Actual packed-cache and rotation row bodies, with test-only faults in their loads and position arithmetic.
const std = @import("std");
const sources = @import("kernel_sources");

pub const Kind = enum { rope, glue, publish };
pub const Fault = enum { clip, input_write, position, half_sum };

pub fn raw(kind: Kind) []const u8 {
    return switch (kind) {
        .rope => sources.deepseek_v41_rope,
        .glue => sources.deepseek_v41_kv_glue,
        .publish => sources.deepseek_v41_kv,
    };
}

pub fn entry(kind: Kind, old: bool) [:0]const u8 {
    return switch (kind) {
        .rope => if (old) "dsv41_rope" else "dsv41_rope_rows",
        .glue => if (old) "dsv41_kv_glue" else "dsv41_kv_glue_rows",
        .publish => if (old) "dsv41_publish_window" else "dsv41_publish_rows",
    };
}

fn replace(a: std.mem.Allocator, text: *[]const u8, old: []const u8, new: []const u8, count: usize) !void {
    if (std.mem.count(u8, text.*, old) != count) return error.NativeRowsFaultAnchorChanged;
    const next = try std.mem.replaceOwned(u8, a, text.*, old, new);
    a.free(text.*);
    text.* = next;
}

pub fn source(a: std.mem.Allocator, kind: Kind, fault: ?Fault) ![]const u8 {
    var text: []const u8 = try a.dupe(u8, raw(kind));
    errdefer a.free(text);
    switch (fault orelse return text) {
        .clip => switch (kind) {
            .rope => try replace(a, &text, "float a = as_type<float>(uint(x[at]) << 16), b = as_type<float>(uint(x[at + 1]) << 16);", "float a = as_type<float>(uint(x[at]) << 16), b = as_type<float>(uint(x[at + 1]) << 16); if (!isfinite(a)) a=0; if (!isfinite(b)) b=0;", 2),
            .glue => try replace(a, &text, "inline float dsv41kg_bf(ushort x) { return as_type<float>(uint(x) << 16); }", "inline float dsv41kg_bf(ushort x) { float v=as_type<float>(uint(x)<<16); return isfinite(v)?v:0; }", 1),
            .publish => try replace(a, &text, "ring[(pos[i.y] % 128) * p.y + i.x] = chunk[i.y * p.y + i.x];", "uchar v=chunk[i.y*p.y+i.x]; ring[(pos[i.y]%128)*p.y+i.x]=((v&127u)==127u)?0:v;", 1),
        },
        .input_write => switch (kind) {
            .rope => try replace(a, &text, "float2 rotation = factors[ulong(pos[i.z]) * (p.rotary / 2) + i.x];", "float2 rotation = factors[ulong(pos[i.z]) * (p.rotary / 2) + i.x]; const_cast<device float2*>(factors)[0]=float2(0);", 1),
            .glue => try replace(a, &text, "const device ushort* x = X + ulong(g.y) * p.stride;", "const device ushort* x = X + ulong(g.y) * p.stride; const_cast<device ushort*>(W)[0]=0;", 1),
            .publish => try replace(a, &text, "device uchar *ring = reinterpret_cast<device uchar *>(addr[4 * i.y]) + p.x;", "device uchar *ring = reinterpret_cast<device uchar *>(addr[4 * i.y]) + p.x; const_cast<device uchar*>(chunk)[0]=0;", 1),
        },
        .position => switch (kind) {
            .rope => try replace(a, &text, "pos[i.z]", "pos[0]", 1),
            .glue => try replace(a, &text, "lane, sg, pos[g.y]);", "lane, sg, pos[0]);", 1),
            .publish => try replace(a, &text, "pos[i.y]", "pos[0]", 1),
        },
        .half_sum => {
            if (kind != .glue) return error.NoNativeRowsReduction;
            try replace(a, &text, "float s = fma(v0, v0, 0.0f);", "float s = float(half(fma(v0, v0, 0.0f)));", 1);
            try replace(a, &text, "s = fma(v1, v1, s);", "s = float(half(fma(v1, v1, s)));", 1);
        },
    }
    return text;
}
