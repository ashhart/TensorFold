//! Uploading core exl3_load groups to the device as the exl3 kernels' weights (the engine's load path).
const std = @import("std");
const driver = @import("driver.zig");
const memory = @import("memory.zig");
const exl3 = @import("exl3.zig");
const exl3e = @import("exl3_experts.zig");
const Driver = driver.Driver;

/// The tensors one upload holds: appended to `bufs` and freed together.
pub const Uploader = struct {
    d: *const Driver,
    gpa: std.mem.Allocator,
    bufs: std.ArrayList(memory.DeviceBuffer) = .empty,
    bytes: usize = 0, // the tensors' total device bytes

    pub fn deinit(u: *Uploader) void {
        for (u.bufs.items) |*b| b.free();
        u.bufs.deinit(u.gpa);
    }

    pub fn dev(u: *Uploader, bytes: []const u8) !u64 {
        const b = try memory.DeviceBuffer.fromHost(u.d, bytes);
        try u.bufs.append(u.gpa, b);
        u.bytes += bytes.len;
        return b.ptr;
    }

    /// One dense group: `parts` (exl3_load.Parts) and `g` (exl3_load.Group) as the Weight; the scales must be fp16.
    pub fn dense(u: *Uploader, parts: anytype, g: anytype) !exl3.Weight {
        if (parts.suh_packed) return error.PackedScales;
        const st = exl3.strides(g.k, g.n, g.halves, false);
        return .{
            .trellis = try u.dev(parts.trellis.bytes),
            .suh = try u.dev(parts.suh.bytes),
            .svh = try u.dev(parts.svh.bytes),
            .bias = if (parts.bias) |b| try u.dev(b.bytes) else 0,
            .k = @intCast(g.k),
            .n = @intCast(g.n),
            .halves = g.halves,
            .codebook = switch (g.codebook) {
                .inst3 => .inst3,
                .mcg => .mcg,
                .mul1 => .mul1,
            },
            .stride_k = st.k,
            .stride_nb = st.nb,
        };
    }
};

/// One expert projection's tensors (fp16 scales) and half-bit width: the exl3_load Parts' bytes, the group's halves.
pub const Proj = struct { trellis: []const u8, suh: []const u8, svh: []const u8, k2: u32 };

/// One layer's routed experts: trellises in place (pointers stacked), scales stacked [E, n] fp16, widths int32 [E].
pub fn layer(u: *Uploader, a: std.mem.Allocator, dims: usize, width: usize, cb: exl3.Codebook, limit: f32, act_mode: i32, gate: []const Proj, up: []const Proj, down: []const Proj) !exl3e.Layer {
    const e = gate.len;
    if (up.len != e or down.len != e or e == 0) return error.BadExperts;
    const gt = try pointers(u, a, gate);
    const ut = try pointers(u, a, up);
    const dt = try pointers(u, a, down);
    return .{
        .gate = try u.dev(std.mem.sliceAsBytes(gt)),
        .up = try u.dev(std.mem.sliceAsBytes(ut)),
        .down = try u.dev(std.mem.sliceAsBytes(dt)),
        .gate_k2 = try k2s(u, a, gate),
        .up_k2 = try k2s(u, a, up),
        .down_k2 = try k2s(u, a, down),
        .suh_g = try stacked(u, a, gate, dims, "suh"),
        .suh_u = try stacked(u, a, up, dims, "suh"),
        .svh_g = try stacked(u, a, gate, width, "svh"),
        .svh_u = try stacked(u, a, up, width, "svh"),
        .suh_d = try stacked(u, a, down, width, "suh"),
        .svh_d = try stacked(u, a, down, dims, "svh"),
        .experts = e,
        .dims = dims,
        .width = width,
        .codebook = cb,
        .gu_stride = 0,
        .limit = limit,
        .act_mode = act_mode,
    };
}

fn pointers(u: *Uploader, a: std.mem.Allocator, projs: []const Proj) ![]u64 {
    const out = try a.alloc(u64, projs.len);
    for (projs, 0..) |p, i| out[i] = try u.dev(p.trellis);
    return out;
}

fn k2s(u: *Uploader, a: std.mem.Allocator, projs: []const Proj) !u64 {
    const host = try a.alloc(i32, projs.len);
    for (projs, 0..) |p, i| host[i] = @intCast(p.k2);
    return u.dev(std.mem.sliceAsBytes(host));
}

/// Every expert's fp16 scale of `name` ("suh" or "svh"), `n` values each, stacked [E, n].
fn stacked(u: *Uploader, a: std.mem.Allocator, projs: []const Proj, n: usize, comptime name: []const u8) !u64 {
    const host = try a.alloc(u8, projs.len * n * 2);
    for (projs, 0..) |p, i| {
        const s = @field(p, name);
        if (s.len != n * 2) return error.BadScales;
        @memcpy(host[i * n * 2 ..][0..s.len], s);
    }
    return u.dev(host);
}
