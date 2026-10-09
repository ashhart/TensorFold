//! The Linear seam: a quantized projection as one value (format tag, buffers, shape); Ops launches per format.

const xpu = @import("xpu");

const exl3 = xpu.exl3;
const gg = xpu.ggml;
const Buf = xpu.rt.Buffer;

/// Weight formats a Linear can hold; a new one adds a tag, a loader and a branch in Ops.matvec / Ops.embed.
pub const Format = enum { mlx_affine4_g64, exl3, f16, bf16, q2_k, q4_k, iq4_xs, iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s, iq1_m, q6_k, q5_k, q8_0, iq4_nl };

/// The ggml block type of a format (the ggml tags follow `bf16` in the same order as gg.Type), or null.
pub fn ggType(f: Format) ?gg.Type {
    const base = @intFromEnum(Format.q2_k);
    return if (@intFromEnum(f) >= base) @enumFromInt(@intFromEnum(f) - base) else null;
}

pub fn ggFormat(t: gg.Type) Format {
    return @enumFromInt(@intFromEnum(Format.q2_k) + @intFromEnum(t));
}

/// One quantized projection: its device buffers, shape and format.
pub const Linear = struct {
    format: Format = .mlx_affine4_g64,
    w: Buf,
    s: Buf,
    b: Buf,
    rows: u32,
    in: u32,
    /// MLX weights repacked block-interleaved at load (MLX4_BLOCK=0 keeps rows): matvecs use the systolic kernel.
    block: bool = false,
    /// EXL3 trellis layer (format .exl3); the other formats use w, s, b.
    ex: ?*exl3.Layer = null,
};
