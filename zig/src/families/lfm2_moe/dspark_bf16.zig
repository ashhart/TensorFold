//! Native BF16 projection regimes and the row-fused gate/up activation.
const std = @import("std");
const mtl = @import("metal");
const kernels = @import("kernels.zig");
const dw = @import("dspark_weights.zig");
const c = @import("config.zig");

const Args = extern struct { rows: u32, n: u32, k: u32 };

pub fn tensorRows(n: usize) usize {
    return if (n > c.hidden) 8 else 16;
}

/// Physical projection rows select the measured tensor regime; the caller retains its scalar-row limit.
pub fn project(e: mtl.ComputeEncoder, pipes: *const kernels.Kernels, m: dw.Bf16, in: anytype, out: anytype, rows: usize, row_limit: usize) void {
    const args = Args{ .rows = @intCast(rows), .n = @intCast(m.n), .k = @intCast(m.k) };
    const tensor = if (rows >= tensorRows(m.n)) pipes.bf16_nax else null;
    if (tensor) |p| {
        e.setPipeline(p);
        e.setBuffer(in.buf, in.off, 0);
        e.setBuffer(m.w, 0, 1);
        e.setValue(args, 2);
        e.setBuffer(out.buf, out.off, 3);
        e.dispatchGroups(mtl.Size.of((m.n + 127) / 128, (rows + 15) / 16, 1), mtl.Size.of(128, 1, 1));
        return;
    }
    const few = rows <= row_limit;
    if (few) e.setPipeline(pipes.get("lfm2_bf16_rows")) else e.setPipeline(pipes.get("lfm2_bf16_mma"));
    e.setBuffer(in.buf, in.off, 0);
    e.setBuffer(m.w, 0, 1);
    e.setBytes(std.mem.asBytes(&args), 2);
    e.setBuffer(out.buf, out.off, 3);
    if (few) {
        e.dispatchGroups(mtl.Size.of((m.n + 7) / 8, (rows + 15) / 16, 1), mtl.Size.of(256, 1, 1));
    } else e.dispatchGroups(mtl.Size.of((m.n + 63) / 64, (rows + 63) / 64, 1), mtl.Size.of(128, 1, 1));
}

/// `in` and `out` are buffer/byte-offset refs; out is the existing activation scratch, never gate/up scratch.
pub fn glu(e: mtl.ComputeEncoder, pipes: *const kernels.Kernels, gate: dw.Bf16, up: dw.Bf16, in: anytype, out: anytype, rows: usize) void {
    std.debug.assert(rows > 0 and rows <= 16 and gate.n == up.n and gate.k == up.k);
    e.setPipeline(pipes.get("lfm2_bf16_glu_rows"));
    e.setBuffer(in.buf, in.off, 0);
    e.setBuffer(gate.w, 0, 1);
    e.setValue(Args{ .rows = @intCast(rows), .n = @intCast(gate.n), .k = @intCast(gate.k) }, 2);
    e.setBuffer(out.buf, out.off, 3);
    e.setBuffer(up.w, 0, 4);
    e.dispatchGroups(mtl.Size.of((gate.n + 7) / 8, (rows + 15) / 16, 1), mtl.Size.of(256, 1, 1));
}
