//! Affine-4 SwiGLU experts over pair rows (experts_rows.cu), the rows' bits each as alone; experts.zig re-exports it.
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const Stream = @import("stream.zig").Stream;
const Plan = @import("grouped.zig").Plan;
const Affine4Experts = @import("experts.zig").Affine4Experts;
const Module = module.Module;
const Function = module.Function;

/// Affine-4 SwiGLU experts at groups of 32 over pair rows (experts_rows.cu): each row's bits are those it gets alone.
pub const RowsExperts = struct {
    up_fn: Function,
    down_fn: Function,

    /// Pairs a plan item holds: the kernel's two row tiles of 16.
    pub const tile: usize = 32;

    /// The instances' names (cuobjdump -symbols of the experts_rows fatbin): gate and up with SwiGLU, then down.
    pub const symbols = struct {
        pub const up = "_ZN15tf_experts_rows11expert_rowsILi32ELi2ELi2ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif";
        pub const down = "_ZN15tf_experts_rows11expert_rowsILi32ELi1ELi0ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif";
        /// experts.cu's one-row forms at groups of 32, whose bits these give every row (tf-cuda-test experts-rows).
        pub const decode_up = "_ZN10tf_experts13expert_kernelILi32ELi2ELi2ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif";
        pub const decode_down = "_ZN10tf_experts13expert_kernelILi32ELi1ELi0ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif";
    };

    /// From a loaded experts_rows module.
    pub fn resolve(mod: Module) !RowsExperts {
        return .{ .up_fn = try mod.function(symbols.up), .down_fn = try mod.function(symbols.down) };
    }

    /// `items` plan items: `up` maps token rows to SwiGLU's bf16, else pair rows to fp32; K in `kg` groups of 32.
    pub fn run(e: *const RowsExperts, s: Stream, up: bool, x: u64, x_stride: usize, slots: usize, w: u64, kg: usize, nb: usize, p: Plan, out: u64, n: usize, items: usize) !void {
        const grid = @min((items * nb + 3) / 4, 4096);
        if (grid < 1) return;
        var a = Affine4Experts.argsOf(x, x_stride, slots, w, kg, nb, p, out, n);
        const cfg: launch_.Config = .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = 128 }, .shared = 0 };
        try launch_.launch(if (up) e.up_fn else e.down_fn, cfg, s, &a);
    }
};
