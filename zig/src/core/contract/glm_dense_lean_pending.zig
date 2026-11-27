//! Dense lean geometry is declared; nvcc register/static/bound receipts have not been measured.
const r = @import("registry.zig");
pub const records = [_]r.CudaRecord{
    .{ .launch = .{ .name = "prompt16 GLM dense lean bf16", .symbol = "_ZN11tf_prompt1611prompt_leanILi0ELi64ELi128ELi2ELi4ELi3ELi0EEEvPK13__nv_bfloat16PKhS5_fPviiiiiiS5_S5_fPKi", .image = "prompt16", .threads = 256, .dynamic = 38448, .min_sm = 80 }, .usage = r.none },
    .{ .launch = .{ .name = "prompt16 GLM dense lean fp32", .symbol = "_ZN11tf_prompt1611prompt_leanILi0ELi64ELi128ELi2ELi4ELi3ELi1EEEvPK13__nv_bfloat16PKhS5_fPviiiiiiS5_S5_fPKi", .image = "prompt16", .threads = 256, .dynamic = 38448, .min_sm = 80 }, .usage = r.none },
};
