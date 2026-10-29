//! LFM2.5-24B-A2B on CUDA behind lanes.Backend: liquid's lane backend over this family's model and forward.
const liquid = @import("liquid_cuda");

const B = liquid.lanes.Backend(struct {
    pub const cfg = @import("config.zig");
    pub const fwd = @import("cuda_forward.zig");
    pub const mdl = @import("cuda_model.zig");
    pub const kern = @import("cuda_kernels.zig");
    pub const snap = @import("cuda_snap.zig");
    pub const text_only = error.LfmTextOnly;
});

pub const Liquid = B.Liquid;
pub const prompt_rows = B.prompt_rows;
pub const round_rows = B.round_rows;
pub const max_window = B.max_window;
pub const pagesFor = B.pagesFor;
