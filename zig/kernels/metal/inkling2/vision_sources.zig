//! Inkling 2's image tower: the prefill GEMM template, the towers' shared glue, the tower's GEMMs and its glue kernels.

pub const vision = @embedFile("../prefill/gemm_nax.metal") ++ @embedFile("../prefill/gemm_nax_kernel.metal") ++
    @embedFile("../prefill/vision_glue.metal") ++ @embedFile("vision_gemm.metal") ++ @embedFile("vision.metal");
