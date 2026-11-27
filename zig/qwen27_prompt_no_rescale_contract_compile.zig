//! Force the Qwen no-rescale contract bodies to compile without linking or running Metal.
const rows = @import("q27_rows");

export fn compileQwen27PromptNoRescalePrepare() usize {
    return @intFromPtr(&rows.Attn.prepare);
}

export fn compileQwen27PromptNoRescaleKernel() usize {
    return @intFromPtr(&rows.Attn.kernel);
}
