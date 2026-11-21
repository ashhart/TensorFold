//! Rows required by the decoder's sliding-window state at this chunk's end.
const c = @import("config.zig");

// This layer projects global latents and the candidate index for every prompt row.
pub const global_kv_layer = c.kv_sources[c.kv_sources.len - 1];

// The draft targets write features at the same scratch coordinates. Freeze the

// tail width before the first target so every DSpark stage sees the same rows.
const last_cut_layer = c.draft_targets[0] - 1;

pub fn inputRows(chunk: usize, layer: usize) usize {
    if (layer <= global_kv_layer) return chunk;
    return chunk - inputStart(chunk, layer);
}

/// Earliest decoder input for the export's final window. Global KV still runs at every position.
pub fn inputStart(export_at: usize, layer: usize) usize {
    const last: usize = @min(layer, last_cut_layer);
    return export_at -| (c.window + (c.layers - last) * (c.window - 1));
}
