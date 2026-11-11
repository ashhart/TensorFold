//! Each prompt chunk chooses its width from the streams that can decode beside it, and a background pass's depth.
pub fn choose(b: anytype, current: anytype, at: usize) usize {
    const width = for (b.owners[0..b.runner.slots]) |owner| {
        const stream = owner orelse continue;
        if (stream != current and !stream.finished and !stream.paused and !b.fills.contains(stream)) break @min(b.chunk, @as(usize, if (b.runner.model.device.tensorUnits()) 512 else 128));
    } else b.chunk;
    return if (current.yield_check != null) @min(width, deep(b.chunk, at)) else width;
}

/// Decoded spans keep the runner's decode bound even when a solo prompt can take its whole frame.
pub fn chunk(b: anytype, current: anytype, at: usize, decoded: bool) usize {
    const rows = choose(b, current, at);
    return if (decoded) @min(rows, @as(usize, b.runner.decodeRows())) else rows;
}

/// A background chunk's width at offset `at`: narrower with depth, each about as long (256 at least).
pub fn deep(chunk: usize, at: usize) usize {
    const scaled = @min(chunk, chunk * 16384 / @max(at, 16384));
    return @max(256, scaled / 128 * 128);
}
