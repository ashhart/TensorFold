//! Each prompt chunk chooses its width from the streams that can decode beside it, and a background pass's depth.
pub fn choose(b: anytype, current: anytype, at: usize) usize {
    const width = for (b.owners[0..b.runner.slots]) |owner| {
        const stream = owner orelse continue;
        if (stream != current and !stream.finished and !stream.paused and !b.fills.contains(stream)) break @min(b.chunk, 512);
    } else b.chunk;
    return if (current.yield_check != null) @min(width, deep(b.chunk, at)) else width;
}

/// A background chunk's width at offset `at`: narrower with depth, each about as long (256 at least).
pub fn deep(chunk: usize, at: usize) usize {
    const scaled = @min(chunk, chunk * 16384 / @max(at, 16384));
    return @max(256, scaled / 128 * 128);
}
