//! Each prompt chunk chooses its width from the streams that can decode beside it.
pub fn choose(b: anytype, current: anytype) usize {
    for (b.owners[0..b.runner.slots]) |owner| if (owner) |stream| {
        if (stream != current and !stream.finished and !stream.paused and !b.fills.contains(stream)) {
            return @min(b.chunk, 512);
        }
    };
    return b.chunk;
}
