//! Leave room for a full calibration sweep before a tight scratch context needs another prefill.
const std = @import("std");

pub fn promptRows(capacity: usize, widest: usize, sweep: usize) usize {
    std.debug.assert(capacity > widest);
    const room = capacity - widest;
    return @min(192, room -| @min(sweep, room - 1));
}
