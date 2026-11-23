//! MiniMax-M3 Metal's decode expert path, independent of pipeline loading and launch encoding.

/// The grouped pipeline slot for `bits`, or null for indexed decode. Q8 requires its explicit opt-in.
pub fn groupedExpertSlot(bits: u32, q8_opt_in: bool) ?usize {
    return switch (bits) {
        3 => 0,
        4 => 1,
        8 => if (q8_opt_in) 2 else null,
        else => null,
    };
}
