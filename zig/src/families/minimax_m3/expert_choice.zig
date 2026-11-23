//! Decode expert width selection without Metal dependencies.
/// The grouped expert pipeline for `bits`, or null for indexed. Q8 stays indexed before its opt-in pipeline exists.
pub fn groupedExpertSlot(bits: u32, q8_opt_in: bool) ?usize {
    _ = q8_opt_in;
    return switch (bits) {
        3 => 0,
        4 => 1,
        else => null,
    };
}
