//! Long rotary fixtures use actual admitted positions, including the allocation margin.
const maximum = @import("tensorfold").flashnext_vision_contract.maximum;
pub const positions = [_]i32{ 0, 1, 850, 262143, 262144, 1048575, 1048576, maximum - 1 };

pub fn token(row: usize, axis: usize) i32 {
    return positions[(row + axis) % positions.len];
}

pub fn firstBlock(rows: usize) usize {
    return maximum / 4 - rows;
}
