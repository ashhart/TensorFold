//! Mini DSpark's arithmetic shader and its ten Metal library entry points.

pub const dspark = @embedFile("dspark.metal");
pub const dspark_units = @embedFile("../nax.h") ++ @embedFile("dspark_units.metal");
