//! The affine metadata's stored element type; packing copies its native bytes without narrowing.
pub const Type = enum {
    bf16,
    f32,

    pub fn size(t: Type) usize {
        return if (t == .f32) 4 else 2;
    }
};
