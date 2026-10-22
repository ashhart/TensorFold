//! Kimi's safetensors names taken from core's own reader, so a program that links core owns the file once.
const st = @import("core").safetensors;
pub const DType = st.DType;
pub const headerSize = st.headerSize;
pub const index = st.index;
pub const parseHeaderPrefix = st.parseHeaderPrefix;
