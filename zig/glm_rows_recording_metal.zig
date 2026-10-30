//! The existing recording encoder, with the queue type needed by the real contract adapter.
const recording = @import("gemma4_recording_metal.zig");
pub const Pipeline = recording.Pipeline;
pub const Device = recording.Device;
pub const Buffer = recording.Buffer;
pub const Size = recording.Size;
pub const Record = recording.Record;
pub const ComputeEncoder = recording.ComputeEncoder;
pub const Queue = struct {};
