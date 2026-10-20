//! Startup text reads the loaded pass size when a family returns one; other families keep their static value.
pub const format = "CUDA sm_{d} device {d} ({s}{s}): model {d:.2} GiB{s}; {d} stream{s} at once, {s}, of {d:.1} GiB left after a {d:.1} GiB reserve; prompts in passes of up to {d} rows{s}{s}; {s}";

pub fn rows(comptime F: type, loaded: anytype) u32 {
    return if (@hasField(@TypeOf(loaded), "prompt_rows")) loaded.prompt_rows else F.prompt_rows;
}
