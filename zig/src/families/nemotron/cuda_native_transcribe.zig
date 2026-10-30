//! Nemotron 3.5 Transcribe (nemo_speechlm) on CUDA: the Nemotron engine with its audio encoder (cuda_audio.zig).
const native = @import("cuda_native.zig");

pub const model_type = "nemo_speechlm";
pub const formats = native.formats;
pub const default_context = native.default_context;
pub const max_segments = native.max_segments;
pub const prompt_rows = native.prompt_rows;
pub const Options = native.Options;
pub const explains_open = true;
pub const Loaded = native.Loaded;
pub const LoneRun = native.LoneRun;
pub const open = native.open;
pub const explain = native.explain;
pub const warm_quiet = native.warm_quiet;
pub const lone_min_budget = native.lone_min_budget;
pub const snap_bytes = native.snap_bytes;
pub const snap_save = native.snap_save;
pub const snap_restore = native.snap_restore;
pub const snap_drop = native.snap_drop;
pub const snap_slots = native.snap_slots;
pub const snap_place = native.snap_place;
pub const window_rows = native.window_rows;
pub const attach = native.attach;
pub const fits = native.fits;
pub const tensorWeightBytes = native.tensorWeightBytes;
