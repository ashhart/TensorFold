const image = @import("image_encoder.zig");
const language = @import("text_encoder.zig");
const concepts = @import("concept_detector.zig");

pub const trunk = image.encode;
pub const pyramid = image.pyramid;
pub const text = language.encode;
pub const detect = concepts.detect;
pub const Detection = concepts.Detection;
pub const DetectionOptions = concepts.Options;
