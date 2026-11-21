//! Take actual learner and trainer entry addresses to compile their bodies without loading a device or shaders.
const root = @import("family");
const F = if (@hasDecl(root, "nemotron")) root.nemotron else root;
export fn compiledLocate() usize {
    return @intFromPtr(&F.train.Trainer.causalLogit) ^ @intFromPtr(&F.slide.Learner.begin) ^ @intFromPtr(&F.slide.Learner.step);
}
