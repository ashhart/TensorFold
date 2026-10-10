//! Nemotron's Sliding Weights learner on CUDA: the backend-neutral learner over the CUDA trainer.
const learner = @import("learner.zig");
const train = @import("cuda_train.zig");

pub const Backend = train.Backend;
pub const Example = learner.Example;
pub const Lesson = learner.Lesson;
pub const Report = learner.Report;
pub const Step = learner.Step;
pub const Learner = learner.Of(train);
