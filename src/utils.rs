//! Utility functions

pub fn freq_to_period(sample_rate: f32, freq: f32) -> f32 {
    sample_rate / freq
}

pub fn lerp(x: f32, length: f32) -> f32 {
    x / length
}
