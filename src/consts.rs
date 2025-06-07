//! Constants

// TODO: don't hardcode sample rate
pub const SAMPLE_RATE: f32 = 44100.0;

// Test signals for testing DSP components
#[cfg(test)]
pub const IMPULSE_SIGNAL: [f32; 7] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
