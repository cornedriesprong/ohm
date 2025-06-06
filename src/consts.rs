//! Constants

// TODO: don't hardcode sample rate
pub const SAMPLE_RATE: f32 = 44100.0;

// Test signals for testing DSP components
#[cfg(test)]
pub const DC_SIGNAL: [f32; 7] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
#[cfg(test)]
pub const IMPULSE_SIGNAL: [f32; 7] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
#[cfg(test)]
pub const NYQUIST_SIGNAL: [f32; 7] = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
#[cfg(test)]
pub const HALF_NYQUIST_SIGNAL: [f32; 7] = [1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0];
#[cfg(test)]
pub const QUARTER_NYQUIST_SIGNAL: [f32; 9] = [0.0, 0.7071, 1.0, 0.7071, 0.0, -0.7071, -1.0, -0.7071, 0.0];
