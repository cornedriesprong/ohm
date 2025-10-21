//! Utility functions

use fundsp::wave::Wave;

use crate::nodes::Frame;

#[inline(always)]
pub fn cubic_interpolate(buffer: &[Frame], read_head: f32) -> Frame {
    let len = buffer.len() as isize;
    if len == 0 {
        return [0.0; 2];
    }

    let index_floor = read_head.floor() as isize;
    let t = read_head.fract(); // Fractional part for interpolation
    let t2 = t * t;
    let t3 = t2 * t;

    let mut output_frame: Frame = [0.0; 2];

    for ch in 0..2 {
        // Get four neighboring sample indices, using `rem_euclid` for safe wrapping
        let idx0 = (index_floor - 1).rem_euclid(len) as usize;
        let idx1 = index_floor.rem_euclid(len) as usize;
        let idx2 = (index_floor + 1).rem_euclid(len) as usize;
        let idx3 = (index_floor + 2).rem_euclid(len) as usize;

        // Fetch the four points (p0, p1, p2, p3) for the current channel
        let p0 = buffer[idx0][ch];
        let p1 = buffer[idx1][ch];
        let p2 = buffer[idx2][ch];
        let p3 = buffer[idx3][ch];

        // Apply the Catmull-Rom interpolation formula:
        // \[
        // y(t) = 0.5 \times ( (2P_1) + (-P_0+P_2)t + (2P_0-5P_1+4P_2-P_3)t^2 + (-P_0+3P_1-3P_2+P_3)t^3 )
        // \]
        let sample = 0.5
            * ((2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);

        output_frame[ch] = sample;
    }

    output_frame
}

#[inline(always)]
pub fn soft_limit_poly(frames: &mut [Frame]) {
    for frame in frames.iter_mut() {
        let x0 = frame[0];
        let x1 = frame[1];
        let x0_2 = x0 * x0;
        let x1_2 = x1 * x1;
        frame[0] = x0 * (27.0 + x0_2 * (-27.0 + 9.0 * x0_2)) / 27.0;
        frame[1] = x1 * (27.0 + x1_2 * (-27.0 + 9.0 * x1_2)) / 27.0;
    }
}

#[inline(always)]
pub fn hard_clip(frames: &mut [Frame]) {
    for frame in frames.iter_mut() {
        frame[0] = frame[0].max(-1.0).min(1.0);
        frame[1] = frame[1].max(-1.0).min(1.0);
    }
}

#[inline(always)]
pub fn scale_buffer(frames: &mut [Frame], gain: f32) {
    for frame in frames.iter_mut() {
        frame[0] *= gain;
        frame[1] *= gain;
    }
}

pub fn get_audio_frames(name: &str) -> Vec<Frame> {
    let filename = if name.ends_with(".wav") {
        format!("samples/{}", name)
    } else {
        format!("samples/{}.wav", name)
    };

    let wave = Wave::load(filename)
        .map_err(|e| format!("Failed to load '{name}': {e}"))
        .unwrap();

    match wave.channels() {
        1 => (0..wave.len())
            .map(|i| {
                let sample = wave.at(0, i);
                [sample, sample]
            })
            .collect(),
        _ => (0..wave.len())
            .map(|i| [wave.at(0, i), wave.at(1, i)])
            .collect(),
    }
}
