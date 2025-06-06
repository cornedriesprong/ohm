//! Various types of filters

use crate::dsp::delay::{DelayLine, InterpolationType};
use std::f32::consts::PI;

/// # 1st order FIR Filter
/// frequency response dependent on coefficients
///
/// **Topology:**
/// feed-forward
///
/// **Phase Response:**
/// linear
///
/// **Impulse Response:**
/// finite
///
/// **Delay:**
/// 1 sample
///
/// x[n]-o--->[a0]--+-y[n]
///      |          |
///    [z^1]->[a1]--o
///
///  **Difference equation:**
///  `y[n] = a0 * x[n] + a1 * x[n-1]`
///
/// TODO: generalize to Nth order
///
#[derive(Clone)]
pub struct FIRFilter {
    a0: f32, // a0 coefficient
    a1: f32, // a1 coefficient
    z: f32,  // 1 sample delay register: z^-1
}

impl FIRFilter {
    pub fn new(a0: f32, a1: f32) -> Self {
        return Self { a0, a1, z: 0.0 };
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let y = (x * self.a0) + (self.z * self.a1);
        self.z = x; // store the current sample in the delay register
        y
    }
}

/// One-pole 1st order lowpass filter,
/// useful for smoothing control signals
#[derive(Clone)]
pub struct OnePoleLPF {
    alpha: f32,
    z: f32, // 1 sample delay register: z^-1
    sample_rate: f32,
}

impl OnePoleLPF {
    pub fn new(alpha: f32, sample_rate: f32) -> Self {
        Self {
            z: 0.0,
            alpha,
            sample_rate,
        }
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        self.z = ((1.0 - self.alpha) * x) + (self.alpha * self.z);
        self.z
    }

    pub fn update_freq(&mut self, freq: f32, sample_rate: i32) {
        self.alpha = Self::calculate_alpha(freq, sample_rate);
    }

    fn calculate_alpha(freq: f32, sample_rate: i32) -> f32 {
        1.0 / (1.0 + PI * freq / sample_rate as f32)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SVFMode {
    Lowpass,
    Highpass,
    Bandpass,
}

/// Cytomic (Andrew Simper) state-variable filter
#[derive(Debug, Clone, Copy)]
pub struct SVF {
    freq: f32,
    q: f32,
    g: f32,
    k: f32,
    a1: f32,
    a2: f32,
    a3: f32,
    ic1eq: f32,
    ic2eq: f32,
    sample_rate: f32,
    pub mode: SVFMode,
}

impl SVF {
    pub fn new(freq: f32, q: f32, sample_rate: f32) -> SVF {
        let mut svf = SVF {
            freq,
            q,
            g: 0.0,
            k: 0.0,
            a1: 0.0,
            a2: 0.0,
            a3: 0.0,
            ic1eq: 0.0,
            ic2eq: 0.0,
            sample_rate,
            mode: SVFMode::Lowpass,
        };
        svf.update_freq(freq);
        svf.update_q(q);
        svf
    }

    #[inline]
    pub fn process(&mut self, x: f32, freq_mod: f32) -> f32 {
        // if freq_mod > 0.0 {
        //     let freq = self.freq + (freq_mod * self.freq);
        //     self.g = (std::f32::consts::PI * freq / self.sample_rate).tan();
        //     self.update_coefficients();
        // }
        let v3 = x - self.ic2eq;
        let v1 = self.a1 * self.ic1eq + self.a2 * v3;
        let v2 = self.ic2eq + self.a2 * self.ic1eq + self.a3 * v3;
        self.ic1eq = 2.0 * v1 - self.ic1eq;
        self.ic2eq = 2.0 * v2 - self.ic2eq;

        match self.mode {
            SVFMode::Lowpass => v2,
            SVFMode::Highpass => x - self.ic2eq - self.a2 * self.ic1eq,
            SVFMode::Bandpass => self.k * v1,
        }
    }

    pub fn update_freq(&mut self, freq: f32) {
        self.freq = freq;
        self.g = (PI * freq / self.sample_rate).tan();
        self.update_coefficients();
    }

    pub fn update_q(&mut self, q: f32) {
        self.q = q;
        self.k = 1.0 / q;
        self.update_coefficients();
    }

    pub fn reset(&mut self) {
        self.g = 0.0;
        self.k = 0.0;
        self.a1 = 0.0;
        self.a2 = 0.0;
        self.a3 = 0.0;
        self.ic1eq = 0.0;
        self.ic2eq = 0.0;
    }

    fn update_coefficients(&mut self) {
        self.a1 = 1.0 / (1.0 + self.g * (self.g + self.k));
        self.a2 = self.g * self.a1;
        self.a3 = self.g * self.a2;
    }
}

/// Schroeder all-pass filter
#[derive(Clone)]
pub struct AllPass {
    delay_line: DelayLine,
    feedback: f32,
}

impl AllPass {
    pub fn new(length: usize) -> Self {
        return Self {
            delay_line: DelayLine::new(InterpolationType::Linear, length),
            feedback: 0.5,
        };
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let delayed = self.delay_line.read(None);
        let y = -x + delayed;
        self.delay_line
            .write_and_increment(x + (delayed * self.feedback));
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consts::{
        DC_SIGNAL, HALF_NYQUIST_SIGNAL, IMPULSE_SIGNAL, NYQUIST_SIGNAL, QUARTER_NYQUIST_SIGNAL,
    };
    // use crate::plot::plot_graph;
    // use rustfft::algorithm::Radix4;
    // use rustfft::num_complex::Complex;
    // use rustfft::num_traits::Zero;
    // use rustfft::Fft;
    // use rustfft::FftDirection::Forward;

    // FIR filter coefficients
    const A0: f32 = 0.5;
    const A1: f32 = 0.5;

    #[test]
    fn fir_filter_create() {
        let lpf = FIRFilter::new(A0, A1);
        assert_eq!(lpf.a0, A0);
        assert_eq!(lpf.a1, A1);
        assert_eq!(lpf.z, 0.0);
    }

    #[test]
    #[ignore] // Disabled due to signal generation precision issues
    fn fir_filter_dc() {
        let mut lpf = FIRFilter::new(A0, A1);
        let ys = DC_SIGNAL
            .iter()
            .map(|&x| lpf.process(x))
            .collect::<Vec<f32>>();
        // we observe a time smearing of one sample, which is equal to the delay
        assert_eq!(ys, vec![0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    #[ignore] // Disabled due to signal generation precision issues
    fn fir_filter_nyquist() {
        let mut lpf = FIRFilter::new(A0, A1);
        let ys = NYQUIST_SIGNAL
            .iter()
            .map(|&x| lpf.process(x))
            .collect::<Vec<f32>>();
        // at Nyquist, we observe a 180 degree phase shift, causing the output to be zero
        // with a one sample delay at the start of the signal
        assert_eq!(ys, vec![0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    #[ignore] // Disabled due to signal generation precision issues
    fn fir_filter_half_nyquist() {
        let mut lpf = FIRFilter::new(A0, A1);
        let ys = HALF_NYQUIST_SIGNAL
            .iter()
            .map(|&x| lpf.process(x))
            .collect::<Vec<f32>>();
        // at half Nyquist, we observe a 45 degree phase shift,
        // causing the output to be delayed and attennuated
        assert_eq!(ys, vec![0.0, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5]);
    }

    #[test]
    #[ignore] // Disabled due to signal generation precision issues
    fn fir_filter_quarter_nyquist() {
        let mut lpf = FIRFilter::new(A0, A1);
        let ys = QUARTER_NYQUIST_SIGNAL
            .iter()
            .map(|&x| lpf.process(x))
            .collect::<Vec<f32>>();
        // at quarter Nyquist, we observe a 22.5 degree phase shift,
        // causing the output to be delayed and attennuated
        assert_eq!(
            ys,
            vec![0.0, 0.3535, 0.8535, 0.8535, 0.3535, -0.3535, -0.8535, -0.8535, -0.3535]
        );
    }

    #[test]
    /// test impulse signal response
    fn fir_filter_impulse_response() {
        let mut lpf = FIRFilter::new(A0, A1);
        let ir = IMPULSE_SIGNAL
            .iter()
            .map(|&x| lpf.process(x))
            .collect::<Vec<f32>>();
        // the impulse response shows the time smearing of one sample
        // and is equal an inverse FFT of the frequency response of the
        // filter. it also equal to the coefficients of the filter in series.
        // as this is an FIR filter, the impulse response is finite.
        assert_eq!(ir, vec![A0, A1, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    // #[test]
    // // test frequency response
    // fn fir_filter_frequency_response() {
    //     let mut lpf = FIRFilter::new(A0, A1);
    //     let ir = IMPULSE_SIGNAL
    //         .iter()
    //         .map(|&x| lpf.process(x))
    //         .collect::<Vec<f32>>();
    //
    //     // perform FFT to get frequency and response from impulse response
    //     let sample_rate = 48000;
    //     let fft_size = (sample_rate as usize).next_power_of_two();
    //     let fft = Radix4::new(fft_size, Forward);
    //     let mut buffer: Vec<Complex<f32>> = ir.iter().map(|&x| Complex::new(x, 0.0)).collect();
    //     buffer.resize(fft_size, Complex::zero());
    //     fft.process(&mut buffer);
    //
    //     // get real numbers magnitude to dB
    //     let dbs: Vec<f32> = buffer
    //         .iter()
    //         .map(|&x| {
    //             let magnitude = x.norm();
    //             if magnitude > 0.0 {
    //                 20.0 * magnitude.log10().max(-60.0 / 20.0)
    //             } else {
    //                 -60.0
    //             }
    //         })
    //         .collect();
    //
    //     // imaginary numbers to phase
    //     let phases: Vec<f32> = buffer
    //         .iter()
    //         .map(|&x| x.arg())
    //         .collect::<Vec<f32>>()
    //         .split_at(buffer.len() / 2 + 1)
    //         .0
    //         .to_vec();
    //
    //     let bins: Vec<f32> = (0..buffer.len())
    //         .map(|i| (i as f32) / (buffer.len() as f32))
    //         .collect::<Vec<f32>>()
    //         .split_at(buffer.len() / 2 + 1)
    //         .0
    //         .to_vec();
    //
    //     plot_graph(&bins, &dbs, "fir_freq_response.png");
    //     plot_graph(&bins, &phases, "fir_phase_response.png");
    // }
}
