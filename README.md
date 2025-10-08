# ohm

ohm is a live coding language/DSL for audio synthesis and processing. It is based on the [Koto](https://koto.dev/) scripting language, which is embedded into a realtime audio synthesis engine written in Rust, and can be used to define audio graphs and play back their output. Audio graphs can be updated and re-rendered in real-time, allowing for fast iteration and experimentation. ohm is inspired by and similar to projects like [Faust](https://faust.grame.fr/) (but interpreted, instead of compiled) or Max's [Gen~](https://docs.cycling74.com/legacy/max8/vignettes/gen_overview) (but primarily text-based, instead of graphical). 

The Koto language has an elegant and minimal syntax. For instance, a simple 2-operator FM synthesizer can we defined in ohm as:

```
sine(sine(400) * 200 + 400)
```

...or, when leaving out the optional parentheses and using Koto's [function piping](https://koto.dev/docs/0.15/language/#function-piping) operator (`->`):

```
(sine 400) * 200 + 400 -> sine
```

See the `examples` directory for more examples.

There is no distinction between audio- and control signals; any can modulate any other signal. ohm functions operate on a per-sample level, which allows for creating DSP structures that require single-sample feedback loops, like filters and delay-based effects (chorus, flanger, etc.).

WARNING: ohm currently doesn't have a built-in limiter, and it's trivial to generate a signal that will massively clip the output. So watch your output levels!

## Usage

The way it works is you start the engine with the path to a koto file as an argument:

```
cargo run --release -- examples/sine.koto
```

ohm will parse the code in the file and, if it's valid, will start playing back the audio. It will listen for any changes in the file and reconstruct the audio graph on save.

Note that re-evaluating the code currently causes discontinuities in the audio output, since the new graph is reconstructed in place of the old one.

## Functions

ohm currently contains the following high-level synthesis functions and effects:

### Basics

- `ramp(freq=100)` a ramp generator going from 0 to 1, useful for driving sequencers and envelopes
- `lfo(freq=1)` a low-frequency sine oscillator, maintains its phase between graph evaluations
- `mul(lhs, rhs)` multiplies two signals
- `add(lhs, rhs)` adds two signals
- `sub(lhs, rhs)` subtracts one signal from another
- `div(lhs, rhs)` divides one signal by another
- `pow(lhs, rhs=2)` raises one signal to the power of another
- `mod(lhs, rhs=1)` modulo of one signal by another
- `eq(lhs, rhs)` equality comparison, outputs 1 if lhs == rhs, 0 otherwise
- `gt(lhs, rhs)` greater-than comparison, outputs 1 if lhs > rhs, 0 otherwise
- `lt(lhs, rhs)` less-than comparison, outputs 1 if lhs < rhs, 0 otherwise
- `clip(input)` clips a signal between -1 and 1
- `sh(input, trig)` sample & hold, trigger is the rising or falling segment of a ramp
- `euclid(pulses, steps)` euclidean rhythm, with 1s for pulses and 0s for rests
- `pan(input, value(0-1)=0.5)` euclidean rhythm, with 1s for pulses and 0s for rests
 
### Buffers / recording
- `file("filename")` load a .wav file from the `samples` directory by specifying its filename (`.wav` is optional)
- `buf(length={sample rate})` define a buffer object of a given length in samples. default length is the sample rate, i.e., 1 second
- `rec(input, buf)` takes a reference to a buffer object and records into it. input is the signal to be recorded.
- `play(input, buf)` takes a reference to a buffer object and plays it back. input is a ramp that control the playback position.
- `tap(buf, offset)` plays back a buffer at a give offset in samples. playback speed is constant.

### Filters

- `lp(freq=500, q=7.07)` state-variable lowpass filter  
- `bp(freq=500, q=7.07)` state-variable bandpass filter  
- `hp(freq=500, q=7.07)` state-variable highpass filter  
- `notch(freq=500, q=7.07)` state-variable notch filter  
- `peak(freq=500, q=7.07)` state-variable peak filter  
- `allpass(freq=500, q=7.07)` state-variable allpass filter  
- `moog(freq=500, q=0.0)` moog filter

### Oscillators

- `sin(freq=100)` a sine wave
- `sqr(freq=100)` a square wave
- `saw(freq=100)` a bandlimited sawtooth wave
- `tri(freq=100)` a triangle wave
- `noise` white noise, also useful as a randomization source

### Effects
- `flanger(input, mix=0.5, amount=0.5, speed = 0.1)` a flanger effect
- `rev(input, mix=0.5)` a reverb effect

### Sequencers
- `seq(list, trig)` a sequencer that outputs the next value in the list every time it receives a trigger
- `delay(delay, input)` delays a signal by *n* samples

### Drum synthesis
- `hh(ramp, curve=3)` a hihat
 
### Randomness
- `rand(ramp)` output a random value between 0 and 1 on every trigger
- `bernoulli(ramp, probability(0-1)=0.5)` bernoulli gate, outputs 1 or 0 with probability on every trigger
 
### Utilities / conversions
- `log(input)` print the value of a signal to stdout
- `mix([inputs])` mix an array of signals
- `ptof(pitch)` converts pitch to frequency
- `fotp(freq)` converts frequency to pitch
- `stof(samples)` samples to frequency
- `ftos(freq)` frequency to samples
- `mstos(ms)` milliseconds to samples
- `stoms(samples)` samples to milliseconds
- `btou(input)` scale a bipolar (-1 to 1) signal to unipolar (0 to 1)
- `utob(input)` scale a unipolar (0 to 1) signal to bipolar (-1 to 1) 

### Constants
- MIDI pitches `C0` to `G#9`
- musical scales (see `scales.koto`)
- `sr` sample rate

Of course, there is nothing stopping you from defining your own functions in Koto, and using them in your audio graph.

## Similar projects
- [Faust](https://faust.grame.fr/)
- [Gen~](https://docs.cycling74.com/legacy/max8/vignettes/gen_overview)
- [Csound](https://csound.com/)
- [SuperCollider](https://supercollider.github.io/)
- [TidalCycles](https://tidalcycles.org/)
- [Sonic Pi](https://sonic-pi.net/)
- [Chuck](http://chuck.stanford.edu/)
- [Glicol](https://glicol.org/)
