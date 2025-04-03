# Ohm

Ohm is a live coding language/DSL for audio synthesis and processing. It is based on the [Koto](https://koto.dev/) scripting language, which is embedded into a realtime audio synthesis engine written in Rust, and can be used to define audio graphs and play back their output. Audio graphs can be updated and re-rendered in real-time, allowing for fast iteration and experimentation. Ohm is inspired by and similar to projects like [Faust](https://faust.grame.fr/) (but interpreted, instead of compiled) or Max's [Gen~](https://docs.cycling74.com/legacy/max8/vignettes/gen_overview) (but primarily text-based, instead of graphical). 

The Koto language has an elegant and minimal syntax. For instance, a simple 2-operator FM synthesizer can we defined in Ohm as:

```
sine(sine(400) * 200 + 400)
```

...or, when leaving out the optional parentheses and using Koto's [function piping](https://koto.dev/docs/0.15/language/#function-piping) operator (`->`):

```
(sine 400) * 200 + 400 -> sine
```

See the `examples` directory for more examples.

There is no distinction between audio- and control signals; any can modulate any other signal. Ohm functions operate on a per-sample level, which allows for creating DSP structures that require single-sample feedback loops, like filters and delay-based effects (chorus, flanger, etc.).

WARNING: Ohm currently doesn't have a built-in limiter, and it's trivial to generate a signal that will massively clip the output. So watch your output levels!

## Usage

The way it works is you start the engine with the path to a koto file as an argument:

```
cargo run --release -- examples/sine.koto
```

Ohm will parse the code in the file and, if it's valid, will start playing back the audio. It will listen for any changes in the file and reconstruct the audio graph on save.

Note that re-evaluating the code currently causes discontinuities in the audio output, since the new graph is reconstructed in place of the old one.

## Functions

Ohm currently contains the following high-level synthesis functions and effects:

- `sine(freq)` a sine wave
- `square(freq)` a square wave
- `saw(freq)` a bandlimited sawtooth wave
- `pulse(freq)` a pulse wave with a single-sample impulse, primarily useful for triggering `ar`, `env` or `pluck` functions
- `noise()` white noise, also useful as a randomization source
- `ar(attack, release, trig)` a simple attack-release envelope, which can be triggered by a pulse function
- `svf(mode ("lp" | "hp" | "bp"), cutoff, resonance, input)` a state-variable filter with lowpass (lp), highpass (hp) and bandpass (bp) modes.
- `seq(list, trig)` a sequencer that outputs the next value in the list every time it receives a trigger
- `pipe(delay, input)` delays a signal by *n* samples
- `pluck(frequency, tone, damping, trig)` a Karplus-Strong pluck string synthesis voice
- `reverb(input)` a FDN-reverb effect
- `delay(input)` an echo/delay effect
- basic arithmetic operators `+`, `-`, `*` and `/` can be applied to any signal

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
