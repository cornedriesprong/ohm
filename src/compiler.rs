use crate::graph::Graph;
use crate::nodes::Node;
use crate::parser::Expr;
use fundsp::hacker32::*;
use petgraph::graph::NodeIndex;
use std::collections::HashMap;

/// compile a graph of Expr's into a graph DSP Nodes
pub(crate) struct Compiler {
    graph: Graph,
    env: HashMap<String, NodeIndex>,
    sample_rate: u32,
}

impl Compiler {
    pub(crate) fn new(sample_rate: u32) -> Self {
        Self {
            graph: Graph::new(),
            env: HashMap::new(),
            sample_rate,
        }
    }

    pub(crate) fn compile(mut self, expr: &Expr) -> Graph {
        if let Some(node_idx) = self.compile_expr(expr) {
            let _ = node_idx;
        }
        self.graph
    }

    fn compile_expr(&mut self, expr: &Expr) -> Option<NodeIndex> {
        match expr {
            Expr::Number(n) => Some(self.add_node(Node::Constant(*n), vec![])),
            Expr::Ref(name) => self.env.get(name).copied(),
            Expr::Call { func, args } => self.compile_call(func, args),
            Expr::Func { .. } => None,
        }
    }

    fn compile_call(&mut self, func: &Expr, args: &[Box<Expr>]) -> Option<NodeIndex> {
        // Compile all arguments, return None if any fail
        let compiled_args: Vec<NodeIndex> = args
            .iter()
            .map(|arg| self.compile_expr(arg))
            .collect::<Option<_>>()?;

        match func {
            Expr::Ref(name) => self.compile_builtin(name, compiled_args),
            Expr::Func { params, body } => self.apply_user_function(params, body, compiled_args),
            _ => {
                println!("Warning: higher-order functions not yet supported");
                None
            }
        }
    }

    fn apply_user_function(
        &mut self,
        params: &[String],
        body: &Expr,
        args: Vec<NodeIndex>,
    ) -> Option<NodeIndex> {
        if params.len() != args.len() {
            println!(
                "Warning: function expects {} arguments but got {}",
                params.len(),
                args.len()
            );
            return None;
        }

        let saved_env = self.env.clone();
        self.env.extend(params.iter().zip(&args).map(|(p, &n)| (p.clone(), n)));
        let result = self.compile_expr(body);
        self.env = saved_env;
        result
    }

    fn compile_builtin(&mut self, name: &str, args: Vec<NodeIndex>) -> Option<NodeIndex> {
        match name {
            "ramp" => {
                let freq = self.arg(&args, 0)?;
                Some(self.add_node(
                    Node::Ramp {
                        phase: 0.0,
                        sample_rate: self.sample_rate,
                    },
                    vec![freq],
                ))
            }
            "sin" | "lfo" => {
                let freq = self.arg(&args, 0)?;
                let mut node_args = vec![freq];
                if let Some(phase_offset) = self.arg(&args, 1) {
                    node_args.push(phase_offset);
                    if let Some(fb) = self.arg(&args, 2) {
                        node_args.push(fb);
                    }
                }
                Some(self.add_node(
                    Node::Osc {
                        phase: 0.0,
                        z: 0.0,
                        sample_rate: self.sample_rate,
                    },
                    node_args,
                ))
            }
            "saw" => self.unary_fundsp(&args, saw()),
            "sqr" => self.unary_fundsp(&args, square()),
            "tri" => self.unary_fundsp(&args, triangle()),
            "organ" => self.unary_fundsp(&args, organ()),
            "softsaw" => self.unary_fundsp(&args, soft_saw()),
            "rossler" => self.unary_fundsp(&args, rossler()),
            "lorenz" => self.unary_fundsp(&args, lorenz()),
            "noise" => Some(self.add_fundsp_node(Box::new(noise()), false, vec![])),
            "sum" => self.binary(&args, Node::Sum),
            "diff" => self.binary(&args, Node::Diff),
            "mul" | "gain" => self.binary(&args, Node::Gain),
            "divide" => self.binary(&args, Node::Divide),
            "wrap" => self.binary(&args, Node::Wrap),
            "power" => self.binary(&args, Node::Power),
            "greater" => self.binary(&args, Node::Greater),
            "less" => self.binary(&args, Node::Less),
            "equal" => self.binary(&args, Node::Equal),
            "clip" => self.unary_fundsp(&args, clip()),
            "round" => self.unary(&args, Node::Round),
            "floor" => self.unary(&args, Node::Floor),
            "ceil" => self.unary(&args, Node::Ceil),
            "tanh" => self.unary_fundsp(&args, shape(Tanh(1.0))),
            "log" => self.unary(&args, Node::Log),
            "sh" => self.binary(
                &args,
                Node::SampleAndHold {
                    value: [0.0; 2],
                    prev: 0.0,
                },
            ),
            "pan" => self.binary_fundsp_stereo(&args, panner()),
            "seq" | "mix" => {
                Some(self.add_node(if name == "seq" { Node::Seq } else { Node::Mix }, args))
            }
            "onepole" | "smooth" => self.binary_fundsp(&args, lowpole()),
            "lp" => self.ternary_fundsp(&args, lowpass()),
            "bp" => self.ternary_fundsp(&args, bandpass()),
            "hp" => self.ternary_fundsp(&args, highpass()),
            "moog" => self.ternary_fundsp(&args, moog()),
            "delay" => self.binary(
                &args,
                Node::Delay {
                    buffer: Box::new([[0.0; 2]; 48000]),
                    write_pos: 0,
                },
            ),
            "reverb" => self.unary_fundsp_stereo(
                &args,
                reverb2_stereo(10.0, 2.0, 0.9, 1.0, lowpole_hz(18000.0)),
            ),
            "buf" | "file" | "play" | "tap" | "rec" => {
                println!("Warning: {} not yet implemented in compiler", name);
                None
            }
            _ => {
                println!("Unknown builtin: {}", name);
                None
            }
        }
    }

    fn arg(&self, args: &[NodeIndex], index: usize) -> Option<NodeIndex> {
        args.get(index).copied()
    }

    fn unary(&mut self, args: &[NodeIndex], node: Node) -> Option<NodeIndex> {
        let input = self.arg(args, 0)?;
        Some(self.add_node(node, vec![input]))
    }

    fn binary(&mut self, args: &[NodeIndex], node: Node) -> Option<NodeIndex> {
        let arg1 = self.arg(args, 0)?;
        let arg2 = self.arg(args, 1)?;
        Some(self.add_node(node, vec![arg1, arg2]))
    }

    fn unary_fundsp(
        &mut self,
        args: &[NodeIndex],
        audio_unit: impl AudioUnit + 'static,
    ) -> Option<NodeIndex> {
        let input = self.arg(args, 0)?;
        Some(self.add_fundsp_node(Box::new(audio_unit), false, vec![input]))
    }

    fn unary_fundsp_stereo(
        &mut self,
        args: &[NodeIndex],
        audio_unit: impl AudioUnit + 'static,
    ) -> Option<NodeIndex> {
        let input = self.arg(args, 0)?;
        Some(self.add_fundsp_node(Box::new(audio_unit), true, vec![input]))
    }

    fn binary_fundsp(
        &mut self,
        args: &[NodeIndex],
        audio_unit: impl AudioUnit + 'static,
    ) -> Option<NodeIndex> {
        let arg1 = self.arg(args, 0)?;
        let arg2 = self.arg(args, 1)?;
        Some(self.add_fundsp_node(Box::new(audio_unit), false, vec![arg1, arg2]))
    }

    fn binary_fundsp_stereo(
        &mut self,
        args: &[NodeIndex],
        audio_unit: impl AudioUnit + 'static,
    ) -> Option<NodeIndex> {
        let arg1 = self.arg(args, 0)?;
        let arg2 = self.arg(args, 1)?;
        Some(self.add_fundsp_node(Box::new(audio_unit), true, vec![arg1, arg2]))
    }

    fn ternary_fundsp(
        &mut self,
        args: &[NodeIndex],
        audio_unit: impl AudioUnit + 'static,
    ) -> Option<NodeIndex> {
        let arg1 = self.arg(args, 0)?;
        let arg2 = self.arg(args, 1)?;
        let arg3 = self.arg(args, 2)?;
        Some(self.add_fundsp_node(Box::new(audio_unit), false, vec![arg1, arg2, arg3]))
    }

    fn add_node(&mut self, node: Node, args: Vec<NodeIndex>) -> NodeIndex {
        let node_idx = self.graph.add_node(node);
        for arg in args {
            self.graph.connect_node(arg, node_idx);
        }
        node_idx
    }

    fn add_fundsp_node(
        &mut self,
        audio_unit: Box<dyn AudioUnit>,
        is_stereo: bool,
        args: Vec<NodeIndex>,
    ) -> NodeIndex {
        let num_inputs = audio_unit.inputs();
        let node = Node::FunDSP {
            audio_unit,
            is_stereo,
            input_buffer: vec![0.0; num_inputs],
        };
        self.add_node(node, args)
    }
}
