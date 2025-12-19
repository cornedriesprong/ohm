use crate::graph::Graph;
use crate::nodes::Node;
use crate::parser::Expr;
use fundsp::hacker32::*;
use petgraph::graph::NodeIndex;
use std::collections::HashMap;

/// compile a graph of Expr's into a graph DSP Nodes
pub(crate) struct Compiler {
    graph: Graph,
    env: HashMap<String, usize>,
    binding_to_node: HashMap<usize, NodeIndex>,
    cycles: Vec<(NodeIndex, usize)>,
    next_id: usize,
    sample_rate: u32,
}

impl Compiler {
    pub(crate) fn new(sample_rate: u32) -> Self {
        Self {
            graph: Graph::new(),
            env: HashMap::new(),
            binding_to_node: HashMap::new(),
            cycles: Vec::new(),
            next_id: 0,
            sample_rate,
        }
    }

    pub(crate) fn compile(mut self, exprs: &[Expr]) -> Graph {
        for expr in exprs {
            if let Some(node_idx) = self.compile_expr(expr) {
                let _ = node_idx;
            }
        }
        self.connect_cycles();
        self.graph
    }

    fn compile_expr(&mut self, expr: &Expr) -> Option<NodeIndex> {
        match expr {
            Expr::Number(n) => Some(self.add_node(Node::Constant(*n), vec![])),
            Expr::Ref(name) => self.binding_to_node.get(self.env.get(name)?).copied(),
            Expr::FbRef(name) => {
                let z1 = self.add_node(Node::Z1 { z: [0.0; 2] }, vec![]);
                if let Some(&id) = self.env.get(name) {
                    self.cycles.push((z1, id));
                }
                Some(z1)
            }
            Expr::Assign { name, value } => {
                let id = self.next_id;
                self.next_id += 1;

                let node = self.compile_expr(value)?;
                self.env.insert(name.clone(), id);
                self.binding_to_node.insert(id, node);

                None
            }
            Expr::Call { func, args } => self.compile_call(func, args),
            Expr::Func { .. } => None,
        }
    }

    fn connect_cycles(&mut self) {
        for (z1_node, binding_id) in &self.cycles {
            if let Some(&target_node) = self.binding_to_node.get(binding_id) {
                self.graph.connect_node(target_node, *z1_node);
            }
        }
    }

    fn compile_call(&mut self, func: &Expr, args: &[Box<Expr>]) -> Option<NodeIndex> {
        let args: Vec<NodeIndex> = args
            .iter()
            .map(|arg| self.compile_expr(arg))
            .collect::<Option<_>>()?;

        match func {
            Expr::Ref(name) => self.compile_builtin(name, args),
            Expr::Func { params, body } => self.apply_user_function(params, body, args),
            _ => None,
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
        for (param, &arg_node) in params.iter().zip(&args) {
            let binding_id = self.next_id;
            self.next_id += 1;
            self.env.insert(param.clone(), binding_id);
            self.binding_to_node.insert(binding_id, arg_node);
        }
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
                        sample_rate: self.sample_rate,
                    },
                    node_args,
                ))
            }
            "saw" => self.unary_fundsp(&args, saw(), false),
            "sqr" => self.unary_fundsp(&args, square(), false),
            "tri" => self.unary_fundsp(&args, triangle(), false),
            "organ" => self.unary_fundsp(&args, organ(), false),
            "softsaw" => self.unary_fundsp(&args, soft_saw(), false),
            "rossler" => self.unary_fundsp(&args, rossler(), false),
            "lorenz" => self.unary_fundsp(&args, lorenz(), false),
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
            "clip" => self.unary_fundsp(&args, clip(), false),
            "round" => self.unary(&args, Node::Round),
            "floor" => self.unary(&args, Node::Floor),
            "ceil" => self.unary(&args, Node::Ceil),
            "tanh" => self.unary_fundsp(&args, shape(Tanh(1.0)), false),
            "log" => self.unary(&args, Node::Log),
            "sh" => self.binary(
                &args,
                Node::SampleAndHold {
                    value: [0.0; 2],
                    prev: 0.0,
                },
            ),
            "pan" => self.binary_fundsp(&args, panner(), true),
            "seq" | "mix" => {
                Some(self.add_node(if name == "seq" { Node::Seq } else { Node::Mix }, args))
            }
            "onepole" | "smooth" => self.binary_fundsp(&args, lowpole(), false),
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
            "reverb" => self.unary_fundsp(
                &args,
                reverb2_stereo(10.0, 2.0, 0.9, 1.0, lowpole_hz(18000.0)),
                true,
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
        stereo: bool,
    ) -> Option<NodeIndex> {
        let input = self.arg(args, 0)?;
        Some(self.add_fundsp_node(Box::new(audio_unit), stereo, vec![input]))
    }

    fn binary_fundsp(
        &mut self,
        args: &[NodeIndex],
        audio_unit: impl AudioUnit + 'static,
        stereo: bool,
    ) -> Option<NodeIndex> {
        let arg1 = self.arg(args, 0)?;
        let arg2 = self.arg(args, 1)?;
        Some(self.add_fundsp_node(Box::new(audio_unit), stereo, vec![arg1, arg2]))
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
