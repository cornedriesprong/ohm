use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::nodes::*;
use crate::parser::{Expr, OperatorType};

#[derive(Clone, Debug)]
enum EdgeType {
    Audio,
    Control,
}

pub(crate) struct AudioGraph {
    graph: DiGraph<Arc<Mutex<Box<dyn Node>>>, EdgeType>,
    sorted_nodes: Vec<NodeIndex>, // Store sorted node order
}

impl AudioGraph {
    pub(crate) fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            sorted_nodes: Vec::new(),
        }
    }

    pub(crate) fn add_node(&mut self, node: Box<dyn Node>) -> NodeIndex {
        let index = self.graph.add_node(Arc::new(Mutex::new(node)));
        self.update_processing_order();
        index
    }

    pub(crate) fn connect_audio(&mut self, from: NodeIndex, to: NodeIndex) {
        self.graph.add_edge(from.into(), to.into(), EdgeType::Audio);
        self.update_processing_order();
    }

    pub(crate) fn connect_control(&mut self, from: NodeIndex, to: NodeIndex) {
        self.graph
            .add_edge(from.into(), to.into(), EdgeType::Control);
        self.update_processing_order();
    }

    pub(crate) fn process(&mut self) -> f32 {
        let mut outputs: HashMap<NodeIndex, f32> = HashMap::new();

        for &node_index in &self.sorted_nodes {
            // Get audio inputs
            let audio_input = self
                .graph
                .edges_directed(node_index, petgraph::Direction::Incoming)
                .filter(|edge| matches!(edge.weight(), EdgeType::Audio))
                .map(|edge| outputs[&edge.source()])
                .sum();

            // Get control input
            let control_input = self
                .graph
                .edges_directed(node_index, petgraph::Direction::Incoming)
                .find(|edge| matches!(edge.weight(), EdgeType::Control))
                .map(|edge| outputs[&edge.source()]);

            let node = self.graph[node_index].clone();
            let mut node = node.lock().unwrap();
            let output_value = node.process(audio_input, control_input);
            outputs.insert(node_index, output_value);
        }

        outputs[&self.sorted_nodes.last().unwrap()]
    }

    fn update_processing_order(&mut self) {
        self.sorted_nodes = toposort(&self.graph, None).expect("Graph has cycles");
    }
}

pub(crate) fn parse_to_audio_graph(expr: Expr) -> AudioGraph {
    let mut graph = AudioGraph::new();

    fn add_expr_to_graph(expr: Expr, graph: &mut AudioGraph) -> NodeIndex {
        match expr {
            Expr::Operator { kind, input, args } => {
                let input_node = add_expr_to_graph(*input, graph);
                match kind {
                    OperatorType::Sine => {
                        let node = Box::new(Sine::new());
                        let node_index = graph.add_node(node);
                        graph.connect_audio(input_node, node_index);
                        node_index
                    }
                    OperatorType::Square => {
                        let node = Box::new(Square::new());
                        let node_index = graph.add_node(node);
                        graph.connect_audio(input_node, node_index);
                        node_index
                    }
                    OperatorType::Noise => {
                        let node = Box::new(Noise::new());
                        let node_index = graph.add_node(node);
                        graph.connect_audio(input_node, node_index);
                        node_index
                    }
                    OperatorType::Gain => {
                        let node = Box::new(Gain::new());
                        let node_index = graph.add_node(node);

                        // Connect the left-hand side as the audio input
                        graph.connect_audio(input_node, node_index);

                        // Process and connect the right-hand side (gain amount) as control input
                        if let Some(expr) = args.get(0) {
                            match expr {
                                Expr::Number(n) => {
                                    let constant = Box::new(Constant::new(*n));
                                    let constant_index = graph.add_node(constant);
                                    graph.connect_control(constant_index, node_index);
                                }
                                Expr::Operator { .. } => {
                                    let node = add_expr_to_graph(expr.clone(), graph);
                                    graph.connect_control(node, node_index);
                                }
                            }
                        }

                        node_index
                    }
                    OperatorType::Offset => {
                        let node = Box::new(Offset::new());
                        let node_index = graph.add_node(node);

                        graph.connect_audio(input_node, node_index);

                        if let Some(expr) = args.get(0) {
                            match expr {
                                Expr::Number(n) => {
                                    println!("offset: {:?}", n);
                                    let constant = Box::new(Constant::new(*n));
                                    let constant_index = graph.add_node(constant);
                                    graph.connect_control(constant_index, node_index);
                                }
                                Expr::Operator { .. } => {
                                    let node = add_expr_to_graph(expr.clone(), graph);
                                    graph.connect_control(node, node_index);
                                }
                            }
                        }

                        node_index
                    }
                    _ => unimplemented!(),
                }
            }
            Expr::Number(n) => graph.add_node(Box::new(Constant::new(n))),
        }
    }

    add_expr_to_graph(expr, &mut graph);

    graph
}
