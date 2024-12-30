use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::nodes::*;
use crate::parser::{Expr, OperatorType};

pub(crate) struct AudioGraph {
    graph: DiGraph<Arc<Mutex<Box<dyn Node>>>, ()>,
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

    pub(crate) fn connect(&mut self, from: NodeIndex, to: NodeIndex) {
        self.graph.add_edge(from.into(), to.into(), ());
        self.update_processing_order();
    }

    pub(crate) fn process(&mut self) -> f32 {
        let mut outputs: HashMap<NodeIndex, f32> = HashMap::new();

        for &node_index in &self.sorted_nodes {
            let input_value = self
                .graph
                .neighbors_directed(node_index, petgraph::Incoming)
                .map(|pred| outputs[&pred])
                .sum();

            let node = self.graph[node_index].clone();
            let mut node = node.lock().unwrap();
            let output_value = node.process(input_value);
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
                let node: Box<dyn Node> = match kind {
                    OperatorType::Sine => Box::new(Sine::new()),
                    OperatorType::Square => Box::new(Square::new()),
                    OperatorType::Gain => {
                        let gain_value = if let Some(Expr::Number(n)) = args.get(0) {
                            *n
                        } else {
                            1.0 // Default gain
                        };
                        Box::new(Gain::new(gain_value))
                    }
                    OperatorType::Offset => {
                        let offset_value = if let Some(Expr::Number(n)) = args.get(0) {
                            *n
                        } else {
                            0.0 // Default offset
                        };
                        Box::new(Offset::new(offset_value))
                    }
                    _ => unimplemented!(),
                };
                let node_index = graph.add_node(node);
                graph.connect(input_node, node_index);
                node_index
            }
            Expr::Number(n) => graph.add_node(Box::new(Constant::new(n))),
        }
    }

    add_expr_to_graph(expr, &mut graph);

    graph
}
