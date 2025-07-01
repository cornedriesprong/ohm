use petgraph::{graph::NodeIndex, prelude::StableDiGraph, visit::EdgeRef};
use rtsan_standalone::nonblocking;
use std::collections::{HashMap, HashSet};

use crate::nodes::{Frame, Node, Op};

type Graph = StableDiGraph<Box<Op>, ()>;

#[derive(Clone)]
pub(crate) struct AudioGraph {
    graph: Graph,
    sorted_nodes: Vec<NodeIndex>,
    inputs: Vec<Frame>,
    outputs: Vec<Frame>,
}

impl AudioGraph {
    pub(crate) fn new() -> Self {
        Self {
            graph: Graph::new(),
            sorted_nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    pub(crate) fn add_node(&mut self, node: Op) -> NodeIndex {
        let index = self.graph.add_node(Box::new(node));
        self.update_processing_order();
        index
    }

    fn connect_node(&mut self, from: NodeIndex, to: NodeIndex) {
        self.graph.add_edge(from.into(), to.into(), ());
        self.update_processing_order();
    }

    #[inline]
    #[nonblocking]
    pub(crate) fn tick(&mut self) -> Frame {
        for &node_index in &self.sorted_nodes {
            self.inputs.clear();

            self.inputs.extend(
                self.graph
                    .edges_directed(node_index, petgraph::Direction::Incoming)
                    .map(|edge| self.outputs[edge.source().index()]),
            );

            self.outputs[node_index.index()] = self.graph[node_index].tick(&self.inputs);
        }

        self.outputs[self.sorted_nodes.len() - 1]
    }

    fn update_processing_order(&mut self) {
        self.sorted_nodes = petgraph::algo::toposort(&self.graph, None).expect("Graph has cycles");
        self.inputs.resize(self.graph.node_count(), [0.0, 0.0]);
        self.outputs.resize(self.graph.node_count(), [0.0, 0.0]);
    }

    pub(crate) fn apply_diff(&mut self, mut new_graph: AudioGraph) {
        let mut old_nodes: HashMap<u64, NodeIndex> = HashMap::new();
        for &node_idx in &self.sorted_nodes {
            let hash = self.graph[node_idx].compute_hash();
            old_nodes.insert(hash, node_idx);
        }

        for &new_node_idx in &new_graph.sorted_nodes {
            let new_hash = new_graph.graph[new_node_idx].compute_hash();
            if let Some(&old_node_idx) = old_nodes.get(&new_hash) {
                let old_node = &self.graph[old_node_idx];
                let new_node = &mut new_graph.graph[new_node_idx];

                if let (Op::Node { node: new, .. }, Op::Node { node: old, .. }) = 
                    (&mut **new_node, &**old_node) {
                    *new = old.clone();
                }
            }
        }

        // Replace current graph with the state-transferred new graph
        *self = new_graph;
    }
}

pub(crate) fn parse_to_audio_graph(expr: Op) -> AudioGraph {
    let mut graph = AudioGraph::new();

    fn add_expr_to_graph(expr: &Op, graph: &mut AudioGraph) -> NodeIndex {
        match expr {
            Op::Constant { .. } => add_node(vec![], expr, graph),
            Op::Node { inputs, .. } => add_node(inputs.iter().collect::<Vec<_>>(), expr, graph),
            Op::Gain(lhs, rhs) => add_node(vec![lhs, rhs], expr, graph),
            Op::Mix(lhs, rhs) => add_node(vec![lhs, rhs], expr, graph),
        }
    }

    fn add_node(inputs: Vec<&Op>, kind: &Op, graph: &mut AudioGraph) -> NodeIndex {
        let mut input_indices: Vec<_> = inputs
            .into_iter()
            .map(|input| add_expr_to_graph(&*input, graph))
            .collect();

        let node_idx = graph.add_node(kind.clone());

        // inputs need to be connected in reverse order
        input_indices.reverse();

        for &input_idx in &input_indices {
            graph.connect_node(input_idx, node_idx);
        }

        node_idx
    }

    add_expr_to_graph(&expr, &mut graph);

    graph
}
