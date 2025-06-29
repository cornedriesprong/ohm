use petgraph::{graph::NodeIndex, prelude::StableDiGraph, visit::EdgeRef};
use rtsan_standalone::nonblocking;
use std::collections::{HashMap, HashSet};

use crate::nodes::{Frame, Node, NodeKind};

type Graph = StableDiGraph<Box<NodeKind>, ()>;

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

    pub(crate) fn add_node(&mut self, node: NodeKind) -> NodeIndex {
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
        // Build hash map of old nodes by their structure hash
        let mut old_nodes: HashMap<u64, NodeIndex> = HashMap::new();
        for &node_idx in &self.sorted_nodes {
            let hash = self.graph[node_idx].compute_hash();
            old_nodes.insert(hash, node_idx);
        }

        // Track which old nodes have been used for state transfer
        let mut used_old_nodes = HashSet::new();

        // First pass: Transfer state from nodes with exact hash matches
        for &new_node_idx in &new_graph.sorted_nodes {
            let new_hash = new_graph.graph[new_node_idx].compute_hash();
            if let Some(&old_node_idx) = old_nodes.get(&new_hash) {
                // Found an exact matching node - transfer state
                let old_node = &self.graph[old_node_idx];
                let new_node = &mut new_graph.graph[new_node_idx];
                new_node.transfer_state_from(old_node);
                used_old_nodes.insert(old_node_idx);
            }
        }

        // Second pass: For unmatched new nodes, try to find compatible old nodes by type
        for &new_node_idx in &new_graph.sorted_nodes {
            let new_hash = new_graph.graph[new_node_idx].compute_hash();
            if old_nodes.get(&new_hash).is_some() {
                continue; // Already handled in first pass
            }

            // Look for an unused old node of the same type
            if let Some(compatible_old_idx) =
                self.find_compatible_unused_node(&new_graph.graph[new_node_idx], &used_old_nodes)
            {
                let old_node = &self.graph[compatible_old_idx];
                let new_node = &mut new_graph.graph[new_node_idx];
                new_node.transfer_state_from(old_node);
                used_old_nodes.insert(compatible_old_idx);
            }
        }

        // Replace current graph with the state-transferred new graph
        *self = new_graph;
    }

    fn find_compatible_unused_node(
        &self,
        new_node: &NodeKind,
        used_nodes: &HashSet<NodeIndex>,
    ) -> Option<NodeIndex> {
        for &old_node_idx in &self.sorted_nodes {
            if used_nodes.contains(&old_node_idx) {
                continue; // Already used
            }

            let old_node = &self.graph[old_node_idx];
            if self.nodes_are_type_compatible(old_node, new_node) {
                return Some(old_node_idx);
            }
        }
        None
    }

    fn nodes_are_type_compatible(&self, old_node: &NodeKind, new_node: &NodeKind) -> bool {
        use NodeKind::*;
        match (old_node, new_node) {
            (Osc { .. }, Osc { .. }) => true,
            (Pulse { .. }, Pulse { .. }) => true,
            (Env { .. }, Env { .. }) => true,
            (Moog { .. }, Moog { .. }) => true,
            (SVF { .. }, SVF { .. }) => true,
            _ => false, // For now, only support oscillators, filters, and effects
        }
    }
}

pub(crate) fn parse_to_audio_graph(expr: NodeKind) -> AudioGraph {
    let mut graph = AudioGraph::new();

    fn add_expr_to_graph(expr: &NodeKind, graph: &mut AudioGraph) -> NodeIndex {
        match expr {
            NodeKind::Constant { .. } => add_node(vec![], expr, graph),
            NodeKind::Osc { freq, .. } => add_node(vec![freq], expr, graph),
            NodeKind::Pulse { freq, .. } => add_node(vec![freq], expr, graph),
            NodeKind::Noise(_) => add_node(vec![], expr, graph),
            NodeKind::Gain(lhs, rhs) => add_node(vec![lhs, rhs], expr, graph),
            NodeKind::Mix(lhs, rhs) => add_node(vec![lhs, rhs], expr, graph),
            NodeKind::Env { segments, trig, .. } => {
                let mut inputs = Vec::new();
                for (value, duration) in segments {
                    inputs.push(value);
                    inputs.push(duration);
                }
                inputs.push(trig);
                add_node(inputs, expr, graph)
            }
            NodeKind::SVF {
                input,
                cutoff,
                resonance,
                ..
            } => add_node(vec![input, cutoff, resonance], &expr, graph),
            NodeKind::Moog {
                input,
                cutoff,
                resonance,
                ..
            } => add_node(vec![input, cutoff, resonance], expr, graph),
            NodeKind::Seq { trig, values, .. } => {
                let mut inputs = values.iter().collect::<Vec<_>>();
                inputs.push(trig);
                add_node(inputs, expr, graph)
            }
            NodeKind::Pan { input, value, .. } => add_node(vec![input, value], &expr, graph),
            NodeKind::Pluck {
                freq,
                tone,
                damping,
                trig,
                ..
            } => add_node(vec![freq, tone, damping, trig], expr, graph),
            NodeKind::Reverb { input, .. } => add_node(vec![input], expr, graph),
            NodeKind::Delay { input, .. } => add_node(vec![input], expr, graph),
            NodeKind::Sampler { .. } => add_node(vec![], expr, graph),
        }
    }

    fn add_node(inputs: Vec<&NodeKind>, kind: &NodeKind, graph: &mut AudioGraph) -> NodeIndex {
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
