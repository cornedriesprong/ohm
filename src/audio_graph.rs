use petgraph::{
    graph::NodeIndex,
    prelude::StableDiGraph,
    visit::{EdgeRef, IntoEdgeReferences},
};
use rtsan_standalone::nonblocking;
use std::collections::HashMap;
use std::fmt;

use crate::nodes::{Node, NodeKind};

pub(crate) type BoxedNode = Box<NodeKind>;
type Graph = StableDiGraph<BoxedNode, ()>;

#[derive(Clone)]
pub(crate) struct AudioGraph {
    graph: Graph,
    sorted_nodes: Vec<NodeIndex>,
    inputs: Vec<f32>,
    outputs: Vec<f32>,
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
    pub(crate) fn tick(&mut self) -> f32 {
        for &node_index in &self.sorted_nodes {
            self.inputs.clear();

            // assert that the inputs vector is large enough to hold all inputs
            // so we don't need to allocate
            assert!(self.inputs.capacity() >= self.graph.node_count());

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
        self.inputs.resize(self.graph.node_count(), 0.0);
        self.outputs.resize(self.graph.node_count(), 0.0);
    }

    pub(crate) fn apply_diff(&mut self, mut new_graph: AudioGraph) {
        // Build hash map of old nodes by their structure hash
        let mut old_nodes: HashMap<u64, NodeIndex> = HashMap::new();
        for &node_idx in &self.sorted_nodes {
            let hash = self.graph[node_idx].compute_hash();
            old_nodes.insert(hash, node_idx);
        }

        // Transfer state from matching old nodes to new nodes
        for &new_node_idx in &new_graph.sorted_nodes {
            let new_hash = new_graph.graph[new_node_idx].compute_hash();
            if let Some(&old_node_idx) = old_nodes.get(&new_hash) {
                // Found a matching node - transfer state
                let old_node = &self.graph[old_node_idx];
                let new_node = &mut new_graph.graph[new_node_idx];
                new_node.transfer_state_from(old_node);
            }
        }

        // Replace current graph with the state-transferred new graph
        *self = new_graph;
    }
}

impl fmt::Debug for AudioGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // print out the graph in DOT format so we can visualize it with Graphviz
        writeln!(f, "digraph AudioGraph {{")?;
        writeln!(f, "  rankdir=LR;")?;
        writeln!(f, "  node [shape=circle style=filled];\n")?;

        // nodes
        for (order, &node) in self.sorted_nodes.iter().enumerate() {
            writeln!(
                f,
                "  n{} [label=\"{:?} ({})\"];",
                node.index(),
                self.graph[node],
                order
            )?;
        }

        // edges
        for edge in self.graph.edge_references() {
            writeln!(
                f,
                "  n{} -> n{};",
                edge.source().index(),
                edge.target().index()
            )?;
        }
        write!(f, "}}")
    }
}

pub(crate) fn parse_to_audio_graph(expr: NodeKind) -> AudioGraph {
    let mut graph = AudioGraph::new();

    fn add_expr_to_graph(expr: &NodeKind, graph: &mut AudioGraph) -> NodeIndex {
        match expr {
            NodeKind::Constant(_) => add_node(vec![], expr, graph),
            NodeKind::Sine { freq, .. } => add_node(vec![freq], expr, graph),
            NodeKind::Square { freq, .. } => add_node(vec![freq], expr, graph),
            NodeKind::Saw { freq, .. } => add_node(vec![freq], expr, graph),
            NodeKind::Noise(_) => add_node(vec![], expr, graph),
            NodeKind::Pulse { freq, .. } => add_node(vec![freq], expr, graph),
            NodeKind::Gain { lhs, rhs, .. } => add_node(vec![lhs, rhs], expr, graph),
            NodeKind::Mix { lhs, rhs, .. } => add_node(vec![lhs, rhs], expr, graph),
            NodeKind::AR {
                attack,
                release,
                trig,
                ..
            } => add_node(vec![attack, release, trig], &expr, graph),
            NodeKind::SVF {
                mode,
                cutoff,
                resonance,
                input,
                ..
            } => add_node(vec![mode, cutoff, resonance, input], &expr, graph),
            NodeKind::Seq { trig, .. } => add_node(vec![trig], expr, graph),
            NodeKind::Pipe { delay, input, .. } => add_node(vec![delay, input], expr, graph),
            NodeKind::Pluck {
                freq,
                tone,
                damping,
                trig,
                ..
            } => add_node(vec![freq, tone, damping, trig], expr, graph),
            NodeKind::Reverb { input, .. } => add_node(vec![input], expr, graph),
            NodeKind::Delay { input, .. } => add_node(vec![input], expr, graph),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nodes::*;

    #[test]
    fn test_parse_graph_1() {
        // 440
        let expr = constant(440.0);

        let mut graph = parse_to_audio_graph(expr);

        assert_eq!(graph.graph.node_count(), 1);
        assert_eq!(graph.graph.edge_count(), 0);
        assert_eq!(graph.tick(), 440.0);
    }

    #[test]
    fn test_parse_graph_2() {
        // sine(440)
        let expr = sine(constant(440.0));
        let mut graph = parse_to_audio_graph(expr);

        assert_eq!(graph.graph.node_count(), 2);
        assert_eq!(graph.graph.edge_count(), 1);
        assert_eq!(graph.tick(), 0.0);
    }

    #[test]
    fn test_parse_graph_3() {
        // sine(sine(0.1) * 100)
        let expr = sine(gain(sine(constant(0.1)), constant(100.0)));
        let mut graph = parse_to_audio_graph(expr);

        assert_eq!(graph.graph.node_count(), 5);
        assert_eq!(graph.graph.edge_count(), 4);
        assert_eq!(graph.tick(), 0.0);
    }

    #[test]
    fn test_node_comparison() {
        let node1 = constant(1.0);
        let node2 = constant(1.0);
        let node3 = constant(2.0);

        assert_eq!(node1, node2);
        assert_ne!(node1, node3);
    }

    #[test]
    fn test_node_tick() {
        let mut node = constant(42.0);
        assert_eq!(node.tick(&[]), 42.0);
    }

    #[test]
    fn test_node_hashing_identical_nodes() {
        let node1 = sine(constant(440.0));
        let node2 = sine(constant(440.0));
        let node3 = sine(constant(880.0));

        assert_eq!(node1.compute_hash(), node2.compute_hash());
        assert_ne!(node1.compute_hash(), node3.compute_hash());
    }

    #[test]
    fn test_node_hashing_different_types() {
        let sine_node = sine(constant(440.0));
        let square_node = square(constant(440.0));
        let constant_node = constant(440.0);

        let sine_hash = sine_node.compute_hash();
        let square_hash = square_node.compute_hash();
        let constant_hash = constant_node.compute_hash();

        assert_ne!(sine_hash, square_hash);
        assert_ne!(sine_hash, constant_hash);
        assert_ne!(square_hash, constant_hash);
    }

    #[test]
    fn test_node_hashing_complex_graphs() {
        // sine(440) + sine(880)
        let graph1 = mix(sine(constant(440.0)), sine(constant(880.0)));
        let graph2 = mix(sine(constant(440.0)), sine(constant(880.0)));
        let graph3 = mix(sine(constant(880.0)), sine(constant(440.0))); // swapped order

        assert_eq!(graph1.compute_hash(), graph2.compute_hash());
        assert_ne!(graph1.compute_hash(), graph3.compute_hash());
    }

    #[test]
    fn test_state_transfer_sine_node() {
        let mut old_node = sine(constant(440.0));
        let mut new_node = sine(constant(440.0));

        // Advance old node's phase by ticking it
        for _ in 0..100 {
            old_node.tick(&[440.0]);
        }

        // Get the old phase
        let old_phase = match &old_node {
            NodeKind::Sine { node, .. } => node.phase,
            _ => panic!("Expected sine node"),
        };

        // Transfer state
        new_node.transfer_state_from(&old_node);

        // Check that phase was transferred
        let new_phase = match &new_node {
            NodeKind::Sine { node, .. } => node.phase,
            _ => panic!("Expected sine node"),
        };

        assert_eq!(old_phase, new_phase);
    }

    #[test]
    fn test_state_transfer_ar_node() {
        let mut old_node = ar(constant(1000.0), constant(1000.0), constant(1.0));
        let mut new_node = ar(constant(1000.0), constant(1000.0), constant(1.0));

        // Trigger and advance old node
        old_node.tick(&[1000.0, 1000.0, 1.0]); // trigger
        for _ in 0..50 {
            old_node.tick(&[1000.0, 1000.0, 0.0]); // advance attack phase
        }

        // Get old state
        let (old_state, old_value, old_time) = match &old_node {
            NodeKind::AR { node, .. } => (node.state, node.value, node.time),
            _ => panic!("Expected AR node"),
        };

        // Transfer state
        new_node.transfer_state_from(&old_node);

        // Check that state was transferred
        let (new_state, new_value, new_time) = match &new_node {
            NodeKind::AR { node, .. } => (node.state, node.value, node.time),
            _ => panic!("Expected AR node"),
        };

        assert_eq!(format!("{:?}", old_state), format!("{:?}", new_state));
        assert_eq!(old_value, new_value);
        assert_eq!(old_time, new_time);
    }

    #[test]
    fn test_apply_diff_preserves_state() {
        // Create initial graph: sine(440)
        let mut old_graph = parse_to_audio_graph(sine(constant(440.0)));

        // Advance the sine node's phase
        for _ in 0..100 {
            old_graph.tick();
        }

        // Get the phase from the sine node
        let old_phase = get_sine_phase(&old_graph, 1); // sine node is at index 1

        // Create "new" graph with same structure
        let new_graph = parse_to_audio_graph(sine(constant(440.0)));

        // Apply diff
        old_graph.apply_diff(new_graph);

        // Check that phase was preserved
        let new_phase = get_sine_phase(&old_graph, 1);
        assert_eq!(old_phase, new_phase);
    }

    #[test]
    fn test_apply_diff_different_structure_no_transfer() {
        // Create initial graph: sine(440)
        let mut old_graph = parse_to_audio_graph(sine(constant(440.0)));

        // Advance the sine node's phase
        for _ in 0..100 {
            old_graph.tick();
        }

        // Create new graph with different frequency: sine(880)
        let new_graph = parse_to_audio_graph(sine(constant(880.0)));

        // Apply diff
        old_graph.apply_diff(new_graph);

        // Check that phase was reset (new node starts at 0)
        let new_phase = get_sine_phase(&old_graph, 1);
        assert_eq!(new_phase, 0.0);
    }

    #[test]
    fn test_apply_diff_partial_match() {
        // Create initial graph: sine(440) + sine(880)
        let mut old_graph = parse_to_audio_graph(mix(sine(constant(440.0)), sine(constant(880.0))));

        // Advance both sine nodes
        for _ in 0..100 {
            old_graph.tick();
        }

        // Get phases by frequency
        let phase_440 = find_sine_phase_by_freq(&old_graph, 440.0).unwrap();

        // Create new graph where only one frequency changes: sine(440) + sine(1760)
        let new_graph = parse_to_audio_graph(mix(sine(constant(440.0)), sine(constant(1760.0))));

        // Apply diff
        old_graph.apply_diff(new_graph);

        // Check that 440Hz sine kept its phase, but 1760Hz sine started fresh
        let new_phase_440 = find_sine_phase_by_freq(&old_graph, 440.0).unwrap();
        let new_phase_1760 = find_sine_phase_by_freq(&old_graph, 1760.0).unwrap();

        assert_eq!(phase_440, new_phase_440); // preserved
        assert_eq!(new_phase_1760, 0.0); // reset
    }

    #[test]
    fn test_apply_diff_simple_preservation() {
        // Test that identical sine nodes preserve state
        let mut old_graph = parse_to_audio_graph(sine(constant(440.0)));

        // Advance the sine
        for _ in 0..100 {
            old_graph.tick();
        }

        let phase_440 = find_sine_phase_by_freq(&old_graph, 440.0).unwrap();
        assert!(phase_440 > 0.0); // Should have advanced

        // Create identical graph
        let new_graph = parse_to_audio_graph(sine(constant(440.0)));

        // Apply diff
        old_graph.apply_diff(new_graph);

        // Phase should be preserved
        let new_phase_440 = find_sine_phase_by_freq(&old_graph, 440.0).unwrap();
        assert_eq!(phase_440, new_phase_440);
    }

    #[test]
    fn test_apply_diff_different_frequencies() {
        // Test that different sine frequencies get reset
        let mut old_graph = parse_to_audio_graph(sine(constant(440.0)));

        // Advance the sine
        for _ in 0..100 {
            old_graph.tick();
        }

        let phase_440 = find_sine_phase_by_freq(&old_graph, 440.0).unwrap();
        assert!(phase_440 > 0.0);

        // Create graph with different frequency
        let new_graph = parse_to_audio_graph(sine(constant(880.0)));

        // Apply diff
        old_graph.apply_diff(new_graph);

        // Should have a fresh sine at 880Hz
        let new_phase_880 = find_sine_phase_by_freq(&old_graph, 880.0).unwrap();
        assert_eq!(new_phase_880, 0.0); // Should start fresh
    }

    // Helper function to find and extract phase from first sine node with given frequency
    fn find_sine_phase_by_freq(graph: &AudioGraph, target_freq: f32) -> Option<f32> {
        for &node_idx in &graph.sorted_nodes {
            if let Some(node) = graph.graph.node_weight(node_idx) {
                match &**node {
                    NodeKind::Sine { freq, node } => {
                        if let NodeKind::Constant(const_node) = &**freq {
                            if (const_node.value - target_freq).abs() < 0.01 {
                                return Some(node.phase);
                            }
                        }
                    }
                    _ => continue,
                }
            }
        }
        None
    }

    // Helper function to extract phase from sine node at given index
    fn get_sine_phase(graph: &AudioGraph, node_index: usize) -> f32 {
        if node_index >= graph.sorted_nodes.len() {
            panic!(
                "Node index {} out of bounds (graph has {} nodes)",
                node_index,
                graph.sorted_nodes.len()
            );
        }
        let node_idx = graph.sorted_nodes[node_index];
        match &**graph.graph.node_weight(node_idx).unwrap() {
            NodeKind::Sine { node, .. } => node.phase,
            other => panic!(
                "Expected sine node at index {}, found {:?}",
                node_index, other
            ),
        }
    }
}
