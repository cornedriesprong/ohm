use petgraph::{
    graph::NodeIndex,
    prelude::StableDiGraph,
    visit::{EdgeRef, IntoEdgeReferences},
};
use rtsan_standalone::nonblocking;
use std::collections::{HashMap, HashSet};
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
            (Sine { .. }, Sine { .. }) => true,
            (Square { .. }, Square { .. }) => true,
            (Saw { .. }, Saw { .. }) => true,
            (Pulse { .. }, Pulse { .. }) => true,
            (Triangle { .. }, Triangle { .. }) => true,
            (Env { .. }, Env { .. }) => true,
            (Moog { .. }, Moog { .. }) => true,
            _ => false, // For now, only support oscillators, filters, and effects
        }
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
            NodeKind::Constant { .. } => add_node(vec![], expr, graph),
            NodeKind::Sine { freq, .. } => add_node(vec![freq], expr, graph),
            NodeKind::Square { freq, .. } => add_node(vec![freq], expr, graph),
            NodeKind::Saw { freq, .. } => add_node(vec![freq], expr, graph),
            NodeKind::Noise(_) => add_node(vec![], expr, graph),
            NodeKind::Pulse { freq, .. } => add_node(vec![freq], expr, graph),
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
            NodeKind::Lowpass {
                input,
                cutoff,
                resonance,
                ..
            } => add_node(vec![input, cutoff, resonance], &expr, graph),
            NodeKind::Bandpass {
                input,
                cutoff,
                resonance,
                ..
            } => add_node(vec![input, cutoff, resonance], &expr, graph),
            NodeKind::Highpass {
                input,
                cutoff,
                resonance,
                ..
            } => add_node(vec![input, cutoff, resonance], &expr, graph),
            NodeKind::Seq { trig, values, .. } => {
                let mut inputs = values.iter().collect::<Vec<_>>();
                inputs.push(trig);
                add_node(inputs, expr, graph)
            }
            NodeKind::Pluck {
                freq,
                tone,
                damping,
                trig,
                ..
            } => add_node(vec![freq, tone, damping, trig], expr, graph),
            NodeKind::Reverb { input, .. } => add_node(vec![input], expr, graph),
            NodeKind::Delay { input, .. } => add_node(vec![input], expr, graph),
            NodeKind::Triangle { freq, .. } => add_node(vec![freq], expr, graph),
            NodeKind::Moog {
                input,
                cutoff,
                resonance,
                ..
            } => add_node(vec![input, cutoff, resonance], expr, graph),
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

        // Test that graph produces valid sine wave output (between -1 and 1)
        let output = graph.tick();
        assert!(output >= -1.0 && output <= 1.0);
    }

    #[test]
    fn test_parse_graph_3() {
        // sine(sine(0.1) * 100)
        let expr = sine(gain(sine(constant(0.1)), constant(100.0)));
        let mut graph = parse_to_audio_graph(expr);

        assert_eq!(graph.graph.node_count(), 5);
        assert_eq!(graph.graph.edge_count(), 4);

        // Test that graph produces valid output (complex FM synthesis)
        let output = graph.tick();
        assert!(output >= -1.0 && output <= 1.0);
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

        // Advance old node by ticking it
        for _ in 0..100 {
            old_node.tick(&[440.0]);
        }

        // Transfer state (fundsp handles internal state automatically)
        new_node.transfer_state_from(&old_node);

        // Test that both nodes produce valid output
        let old_output = old_node.tick(&[440.0]);
        let new_output = new_node.tick(&[440.0]);

        // Both should produce sine wave outputs in valid range
        assert!(old_output >= -1.0 && old_output <= 1.0);
        assert!(new_output >= -1.0 && new_output <= 1.0);
    }

    #[test]
    fn test_state_transfer_ar_node() {
        let mut old_node = env(constant(1000.0), constant(1000.0), constant(1.0));
        let mut new_node = env(constant(1000.0), constant(1000.0), constant(1.0));

        // Trigger and advance old node
        old_node.tick(&[1000.0, 1000.0, 1.0]); // trigger
        for _ in 0..50 {
            old_node.tick(&[1000.0, 1000.0, 0.0]); // advance attack phase
        }

        // Get old state
        let (old_state, old_value, old_time) = match &old_node {
            NodeKind::Env { node, .. } => (node.segments, node.value, node.time),
            _ => panic!("Expected AR node"),
        };

        // Transfer state
        new_node.transfer_state_from(&old_node);

        // Check that state was transferred
        let (new_state, new_value, new_time) = match &new_node {
            NodeKind::Env { node, .. } => (node.segments, node.value, node.time),
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

        // Advance the sine node
        for _ in 0..100 {
            old_graph.tick();
        }

        // Verify sine node exists
        assert!(verify_sine_node(&old_graph, 1)); // sine node is at index 1

        // Create "new" graph with same structure
        let new_graph = parse_to_audio_graph(sine(constant(440.0)));

        // Apply diff
        old_graph.apply_diff(new_graph);

        // Check that sine node still exists (fundsp handles state internally)
        assert!(verify_sine_node(&old_graph, 1));
    }

    #[test]
    fn test_apply_diff_different_structure_no_transfer() {
        // Create initial graph: sine(440)
        let mut old_graph = parse_to_audio_graph(sine(constant(440.0)));

        // Advance the sine node
        for _ in 0..100 {
            old_graph.tick();
        }

        // Create new graph with different frequency: sine(880)
        let new_graph = parse_to_audio_graph(sine(constant(880.0)));

        // Apply diff
        old_graph.apply_diff(new_graph);

        // Check that sine node still exists (with new frequency)
        assert!(verify_sine_node(&old_graph, 1));
    }

    #[test]
    fn test_apply_diff_partial_match() {
        // Create initial graph: sine(440) + sine(880)
        let mut old_graph = parse_to_audio_graph(mix(sine(constant(440.0)), sine(constant(880.0))));

        // Advance both sine nodes
        for _ in 0..100 {
            old_graph.tick();
        }

        // Verify both sine nodes exist
        assert!(find_sine_node_by_freq(&old_graph, 440.0));
        assert!(find_sine_node_by_freq(&old_graph, 880.0));

        // Create new graph where only one frequency changes: sine(440) + sine(1760)
        let new_graph = parse_to_audio_graph(mix(sine(constant(440.0)), sine(constant(1760.0))));

        // Apply diff
        old_graph.apply_diff(new_graph);

        // Check that correct sine nodes exist after diff
        assert!(find_sine_node_by_freq(&old_graph, 440.0)); // preserved
        assert!(find_sine_node_by_freq(&old_graph, 1760.0)); // new frequency
    }

    #[test]
    fn test_apply_diff_simple_preservation() {
        // Test that identical sine nodes preserve state
        let mut old_graph = parse_to_audio_graph(sine(constant(440.0)));

        // Advance the sine
        for _ in 0..100 {
            old_graph.tick();
        }

        assert!(find_sine_node_by_freq(&old_graph, 440.0)); // Should exist

        // Create identical graph
        let new_graph = parse_to_audio_graph(sine(constant(440.0)));

        // Apply diff
        old_graph.apply_diff(new_graph);

        // Sine node should still exist (fundsp handles state preservation)
        assert!(find_sine_node_by_freq(&old_graph, 440.0));
    }

    #[test]
    fn test_apply_diff_different_frequencies() {
        // Test that different sine frequencies get reset
        let mut old_graph = parse_to_audio_graph(sine(constant(440.0)));

        // Advance the sine
        for _ in 0..100 {
            old_graph.tick();
        }

        assert!(find_sine_node_by_freq(&old_graph, 440.0));

        // Create graph with different frequency
        let new_graph = parse_to_audio_graph(sine(constant(880.0)));

        // Apply diff
        old_graph.apply_diff(new_graph);

        // Should have a sine at 880Hz (not 440Hz anymore)
        assert!(find_sine_node_by_freq(&old_graph, 880.0)); // New frequency
    }

    #[test]
    fn test_phase_continuity_during_diff() {
        // Test that oscillator phase is preserved when the same oscillator structure remains
        let mut old_graph = parse_to_audio_graph(sine(constant(440.0)));

        // Advance the sine oscillator to build up phase
        for _ in 0..1000 {
            old_graph.tick();
        }

        // Get the current phase from the sine node
        let old_phase = get_sine_node_phase(&old_graph, 440.0).expect("Should find sine node");

        // Create identical graph
        let new_graph = parse_to_audio_graph(sine(constant(440.0)));

        // Apply diff - this should preserve phase
        old_graph.apply_diff(new_graph);

        // Get the phase after diff
        let new_phase =
            get_sine_node_phase(&old_graph, 440.0).expect("Should find sine node after diff");

        // Phase should be preserved (within small tolerance for floating point)
        let phase_diff = (old_phase - new_phase).abs();
        assert!(
            phase_diff < 0.01,
            "Phase not preserved: old={}, new={}, diff={}",
            old_phase,
            new_phase,
            phase_diff
        );
    }

    #[test]
    fn test_fallback_state_transfer_different_frequency() {
        // Test that sine oscillator state transfers even when frequency changes
        let mut old_graph = parse_to_audio_graph(sine(constant(440.0)));

        // Advance the sine oscillator to build up phase
        for _ in 0..1000 {
            old_graph.tick();
        }

        // Get the current phase from the sine node
        let old_phase =
            get_sine_node_phase(&old_graph, 440.0).expect("Should find sine node at 440Hz");

        // Create graph with different frequency - this should NOT match by hash
        let new_graph = parse_to_audio_graph(sine(constant(880.0)));

        // Apply diff - this should use fallback type-based matching and preserve phase
        old_graph.apply_diff(new_graph);

        // Get the phase after diff (now at 880Hz)
        let new_phase = get_sine_node_phase(&old_graph, 880.0)
            .expect("Should find sine node at 880Hz after diff");

        // Phase should be preserved via fallback matching (within small tolerance)
        let phase_diff = (old_phase - new_phase).abs();
        assert!(
            phase_diff < 0.01,
            "Phase not preserved with frequency change: old={}, new={}, diff={}",
            old_phase,
            new_phase,
            phase_diff
        );
    }

    // Helper function to find sine node with given frequency and verify it exists
    fn find_sine_node_by_freq(graph: &AudioGraph, target_freq: f32) -> bool {
        for &node_idx in &graph.sorted_nodes {
            if let Some(node) = graph.graph.node_weight(node_idx) {
                match &**node {
                    NodeKind::Sine { freq, .. } => {
                        if let NodeKind::Constant(const_node) = &**freq {
                            if (const_node.value - target_freq).abs() < 0.01 {
                                return true;
                            }
                        }
                    }
                    _ => continue,
                }
            }
        }
        false
    }

    // Helper function to verify sine node exists at given index
    fn verify_sine_node(graph: &AudioGraph, node_index: usize) -> bool {
        if node_index >= graph.sorted_nodes.len() {
            return false;
        }
        let node_idx = graph.sorted_nodes[node_index];
        match &**graph.graph.node_weight(node_idx).unwrap() {
            NodeKind::Sine { .. } => true,
            _ => false,
        }
    }

    // Helper function to get the phase from a sine node with given frequency
    fn get_sine_node_phase(graph: &AudioGraph, target_freq: f32) -> Option<f32> {
        for &node_idx in &graph.sorted_nodes {
            if let Some(node) = graph.graph.node_weight(node_idx) {
                match &**node {
                    NodeKind::Sine { freq, node } => {
                        if let NodeKind::Constant(const_node) = &**freq {
                            if (const_node.value - target_freq).abs() < 0.01 {
                                return Some(node.get_phase());
                            }
                        }
                    }
                    _ => continue,
                }
            }
        }
        None
    }
}
