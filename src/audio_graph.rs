use petgraph::graph::NodeIndex;
use petgraph::prelude::StableDiGraph;
use petgraph::visit::EdgeRef;

use crate::nodes::*;
use crate::parser::{Expr, OperatorType};

#[derive(Clone, Debug)]
pub(crate) enum EdgeKind {
    Audio,
    Control,
}

type GraphType = StableDiGraph<Box<NodeKind>, EdgeKind>;

pub(crate) struct AudioGraph {
    pub(crate) graph: GraphType,
    sorted_nodes: Vec<NodeIndex>,
    outputs: Vec<f32>,
    control_buffer: Vec<f32>,
}

impl AudioGraph {
    pub(crate) fn new() -> Self {
        Self {
            graph: StableDiGraph::new(),
            sorted_nodes: Vec::new(),
            outputs: Vec::new(),
            control_buffer: Vec::new(),
        }
    }

    pub(crate) fn add_node(&mut self, node: NodeKind) -> NodeIndex {
        let index = self.graph.add_node(Box::new(node));
        self.update_processing_order();
        index
    }

    pub(crate) fn remove_node(&mut self, node: NodeIndex) {
        self.graph.remove_node(node);
        self.update_processing_order();
    }

    pub(crate) fn replace_node(&mut self, at: NodeIndex, new: NodeKind) {
        self.graph[at] = Box::new(new);
        self.update_processing_order();
    }

    pub(crate) fn connect_generator(&mut self, from: NodeIndex, to: NodeIndex) {
        self.graph.add_edge(from.into(), to.into(), EdgeKind::Audio);
        self.update_processing_order();
    }

    pub(crate) fn connect_control(&mut self, from: NodeIndex, to: NodeIndex) {
        self.graph
            .add_edge(from.into(), to.into(), EdgeKind::Control);
    }

    pub fn reconnect_edges(&mut self, new: &AudioGraph) {
        self.graph.clear_edges();

        // Copy all edges from the new graph configuration
        for edge_idx in new.graph.edge_indices() {
            if let Some((source, target)) = new.graph.edge_endpoints(edge_idx) {
                // Only add edge if both nodes exist in our graph
                if self.graph.contains_node(source) && self.graph.contains_node(target) {
                    let edge_weight = new.graph.edge_weight(edge_idx).unwrap();
                    self.graph.add_edge(source, target, edge_weight.clone());
                }
            }
        }

        self.update_processing_order();
    }

    #[inline(always)]
    pub(crate) fn tick(&mut self) -> f32 {
        for &node_index in &self.sorted_nodes {
            let audio_input = self
                .graph
                .edges_directed(node_index, petgraph::Direction::Incoming)
                .filter(|edge| matches!(edge.weight(), EdgeKind::Audio))
                .map(|edge| self.outputs[edge.source().index()])
                .sum();

            self.control_buffer.clear();
            self.control_buffer.extend(
                self.graph
                    .edges_directed(node_index, petgraph::Direction::Incoming)
                    .filter(|edge| matches!(edge.weight(), EdgeKind::Control))
                    .map(|edge| self.outputs[edge.source().index()]),
            );

            self.outputs[node_index.index()] =
                self.graph[node_index].tick(audio_input, &self.control_buffer);
        }

        self.outputs[self.sorted_nodes.last().unwrap().index()]
    }

    fn update_processing_order(&mut self) {
        self.sorted_nodes = petgraph::algo::toposort(&self.graph, None).expect("Graph has cycles");
        self.outputs.resize(self.graph.node_count(), 0.0);
    }
}

pub(crate) fn parse_to_audio_graph(expr: Expr) -> AudioGraph {
    let mut graph = AudioGraph::new();

    fn connect_generator(
        graph: &mut AudioGraph,
        node: NodeKind,
        input_node: NodeIndex,
    ) -> NodeIndex {
        let node_index = graph.add_node(node);
        graph.connect_generator(input_node, node_index);
        node_index
    }

    fn connect_control(graph: &mut AudioGraph, expr: &Expr, target_idx: NodeIndex) {
        match expr {
            Expr::Number(n) => {
                let constant = NodeKind::Constant(Constant::new(*n));
                let constant_index = graph.add_node(constant);
                graph.connect_control(constant_index, target_idx);
            }
            Expr::Operator { .. } => {
                let node = add_expr_to_graph(expr.clone(), graph);
                graph.connect_control(node, target_idx);
            }
            Expr::List(list) => {
                for &n in list {
                    let constant = NodeKind::Constant(Constant::new(n));
                    let constant_index = graph.add_node(constant);
                    graph.connect_control(constant_index, target_idx);
                }
            }
        }
    }

    fn add_expr_to_graph(expr: Expr, graph: &mut AudioGraph) -> NodeIndex {
        match expr {
            Expr::Operator { kind, input, args } => {
                let input_node = add_expr_to_graph(*input, graph);

                use OperatorType as OT;
                match kind {
                    OT::Sine => {
                        let node_idx = connect_generator(graph, NodeKind::sine(), input_node);
                        if let Some(expr) = args.get(0) {
                            connect_control(graph, expr, node_idx);
                        }

                        node_idx
                    }
                    OT::Square => {
                        let node_idx = connect_generator(graph, NodeKind::square(), input_node);
                        if let Some(expr) = args.get(0) {
                            connect_control(graph, expr, node_idx);
                        }

                        node_idx
                    }
                    OT::Saw => {
                        let node_idx = connect_generator(graph, NodeKind::saw(), input_node);
                        if let Some(expr) = args.get(0) {
                            connect_control(graph, expr, node_idx);
                        }

                        node_idx
                    }
                    OT::Noise => {
                        let node_idx = connect_generator(graph, NodeKind::noise(), input_node);
                        if let Some(expr) = args.get(0) {
                            connect_control(graph, expr, node_idx);
                        }

                        node_idx
                    }
                    OT::Pulse => {
                        let node_idx = connect_generator(graph, NodeKind::pulse(), input_node);
                        if let Some(expr) = args.get(0) {
                            connect_control(graph, expr, node_idx);
                        }

                        node_idx
                    }
                    OT::Gain => {
                        let node_idx = connect_generator(graph, NodeKind::gain(), input_node);
                        if let Some(expr) = args.get(0) {
                            connect_control(graph, expr, node_idx);
                        }

                        node_idx
                    }
                    OT::Mix => {
                        let node_idx = connect_generator(graph, NodeKind::mix(), input_node);
                        if let Some(expr) = args.get(0) {
                            connect_control(graph, expr, node_idx);
                        }

                        node_idx
                    }
                    OT::AR => {
                        let node_idx = connect_generator(graph, NodeKind::ar(), input_node);
                        // nb: notes need to be connected in reverse order
                        if let Some(release_expr) = args.get(1) {
                            connect_control(graph, release_expr, node_idx);
                        }

                        if let Some(attack_expr) = args.get(0) {
                            connect_control(graph, attack_expr, node_idx);
                        }

                        node_idx
                    }
                    OT::SVF => {
                        let node_idx = connect_generator(graph, NodeKind::svf(), input_node);

                        if let Some(resonance_expr) = args.get(1) {
                            connect_control(graph, resonance_expr, node_idx);
                        }

                        if let Some(cutoff_expr) = args.get(0) {
                            connect_control(graph, cutoff_expr, node_idx);
                        }

                        node_idx
                    }
                    OT::Seq => args
                        .get(0)
                        .and_then(|expr| match expr {
                            Expr::List(list) => Some(connect_generator(
                                graph,
                                NodeKind::seq(list.to_vec()),
                                input_node,
                            )),
                            _ => None,
                        })
                        .expect("Seq operator requires a list of nodes"),
                }
            }
            Expr::Number(n) => graph.add_node(NodeKind::constant(n)),
            _ => panic!("Invalid expression"),
        }
    }

    add_expr_to_graph(expr, &mut graph);

    graph
}

pub(crate) fn diff_graph<'a>(
    old: &'a AudioGraph,
    new: &'a AudioGraph,
) -> (
    Vec<(NodeIndex, NodeKind)>,
    Vec<(NodeIndex, NodeKind)>,
    Vec<NodeIndex>,
) {
    let mut to_update = Vec::new();
    let mut to_add = Vec::new();
    let mut to_remove = Vec::new();

    for old_idx in old.graph.node_indices() {
        let old_node = &old.graph[old_idx];

        // check if node is in both the old and the new graph
        let matching_node = new.graph.node_indices().find(|&idx| idx == old_idx);

        match matching_node {
            Some(new_idx) => {
                // ...if it is, then compare the nodes to see if it needs updating
                if **old_node != *new.graph[new_idx] {
                    to_update.push((old_idx, *new.graph[new_idx].clone()));
                }
            }
            // ...if not, we can remove it
            None => to_remove.push(old_idx),
        }
    }

    for new_idx in new.graph.node_indices() {
        // nodes that are in the new graph but not in the old graph need to be added
        if !old.graph.node_indices().any(|idx| idx == new_idx) {
            to_add.push((new_idx, *new.graph[new_idx].clone()));
        }
    }

    (to_update, to_add, to_remove)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_comparison() {
        let node1 = NodeKind::Constant(Constant::new(1.0));
        let node2 = NodeKind::Constant(Constant::new(1.0));
        let node3 = NodeKind::Constant(Constant::new(2.0));

        assert_eq!(node1, node2);
        assert_ne!(node1, node3);
    }

    #[test]
    fn test_graph_diff() {
        let mut old = AudioGraph::new();
        let mut new = AudioGraph::new();

        let const_idx1 = old.add_node(NodeKind::Constant(Constant::new(1.0)));
        let sine_idx1 = old.add_node(NodeKind::Sine(Sine::new()));
        old.connect_generator(const_idx1, sine_idx1);

        let const_idx2 = new.add_node(NodeKind::Constant(Constant::new(2.0))); // Same
        let sine_idx2 = new.add_node(NodeKind::Sine(Sine::new()));
        new.connect_generator(const_idx2, sine_idx2);

        let (updates, additions, removals) = diff_graph(&old, &new);

        println!("updates {:?}", updates);
        println!("additions {:?}", additions);
        println!("removals {:?}", removals);

        assert_eq!(updates.len(), 1, "Should have one update");
        assert!(additions.is_empty(), "Should have no additions");
        assert!(removals.is_empty(), "Should have no removals");
    }

    #[test]
    fn test_node_tick() {
        let mut node = NodeKind::Constant(Constant::new(42.0));
        assert_eq!(node.tick(0.0, &[]), 42.0);
    }

    fn create_complex_graph() -> AudioGraph {
        let mut graph = AudioGraph::new();

        // Create a synth voice with envelope
        let freq = graph.add_node(NodeKind::Constant(Constant::new(440.0)));
        let amp = graph.add_node(NodeKind::Constant(Constant::new(0.5)));
        let osc = graph.add_node(NodeKind::Sine(Sine::new()));
        let env = graph.add_node(NodeKind::AR(AR::new()));
        let gain = graph.add_node(NodeKind::Gain(Gain::new()));

        // Create a filter
        let cutoff = graph.add_node(NodeKind::Constant(Constant::new(1000.0)));
        let resonance = graph.add_node(NodeKind::Constant(Constant::new(0.7)));
        let filter = graph.add_node(NodeKind::SVF(SVF::new()));

        // Connect everything
        graph.connect_generator(freq, osc);
        graph.connect_generator(amp, gain);
        graph.connect_generator(osc, gain);
        graph.connect_generator(env, gain);
        graph.connect_generator(gain, filter);
        graph.connect_generator(cutoff, filter);
        graph.connect_generator(resonance, filter);

        graph
    }

    #[test]
    fn test_simple_update() {
        let mut old = AudioGraph::new();
        let mut new = AudioGraph::new();

        let const_idx1 = old.add_node(NodeKind::Constant(Constant::new(1.0)));
        let sine_idx1 = old.add_node(NodeKind::Sine(Sine::new()));
        old.connect_generator(const_idx1, sine_idx1);

        let const_idx2 = new.add_node(NodeKind::Constant(Constant::new(2.0)));
        let sine_idx2 = new.add_node(NodeKind::Sine(Sine::new()));
        new.connect_generator(const_idx2, sine_idx2);

        let (updates, additions, removals) = diff_graph(&old, &new);
        assert_eq!(updates.len(), 1, "Should have one update");
        assert!(additions.is_empty(), "Should have no additions");
        assert!(removals.is_empty(), "Should have no removals");
    }

    #[test]
    fn test_change_oscillator_type() {
        let mut old = AudioGraph::new();
        let mut new = AudioGraph::new();

        // Create original graph with Sine
        let freq1 = old.add_node(NodeKind::constant(440.0));
        let osc1 = old.add_node(NodeKind::sine());
        old.connect_generator(freq1, osc1);

        // Create new graph with Square
        let freq2 = new.add_node(NodeKind::Constant(Constant::new(440.0)));
        let osc2 = new.add_node(NodeKind::Square(Square::new()));
        new.connect_generator(freq2, osc2);

        let (updates, additions, removals) = diff_graph(&old, &new);
        assert_eq!(updates.len(), 1, "Should update oscillator type");
        assert_eq!(additions.len(), 0);
        assert_eq!(removals.len(), 0);
    }

    #[test]
    fn test_add_filter() {
        let mut old = AudioGraph::new();
        let mut new = AudioGraph::new();

        // Original simple oscillator
        let freq1 = old.add_node(NodeKind::Constant(Constant::new(440.0)));
        let osc1 = old.add_node(NodeKind::Sine(Sine::new()));
        old.connect_generator(freq1, osc1);

        // New graph with added filter
        let freq2 = new.add_node(NodeKind::Constant(Constant::new(440.0)));
        let osc2 = new.add_node(NodeKind::Sine(Sine::new()));
        let cutoff = new.add_node(NodeKind::Constant(Constant::new(1000.0)));
        let filter = new.add_node(NodeKind::SVF(SVF::new()));
        new.connect_generator(freq2, osc2);
        new.connect_generator(osc2, filter);
        new.connect_generator(cutoff, filter);

        let (updates, additions, removals) = diff_graph(&old, &new);
        assert!(updates.is_empty(), "No updates expected");
        assert_eq!(additions.len(), 2, "Should add filter and cutoff nodes");
        assert!(removals.is_empty());
    }

    #[test]
    fn test_complex_modifications() {
        let mut old = create_complex_graph();
        let mut new = create_complex_graph();

        // Modify new graph
        // 1. Change frequency
        let new_freq = new.add_node(NodeKind::Constant(Constant::new(880.0)));

        // 2. Add sequence modulation
        let seq = new.add_node(NodeKind::Seq(Seq::new([100.0, 200.0].to_vec())));
        let seq_amp = new.add_node(NodeKind::Constant(Constant::new(0.3)));

        // 3. Change oscillator type
        let new_osc = new.add_node(NodeKind::Saw(Saw::new()));

        // 4. Add noise mix
        let noise = new.add_node(NodeKind::Noise(Noise::new()));
        let mix = new.add_node(NodeKind::Mix(Mix::new()));

        // Connect new components
        new.connect_generator(new_freq, new_osc);
        new.connect_generator(seq_amp, seq);
        new.connect_generator(seq, mix);
        new.connect_generator(noise, mix);

        let (updates, additions, removals) = diff_graph(&old, &new);

        // Verify changes
        // assert!(!updates.is_empty(), "Should have updates");
        assert!(!additions.is_empty(), "Should have additions");
        // assert!(!removals.is_empty(), "Should have removals");

        // Apply changes
        old.graph.clear_edges();

        for (id, node) in updates {
            old.replace_node(id, node);
        }

        for (_, node) in additions {
            old.add_node(node);
        }

        for id in removals {
            old.remove_node(id);
        }

        old.reconnect_edges(&new);

        // Verify final state
        assert_eq!(old.graph.node_count(), new.graph.node_count());
        assert_eq!(old.graph.edge_count(), new.graph.edge_count());
    }

    #[test]
    fn test_remove_components() {
        let mut old = create_complex_graph();
        let mut new = AudioGraph::new();

        // Create simplified version with just oscillator
        let freq = new.add_node(NodeKind::Constant(Constant::new(440.0)));
        let osc = new.add_node(NodeKind::Sine(Sine::new()));
        new.connect_generator(freq, osc);

        let (updates, additions, removals) = diff_graph(&old, &new);

        assert_eq!(updates.len(), 1, "Update constant");
        assert!(additions.is_empty(), "No additions expected");
        assert!(!removals.is_empty(), "Should remove multiple nodes");

        // Apply changes
        old.graph.clear_edges();

        for id in removals {
            old.remove_node(id);
        }

        old.reconnect_edges(&new);

        assert_eq!(
            old.graph.node_count(),
            2,
            "Should only have freq and osc nodes"
        );
    }
}
