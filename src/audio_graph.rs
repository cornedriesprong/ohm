use petgraph::graph::NodeIndex;
use petgraph::prelude::StableDiGraph;
use petgraph::visit::EdgeRef;

use crate::nodes::*;
use crate::parser::Expr;

type GraphType = StableDiGraph<Box<NodeKind>, ()>;

pub(crate) struct AudioGraph {
    pub(crate) graph: GraphType,
    sorted_nodes: Vec<NodeIndex>,
    inputs: Vec<f32>,
    outputs: Vec<f32>,
}

impl AudioGraph {
    pub(crate) fn new() -> Self {
        Self {
            graph: StableDiGraph::new(),
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

    pub(crate) fn remove_node(&mut self, node: NodeIndex) {
        self.graph.remove_node(node);
        self.update_processing_order();
    }

    pub(crate) fn replace_node(&mut self, at: NodeIndex, new: NodeKind) {
        self.graph[at] = Box::new(new);
        self.update_processing_order();
    }

    pub(crate) fn connect_node(&mut self, from: NodeIndex, to: NodeIndex) {
        self.graph.add_edge(from.into(), to.into(), ());
        self.update_processing_order();
    }

    #[inline(always)]
    pub(crate) fn tick(&mut self) -> f32 {
        for &node_index in &self.sorted_nodes {
            self.inputs.clear();

            // TODO: make sure we don't allocate, maybe resize inputs vec on graph update
            self.inputs.extend(
                self.graph
                    .edges_directed(node_index, petgraph::Direction::Incoming)
                    .filter(|edge| matches!(edge.weight(), ()))
                    .map(|edge| self.outputs[edge.source().index()]),
            );

            self.outputs[node_index.index()] = self.graph[node_index].tick(&self.inputs);
        }

        self.outputs[self.sorted_nodes.last().unwrap().index()]
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

    fn update_processing_order(&mut self) {
        self.sorted_nodes = petgraph::algo::toposort(&self.graph, None).expect("Graph has cycles");
        self.outputs.resize(self.graph.node_count(), 0.0);
    }
}

pub(crate) fn parse_to_audio_graph(expr: Expr) -> AudioGraph {
    let mut graph = AudioGraph::new();

    fn add_expr_to_graph(expr: Expr, graph: &mut AudioGraph) -> NodeIndex {
        match expr {
            Expr::Constant(n) => graph.add_node(NodeKind::constant(n)),
            Expr::Sine(input) => add_node(input, NodeKind::sine(), graph),
            Expr::Square(input) => add_node(input, NodeKind::square(), graph),
            Expr::Saw(input) => add_node(input, NodeKind::saw(), graph),
            Expr::Pulse(input) => add_node(input, NodeKind::pulse(), graph),
            Expr::Noise => graph.add_node(NodeKind::noise()),
            Expr::Gain(input, mult) => {
                let input_idx = add_expr_to_graph(*input, graph);
                let mult_idx = add_expr_to_graph(*mult, graph);
                let gain_idx = graph.add_node(NodeKind::gain());
                graph.connect_node(input_idx, gain_idx);
                graph.connect_node(mult_idx, gain_idx);
                gain_idx
            }
            Expr::Mix(a, b) => {
                let a_idx = add_expr_to_graph(*a, graph);
                let b_idx = add_expr_to_graph(*b, graph);
                let mix_idx = graph.add_node(NodeKind::mix());
                graph.connect_node(a_idx, mix_idx);
                graph.connect_node(b_idx, mix_idx);
                mix_idx
            }
            Expr::AR {
                attack,
                release,
                trig,
            } => {
                let attack_idx = add_expr_to_graph(*attack, graph);
                let release_idx = add_expr_to_graph(*release, graph);
                let trig_idx = add_expr_to_graph(*trig, graph);
                let ar_idx = graph.add_node(NodeKind::ar());

                graph.connect_node(trig_idx, ar_idx);
                graph.connect_node(release_idx, ar_idx);
                graph.connect_node(attack_idx, ar_idx);

                ar_idx
            }
            Expr::SVF {
                cutoff,
                resonance,
                input,
            } => {
                let cutoff_idx = add_expr_to_graph(*cutoff, graph);
                let resonance_idx = add_expr_to_graph(*resonance, graph);
                let input_idx = add_expr_to_graph(*input, graph);
                let svf_idx = graph.add_node(NodeKind::svf());

                graph.connect_node(input_idx, svf_idx);
                graph.connect_node(resonance_idx, svf_idx);
                graph.connect_node(cutoff_idx, svf_idx);

                svf_idx
            }
            _ => panic!("Invalid expression"),
        }
    }

    fn add_node(input: Box<Expr>, kind: NodeKind, graph: &mut AudioGraph) -> NodeIndex {
        let input_idx = add_expr_to_graph(*input, graph);
        let node_idx = graph.add_node(kind);
        graph.connect_node(input_idx, node_idx);
        node_idx
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
        old.connect_node(const_idx1, sine_idx1);

        let const_idx2 = new.add_node(NodeKind::Constant(Constant::new(2.0))); // Same
        let sine_idx2 = new.add_node(NodeKind::Sine(Sine::new()));
        new.connect_node(const_idx2, sine_idx2);

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
        assert_eq!(node.tick(&[]), 42.0);
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
        graph.connect_node(freq, osc);
        graph.connect_node(amp, gain);
        graph.connect_node(osc, gain);
        graph.connect_node(env, gain);
        graph.connect_node(gain, filter);
        graph.connect_node(cutoff, filter);
        graph.connect_node(resonance, filter);

        graph
    }

    #[test]
    fn test_simple_update() {
        let mut old = AudioGraph::new();
        let mut new = AudioGraph::new();

        let const_idx1 = old.add_node(NodeKind::Constant(Constant::new(1.0)));
        let sine_idx1 = old.add_node(NodeKind::Sine(Sine::new()));
        old.connect_node(const_idx1, sine_idx1);

        let const_idx2 = new.add_node(NodeKind::Constant(Constant::new(2.0)));
        let sine_idx2 = new.add_node(NodeKind::Sine(Sine::new()));
        new.connect_node(const_idx2, sine_idx2);

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
        old.connect_node(freq1, osc1);

        // Create new graph with Square
        let freq2 = new.add_node(NodeKind::Constant(Constant::new(440.0)));
        let osc2 = new.add_node(NodeKind::Square(Square::new()));
        new.connect_node(freq2, osc2);

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
        old.connect_node(freq1, osc1);

        // New graph with added filter
        let freq2 = new.add_node(NodeKind::Constant(Constant::new(440.0)));
        let osc2 = new.add_node(NodeKind::Sine(Sine::new()));
        let cutoff = new.add_node(NodeKind::Constant(Constant::new(1000.0)));
        let filter = new.add_node(NodeKind::SVF(SVF::new()));
        new.connect_node(freq2, osc2);
        new.connect_node(osc2, filter);
        new.connect_node(cutoff, filter);

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
        new.connect_node(new_freq, new_osc);
        new.connect_node(seq_amp, seq);
        new.connect_node(seq, mix);
        new.connect_node(noise, mix);

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
        new.connect_node(freq, osc);

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
