use petgraph::{graph::NodeIndex, prelude::DiGraph, visit::EdgeRef};
use std::collections::HashSet;
use std::fmt;

use crate::nodes::{Node, NodeKind};
use crate::parser::Expr;

type BoxedNode = Box<NodeKind>;
type Graph = DiGraph<BoxedNode, ()>;

pub(crate) struct AudioGraph {
    graph: Graph,
    sorted_nodes: Vec<NodeIndex>,
    inputs: Vec<f32>,
    outputs: Vec<f32>,
}

impl AudioGraph {
    pub(crate) fn new() -> Self {
        Self {
            graph: DiGraph::new(),
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

    #[inline]
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

        self.outputs[self.sorted_nodes.last().unwrap().index()]
    }

    pub(crate) fn clear_edges(&mut self) {
        self.graph.clear_edges();
    }

    pub(crate) fn reconnect_edges(&mut self, new: &AudioGraph) {
        let old_edges: HashSet<_> = self.graph.edge_indices().collect();
        let new_edges: HashSet<_> = new.graph.edge_indices().collect();

        for edge in old_edges.difference(&new_edges) {
            self.graph.remove_edge(*edge);
        }

        for edge in new.graph.edge_references() {
            let (source, target) = (edge.source(), edge.target());
            if !self.graph.contains_edge(source, target) {
                self.graph.add_edge(source, target, edge.weight().clone());
            }
        }

        for edge in old_edges.difference(&new_edges) {
            self.graph.remove_edge(*edge);
        }

        self.update_processing_order();
    }

    fn update_processing_order(&mut self) {
        self.sorted_nodes = petgraph::algo::toposort(&self.graph, None).expect("Graph has cycles");
        self.inputs.resize(self.graph.node_count(), 0.0);
        self.outputs.resize(self.graph.node_count(), 0.0);
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
                "  n{} [label=\"{} ({})\"];",
                node.index(),
                node.index(),
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

pub(crate) fn parse_to_audio_graph(expr: Expr) -> AudioGraph {
    let mut graph = AudioGraph::new();

    fn add_expr_to_graph(expr: Expr, graph: &mut AudioGraph) -> NodeIndex {
        match expr {
            Expr::Constant(n) => graph.add_node(NodeKind::constant(n)),
            Expr::Sine { freq } => add_node(vec![freq], NodeKind::sine(), graph),
            Expr::Square { freq } => add_node(vec![freq], NodeKind::square(), graph),
            Expr::Saw { freq } => add_node(vec![freq], NodeKind::saw(), graph),
            Expr::Pulse { freq } => add_node(vec![freq], NodeKind::pulse(), graph),
            Expr::Noise => graph.add_node(NodeKind::noise()),
            Expr::Gain { lhs: a, rhs: b } => add_node(vec![a, b], NodeKind::gain(), graph),
            Expr::Mix { lhs: a, rhs: b } => add_node(vec![a, b], NodeKind::mix(), graph),
            Expr::AR {
                attack,
                release,
                trig,
            } => add_node(vec![attack, release, trig], NodeKind::ar(), graph),
            Expr::SVF {
                cutoff,
                resonance,
                input,
            } => add_node(vec![cutoff, resonance, input], NodeKind::svf(), graph),
            Expr::Seq { seq, trig } => add_node(vec![trig], NodeKind::seq(seq), graph),
        }
    }

    fn add_node(inputs: Vec<Box<Expr>>, kind: NodeKind, graph: &mut AudioGraph) -> NodeIndex {
        let mut input_indices: Vec<_> = inputs
            .into_iter()
            .map(|input| add_expr_to_graph(*input, graph))
            .collect();

        let node_idx = graph.add_node(kind);

        // inputs need to be connected in reverse order
        input_indices.reverse();

        for &input_idx in &input_indices {
            graph.connect_node(input_idx, node_idx);
        }

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
    use assert_no_alloc::*;

    #[test]
    fn test_parse_graph_1() {
        // 440
        let expr = Expr::Constant(440.0);

        let mut graph = parse_to_audio_graph(expr);

        assert_eq!(graph.graph.node_count(), 1);
        assert_eq!(graph.graph.edge_count(), 0);
        assert_eq!(graph.tick(), 440.0);
    }

    #[test]
    fn test_parse_graph_2() {
        // sine(440)
        let expr = Expr::Sine {
            freq: Box::new(Expr::Constant(440.0)),
        };

        let mut graph = parse_to_audio_graph(expr);

        assert_eq!(graph.graph.node_count(), 2);
        assert_eq!(graph.graph.edge_count(), 1);
        assert_eq!(graph.tick(), 0.0);
    }

    #[test]
    fn test_parse_graph_3() {
        // sine((sine(0.1) + 2) * 100) * 0.2
        let expr = Expr::Gain {
            lhs: Box::new(Expr::Sine {
                freq: Box::new(Expr::Gain {
                    lhs: Box::new(Expr::Mix {
                        lhs: Box::new(Expr::Sine {
                            freq: Box::new(Expr::Constant(0.1)),
                        }),
                        rhs: Box::new(Expr::Constant(2.0)),
                    }),
                    rhs: Box::new(Expr::Constant(100.0)),
                }),
            }),
            rhs: Box::new(Expr::Constant(0.2)),
        };

        let mut graph = parse_to_audio_graph(expr);

        assert_eq!(graph.graph.node_count(), 9);
        assert_eq!(graph.graph.edge_count(), 8);
        assert_eq!(graph.tick(), 0.0);
    }

    #[test]
    fn test_node_comparison() {
        let node1 = NodeKind::constant(1.0);
        let node2 = NodeKind::constant(1.0);
        let node3 = NodeKind::constant(2.0);

        assert_eq!(node1, node2);
        assert_ne!(node1, node3);
    }

    #[test]
    fn test_graph_diff() {
        let mut old = AudioGraph::new();
        let mut new = AudioGraph::new();

        let const_idx1 = old.add_node(NodeKind::constant(1.0));
        let sine_idx1 = old.add_node(NodeKind::sine());
        old.connect_node(const_idx1, sine_idx1);

        let const_idx2 = new.add_node(NodeKind::constant(2.0)); // Same
        let sine_idx2 = new.add_node(NodeKind::sine());
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
        let mut node = NodeKind::constant(42.0);
        assert_eq!(node.tick(&[]), 42.0);
    }

    fn create_complex_graph() -> AudioGraph {
        let mut graph = AudioGraph::new();

        let freq = graph.add_node(NodeKind::constant(440.0));
        let amp = graph.add_node(NodeKind::constant(0.5));
        let osc = graph.add_node(NodeKind::sine());
        let env = graph.add_node(NodeKind::ar());
        let gain = graph.add_node(NodeKind::gain());

        let cutoff = graph.add_node(NodeKind::constant(1000.0));
        let resonance = graph.add_node(NodeKind::constant(0.7));
        let filter = graph.add_node(NodeKind::svf());

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

        let const_idx1 = old.add_node(NodeKind::constant(1.0));
        let sine_idx1 = old.add_node(NodeKind::sine());
        old.connect_node(const_idx1, sine_idx1);

        let const_idx2 = new.add_node(NodeKind::constant(2.0));
        let sine_idx2 = new.add_node(NodeKind::sine());
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
        let freq2 = new.add_node(NodeKind::constant(440.0));
        let osc2 = new.add_node(NodeKind::square());
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
        let freq1 = old.add_node(NodeKind::constant(440.0));
        let osc1 = old.add_node(NodeKind::sine());
        old.connect_node(freq1, osc1);

        // New graph with added filter
        let freq2 = new.add_node(NodeKind::constant(440.0));
        let osc2 = new.add_node(NodeKind::sine());
        let cutoff = new.add_node(NodeKind::constant(1000.0));
        let filter = new.add_node(NodeKind::svf());
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
        let new_freq = new.add_node(NodeKind::constant(880.0));

        // 2. Add sequence modulation
        let seq = new.add_node(NodeKind::seq([100.0, 200.0].to_vec()));
        let seq_amp = new.add_node(NodeKind::constant(0.3));

        // 3. Change oscillator type
        let new_osc = new.add_node(NodeKind::saw());

        // 4. Add noise mix
        let noise = new.add_node(NodeKind::noise());
        let mix = new.add_node(NodeKind::mix());

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
        let freq = new.add_node(NodeKind::constant(440.0));
        let osc = new.add_node(NodeKind::sine());
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
