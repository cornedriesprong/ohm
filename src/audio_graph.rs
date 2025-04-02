use petgraph::{
    graph::NodeIndex,
    prelude::StableDiGraph,
    visit::{EdgeRef, IntoEdgeReferences},
};
use rtsan_standalone::nonblocking;
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
}
