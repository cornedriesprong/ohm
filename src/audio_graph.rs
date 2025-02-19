use crossbeam::channel::{Receiver, Sender};
use petgraph::{
    graph::NodeIndex,
    prelude::StableDiGraph,
    visit::{EdgeRef, IntoEdgeReferences},
};
use rtsan_standalone::nonblocking;
use std::collections::HashSet;
use std::fmt;

use crate::nodes::{Node, NodeKind};

pub(crate) type BoxedNode = Box<NodeKind>;
type Graph = StableDiGraph<BoxedNode, ()>;

enum Message {
    AddNode(NodeKind),
    RemoveNode { at: NodeIndex },
    ReplaceNode { at: NodeIndex, with: NodeKind },
    ClearEdges,
    ReconnectEdges(AudioGraph),
}

pub(crate) struct GraphPlayer {
    graph: AudioGraph,
    sender: Sender<Message>,
    receiver: Receiver<Message>,
}

impl GraphPlayer {
    pub(crate) fn new(graph: AudioGraph) -> Self {
        let (sender, receiver) = crossbeam::channel::bounded(1024);
        return Self {
            graph,
            sender,
            receiver,
        };
    }

    #[inline]
    #[nonblocking]
    pub(crate) fn tick(&mut self) -> f32 {
        self.fetch_updates();

        self.graph.tick()
    }

    pub(crate) fn replace_graph(&self, new: AudioGraph) {
        let mut update = Vec::new();
        let mut add = Vec::new();
        let mut remove = Vec::new();
        let old = &self.graph;

        for old_idx in old.graph.node_indices() {
            let old_node = &old.graph[old_idx];

            // check if node is in both the old and the new graph
            let matching_node = new.graph.node_indices().find(|&idx| idx == old_idx);

            match matching_node {
                Some(new_idx) => {
                    // ...if it is, then compare the nodes to see if it needs updating
                    if **old_node != *new.graph[new_idx] {
                        update.push((old_idx, *new.graph[new_idx].clone()));
                    }
                }
                // ...if not, we can remove it
                None => remove.push(old_idx),
            }
        }

        for new_idx in new.graph.node_indices() {
            // nodes that are in the new graph but not in the old one need to be added
            if !old.graph.node_indices().any(|idx| idx == new_idx) {
                add.push((new_idx, *new.graph[new_idx].clone()));
            }
        }

        self.sender.send(Message::ClearEdges).unwrap();

        for (id, node) in update {
            self.sender
                .send(Message::ReplaceNode { at: id, with: node })
                .unwrap();
        }

        for (_, node) in add {
            self.sender.send(Message::AddNode(node)).unwrap();
        }

        for id in remove {
            self.sender.send(Message::RemoveNode { at: id }).unwrap();
        }

        self.sender.send(Message::ReconnectEdges(new)).unwrap();
    }

    fn fetch_updates(&mut self) {
        // fetch pending graph updates from channel
        for tx in self.receiver.try_iter() {
            match tx {
                Message::AddNode(node) => {
                    self.graph.add_node(node);
                }
                Message::RemoveNode { at } => {
                    self.graph.remove_node(at);
                }
                Message::ReplaceNode { at, with } => {
                    self.graph.replace_node(at, with);
                }
                Message::ClearEdges => {
                    self.graph.clear_edges();
                }
                Message::ReconnectEdges(new) => {
                    self.graph.reconnect_edges(&new);
                }
            }
        }
    }
}

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

    pub(crate) fn remove_node(&mut self, node: NodeIndex) {
        self.graph.remove_node(node);
        self.update_processing_order();
    }

    pub(crate) fn replace_node(&mut self, at: NodeIndex, new: NodeKind) {
        self.graph[at] = Box::new(new);
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

    fn connect_node(&mut self, from: NodeIndex, to: NodeIndex) {
        self.graph.add_edge(from.into(), to.into(), ());
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

pub(crate) fn parse_to_audio_graph(expr: NodeKind) -> AudioGraph {
    let mut graph = AudioGraph::new();

    fn add_expr_to_graph(expr: &NodeKind, graph: &mut AudioGraph) -> NodeIndex {
        match expr {
            NodeKind::Constant(_) | NodeKind::Noise(_) => add_node(vec![], expr, graph),
            NodeKind::Sine { freq, .. } => add_node(vec![freq], expr, graph),
            NodeKind::Square { freq, .. } => add_node(vec![freq], expr, graph),
            NodeKind::Saw { freq, .. } => add_node(vec![freq], expr, graph),
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
                cutoff,
                resonance,
                input,
                ..
            } => add_node(vec![cutoff, resonance, input], &expr, graph),
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

    fn create_complex_graph() -> AudioGraph {
        let expr = svf(
            constant(1000.0),
            constant(0.7),
            gain(constant(0.5), sine(constant(440.0))),
        );
        parse_to_audio_graph(expr)
    }

    #[test]
    fn test_simple_update() {
        let expr1 = sine(constant(1.0));
        let old = parse_to_audio_graph(expr1);
        let mut graph_player = GraphPlayer::new(old);

        let expr2 = sine(constant(2.0));
        let new = parse_to_audio_graph(expr2);

        graph_player.replace_graph(new.clone());
        // we need to tick the player for the changes to propagate
        graph_player.tick();

        assert_eq!(format!("{:?}", graph_player.graph), format!("{:?}", new));
    }

    #[test]
    fn test_change_oscillator_type() {
        let expr1 = sine(constant(440.0));
        let old = parse_to_audio_graph(expr1);
        let mut graph_player = GraphPlayer::new(old);

        let expr2 = square(constant(440.0));
        let new = parse_to_audio_graph(expr2);

        graph_player.replace_graph(new.clone());
        // we need to tick the player for the changes to propagate
        graph_player.tick();

        assert_eq!(format!("{:?}", graph_player.graph), format!("{:?}", new));
    }

    #[test]
    fn test_add_filter() {
        // original simple oscillator
        let expr1 = sine(constant(440.0));
        let old = parse_to_audio_graph(expr1);
        let mut graph_player = GraphPlayer::new(old);

        // new graph with added filter
        let expr2 = svf(constant(440.0), constant(0.717), sine(constant(440.0)));
        let new = parse_to_audio_graph(expr2);

        graph_player.replace_graph(new.clone());
        // we need to tick the player for the changes to propagate
        graph_player.tick();

        assert_eq!(format!("{:?}", graph_player.graph), format!("{:?}", new));
    }

    #[test]
    fn test_complex_modifications() {
        let old = create_complex_graph();
        let mut graph_player = GraphPlayer::new(old);

        let expr2 = svf(constant(440.0), constant(0.717), sine(constant(440.0)));
        let new = parse_to_audio_graph(expr2);

        graph_player.replace_graph(new.clone());
        graph_player.tick();

        assert_eq!(format!("{:?}", graph_player.graph), format!("{:?}", new));
    }

    #[test]
    fn test_remove_components() {
        let old = create_complex_graph();
        let mut graph_player = GraphPlayer::new(old);

        let expr1 = sine(constant(440.0));
        let new = parse_to_audio_graph(expr1);

        graph_player.replace_graph(new.clone());
        graph_player.tick();

        assert_eq!(format!("{:?}", graph_player.graph), format!("{:?}", new));

        let mut new = AudioGraph::new();
        new.add_node(sine(constant(440.0)));

        graph_player.replace_graph(new.clone());
        graph_player.tick();

        assert_eq!(format!("{:?}", graph_player.graph), format!("{:?}", new));

        let new = create_complex_graph();
        graph_player.replace_graph(new.clone());
        graph_player.tick();

        assert_eq!(format!("{:?}", graph_player.graph), format!("{:?}", new));
    }
}
