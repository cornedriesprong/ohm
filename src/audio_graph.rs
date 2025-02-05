use petgraph::graph::NodeIndex;
use petgraph::prelude::StableDiGraph;
use petgraph::visit::EdgeRef;

use crate::nodes::*;
use crate::parser::{Expr, OperatorType};

#[derive(Clone, Debug)]
enum EdgeType {
    Audio,
    Control,
}

type GraphType = StableDiGraph<Box<NodeKind>, EdgeType>;

pub(crate) struct AudioGraph {
    graph: GraphType,
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
        self.graph.add_edge(from.into(), to.into(), EdgeType::Audio);
        self.update_processing_order();
    }

    pub(crate) fn connect_control(&mut self, from: NodeIndex, to: NodeIndex) {
        self.graph
            .add_edge(from.into(), to.into(), EdgeType::Control);
        self.update_processing_order();
    }

    #[inline]
    pub(crate) fn tick(&mut self) -> f32 {
        for &node_index in &self.sorted_nodes {
            let audio_input = self
                .graph
                .edges_directed(node_index, petgraph::Direction::Incoming)
                .filter(|edge| matches!(edge.weight(), EdgeType::Audio))
                .map(|edge| self.outputs[edge.source().index()])
                .sum();

            self.control_buffer.clear();
            self.control_buffer.extend(
                self.graph
                    .edges_directed(node_index, petgraph::Direction::Incoming)
                    .filter(|edge| matches!(edge.weight(), EdgeType::Control))
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
                    OT::Sine => connect_generator(graph, NodeKind::Sine(Sine::new()), input_node),
                    // OT::Square => connect_generator(graph, Box::new(Square::new()), input_node),
                    // OT::Saw => connect_generator(graph, Box::new(Saw::new()), input_node),
                    // OT::Noise => connect_generator(graph, Box::new(Noise::new()), input_node),
                    // OT::Pulse => connect_generator(graph, Box::new(Pulse::new()), input_node),
                    // OT::Gain => {
                    //     let node_index =
                    //         connect_generator(graph, Box::new(Gain::new()), input_node);
                    //     if let Some(expr) = args.get(0) {
                    //         connect_control(graph, expr, node_index);
                    //     }
                    //
                    //     node_index
                    // }
                    // OT::Mix => {
                    //     let node_index = connect_generator(graph, Box::new(Mix::new()), input_node);
                    //     if let Some(expr) = args.get(0) {
                    //         connect_control(graph, expr, node_index);
                    //     }
                    //
                    //     node_index
                    // }
                    // OT::AR => {
                    //     let node_index = connect_generator(graph, Box::new(AR::new()), input_node);
                    //     // nb: notes need to be connected in reverse order
                    //     if let Some(release_expr) = args.get(1) {
                    //         connect_control(graph, release_expr, node_index);
                    //     }
                    //
                    //     if let Some(attack_expr) = args.get(0) {
                    //         connect_control(graph, attack_expr, node_index);
                    //     }
                    //
                    //     node_index
                    // }
                    // OT::SVF => {
                    //     let node_index = connect_generator(graph, Box::new(SVF::new()), input_node);
                    //
                    //     if let Some(resonance_expr) = args.get(1) {
                    //         connect_control(graph, resonance_expr, node_index);
                    //     }
                    //
                    //     if let Some(cutoff_expr) = args.get(0) {
                    //         connect_control(graph, cutoff_expr, node_index);
                    //     }
                    //
                    //     node_index
                    // }
                    // OT::Seq => args
                    //     .get(0)
                    //     .and_then(|expr| match expr {
                    //         Expr::List(list) => Some(connect_generator(
                    //             graph,
                    //             Box::new(Seq::new(list.to_vec())),
                    //             input_node,
                    //         )),
                    //         _ => None,
                    //     })
                    //     .expect("Seq operator requires a list of nodes"),
                }
            }
            Expr::Number(n) => graph.add_node(NodeKind::Constant(Constant::new(n))),
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

        let matching_node = new.graph.node_indices().find(|&idx| idx == old_idx);

        match matching_node {
            Some(new_idx) => {
                if **old_node != *new.graph[new_idx] {
                    to_update.push((old_idx, *new.graph[new_idx].clone()));
                }
            }
            None => to_remove.push(old_idx),
        }
    }

    for new_idx in new.graph.node_indices() {
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
}
