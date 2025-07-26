use crate::nodes::{Frame, Node, NodeKind};
use crate::op::Op;
use petgraph::{graph::NodeIndex, prelude::StableDiGraph, visit::EdgeRef};
use std::collections::HashMap;

pub(crate) struct Container {
    graph: Option<Graph>,
    buffers: Vec<Vec<Frame>>,
}

impl Container {
    pub(crate) fn new() -> Self {
        Self {
            graph: None,
            buffers: Vec::new(),
        }
    }

    pub(crate) fn update_graph(&mut self, new: Graph) {
        if let Some(old) = self.graph.as_mut() {
            old.apply_diff(new)
        } else {
            self.graph = Some(new);
        }
    }

    pub(crate) fn load_frames_to_buffer(&mut self, frames: Vec<Frame>) -> usize {
        self.buffers.push(frames);
        self.buffers.len() - 1
    }

    pub(crate) fn add_buffer(&mut self, length: usize) -> usize {
        self.buffers.push(vec![[0.0, 0.0]; length]);
        self.buffers.len() - 1
    }

    #[inline]
    pub fn tick(&mut self) -> Frame {
        let Some(graph) = &mut self.graph else {
            return [0.0, 0.0];
        };

        let buffers = &mut self.buffers;

        for (writer_idx, buf_name) in &graph.buffer_writers {
            graph.inputs.clear();
            graph.inputs.extend(
                graph
                    .graph
                    .edges_directed(*writer_idx, petgraph::Direction::Incoming)
                    .map(|edge| graph.outputs[edge.source().index()]),
            );

            if let Some(buffer) = buffers.get_mut(*buf_name) {
                let node = &mut graph.graph[*writer_idx];
                node.tick_write_buffer(&graph.inputs, buffer);
            }
        }

        for &node_idx in &graph.sorted_nodes {
            if graph.buffer_writers.contains_key(&node_idx) {
                continue;
            }

            graph.inputs.clear();

            graph.inputs.extend(
                graph
                    .graph
                    .edges_directed(node_idx, petgraph::Direction::Incoming)
                    .map(|edge| graph.outputs[edge.source().index()]),
            );

            let node = &mut graph.graph[node_idx];
            let output_idx = node_idx.index();

            if let Some(buf_name) = graph.buffer_readers.get(&node_idx) {
                if let Some(buffer) = buffers.get(*buf_name) {
                    graph.outputs[output_idx] = node.tick_read_buffer(&graph.inputs, buffer);
                    continue;
                }
            }
            graph.outputs[output_idx] = node.tick(&graph.inputs);
        }

        *graph.outputs.last().unwrap_or(&[0.0, 0.0])
    }
}

#[derive(Clone)]
pub(crate) struct Graph {
    graph: StableDiGraph<Box<Op>, ()>,
    sorted_nodes: Vec<NodeIndex>,
    buffer_writers: HashMap<NodeIndex, usize>,
    buffer_readers: HashMap<NodeIndex, usize>,
    inputs: Vec<Frame>,
    outputs: Vec<Frame>,
}

impl Graph {
    pub(crate) fn new() -> Self {
        Self {
            graph: StableDiGraph::new(),
            sorted_nodes: Vec::new(),
            buffer_writers: HashMap::new(),
            buffer_readers: HashMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    pub(crate) fn add_node(&mut self, node: Op) -> NodeIndex {
        let index = self.graph.add_node(Box::new(node.clone()));

        // if node is a buffer reader or writer, connect it to the corresponding buffer
        match node {
            Op::Node { kind, .. } => match kind {
                NodeKind::BufferWriter { id } => {
                    self.buffer_writers.insert(index, id);
                }
                NodeKind::BufferReader { id } => {
                    self.buffer_readers.insert(index, id);
                }
                _ => {}
            },
            _ => {}
        }

        self.update_processing_order();

        index
    }

    fn connect_node(&mut self, from: NodeIndex, to: NodeIndex) {
        self.graph.add_edge(from.into(), to.into(), ());
        self.update_processing_order();
    }

    fn update_processing_order(&mut self) {
        self.sorted_nodes = petgraph::algo::toposort(&self.graph, None).expect("Graph has cycles");
        self.inputs.resize(self.graph.node_count(), [0.0, 0.0]);
        self.outputs.resize(self.graph.node_count(), [0.0, 0.0]);
    }

    pub(crate) fn apply_diff(&mut self, mut new_graph: Graph) {
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
                    (&mut **new_node, &**old_node)
                {
                    *new = old.clone();
                }
            }
        }

        // Replace current graph with the state-transferred new graph
        *self = new_graph;
    }
}

pub(crate) fn parse_to_graph(expr: Op) -> Graph {
    let mut graph = Graph::new();

    fn add_expr_to_graph(expr: &Op, graph: &mut Graph) -> NodeIndex {
        match expr {
            Op::Constant { .. } => add_node(vec![], expr, graph),
            Op::Node { inputs, .. } => add_node(inputs.iter().collect::<Vec<_>>(), expr, graph),
            Op::Gain(lhs, rhs) | Op::Mix(lhs, rhs) | Op::Wrap(lhs, rhs) => {
                add_node(vec![lhs, rhs], expr, graph)
            }
            Op::Negate(val) => add_node(vec![val], expr, graph),
        }
    }

    fn add_node(inputs: Vec<&Op>, kind: &Op, graph: &mut Graph) -> NodeIndex {
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
