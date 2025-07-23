use crate::nodes::{Frame, Node, NodeKind};
use crate::op::Op;
use petgraph::visit::IntoEdgesDirected;
use petgraph::{graph::NodeIndex, prelude::StableDiGraph, visit::EdgeRef};
use rtsan_standalone::nonblocking;
use std::collections::HashMap;

pub(crate) struct Container {
    graph: Option<Graph>,
    buffers: HashMap<String, Vec<Frame>>,
}

impl Container {
    pub(crate) fn new() -> Self {
        Self {
            graph: None,
            buffers: HashMap::new(),
        }
    }

    pub(crate) fn update_graph(&mut self, new: Graph) {
        if let Some(old) = self.graph.as_mut() {
            old.apply_diff(new)
        } else {
            self.graph = Some(new);
        }
    }

    pub(crate) fn add_buffer(&mut self, name: &str) {
        self.buffers
            .insert(name.to_string(), vec![[0.0, 0.0]; 1024]);
    }

    pub(crate) fn get_buffer(&mut self, name: &str) -> Option<&Vec<Frame>> {
        if let Some(buffer) = self.buffers.get(name) {
            Some(&buffer)
        } else {
            None
        }
    }

    pub(crate) fn get_buffer_mut(&mut self, name: &str) -> Option<&mut Vec<Frame>> {
        if let Some(buffer) = self.buffers.get_mut(name) {
            Some(buffer)
        } else {
            None
        }
    }

    // #[inline]
    // #[nonblocking]
    // pub(crate) fn tick(&mut self) -> Frame {
    //     if let Some(mut g) = self.graph.as_mut() {
    //         for (&writer_idx, buffer_id) in &g.buffer_writers {
    //             // Get mutable access to the specific buffer for this writer.
    //             let buffer = &mut self.buffers.get(buffer_id).unwrap();
    //
    //             g.inputs.clear();
    //             g.inputs.extend(
    //                 g.graph
    //                     .edges_directed(writer_idx, petgraph::Direction::Incoming)
    //                     .map(|edge| g.outputs[edge.source().index()]),
    //             );
    //
    //             let writer_node = &mut g.graph[writer_idx];
    //             g.outputs[writer_idx.index()] = writer_node.tick_write_buffer(
    //                 &g.inputs,
    //                 buffer, // Pass the single mutable buffer slice
    //             );
    //         }
    //
    //         // graph.tick()
    //         for &node_index in &g.sorted_nodes {
    //             g.inputs.clear();
    //
    //             g.inputs.extend(
    //                 g.graph
    //                     .edges_directed(node_index, petgraph::Direction::Incoming)
    //                     .map(|edge| g.outputs[edge.source().index()]),
    //             );
    //
    //             g.outputs[node_index.index()] = g.graph[node_index].tick(&g.inputs);
    //         }
    //
    //         g.outputs[g.sorted_nodes.len() - 1]
    //     } else {
    //         [0.0; 2]
    //     }
    // }

    #[inline]
    pub fn tick(&mut self) -> Frame {
        // 1. Guard against a non-existent graph. This part is correct.
        let Some(graph) = &mut self.graph else {
            return [0.0, 0.0];
        };

        // 2. Get a mutable borrow of the buffers. This is also correct.
        let buffers = &mut self.buffers;

        // --- PASS 1: WRITERS ---

        // 4. Now, iterate over the temporary Vec. `graph` is no longer borrowed by the loop.
        for (writer_idx, buffer_name) in &graph.buffer_writers {
            // Get mutable access to the specific buffer for this writer.
            let buffer = buffers.get_mut(buffer_name).unwrap();

            // Gather inputs for the writer node. This is a mutable borrow of `graph.inputs`
            // and an immutable borrow of `graph.outputs`. This is fine.
            graph.inputs.clear();
            graph.inputs.extend(
                graph
                    .graph
                    .edges_directed(*writer_idx, petgraph::Direction::Incoming)
                    .map(|edge| graph.outputs[edge.source().index()]),
            );

            // This is now valid! We can mutably borrow `graph.graph` because
            // the `for` loop is no longer borrowing any part of `graph`.
            let writer_node = &mut graph.graph[*writer_idx];
            graph.outputs[writer_idx.index()] =
                writer_node.tick_write_buffer(&graph.inputs, buffer);
        }

        // --- PASS 2: READERS & NORMAL NODES ---
        // This logic can remain the same, as it only requires immutable borrows.
        for &node_idx in &graph.sorted_nodes {
            if graph.buffer_writers.contains_key(&node_idx) {
                continue;
            }

            graph.inputs.clear();

            graph.inputs.extend(
                graph.graph
                    .edges_directed(node_idx, petgraph::Direction::Incoming)
                    .map(|edge| graph.outputs[edge.source().index()]),
            );

            let node = &mut graph.graph[node_idx];
            let output_idx = node_idx.index();

            if let Some(buffer_name) = graph.buffer_readers.get(&node_idx) {
                let buffer = buffers.get(buffer_name).unwrap();
                graph.outputs[output_idx] = node.tick_read_buffer(&graph.inputs, buffer);
            } else {
                graph.outputs[output_idx] = node.tick(&graph.inputs);
            }
        }

        graph.outputs.last().copied().unwrap_or([0.0, 0.0])
    }
}

#[derive(Clone)]
pub(crate) struct Graph {
    graph: StableDiGraph<Box<Op>, ()>,
    sorted_nodes: Vec<NodeIndex>,
    // Mappings from a node to the ID of the buffer it uses.
    buffer_writers: HashMap<NodeIndex, String>,
    buffer_readers: HashMap<NodeIndex, String>,

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

        // if node is a buffer reader or writer,
        // connect it to the corresponding buffer
        match node {
            Op::Node { kind, .. } => match kind {
                NodeKind::BufferWriter { name } => {
                    // TODO: assert that there is only one writer per buffer
                    self.buffer_writers.insert(index, name);
                }
                NodeKind::BufferReader { name } => {
                    self.buffer_readers.insert(index, name);
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

    // #[inline]
    // #[nonblocking]
    // pub(crate) fn tick(&mut self) -> Frame {
    //     for &node_index in &self.sorted_nodes {
    //         self.inputs.clear();
    //
    //         self.inputs.extend(
    //             self.graph
    //                 .edges_directed(node_index, petgraph::Direction::Incoming)
    //                 .map(|edge| self.outputs[edge.source().index()]),
    //         );
    //
    //         self.outputs[node_index.index()] = self.graph[node_index].tick(&self.inputs);
    //     }
    //
    //     self.outputs[self.sorted_nodes.len() - 1]
    // }

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
