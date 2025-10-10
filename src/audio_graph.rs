use crate::nodes::{Frame, Node, NodeKind};
use crate::op::Op;
use crate::utils::{hard_clip, scale_buffer, soft_limit_poly};
use petgraph::{graph::NodeIndex, prelude::StableDiGraph, visit::EdgeRef};
use std::collections::HashMap;

pub(crate) struct Container {
    graph: Option<Graph>,
    buffers: Vec<Vec<Frame>>,
    output_buffer: Vec<Frame>,
}

impl Container {
    pub(crate) fn new() -> Self {
        Self {
            graph: None,
            buffers: Vec::new(),
            output_buffer: Vec::new(),
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

    #[inline(always)]
    pub fn process_interleaved(&mut self, data: &mut [f32]) {
        let num_frames = data.len() / 2;

        if self.output_buffer.len() < num_frames {
            self.output_buffer.resize(num_frames, [0.0; 2]);
        }

        // split borrow to avoid lifetime issues
        let (graph, buffers, output_buffer) =
            (&mut self.graph, &mut self.buffers, &mut self.output_buffer);

        if let Some(graph) = graph {
            let output_chunk = &mut output_buffer[..num_frames];
            graph.set_chunk_size(num_frames);

            // process all nodes in topological order
            let sorted_nodes = graph.sorted_nodes.clone();
            for &node_idx in &sorted_nodes {
                if graph.buffer_writers.contains_key(&node_idx) {
                    continue;
                }

                graph.input_indices.clear();
                for edge in graph
                    .graph
                    .edges_directed(node_idx, petgraph::Direction::Incoming)
                {
                    graph.input_indices.push(edge.source().index());
                }

                let output_idx = node_idx.index();

                // SAFETY: We use unsafe here to work around borrow checker limitations.
                // The topological sort guarantees that input nodes are processed before
                // this node, and that no node writes to its own input buffers.
                // Therefore, the input slices and output slice never alias.
                unsafe {
                    let buffers_ptr = graph.output_buffers.as_ptr();
                    let mut input_slices = Vec::with_capacity(graph.max_inputs);
                    for &idx in &graph.input_indices {
                        let buf_ptr = buffers_ptr.add(idx);
                        input_slices
                            .push(std::slice::from_raw_parts((*buf_ptr).as_ptr(), num_frames));
                    }

                    let output_buffer = &mut graph.output_buffers[output_idx][..num_frames];

                    if let Some(buf_name) = graph.buffer_readers.get(&node_idx) {
                        if let Some(buffer) = buffers.get(*buf_name) {
                            let node = &mut graph.graph[node_idx];
                            node.process_read_buffer(&input_slices, buffer, output_buffer);
                            continue;
                        }
                    }

                    let node = &mut graph.graph[node_idx];
                    node.process(&input_slices, output_buffer);
                }
            }

            for (writer_idx, buf_name) in &graph.buffer_writers {
                graph.input_indices.clear();
                for edge in graph
                    .graph
                    .edges_directed(*writer_idx, petgraph::Direction::Incoming)
                {
                    graph.input_indices.push(edge.source().index());
                }

                if let Some(buffer) = buffers.get_mut(*buf_name) {
                    let node = &mut graph.graph[*writer_idx];

                    // SAFETY: We construct slices from raw pointers that are valid and won't alias
                    // with the buffer being written to (different memory regions)
                    unsafe {
                        let buffers_ptr = graph.output_buffers.as_ptr();
                        let mut input_slices = Vec::with_capacity(graph.max_inputs);
                        for &idx in &graph.input_indices {
                            let buf_ptr = buffers_ptr.add(idx);
                            input_slices
                                .push(std::slice::from_raw_parts((*buf_ptr).as_ptr(), num_frames));
                        }
                        node.process_write_buffer(&input_slices, buffer);
                    }
                }
            }

            if let Some(last_node_idx) = graph.sorted_nodes.last() {
                let final_output = &graph.output_buffers[last_node_idx.index()][..num_frames];
                output_chunk.copy_from_slice(final_output);

                scale_buffer(output_chunk, 0.5);
                soft_limit_poly(output_chunk);
                hard_clip(output_chunk);
            } else {
                // if no nodes, return silence
                output_chunk.fill([0.0, 0.0]);
                for i in 0..num_frames {
                    data[i * 2] = 0.0;
                    data[i * 2 + 1] = 0.0;
                }
                return;
            }
        }

        // convert back to interleaved
        let frames = &self.output_buffer[..num_frames];
        for i in 0..num_frames {
            data[i * 2] = frames[i][0];
            data[i * 2 + 1] = frames[i][1];
        }
    }
}

#[derive(Clone)]
pub(crate) struct Graph {
    graph: StableDiGraph<Box<Op>, ()>,
    sorted_nodes: Vec<NodeIndex>,
    buffer_writers: HashMap<NodeIndex, usize>,
    buffer_readers: HashMap<NodeIndex, usize>,
    input_indices: Vec<usize>,
    output_buffers: Vec<Vec<Frame>>,
    chunk_size: usize,
    max_inputs: usize,
}

impl Graph {
    pub(crate) fn new() -> Self {
        Self {
            graph: StableDiGraph::new(),
            sorted_nodes: Vec::new(),
            buffer_writers: HashMap::new(),
            buffer_readers: HashMap::new(),
            input_indices: Vec::new(),
            output_buffers: Vec::new(),
            chunk_size: 0,
            max_inputs: 8,
        }
    }

    fn set_chunk_size(&mut self, chunk_size: usize) {
        if chunk_size > self.chunk_size {
            self.chunk_size = chunk_size;
            for buffer in &mut self.output_buffers {
                buffer.resize(chunk_size, [0.0, 0.0]);
            }
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
                NodeKind::BufferReader { id } | NodeKind::BufferTap { id } => {
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

        // resize output_buffers to match node count
        let node_count = self.graph.node_count();
        if self.output_buffers.len() < node_count {
            self.output_buffers.resize(node_count, Vec::new());
        }

        // ensure each buffer has at least max_chunk_size capacity
        for buffer in &mut self.output_buffers {
            if buffer.len() < self.chunk_size {
                buffer.resize(self.chunk_size, [0.0, 0.0]);
            }
        }

        // track maximum number of inputs across all nodes for pre-allocation
        for &node_idx in &self.sorted_nodes {
            let input_count = self
                .graph
                .edges_directed(node_idx, petgraph::Direction::Incoming)
                .count();
            if input_count > self.max_inputs {
                self.max_inputs = input_count;
            }
        }

        // pre-allocate input_indices vector
        self.input_indices.clear();
        self.input_indices.reserve(self.max_inputs);
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

        // replace current graph with the state-transferred new graph
        *self = new_graph;
    }
}

pub(crate) fn parse_to_graph(expr: Op) -> Graph {
    let mut graph = Graph::new();

    fn add_expr_to_graph(expr: &Op, graph: &mut Graph) -> NodeIndex {
        match expr {
            Op::Constant { .. } => add_node(vec![], expr, graph),
            Op::Node { inputs, .. } => add_node(inputs.iter().collect::<Vec<_>>(), expr, graph),
            Op::Gain(lhs, rhs)
            | Op::Mix(lhs, rhs)
            | Op::Wrap(lhs, rhs)
            | Op::Power(lhs, rhs)
            | Op::Greater(lhs, rhs)
            | Op::Less(lhs, rhs)
            | Op::Equal(lhs, rhs) => add_node(vec![lhs, rhs], expr, graph),
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

    println!(
        "running audio graph with {} nodes",
        graph.graph.node_count()
    );

    graph
}
