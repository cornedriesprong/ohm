use crate::nodes::{Frame, Node};
use crate::utils::{hard_clip, scale_buffer, soft_limit_poly};
use petgraph::{graph::NodeIndex, prelude::StableDiGraph, visit::EdgeRef};
use smallvec::SmallVec;
use std::collections::HashMap;

pub(crate) struct Graph {
    graph: StableDiGraph<Node, ()>,
    sorted_nodes: Vec<NodeIndex>,
    inputs: Vec<Vec<usize>>,
    buffers: HashMap<NodeIndex, Vec<Frame>>,
    output_buffers: Vec<Vec<Frame>>,
    chunk_size: usize,
}

impl Graph {
    pub(crate) fn new() -> Self {
        Self {
            graph: StableDiGraph::new(),
            sorted_nodes: Vec::new(),
            inputs: Vec::new(),
            buffers: HashMap::new(),
            output_buffers: Vec::new(),
            chunk_size: 0,
        }
    }

    #[inline(always)]
    fn set_chunk_size(&mut self, chunk_size: usize) {
        if chunk_size > self.chunk_size {
            self.chunk_size = chunk_size;
            for buffer in &mut self.output_buffers {
                buffer.resize(chunk_size, [0.0, 0.0]);
            }
        }
    }

    pub(crate) fn add_node(&mut self, node: Node) -> NodeIndex {
        let index = self.graph.add_node(node.clone());

        self.update_processing_order();

        index
    }

    pub(crate) fn add_buffer_node(&mut self, frames: Vec<Frame>) -> NodeIndex {
        let index = self.graph.add_node(Node::BufferRef {
            id: NodeIndex::end(),
            length: frames.len(),
        });

        // update node id
        let node = &mut self.graph[index];
        if let Node::BufferRef { id, .. } = node {
            *id = index;
        }

        self.buffers.insert(index, frames);

        self.update_processing_order();

        index
    }

    pub(crate) fn connect_node(&mut self, from: NodeIndex, to: NodeIndex) {
        self.graph.add_edge(from, to, ());
        self.update_processing_order();
    }

    fn update_processing_order(&mut self) {
        self.sorted_nodes = petgraph::algo::toposort(&self.graph, None).expect("Graph has cycles");

        let node_count = self.graph.node_count();
        if self.output_buffers.len() < node_count {
            self.output_buffers.resize(node_count, Vec::new());
        }

        for buffer in &mut self.output_buffers {
            if buffer.len() < self.chunk_size {
                buffer.resize(self.chunk_size, [0.0, 0.0]);
            }
        }

        self.inputs.clear();
        self.inputs.resize(node_count, Vec::new());

        for &node_idx in &self.sorted_nodes {
            let mut inputs: Vec<usize> = self
                .graph
                .edges_directed(node_idx, petgraph::Direction::Incoming)
                .map(|edge| edge.source().index())
                .collect();

            inputs.reverse();

            self.inputs[node_idx.index()] = inputs;
        }
    }

    fn process(&mut self, output: &mut [Frame]) {
        let num_frames = output.len();
        self.set_chunk_size(num_frames);

        for &node_idx in &self.sorted_nodes {
            let output_idx = node_idx.index();

            // if node is a buffer writer, skip processing
            // if self.buffers.contains_key(&node_idx) {
            //     continue;
            // }

            unsafe {
                let buffers_ptr = self.output_buffers.as_ptr();
                let node_inputs = &self.inputs[output_idx];

                let input_slices: SmallVec<[&[Frame]; 8]> = node_inputs
                    .iter()
                    .map(|&idx| {
                        let buf_ptr = buffers_ptr.add(idx);
                        std::slice::from_raw_parts((*buf_ptr).as_ptr(), num_frames)
                    })
                    .collect();

                let output_buffer = &mut self.output_buffers[output_idx][..num_frames];

                let node = &mut self.graph[node_idx];
                match node {
                    Node::BufferReader { id } | Node::BufferTap { id, .. } => {
                        if let Some(buffer) = self.buffers.get(id) {
                            let node = &mut self.graph[node_idx];
                            node.process_read_buffer(&input_slices, buffer, output_buffer);
                            continue;
                        }
                    }
                    _ => {}
                }

                node.process(&input_slices, output_buffer);

                match node {
                    Node::BufferWriter { id, .. } => {
                        if let Some(buffer) = self.buffers.get_mut(id) {
                            let buffers_ptr = self.output_buffers.as_ptr();
                            let node_inputs = &self.inputs[output_idx];
                            let input_slices: SmallVec<[&[Frame]; 8]> = node_inputs
                                .iter()
                                .map(|&idx| {
                                    let buf_ptr = buffers_ptr.add(idx);
                                    std::slice::from_raw_parts((*buf_ptr).as_ptr(), num_frames)
                                })
                                .collect();
                            node.process_write_buffer(&input_slices, buffer);
                        }
                    }
                    _ => {}
                }
            }
        }

        if let Some(last_node_idx) = self.sorted_nodes.last() {
            let final_output = &self.output_buffers[last_node_idx.index()][..num_frames];
            output.copy_from_slice(final_output);
        } else {
            output.fill([0.0, 0.0]);
        }
    }

    pub(crate) fn apply_diff(&mut self, new_graph: Graph) {
        let mut old_nodes: HashMap<String, NodeIndex> = HashMap::new();
        for &node_idx in &self.sorted_nodes {
            let id = self.graph[node_idx].get_id();
            old_nodes.insert(id, node_idx);
        }

        let old_graph = std::mem::replace(self, new_graph);

        // transfer state
        for &new_idx in &self.sorted_nodes {
            let new_id = self.graph[new_idx].get_id();
            if let Some(&old_idx) = old_nodes.get(&new_id) {
                let old = &old_graph.graph[old_idx];
                let new = &mut self.graph[new_idx];
                new.transfer_state(old);
            }
        }
    }
}

pub(crate) struct Container {
    graph: Option<Graph>,
    output_buffer: Vec<Frame>,
}

impl Container {
    pub(crate) fn new() -> Self {
        Self {
            graph: None,
            output_buffer: Vec::new(),
        }
    }

    pub(crate) fn update_graph(&mut self, new_graph: Graph) {
        if let Some(old_graph) = self.graph.as_mut() {
            old_graph.apply_diff(new_graph);
        } else {
            self.graph = Some(new_graph);
        }

        if let Some(graph) = &self.graph {
            println!(
                "rendering audio graph ({} nodes):",
                graph.graph.node_count()
            );
        }
    }

    #[inline(always)]
    pub fn process(&mut self, output: &mut [f32]) {
        let num_frames = output.len() / 2;

        if self.output_buffer.len() < num_frames {
            self.output_buffer.resize(num_frames, [0.0; 2]);
        }

        let output_chunk = &mut self.output_buffer[..num_frames];

        if let Some(graph) = &mut self.graph {
            graph.process(output_chunk);
            scale_buffer(output_chunk, 0.5);
            soft_limit_poly(output_chunk);
            hard_clip(output_chunk);
        } else {
            // if no graph, return silence
            output_chunk.fill([0.0, 0.0]);
        }

        // convert to interleaved
        for i in 0..num_frames {
            output[i * 2] = output_chunk[i][0];
            output[i * 2 + 1] = output_chunk[i][1];
        }
    }
}
