use crate::nodes::{Frame, Node};
use crate::utils::{hard_clip, scale_buffer, soft_limit_poly};
use petgraph::{graph::NodeIndex, prelude::StableDiGraph, visit::EdgeRef};
use std::collections::HashMap;

pub(crate) struct Graph {
    graph: StableDiGraph<Node, ()>,
    sorted_nodes: Vec<NodeIndex>,
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

    pub(crate) fn add_node(&mut self, node: Node) -> NodeIndex {
        let index = self.graph.add_node(node);
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

        for &node_idx in &self.sorted_nodes {
            let input_count = self
                .graph
                .edges_directed(node_idx, petgraph::Direction::Incoming)
                .count();
            if input_count > self.max_inputs {
                self.max_inputs = input_count;
            }
        }

        self.input_indices.clear();
        self.input_indices.reserve(self.max_inputs);
    }

    pub(crate) fn apply_diff(&mut self, new_graph: Graph) {
        let mut old_nodes: HashMap<String, NodeIndex> = HashMap::new();
        for &node_idx in &self.sorted_nodes {
            let id = self.graph[node_idx].get_id();
            old_nodes.insert(id, node_idx);
        }

        let old_graph = std::mem::replace(self, new_graph);

        // transfer state
        for &new_node_idx in &self.sorted_nodes {
            let new_id = self.graph[new_node_idx].get_id();
            if let Some(&old_node_idx) = old_nodes.get(&new_id) {
                let old_node = &old_graph.graph[old_node_idx];
                let new_node = &mut self.graph[new_node_idx];
                new_node.transfer_state(old_node);
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
    pub fn process(&mut self, data: &mut [f32]) {
        let num_frames = data.len() / 2;

        if self.output_buffer.len() < num_frames {
            self.output_buffer.resize(num_frames, [0.0; 2]);
        }

        let output_chunk = &mut self.output_buffer[..num_frames];

        if let Some(graph) = &mut self.graph {
            graph.set_chunk_size(num_frames);

            let sorted_nodes = graph.sorted_nodes.clone();
            for &node_idx in &sorted_nodes {
                graph.input_indices.clear();
                for edge in graph
                    .graph
                    .edges_directed(node_idx, petgraph::Direction::Incoming)
                {
                    graph.input_indices.push(edge.source().index());
                }
                graph.input_indices.reverse();

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
                    let node = &mut graph.graph[node_idx];
                    node.process(&input_slices, output_buffer);
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
            }
        } else {
            output_chunk.fill([0.0, 0.0]);
        }

        // convert to interleaved
        for i in 0..num_frames {
            data[i * 2] = output_chunk[i][0];
            data[i * 2 + 1] = output_chunk[i][1];
        }
    }
}
