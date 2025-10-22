use crate::nodes::{Frame, Node};
use crate::utils::{hard_clip, scale_buffer, soft_limit_poly};
use std::collections::HashMap;

pub struct Arena {
    nodes: Vec<Box<dyn Node>>,
    buffers: HashMap<String, Vec<Frame>>,
}

impl Arena {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            buffers: HashMap::new(),
        }
    }

    pub fn alloc(&mut self, node: Box<dyn Node>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(node);
        id
    }

    pub fn get(&self, id: usize) -> &dyn Node {
        &*self.nodes[id]
    }

    pub fn get_mut(&mut self, id: usize) -> &mut dyn Node {
        &mut *self.nodes[id]
    }

    pub fn store_buffer(&mut self, name: String, buffer: Vec<Frame>) {
        self.buffers.insert(name, buffer);
    }

    pub fn get_buffer(&self, name: &str) -> Option<&[Frame]> {
        self.buffers.get(name).map(|v| v.as_slice())
    }
}

pub(crate) struct Container {
    arena: Arena,
    root_node: Option<usize>,
    output_buffer: Vec<Frame>,
}

impl Container {
    pub(crate) fn new() -> Self {
        Self {
            arena: Arena::new(),
            root_node: None,
            output_buffer: Vec::new(),
        }
    }

    pub(crate) fn update_graph(&mut self, new_arena: Arena, new_root: usize) {
        if let Some(old_id) = self.root_node {
            self.apply_diff(old_id, &new_arena, new_root);
        }
        self.arena = new_arena;
        self.root_node = Some(new_root);
    }

    #[inline(always)]
    pub fn process_interleaved(&mut self, data: &mut [f32]) {
        let num_frames = data.len() / 2;

        if self.output_buffer.len() < num_frames {
            self.output_buffer.resize(num_frames, [0.0; 2]);
        }

        if let Some(root_id) = self.root_node {
            let output_chunk = &mut self.output_buffer[..num_frames];

            unsafe {
                let arena_ptr = &mut self.arena as *mut Arena;
                (*arena_ptr)
                    .get_mut(root_id)
                    .process(&mut *arena_ptr, output_chunk);
            }

            scale_buffer(output_chunk, 0.5);
            soft_limit_poly(output_chunk);
            hard_clip(output_chunk);

            for i in 0..num_frames {
                data[i * 2] = output_chunk[i][0];
                data[i * 2 + 1] = output_chunk[i][1];
            }
        } else {
            data.fill(0.0);
        }
    }

    fn apply_diff(&self, old_id: usize, arena: &Arena, new_id: usize) {
        let old_node = self.arena.get(old_id);
        let new_node = arena.get(new_id);

        if old_node.get_id() == new_node.get_id() {
            // transfer state from old to new
            unsafe {
                let arena_ptr = arena as *const Arena as *mut Arena;
                (*arena_ptr).get_mut(new_id).transfer_state(old_node);
            }

            let old_inputs = old_node.get_inputs();
            let new_inputs = new_node.get_inputs();

            for (&old, &new) in old_inputs.iter().zip(new_inputs.iter()) {
                self.apply_diff(old, arena, new);
            }
        }
    }
}
