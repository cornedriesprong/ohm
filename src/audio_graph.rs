use crate::nodes::{Frame, Node};
use crate::utils::{hard_clip, scale_buffer, soft_limit_poly};

fn apply_diff(old: &dyn Node, new: &mut Box<dyn Node>) {
    fn compare(old: &dyn Node, new: &mut dyn Node) {
        if old.get_id() == new.get_id() {
            // Transfer state without cloning inputs
            new.transfer_state(old);

            // Recursively compare inputs (no clones needed!)
            let old_inputs = old.get_inputs();
            let new_inputs = new.get_inputs_mut();

            for (o, n) in old_inputs.iter().zip(new_inputs.iter_mut()) {
                compare(&**o, &mut **n);
            }
        }
    }
    println!("Applying diff...");

    compare(old, &mut **new);
}

pub(crate) struct Container {
    root: Option<Box<dyn Node>>,
    buffers: Vec<Vec<Frame>>,
    output_buffer: Vec<Frame>,
}

impl Container {
    pub(crate) fn new() -> Self {
        Self {
            root: None,
            buffers: Vec::new(),
            output_buffer: Vec::new(),
        }
    }

    pub(crate) fn update_graph(&mut self, mut new: Box<dyn Node>) {
        if let Some(ref old_root) = self.root {
            apply_diff(&**old_root, &mut new);
        }
        self.root = Some(new);
    }

    // pub(crate) fn load_frames_to_buffer(&mut self, frames: Vec<Frame>) -> usize {
    //     self.buffers.push(frames);
    //     self.buffers.len() - 1
    // }
    //
    // pub(crate) fn add_buffer(&mut self, length: usize) -> usize {
    //     self.buffers.push(vec![[0.0, 0.0]; length]);
    //     self.buffers.len() - 1
    // }

    #[inline(always)]
    pub fn process_interleaved(&mut self, data: &mut [f32]) {
        let num_frames = data.len() / 2;

        if self.output_buffer.len() < num_frames {
            self.output_buffer.resize(num_frames, [0.0; 2]);
        }

        if let Some(root) = &mut self.root {
            let output_chunk = &mut self.output_buffer[..num_frames];
            root.process(output_chunk);

            scale_buffer(output_chunk, 0.5);
            soft_limit_poly(output_chunk);
            hard_clip(output_chunk);

            // Convert to interleaved
            for i in 0..num_frames {
                data[i * 2] = output_chunk[i][0];
                data[i * 2 + 1] = output_chunk[i][1];
            }
        } else {
            // if no nodes, return silence
            data.fill(0.0);
        }
    }
}
