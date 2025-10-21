use crate::nodes::{Frame, Node};
use crate::utils::{hard_clip, scale_buffer, soft_limit_poly};

pub(crate) struct Container {
    root: Option<Box<dyn Node>>,
    output_buffer: Vec<Frame>,
}

impl Container {
    pub(crate) fn new() -> Self {
        Self {
            root: None,
            output_buffer: Vec::new(),
        }
    }

    pub(crate) fn update_graph(&mut self, mut new: Box<dyn Node>) {
        if let Some(ref old_root) = self.root {
            self.apply_diff(&**old_root, &mut *new);
        }
        self.root = Some(new);
    }

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

            for i in 0..num_frames {
                data[i * 2] = output_chunk[i][0];
                data[i * 2 + 1] = output_chunk[i][1];
            }
        } else {
            data.fill(0.0);
        }
    }

    fn apply_diff(&self, old: &dyn Node, new: &mut dyn Node) {
        if old.get_id() == new.get_id() {
            new.transfer_state(old);

            let old_inputs = old.get_inputs();
            let new_inputs = new.get_inputs_mut();

            for (o, n) in old_inputs.iter().zip(new_inputs.iter_mut()) {
                self.apply_diff(&**o, &mut **n);
            }
        }
    }
}
