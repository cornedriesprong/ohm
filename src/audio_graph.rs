use fundsp::{
    hacker::{mul, noise, pulse, square, AudioUnit, Constant, Frame},
    hacker32::{dc, sine},
    net::Net,
    prelude::NodeId,
};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use std::sync::{Arc, Mutex};

use crate::nodes::*;
use crate::parser::{Expr, OperatorType};

#[derive(Clone, Debug)]
enum EdgeType {
    Audio,
    Control,
}

pub(crate) struct AudioGraph {
    graph: DiGraph<Arc<Mutex<Box<dyn Node>>>, EdgeType>,
    sorted_nodes: Vec<NodeIndex>, // store sorted node order
    outputs: Vec<f32>,
    control_buffer: Vec<f32>,
}

impl AudioGraph {
    pub(crate) fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            sorted_nodes: Vec::new(),
            outputs: Vec::new(),
            control_buffer: Vec::new(),
        }
    }

    pub(crate) fn add_node(&mut self, node: Box<dyn Node>) -> NodeIndex {
        let index = self.graph.add_node(Arc::new(Mutex::new(node)));
        self.update_processing_order();
        index
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
    pub(crate) fn process(&mut self) -> f32 {
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

            let node = self.graph[node_index].clone();
            let mut node = node.lock().unwrap();
            let output_value = node.process(audio_input, &self.control_buffer);

            self.outputs[node_index.index()] = output_value;
        }

        self.outputs[self.sorted_nodes.last().unwrap().index()]
    }

    fn update_processing_order(&mut self) {
        self.sorted_nodes = petgraph::algo::toposort(&self.graph, None).expect("Graph has cycles");
        self.outputs.resize(self.graph.node_count(), 0.0);
    }
}

// pub(crate) fn parse_to_audio_graph(expr: Expr) -> AudioGraph {
//     let mut graph = AudioGraph::new();
//
//     fn connect_generator(
//         graph: &mut AudioGraph,
//         node: Box<dyn Node>,
//         input_node: NodeIndex,
//     ) -> NodeIndex {
//         let node_index = graph.add_node(node);
//         graph.connect_generator(input_node, node_index);
//         node_index
//     }
//
//     fn connect_control(expr: &Expr, graph: &mut AudioGraph, target_idx: NodeIndex) {
//         match expr {
//             Expr::Number(n) => {
//                 let constant = Box::new(Constant::new(*n));
//                 let constant_index = graph.add_node(constant);
//                 graph.connect_control(constant_index, target_idx);
//             }
//             Expr::Operator { .. } => {
//                 let node = add_expr_to_graph(expr.clone(), graph);
//                 graph.connect_control(node, target_idx);
//             }
//         }
//     }
//
//     fn add_expr_to_graph(expr: Expr, graph: &mut AudioGraph) -> NodeIndex {
//         match expr {
//             Expr::Operator { kind, input, args } => {
//                 let input_node = add_expr_to_graph(*input, graph);
//
//                 use OperatorType as OT;
//                 match kind {
//                     OT::Sine => connect_generator(graph, Box::new(Sine::new()), input_node),
//                     OT::Square => connect_generator(graph, Box::new(Square::new()), input_node),
//                     OT::Noise => connect_generator(graph, Box::new(Noise::new()), input_node),
//                     OT::Pulse => connect_generator(graph, Box::new(Pulse::new()), input_node),
//                     OT::Gain => {
//                         let node_index =
//                             connect_generator(graph, Box::new(Gain::new()), input_node);
//                         if let Some(expr) = args.get(0) {
//                             connect_control(expr, graph, node_index);
//                         }
//
//                         node_index
//                     }
//                     OT::Mix => {
//                         let node_index = connect_generator(graph, Box::new(Mix::new()), input_node);
//                         if let Some(expr) = args.get(0) {
//                             connect_control(expr, graph, node_index);
//                         }
//
//                         node_index
//                     }
//                     OT::AR => {
//                         let node_index = connect_generator(graph, Box::new(AR::new()), input_node);
//                         // nb: notes need to be connected in reverse order
//                         if let Some(release_expr) = args.get(2) {
//                             connect_control(release_expr, graph, node_index);
//                         }
//
//                         if let Some(attack_expr) = args.get(1) {
//                             connect_control(attack_expr, graph, node_index);
//                         }
//
//                         if let Some(trigger_expr) = args.get(0) {
//                             connect_control(trigger_expr, graph, node_index);
//                         }
//
//                         node_index
//                     }
//                 }
//             }
//             Expr::Number(n) => graph.add_node(Box::new(Constant::new(n))),
//         }
//     }
//
//     add_expr_to_graph(expr, &mut graph);
//
//     graph
// }

pub(crate) fn parse_to_net(expr: Expr) -> Net {
    let mut net = Net::new(0, 1);

    fn connect_generator(net: &mut Net, target: Box<dyn AudioUnit>, src_id: NodeId) -> NodeId {
        let target_id = net.push(target);
        net.connect(src_id, 0, target_id, 0);
        target_id
    }

    fn add_expr_to_net(expr: Expr, net: &mut Net) -> NodeId {
        match expr {
            Expr::Operator { kind, input, args } => {
                let input_node = add_expr_to_net(*input, net);

                use OperatorType as OT;
                match kind {
                    OT::Sine => connect_generator(net, Box::new(sine()), input_node),
                    OT::Square => connect_generator(net, Box::new(square()), input_node),
                    OT::Noise => connect_generator(net, Box::new(noise()), input_node),
                    OT::Pulse => connect_generator(net, Box::new(pulse()), input_node),
                    // OT::Gain => {
                    //     let mut node_index = input_node;
                    //     if let Some(expr) = args.get(0) {
                    //         node_index =
                    //             connect_generator(net, Box::new(mul(expr.into())), input_node);
                    //         // connect_control(expr, graph, node_index);
                    //     }
                    //
                    //     node_index
                    // }
                    OT::Gain => {
                        if let Some(gain_expr) = args.get(0) {
                            // Parse the gain expression recursively
                            let gain_node = add_expr_to_net(gain_expr.clone(), net);
                            // Create a multiplier that combines the input signal with the gain
                            connect_generator(net, Box::new(mul(gain_node)), input_node);
                            let mul_id = net.push(Box::new(mul()));
                            // Connect both the input and the gain control to the multiplier
                            net.connect(input_node, 0, mul_id, 0);
                            net.connect(gain_node, 0, mul_id, 1);
                            mul_id
                        } else {
                            input_node
                        }
                    }
                    // OT::Mix => {
                    //     let node_index = connect_generator(graph, Box::new(mul), input_node);
                    //     if let Some(expr) = args.get(0) {
                    //         connect_control(expr, graph, node_index);
                    //     }
                    //
                    //     node_index
                    // }
                    _ => unimplemented!(),
                }
            }
            Expr::Number(n) => net.push(Box::new(dc(n))),
        }
    }

    let node = add_expr_to_net(expr, &mut net);
    net.connect_output(node, 0, 0);

    net
}
