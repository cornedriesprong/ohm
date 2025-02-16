use crate::nodes::*;
use crate::parser::Expr;

pub(crate) fn parse_to_audio_graph(expr: Expr) -> Box<dyn Node> {
    fn add_expr_to_graph(expr: Expr) -> Box<dyn Node> {
        match expr {
            Expr::Constant(n) => Box::new(NodeKind::constant(n)),
            Expr::Gain { a, b } => {
                let a_node = add_expr_to_graph(*a);
                let b_node = add_expr_to_graph(*b);
                let node = NodeKind::gain(a_node, b_node);

                return Box::new(node);
            }
            Expr::Mix { a, b } => {
                let a_node = add_expr_to_graph(*a);
                let b_node = add_expr_to_graph(*b);
                let node = NodeKind::mix(a_node, b_node);

                return Box::new(node);
            }
            Expr::Sine { freq } => {
                let freq_node = add_expr_to_graph(*freq);
                let node = NodeKind::sine(freq_node);

                return Box::new(node);
            }
            Expr::Square { freq } => {
                let freq_node = add_expr_to_graph(*freq);
                let node = NodeKind::square(freq_node);

                return Box::new(node);
            }
            Expr::Saw { freq } => {
                let freq_node = add_expr_to_graph(*freq);
                let node = NodeKind::saw(freq_node);

                return Box::new(node);
            }
            Expr::Pulse { freq } => {
                let freq_node = add_expr_to_graph(*freq);
                let node = NodeKind::pulse(freq_node);

                return Box::new(node);
            }
            Expr::Noise => {
                let node = NodeKind::noise();
                return Box::new(node);
            }
            Expr::AR {
                attack,
                release,
                trig,
            } => {
                let attack_node = add_expr_to_graph(*attack);
                let release_node = add_expr_to_graph(*release);
                let trig_node = add_expr_to_graph(*trig);
                let node = NodeKind::ar(attack_node, release_node, trig_node);

                return Box::new(node);
            }
            Expr::SVF {
                cutoff,
                resonance,
                input,
            } => {
                let cutoff_node = add_expr_to_graph(*cutoff);
                let resonance_node = add_expr_to_graph(*resonance);
                let input_node = add_expr_to_graph(*input);
                let node = NodeKind::svf(cutoff_node, resonance_node, input_node);

                return Box::new(node);
            }
            _ => panic!("Invalid expression"),
        }
    }

    add_expr_to_graph(expr)
}
