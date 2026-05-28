// build.rs runs before the main crate is compiled.
// It generates Rust source code from our ONNX file and writes the weights to
// a binary record file. Both outputs land in Cargo's OUT_DIR so they don't
// clutter src/, but we can include them with the include! macro.
use burn_import::onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("models/resnet18_encoder.onnx")
        .out_dir("model/")   // relative to OUT_DIR
        .run_from_script();
}
