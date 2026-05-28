use anyhow::Result;
use burn::backend::Wgpu;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod model;
mod preprocess;

// We pin our backend to Wgpu once here. On your M4 Mac this automatically
// targets Metal. Everywhere else in the codebase we just say `MyBackend`
// so switching backends later requires changing only this one line.
type MyBackend = Wgpu;

/// lensme – visual image search engine
#[derive(Parser)]
#[command(name = "lensme", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Walk a folder, embed every image, and write the index to disk
    Index {
        /// Path to the folder of images
        folder: PathBuf,

        /// Where to write the index file (default: lensme.index)
        #[arg(long, default_value = "lensme.index")]
        output: PathBuf,
    },

    /// Embed a query image and return the top-K nearest neighbors
    Query {
        /// Path to the query image
        #[arg(long)]
        image: PathBuf,

        /// Number of nearest neighbors to return
        #[arg(long, short, default_value_t = 5)]
        k: usize,

        /// Path to the index file to search
        #[arg(long, default_value = "lensme.index")]
        index: PathBuf,
    },
}

fn is_image(path: &std::path::Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("jpg" | "jpeg" | "png" | "bmp" | "gif" | "webp" | "tiff")
    )
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Index { folder, output } => {
            println!("Indexing images in: {}", folder.display());
            println!("Writing index to:   {}", output.display());

            let device = burn::backend::wgpu::WgpuDevice::default();

            println!("Loading pretrained ResNet-18 encoder...");
            let net = model::load_encoder::<MyBackend>(&device);

            // Test on the first image we find.
            for entry in walkdir::WalkDir::new(&folder)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| is_image(e.path()))
                .take(1)
            {
                println!("Loading: {}", entry.path().display());
                let img_tensor = preprocess::load_image::<MyBackend>(entry.path(), &device)?;
                println!("Input  shape: {:?}", img_tensor.shape());

                let embedding = net.forward(img_tensor);
                println!("Output shape: {:?}", embedding.shape());

                // Print first 8 values — with random weights these would be
                // near zero; with pretrained weights they are meaningful floats.
                let vals: Vec<f32> = embedding.clone().into_data().to_vec().unwrap();
                println!("First 8 embedding values: {:.4?}", &vals[..8]);
            }
        }
        Command::Query { image, k, index } => {
            println!("Query image: {}", image.display());
            println!("Index file:  {}", index.display());
            println!("Top-K:       {k}");
            // TODO: Step 3 – embed query and search index
        }
    }

    Ok(())
}
