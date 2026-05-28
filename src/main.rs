use anyhow::Result;
use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod index;
mod model;
mod preprocess;

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

        /// Where to write the index file
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
            println!("Loading pretrained encoder...");
            let device = WgpuDevice::default();
            let net = model::load_encoder::<MyBackend>(&device);

            let mut idx = index::Index::new();
            let mut done = 0usize;
            let mut skipped = 0usize;

            // Walk the folder recursively and collect every image path first,
            // so we can show a "N / total" counter while embedding.
            let paths: Vec<_> = walkdir::WalkDir::new(&folder)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| is_image(e.path()))
                .map(|e| e.path().to_path_buf())
                .collect();

            let total = paths.len();
            println!("Found {total} images in {}.", folder.display());

            for path in &paths {
                // Load and preprocess; skip the file on error rather than aborting.
                let tensor = match preprocess::load_image::<MyBackend>(path, &device) {
                    Ok(t) => t,
                    Err(e) => {
                        eprintln!("  skip {}: {e}", path.display());
                        skipped += 1;
                        continue;
                    }
                };

                // Forward pass through ResNet-18 → [1, 512] tensor
                let embedding = net.forward(tensor);

                // Pull the 512 floats off the GPU and into a plain Vec<f32>
                let values: Vec<f32> = embedding.into_data().to_vec().unwrap();

                idx.add(path.to_string_lossy().into_owned(), values);
                done += 1;

                // Progress every 10 images so the user knows it's working
                if done % 10 == 0 || done == total {
                    println!("  [{done}/{total}] indexed");
                }
            }

            idx.save(&output)?;
            println!("Saved index → {} ({done} images)", output.display());
            if skipped > 0 {
                println!("  ({skipped} files skipped due to errors)");
            }
        }

        Command::Query { image, k, index } => {
            // Load the index first — fail early with a clear message if missing.
            let idx = index::Index::load(&index)?;
            if idx.entries.is_empty() {
                anyhow::bail!("Index is empty — run `lensme index <folder>` first.");
            }
            println!("Loaded index: {} images", idx.entries.len());

            // Embed the query image through the same pipeline used at index time.
            let device = WgpuDevice::default();
            let net = model::load_encoder::<MyBackend>(&device);
            let tensor = preprocess::load_image::<MyBackend>(&image, &device)?;
            let query_vec: Vec<f32> = net.forward(tensor).into_data().to_vec().unwrap();

            // Rank every index entry by cosine similarity to the query.
            let results = idx.search(&query_vec, k);

            println!("\nTop-{k} matches for '{}':\n", image.display());
            for (rank, (path, score)) in results.iter().enumerate() {
                println!("  {}. score={:.4}  {}", rank + 1, score, path);
            }
        }
    }

    Ok(())
}
