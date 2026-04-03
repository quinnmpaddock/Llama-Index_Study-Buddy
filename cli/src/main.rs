use clap::{Parser, Subcommand};
use study_buddy::{
    api::ApiClient,
    commands::QueryCommand,
    config::Settings,
    output::{print_error, print_status, OutputFormat},
};

/// Study Buddy - CLI for GraphRAG knowledge graph queries
#[derive(Parser, Debug)]
#[command(name = "study-buddy", author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Commands,

    /// API base URL (overrides config)
    #[arg(short, long, global = true, env = "STUDY_BUDDY_API_URL")]
    url: Option<String>,

    /// Output format: table, json, yaml
    #[arg(short, long, global = true, default_value = "table")]
    format: OutputFormat,

    /// Disable colored output
    #[arg(long, global = true)]
    no_color: bool,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Query the knowledge graph
    Query {
        /// The query string to ask the knowledge graph
        query: String,

        /// Number of similar nodes to retrieve (1-50)
        #[arg(short, long, default_value = "20")]
        top_k: i32,
    },

    /// Start interactive TUI mode
    Tui,

    /// Check API connection status
    Status,

    /// Show current configuration
    Config,
}

#[tokio::main]
async fn main() {
    // Load settings
    let settings = Settings::load().unwrap_or_else(|e| {
        eprintln!("Warning: Could not load config: {}", e);
        Settings::default()
    });

    let args = Args::parse();

    // Handle no-color flag
    if args.no_color {
        colored::control::set_override(false);
    }

    // Execute command
    if let Err(e) = execute_command(args, &settings).await {
        print_error(&e.to_string());
        std::process::exit(1);
    }
}

async fn execute_command(args: Args, settings: &Settings) -> study_buddy::error::Result<()> {
    // Apply global URL override if provided
    let settings = if let Some(ref url) = args.url {
        let mut s = settings.clone();
        s.api.base_url = url.clone();
        s
    } else {
        settings.clone()
    };

    match args.command {
        Commands::Query { query, top_k } => {
            let cmd = QueryCommand::new(&query)
                .with_top_k(top_k)
                .with_format(args.format);

            cmd.execute(&settings).await?;
        }

        Commands::Tui => {
            print_error("TUI mode not yet implemented. Use 'query' command for now.");
            std::process::exit(1);
        }

        Commands::Status => {
            let client = ApiClient::new(&settings)?;

            print_status("Connecting", &format!("{}...", settings.api.base_url));

            match client.health().await {
                Ok(message) => {
                    print_status("Connected", &message);
                    print_status("API URL", &settings.api.base_url);
                    print_status("Timeout", &format!("{}s", settings.api.timeout_seconds));
                }
                Err(e) => {
                    print_error(&format!("Connection failed: {}", e));
                    print_error("Make sure the FastAPI server is running.");
                    print_error(&format!(
                        "Start with: cd src && python -m uvicorn app:app --reload"
                    ));
                }
            }
        }

        Commands::Config => {
            println!("Configuration:");
            println!("  API URL: {}", settings.api.base_url);
            println!("  Timeout: {}s", settings.api.timeout_seconds);
            println!("  Format:  {}", settings.display.default_format);

            match Settings::config_path() {
                Ok(path) => println!("  Config:  {}", path.display()),
                Err(_) => println!("  Config:  (none)"),
            }
        }
    }

    Ok(())
}