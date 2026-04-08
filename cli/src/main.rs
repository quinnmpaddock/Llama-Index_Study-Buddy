use clap::{Parser, Subcommand};
use study_buddy::{
    api::ApiClient,
    commands::{CommunityAction, CommunityCommand, IngestCommand, QueryCommand, SearchCommand, SummariesCommand},
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

    /// Search for entities in the knowledge graph
    Search {
        /// The search term to find matching entities
        query: Option<String>,

        /// Get details for a specific entity by name
        #[arg(short, long)]
        entity: Option<String>,

        /// Maximum number of results (1-200)
        #[arg(short, long, default_value = "50")]
        limit: i32,
    },

    /// Explore graph communities
    Community {
        #[command(subcommand)]
        action: CommunityCommands,
    },

    /// Ingest documents into the knowledge graph
    Ingest {
        /// Directory path containing documents to ingest
        directory: String,

        /// Specific files to ingest (optional, comma-separated or multiple args)
        #[arg(short, long)]
        files: Option<String>,

        /// Skip confirmation prompt
        #[arg(short, long)]
        yes: bool,
    },

    /// Start interactive TUI mode
    Tui,

    /// Check API connection status
    Status,

    /// Show current configuration
    Config,

    /// Manage community summaries versions
    Summaries {
        #[command(subcommand)]
        action: SummariesCommands,
    },
}

#[derive(Subcommand, Debug)]
enum SummariesCommands {
    /// List all summary versions
    List,

    /// Show the current (active) summary version
    Current,

    /// Delete old summary versions, keeping N most recent
    Cleanup {
        /// Number of versions to keep (default: 5)
        #[arg(default_value = "5")]
        keep: i32,
    },
}

#[derive(Subcommand, Debug)]
enum CommunityCommands {
    /// List all communities with summary previews
    List,

    /// Show details for a specific community
    Show {
        /// Community ID
        id: i32,

        /// Include entities in the output
        #[arg(short, long)]
        entities: bool,
    },

    /// Export all community summaries
    Export {
        /// Output file path (prints to stdout if not specified)
        #[arg(short, long)]
        output: Option<String>,
    },
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

        Commands::Search { query, entity, limit } => {
            let cmd = SearchCommand::new()
                .with_limit(limit)
                .with_format(args.format);

            let cmd = if let Some(name) = entity {
                cmd.with_entity(name)
            } else if let Some(q) = query {
                cmd.with_query(q)
            } else {
                cmd
            };

            cmd.execute(&settings).await?;
        }

        Commands::Community { action } => {
            let show_entities = matches!(action, CommunityCommands::Show { entities: true, .. });
            let community_action = match action {
                CommunityCommands::List => CommunityAction::List,
                CommunityCommands::Show { id, entities: _ } => CommunityAction::Show { id },
                CommunityCommands::Export { output } => CommunityAction::Export { output },
            };

            let cmd = CommunityCommand::new(community_action)
                .with_format(args.format)
                .with_entities(show_entities);

            cmd.execute(&settings).await?;
        }

        Commands::Ingest { directory, files, yes } => {
            let cmd = IngestCommand::new(directory)
                .with_format(args.format)
                .with_yes(yes);

            let cmd = if let Some(files_str) = files {
                // Parse comma-separated files
                let file_list: Vec<String> = files_str
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                cmd.with_files(file_list)
            } else {
                cmd
            };

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

        Commands::Summaries { action } => {
            let cmd = SummariesCommand::new().with_format(args.format);

            match action {
                SummariesCommands::List => {
                    cmd.list(&settings).await?;
                }
                SummariesCommands::Current => {
                    cmd.current(&settings).await?;
                }
                SummariesCommands::Cleanup { keep } => {
                    cmd.cleanup(&settings, keep).await?;
                }
            }
        }
    }

    Ok(())
}