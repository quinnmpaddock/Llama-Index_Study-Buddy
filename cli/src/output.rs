use crate::api::{CommunityId, GraphQueryResponse};
use colored::Colorize;
use tabled::{settings::Style, Table, Tabled};

/// Re-export OutputFormat from config
pub use crate::config::OutputFormat;

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "table" | "t" => Ok(OutputFormat::Table),
            "json" | "j" => Ok(OutputFormat::Json),
            "yaml" | "y" => Ok(OutputFormat::Yaml),
            _ => Err(format!("Invalid output format: {}", s)),
        }
    }
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputFormat::Table => write!(f, "table"),
            OutputFormat::Json => write!(f, "json"),
            OutputFormat::Yaml => write!(f, "yaml"),
        }
    }
}

/// Format a query response for output
pub fn format_query_response(response: &GraphQueryResponse, format: OutputFormat) -> String {
    match format {
        OutputFormat::Table => format_as_table(response),
        OutputFormat::Json => format_as_json(response),
        OutputFormat::Yaml => format_as_yaml(response),
    }
}

fn format_as_table(response: &GraphQueryResponse) -> String {
    let mut output = String::new();

    // Answer section
    output.push_str(&format!("{}\n", "Answer:".green().bold()));
    output.push_str(&format!("{}\n\n", response.answer));

    // Entities found
    if !response.entities_found.is_empty() {
        output.push_str(&format!("{}\n", "Entities Found:".cyan().bold()));
        let rows: Vec<EntityRow> = response
            .entities_found
            .iter()
            .enumerate()
            .map(|(i, e)| EntityRow {
                id: (i + 1) as usize,
                name: e.clone(),
            })
            .collect();

        let table = Table::new(rows).with(Style::rounded()).to_string();
        output.push_str(&format!("{}\n\n", table));
    }

    // Communities consulted
    if !response.communities_consulted.is_empty() {
        output.push_str(&format!("{}\n", "Communities Consulted:".yellow().bold()));
        let ids: Vec<String> = response
            .communities_consulted
            .iter()
            .map(|c: &CommunityId| c.to_string())
            .collect();
        output.push_str(&format!("  [{}]\n", ids.join(", ")));
    }

    output
}

#[derive(Tabled)]
struct EntityRow {
    #[tabled(rename = "#")]
    id: usize,
    #[tabled(rename = "Entity")]
    name: String,
}

fn format_as_json(response: &GraphQueryResponse) -> String {
    serde_json::to_string_pretty(response).unwrap_or_else(|e| format!("{{\"error\": \"{}\"}}", e))
}

fn format_as_yaml(response: &GraphQueryResponse) -> String {
    // Simple YAML formatting without external crate
    let mut yaml = String::new();
    yaml.push_str("answer: |\n");
    for line in response.answer.lines() {
        yaml.push_str(&format!("  {}\n", line));
    }
    yaml.push_str("entities_found:\n");
    for entity in &response.entities_found {
        yaml.push_str(&format!("  - {}\n", entity));
    }
    yaml.push_str("communities_consulted:\n");
    for id in &response.communities_consulted {
        yaml.push_str(&format!("  - {}\n", id));
    }
    yaml
}

/// Print a separator line
pub fn print_separator() {
    println!("{}", "─".repeat(60).dimmed());
}

/// Print a status message
pub fn print_status(status: &str, message: &str) {
    println!("{} {}", status.green().bold(), message);
}

/// Print an error message
pub fn print_error(message: &str) {
    eprintln!("{} {}", "Error:".red().bold(), message);
}