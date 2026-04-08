use crate::api::ApiClient;
use crate::config::Settings;
use crate::error::Result;
use crate::output::{OutputFormat, print_error, print_status};

/// Community command for exploring graph communities
pub struct CommunityCommand {
    action: CommunityAction,
    format: OutputFormat,
    show_entities: bool,
}

#[derive(Debug, Clone)]
pub enum CommunityAction {
    List,
    Show { id: i32 },
    Export { output: Option<String> },
}

impl CommunityCommand {
    pub fn new(action: CommunityAction) -> Self {
        Self {
            action,
            format: OutputFormat::Table,
            show_entities: false,
        }
    }

    pub fn with_format(mut self, format: OutputFormat) -> Self {
        self.format = format;
        self
    }

    pub fn with_entities(mut self, show: bool) -> Self {
        self.show_entities = show;
        self
    }

    pub async fn execute(&self, settings: &Settings) -> Result<()> {
        let client = ApiClient::new(settings)?;

        match &self.action {
            CommunityAction::List => self.list_communities(&client).await?,
            CommunityAction::Show { id } => self.show_community(&client, *id).await?,
            CommunityAction::Export { output } => self.export_communities(&client, output.as_deref()).await?,
        }

        Ok(())
    }

    async fn list_communities(&self, client: &ApiClient) -> Result<()> {
        print_status("Fetching", "community list...");

        let response = client.list_communities().await?;

        if response.communities.is_empty() {
            print_error("No communities found.");
            return Ok(());
        }

        match self.format {
            OutputFormat::Json => {
                println!("{}", serde_json::to_string_pretty(&response).unwrap());
            }
            OutputFormat::Yaml => {
                println!("{}", serde_yaml::to_string(&response).unwrap());
            }
            OutputFormat::Table => {
                use tabled::{Table, Tabled, settings::Style};
                
                #[derive(Tabled)]
                struct CommunityRow {
                    #[tabled(rename = "ID")]
                    id: i32,
                    #[tabled(rename = "ENTITIES")]
                    entity_count: i32,
                    #[tabled(rename = "SUMMARY PREVIEW")]
                    summary_preview: String,
                }

                let rows: Vec<CommunityRow> = response
                    .communities
                    .iter()
                    .map(|c| CommunityRow {
                        id: c.id,
                        entity_count: c.entity_count,
                        summary_preview: c.summary_preview.clone(),
                    })
                    .collect();

                let table = Table::new(rows).with(Style::rounded()).to_string();
                println!("{}", table);
                println!();
                print_status("Results", &format!("{} communities (total: {})", 
                    response.communities.len(), response.total));
            }
        }

        Ok(())
    }

    async fn show_community(&self, client: &ApiClient, id: i32) -> Result<()> {
        print_status("Fetching", &format!("community {}...", id));

        let community = client.get_community(id).await?;

        // Optionally fetch entities
        let entities = if self.show_entities {
            let entities_response = client.get_community_entities(id).await?;
            Some(entities_response.entities)
        } else {
            None
        };

        match self.format {
            OutputFormat::Json => {
                let mut json = serde_json::to_value(&community).unwrap();
                if let Some(ref e) = entities {
                    json["entities"] = serde_json::to_value(e).unwrap();
                }
                println!("{}", serde_json::to_string_pretty(&json).unwrap());
            }
            OutputFormat::Yaml => {
                println!("{}", serde_yaml::to_string(&community).unwrap());
                if let Some(ref e) = entities {
                    println!("\nentities:");
                    for entity in e {
                        println!("  - {}", entity);
                    }
                }
            }
            OutputFormat::Table => {
                use colored::Colorize;
                
                println!();
                println!("{} {}", "Community".bold(), id.to_string().cyan().bold());
                println!();
                println!("{} {}", "Entities:".bold(), community.entity_count);
                println!();
                println!("{}", "Summary:".bold());
                println!("{}", community.summary);
                println!();

                if let Some(ref e) = entities {
                    println!("{}", "Entities in this community:".bold());
                    for entity in e {
                        println!("  {} {}", "•".cyan(), entity);
                    }
                    println!();
                }
            }
        }

        Ok(())
    }

    async fn export_communities(&self, client: &ApiClient, output_path: Option<&str>) -> Result<()> {
        print_status("Exporting", "community summaries...");

        let response = client.list_communities().await?;

        let mut output = String::new();
        output.push_str("# Community Summaries\n\n");

        for info in &response.communities {
            let detail = client.get_community(info.id).await?;
            output.push_str(&format!("\n## Community {}\n\n", info.id));
            output.push_str(&format!("**Entities:** {}\n\n", info.entity_count));
            output.push_str(&format!("{}\n\n---\n\n", detail.summary));
        }

        match output_path {
            Some(path) => {
                std::fs::write(path, &output)?;
                print_status("Exported", &format!("{} communities to {}", response.total, path));
            }
            None => {
                println!("{}", output);
            }
        }

        Ok(())
    }
}