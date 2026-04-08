use anyhow::Context;
use config::{Config, File, FileFormat};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Settings {
    pub api: ApiSettings,
    pub display: DisplaySettings,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ApiSettings {
    #[serde(default = "default_base_url")]
    pub base_url: String,

    #[serde(default = "default_timeout")]
    pub timeout_seconds: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DisplaySettings {
    #[serde(default = "default_format")]
    pub default_format: OutputFormat,

    #[serde(default = "default_color")]
    pub color: bool,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    #[default]
    Table,
    Json,
    Yaml,
}

fn default_base_url() -> String {
    "http://localhost:8000".to_string()
}

fn default_timeout() -> u64 {
    30
}

fn default_format() -> OutputFormat {
    OutputFormat::Table
}

fn default_color() -> bool {
    true
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            api: ApiSettings {
                base_url: default_base_url(),
                timeout_seconds: default_timeout(),
            },
            display: DisplaySettings {
                default_format: default_format(),
                color: default_color(),
            },
        }
    }
}

impl Settings {
    pub fn load() -> anyhow::Result<Self> {
        let config_path = Self::config_path()?;

        if config_path.exists() {
            let config = Config::builder()
                .add_source(File::from(config_path).format(FileFormat::Toml))
                .build()?;

            let settings: Settings = config.try_deserialize().unwrap_or_default();
            Ok(settings)
        } else {
            Ok(Settings::default())
        }
    }

    pub fn config_path() -> anyhow::Result<PathBuf> {
        let proj_dirs = ProjectDirs::from("com", "studybuddy", "study-buddy")
            .context("Could not determine config directory")?;

        Ok(proj_dirs.config_dir().join("config.toml"))
    }

    pub fn save(&self) -> anyhow::Result<()> {
        let config_path = Self::config_path()?;

        // Create parent directories if needed
        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let contents = toml::to_string_pretty(self)?;
        std::fs::write(&config_path, contents)?;

        Ok(())
    }
}