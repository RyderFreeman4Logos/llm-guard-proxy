#![forbid(unsafe_code)]

use std::process::ExitCode;

use llm_guard_proxy_core::{Health, LICENSE, SERVICE_NAME};

fn main() -> ExitCode {
    println!("{}", render_health());
    ExitCode::SUCCESS
}

#[must_use]
fn render_health() -> String {
    let health = Health::current();
    let name = SERVICE_NAME;
    let license = LICENSE;
    let readiness = health.readiness().as_str();

    format!("{name} readiness={readiness} license={license}")
}

#[cfg(test)]
mod tests {
    use super::render_health;

    #[test]
    fn renders_health_placeholder() {
        assert_eq!(
            render_health(),
            "llm-guard-proxy readiness=ready license=Apache-2.0"
        );
    }
}
