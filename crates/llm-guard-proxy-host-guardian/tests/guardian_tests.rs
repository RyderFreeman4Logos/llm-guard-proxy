use llm_guard_proxy_host_guardian::{
    Thresholds, parse_mem_available, parse_registration, should_rearm, should_shed,
};

#[test]
fn public_threshold_contract_preserves_gib_and_mib_units() {
    let thresholds = Thresholds::new(2, 96).expect("thresholds");
    assert_eq!(thresholds.threshold_bytes(), 2 * 1024 * 1024 * 1024);
    assert_eq!(thresholds.reserve_bytes(), 96 * 1024 * 1024);
}

#[test]
fn public_shed_and_rearm_contracts_have_distinct_boundaries() {
    let thresholds = Thresholds::new(1, 64).expect("thresholds");
    assert!(should_shed(thresholds.threshold_bytes() - 1, thresholds));
    assert!(should_shed(thresholds.threshold_bytes(), thresholds));
    assert!(!should_shed(thresholds.threshold_bytes() + 1, thresholds));
    assert!(!should_rearm(thresholds.threshold_bytes(), thresholds));
    assert!(should_rearm(
        thresholds.threshold_bytes() + thresholds.reserve_bytes() as u64,
        thresholds
    ));
}

#[test]
fn public_parsers_reject_untrusted_input_without_panicking() {
    assert!(parse_mem_available(b"MemAvailable: invalid kB\n").is_err());
    assert!(parse_registration(b"version=1\n", 1000).is_err());
}
