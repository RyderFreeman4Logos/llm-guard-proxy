//! SQLite-backed daily request budget counters.

use std::{
    env, fs,
    io::ErrorKind,
    path::{Path, PathBuf},
    sync::{Mutex, MutexGuard},
    time::{SystemTime, UNIX_EPOCH},
};

use rusqlite::{Connection, OpenFlags, params};
use thiserror::Error;

const SECONDS_PER_DAY: u64 = 86_400;
const DAYS_FROM_CIVIL_UNIX_EPOCH: i64 = 719_468;
const RETAIN_DAYS: i64 = 7;

/// Result of one budget check.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BudgetCheck {
    /// Count after increment when allowed, or current stored count when blocked.
    pub current_count: u64,
    /// Configured daily request limit for this profile.
    pub limit: u64,
    /// Whether the request may proceed.
    pub allowed: bool,
}

/// Budget counter storage failures.
#[derive(Debug, Error)]
pub enum BudgetError {
    /// Creating the `SQLite` parent directory failed.
    #[error("failed to create budget directory {path}: {source}")]
    CreateDirectory {
        /// Directory path that could not be created.
        path: PathBuf,
        /// Source I/O error.
        source: std::io::Error,
    },
    /// `SQLite` operation failed.
    #[error("failed to {action}: {source}")]
    Sqlite {
        /// Operation being performed.
        action: &'static str,
        /// Source `SQLite` error.
        source: rusqlite::Error,
    },
    /// Shared `SQLite` connection state was poisoned by a panic.
    #[error("budget store lock is poisoned")]
    LockPoisoned,
    /// Stored counter value is outside the supported range.
    #[error("budget counter value for profile {profile} on {date} is outside the supported range")]
    CounterOutOfRange {
        /// Profile name.
        profile: String,
        /// Budget date.
        date: String,
    },
    /// Counter reached `SQLite`'s signed integer limit.
    #[error("budget counter value for profile {profile} on {date} reached SQLite integer limit")]
    CounterSaturated {
        /// Profile name.
        profile: String,
        /// Budget date.
        date: String,
    },
}

/// SQLite-backed budget counter store.
#[derive(Debug)]
pub struct BudgetStore {
    db: Mutex<Connection>,
}

impl BudgetStore {
    /// Opens a budget counter database and prunes entries older than seven days.
    ///
    /// # Errors
    ///
    /// Returns [`BudgetError`] when the database path cannot be prepared,
    /// `SQLite` cannot open, schema creation fails, or startup pruning fails.
    pub fn open(path: &str) -> Result<Self, BudgetError> {
        let path = resolve_sqlite_path(path);
        prepare_parent_directory(&path)?;
        let db = open_connection(&path)?;
        create_schema(&db)?;
        prune_old_entries(&db, &budget_date_for_time(SystemTime::now(), 0))?;
        Ok(Self { db: Mutex::new(db) })
    }

    /// Atomically checks the profile's daily limit and increments if allowed.
    ///
    /// # Errors
    ///
    /// Returns [`BudgetError`] when the database lock is poisoned, `SQLite`
    /// fails, or a stored count cannot be represented as `u64`.
    pub fn check_and_increment(
        &self,
        profile: &str,
        date: &str,
        limit: u64,
    ) -> Result<BudgetCheck, BudgetError> {
        let mut db = self.lock()?;
        let transaction = db.transaction().map_err(|source| BudgetError::Sqlite {
            action: "start budget counter transaction",
            source,
        })?;
        transaction
            .execute(
                "INSERT INTO budget_counts (profile, date, count) VALUES (?1, ?2, 0)
                 ON CONFLICT(profile, date) DO NOTHING",
                params![profile, date],
            )
            .map_err(|source| BudgetError::Sqlite {
                action: "initialize budget counter row",
                source,
            })?;

        let current = read_count_from_connection(&transaction, profile, date)?;
        if current >= limit {
            transaction.commit().map_err(|source| BudgetError::Sqlite {
                action: "commit budget counter transaction",
                source,
            })?;
            return Ok(BudgetCheck {
                current_count: current,
                limit,
                allowed: false,
            });
        }

        let next = current
            .checked_add(1)
            .ok_or_else(|| BudgetError::CounterSaturated {
                profile: profile.to_owned(),
                date: date.to_owned(),
            })?;
        let next_i64 = i64::try_from(next).map_err(|_error| BudgetError::CounterSaturated {
            profile: profile.to_owned(),
            date: date.to_owned(),
        })?;
        transaction
            .execute(
                "UPDATE budget_counts SET count = ?3 WHERE profile = ?1 AND date = ?2",
                params![profile, date, next_i64],
            )
            .map_err(|source| BudgetError::Sqlite {
                action: "increment budget counter row",
                source,
            })?;
        transaction.commit().map_err(|source| BudgetError::Sqlite {
            action: "commit budget counter transaction",
            source,
        })?;

        Ok(BudgetCheck {
            current_count: next,
            limit,
            allowed: true,
        })
    }

    /// Reads the stored count for one profile/date pair.
    ///
    /// # Errors
    ///
    /// Returns [`BudgetError`] when the database lock is poisoned, `SQLite`
    /// fails, or a stored count cannot be represented as `u64`.
    pub fn get_count(&self, profile: &str, date: &str) -> Result<u64, BudgetError> {
        let db = self.lock()?;
        read_count_from_connection(&db, profile, date)
    }

    fn lock(&self) -> Result<MutexGuard<'_, Connection>, BudgetError> {
        self.db.lock().map_err(|_error| BudgetError::LockPoisoned)
    }
}

fn resolve_sqlite_path(path: &str) -> PathBuf {
    let path = Path::new(path);
    if path == Path::new(":memory:") {
        return path.to_path_buf();
    }
    let Some(suffix) = path.strip_prefix("~").ok() else {
        return path.to_path_buf();
    };
    env::var_os("HOME").map_or_else(
        || path.to_path_buf(),
        |home| PathBuf::from(home).join(suffix),
    )
}

/// Returns the active budget date for the current UTC time and reset hour.
#[must_use]
pub fn current_budget_date(reset_hour_utc: u32) -> String {
    budget_date_for_time(SystemTime::now(), reset_hour_utc)
}

/// Returns the active budget date for a supplied UTC time and reset hour.
#[must_use]
pub fn budget_date_for_time(time: SystemTime, reset_hour_utc: u32) -> String {
    let seconds = time
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs());
    let mut day = i64::try_from(seconds / SECONDS_PER_DAY).unwrap_or(i64::MAX);
    let hour = (seconds % SECONDS_PER_DAY) / 3_600;
    if hour < u64::from(reset_hour_utc) {
        day = day.saturating_sub(1);
    }
    format_utc_day(day)
}

fn prepare_parent_directory(path: &Path) -> Result<(), BudgetError> {
    let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    else {
        return Ok(());
    };
    match fs::create_dir_all(parent) {
        Ok(()) => Ok(()),
        Err(source) if source.kind() == ErrorKind::AlreadyExists => Ok(()),
        Err(source) => Err(BudgetError::CreateDirectory {
            path: parent.to_path_buf(),
            source,
        }),
    }
}

fn open_connection(path: &Path) -> Result<Connection, BudgetError> {
    Connection::open_with_flags(path, OpenFlags::default() | OpenFlags::SQLITE_OPEN_NOFOLLOW)
        .map_err(|source| BudgetError::Sqlite {
            action: "open SQLite budget store",
            source,
        })
}

fn create_schema(db: &Connection) -> Result<(), BudgetError> {
    db.execute_batch(
        "CREATE TABLE IF NOT EXISTS budget_counts (
            profile TEXT NOT NULL,
            date TEXT NOT NULL,
            count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (profile, date)
        );",
    )
    .map_err(|source| BudgetError::Sqlite {
        action: "create budget counter schema",
        source,
    })
}

fn prune_old_entries(db: &Connection, current_date: &str) -> Result<(), BudgetError> {
    let Some(current_day) = parse_utc_day(current_date) else {
        return Ok(());
    };
    let cutoff = format_utc_day(current_day.saturating_sub(RETAIN_DAYS));
    db.execute("DELETE FROM budget_counts WHERE date < ?1", params![cutoff])
        .map_err(|source| BudgetError::Sqlite {
            action: "prune old budget counter rows",
            source,
        })?;
    Ok(())
}

fn read_count_from_connection(
    db: &Connection,
    profile: &str,
    date: &str,
) -> Result<u64, BudgetError> {
    let count = db
        .query_row(
            "SELECT count FROM budget_counts WHERE profile = ?1 AND date = ?2",
            params![profile, date],
            |row| row.get::<_, i64>(0),
        )
        .or_else(|source| match source {
            rusqlite::Error::QueryReturnedNoRows => Ok(0),
            source => Err(source),
        })
        .map_err(|source| BudgetError::Sqlite {
            action: "read budget counter row",
            source,
        })?;
    u64::try_from(count).map_err(|_error| BudgetError::CounterOutOfRange {
        profile: profile.to_owned(),
        date: date.to_owned(),
    })
}

fn parse_utc_day(date: &str) -> Option<i64> {
    let (year, suffix) = date.split_once('-')?;
    let (month, day) = suffix.split_once('-')?;
    days_from_civil(
        year.parse::<i32>().ok()?,
        month.parse::<u32>().ok()?,
        day.parse::<u32>().ok()?,
    )
}

fn format_utc_day(day: i64) -> String {
    let (year, month, day) = civil_from_days(day);
    format!("{year:04}-{month:02}-{day:02}")
}

fn days_from_civil(year: i32, month: u32, day: u32) -> Option<i64> {
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    let year = i64::from(year) - i64::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let yoe = year - era * 400;
    let month = i64::from(month);
    let doy = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + i64::from(day) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some(era * 146_097 + doe - DAYS_FROM_CIVIL_UNIX_EPOCH)
}

fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let days = days.saturating_add(DAYS_FROM_CIVIL_UNIX_EPOCH);
    let era = if days >= 0 { days } else { days - 146_096 } / 146_097;
    let doe = days - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    let year = year + i64::from(month <= 2);
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, UNIX_EPOCH};

    use super::{BudgetStore, budget_date_for_time};

    #[test]
    fn check_and_increment_persists_counts() {
        let store = BudgetStore::open(":memory:").expect("budget store should open");

        let first = store
            .check_and_increment("adult", "2026-07-05", 2)
            .expect("first check should succeed");
        let second = store
            .check_and_increment("adult", "2026-07-05", 2)
            .expect("second check should succeed");
        let third = store
            .check_and_increment("adult", "2026-07-05", 2)
            .expect("third check should succeed");

        assert!(first.allowed);
        assert_eq!(first.current_count, 1);
        assert!(second.allowed);
        assert_eq!(second.current_count, 2);
        assert!(!third.allowed);
        assert_eq!(third.current_count, 2);
        assert_eq!(
            store
                .get_count("adult", "2026-07-05")
                .expect("count should read"),
            2
        );
    }

    #[test]
    fn budget_date_respects_reset_hour() {
        let july_fifth_0030 = UNIX_EPOCH + Duration::from_secs(1_783_209_600 + 1_800);

        assert_eq!(budget_date_for_time(july_fifth_0030, 0), "2026-07-05");
        assert_eq!(budget_date_for_time(july_fifth_0030, 1), "2026-07-04");
    }
}
