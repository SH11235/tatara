//! Versioned loader metadata stored alongside raw optimizer checkpoints.
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::dataloader::TeacherPosition;

pub fn artifact_identity(path: &Path) -> io::Result<serde_json::Value> {
    Ok(artifact_bytes_identity(&std::fs::read(path)?))
}

pub(crate) fn artifact_bytes_identity(bytes: &[u8]) -> serde_json::Value {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(bytes);
    let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    serde_json::json!({"bytes": bytes.len(), "sha256": hex})
}

/// How `--resume` treats the teacher input positions saved next to the checkpoint.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ResumeInput {
    /// Continue from the saved positions; a mismatch with this run is an error.
    #[default]
    Restore,
    /// Ignore the saved positions and read the teachers from the beginning.
    Restart,
}

fn same_files(saved: &[TeacherFileIdentity], current: &[TeacherFileIdentity]) -> bool {
    saved.len() == current.len()
        && saved
            .iter()
            .zip(current)
            .all(|(a, b)| a.bytes == b.bytes && a.path.file_name() == b.path.file_name())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TeacherFileIdentity {
    pub path: PathBuf,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TeacherState {
    pub id: String,
    pub files: Vec<TeacherFileIdentity>,
    pub position: TeacherPosition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoaderState {
    pub version: u32,
    pub superbatch: usize,
    pub teachers: Vec<TeacherState>,
    pub settings: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wdl_jitter_state: Option<WdlJitterState>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WdlJitterState {
    pub completed_steps: u64,
    // Store the bits to preserve the f64 value across JSON implementations.
    pub batch_state_bits: u64,
}

impl LoaderState {
    pub fn identities(paths: &[impl AsRef<Path>]) -> io::Result<Vec<TeacherFileIdentity>> {
        paths
            .iter()
            .map(|p| {
                let path = std::fs::canonicalize(p)?;
                Ok(TeacherFileIdentity {
                    bytes: std::fs::metadata(&path)?.len(),
                    path,
                })
            })
            .collect()
    }

    pub fn load(checkpoint: &Path) -> io::Result<Option<Self>> {
        match std::fs::read(checkpoint.with_extension("loader.json")) {
            Ok(bytes) => {
                let state: Self = serde_json::from_slice(&bytes).map_err(io::Error::other)?;
                if state.version != 1 {
                    return Err(io::Error::other(format!(
                        "unsupported loader state version: {}",
                        state.version
                    )));
                }
                Ok(Some(state))
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }

    pub fn save(&self, checkpoint: &Path) -> io::Result<()> {
        let path = checkpoint.with_extension("loader.json");
        let mut temp = tempfile::NamedTempFile::new_in(path.parent().unwrap_or(Path::new(".")))?;
        serde_json::to_writer_pretty(&mut temp, self).map_err(io::Error::other)?;
        temp.as_file().sync_all()?;
        temp.persist(&path).map_err(|e| e.error)?;
        Ok(())
    }

    pub fn validate(
        &self,
        files: &[TeacherFileIdentity],
        settings: &serde_json::Value,
        superbatch: usize,
    ) -> io::Result<&TeacherPosition> {
        self.validate_mixed(files, &[], settings, superbatch)
    }

    pub fn validate_mixed(
        &self,
        files: &[TeacherFileIdentity],
        mixes: &[Vec<TeacherFileIdentity>],
        settings: &serde_json::Value,
        superbatch: usize,
    ) -> io::Result<&TeacherPosition> {
        let mismatches = self.mismatches(files, mixes, settings, superbatch);
        if !mismatches.is_empty() {
            return Err(io::Error::other(format!(
                "loader state does not match this run, so teacher input positions cannot be restored:\n  {}\n\
                 Add --resume-input restart to resume the weights and optimizer state and read the teachers from the beginning.",
                mismatches.join("\n  ")
            )));
        }
        Ok(&self.teachers[0].position)
    }

    /// Human-readable differences between the saved state and this run, empty when the
    /// saved input positions can be restored.
    pub fn mismatches(
        &self,
        files: &[TeacherFileIdentity],
        mixes: &[Vec<TeacherFileIdentity>],
        settings: &serde_json::Value,
        superbatch: usize,
    ) -> Vec<String> {
        let mut out = Vec::new();
        if self.version != 1 {
            out.push(format!(
                "loader state version: saved {}, supported 1",
                self.version
            ));
        }
        if self.superbatch != superbatch {
            out.push(format!(
                "superbatch: saved {}, resuming after {superbatch}",
                self.superbatch
            ));
        }
        let saved_files: Vec<&[TeacherFileIdentity]> =
            self.teachers.iter().map(|t| t.files.as_slice()).collect();
        let current_files: Vec<&[TeacherFileIdentity]> = std::iter::once(files)
            .chain(mixes.iter().map(Vec::as_slice))
            .collect();
        for (i, current) in current_files.iter().enumerate() {
            let label = if i == 0 {
                "--data".to_string()
            } else {
                format!("--mix-data #{i}")
            };
            match saved_files.get(i) {
                None => out.push(format!("{label}: not in the saved state")),
                Some(saved) if !same_files(saved, current) => out.push(format!(
                    "{label}: saved [{}], now [{}]",
                    describe_files(saved),
                    describe_files(current)
                )),
                Some(_) => {}
            }
        }
        if saved_files.len() > current_files.len() {
            out.push(format!(
                "teacher count: saved {} (--data + --mix-data), now {}",
                saved_files.len(),
                current_files.len()
            ));
        }
        let ids_ok = self.teachers.first().is_some_and(|t| t.id == "primary")
            && self
                .teachers
                .iter()
                .skip(1)
                .enumerate()
                .all(|(i, t)| t.id == format!("mix-{}", i + 1));
        let cursors_ok = self
            .teachers
            .first()
            .is_some_and(|t| t.position.mixes.len() + 1 == self.teachers.len())
            && self
                .teachers
                .iter()
                .skip(1)
                .zip(&self.teachers[0].position.mixes)
                .all(|(teacher, cursor)| &teacher.position == cursor);
        if !ids_ok || !cursors_ok {
            out.push("loader state is internally inconsistent".to_string());
        }
        let mut saved_settings = self.settings.clone();
        let mut expected_settings = settings.clone();
        for value in [&mut saved_settings, &mut expected_settings] {
            if let Some(object) = value.as_object_mut() {
                object.remove("batches_per_superbatch");
                object
                    .entry("tree_sample_eval_only")
                    .or_insert(serde_json::Value::Bool(false));
            }
        }
        match (saved_settings.as_object(), expected_settings.as_object()) {
            (Some(saved), Some(now)) => {
                let mut keys: Vec<&String> = saved.keys().chain(now.keys()).collect();
                keys.sort();
                keys.dedup();
                for key in keys {
                    let (a, b) = (saved.get(key), now.get(key));
                    if a != b {
                        out.push(format!(
                            "{key}: saved {}, now {}",
                            a.map_or("(absent)".to_string(), |v| v.to_string()),
                            b.map_or("(absent)".to_string(), |v| v.to_string())
                        ));
                    }
                }
            }
            _ if saved_settings != expected_settings => {
                out.push(format!(
                    "input settings: saved {saved_settings}, now {expected_settings}"
                ));
            }
            _ => {}
        }
        out
    }
}

fn describe_files(files: &[TeacherFileIdentity]) -> String {
    files
        .iter()
        .map(|f| {
            let name = f.path.file_name().map_or_else(
                || f.path.display().to_string(),
                |n| n.to_string_lossy().into_owned(),
            );
            format!("{name} ({} bytes)", f.bytes)
        })
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_sample_label_setting_must_match_and_defaults_to_disabled() {
        let files = vec![TeacherFileIdentity {
            path: "teacher.psv".into(),
            bytes: 40,
        }];
        for saved in [
            serde_json::json!({}),
            serde_json::json!({"tree_sample_eval_only": false}),
            serde_json::json!({"tree_sample_eval_only": true}),
        ] {
            let state = LoaderState {
                version: 1,
                wdl_jitter_state: None,
                superbatch: 1,
                teachers: vec![TeacherState {
                    id: "primary".into(),
                    files: files.clone(),
                    position: TeacherPosition::default(),
                }],
                settings: saved.clone(),
            };
            for enabled in [false, true] {
                let current = serde_json::json!({"tree_sample_eval_only": enabled});
                let result = state.validate(&files, &current, 1);
                assert_eq!(
                    result.is_ok(),
                    saved["tree_sample_eval_only"].as_bool().unwrap_or(false) == enabled
                );
                if let Err(error) = result {
                    assert!(error.to_string().contains("tree_sample_eval_only"));
                }
            }
        }
    }

    #[test]
    fn mismatch_lists_each_changed_setting_and_teacher() {
        let saved_files = vec![TeacherFileIdentity {
            path: "a.psv".into(),
            bytes: 400,
        }];
        let state = LoaderState {
            version: 1,
            wdl_jitter_state: None,
            superbatch: 4,
            teachers: vec![TeacherState {
                id: "primary".into(),
                files: saved_files,
                position: TeacherPosition::default(),
            }],
            settings: serde_json::json!({"batch_size": 16384, "seed": 0, "batches_per_superbatch": 10}),
        };
        let now_files = vec![TeacherFileIdentity {
            path: "b.psv".into(),
            bytes: 800,
        }];
        let now = serde_json::json!({"batch_size": 32768, "seed": 0, "batches_per_superbatch": 20});
        let diffs = state.mismatches(&now_files, &[], &now, 4);
        assert_eq!(
            diffs,
            vec![
                "--data: saved [a.psv (400 bytes)], now [b.psv (800 bytes)]".to_string(),
                "batch_size: saved 16384, now 32768".to_string(),
            ]
        );
        let message = state.validate(&now_files, &now, 4).unwrap_err().to_string();
        assert!(message.contains("batch_size: saved 16384, now 32768"));
        assert!(message.contains("--resume-input restart"));
    }

    #[test]
    fn chunk_replay_metadata_round_trips() {
        let files = vec![TeacherFileIdentity {
            path: "teacher.psv".into(),
            bytes: 4000,
        }];
        let settings = serde_json::json!({"batch_size": 7});
        let position = TeacherPosition {
            next_record: 42,
            bucket_skip: 3,
            ..Default::default()
        };
        let state = LoaderState {
            version: 1,
            wdl_jitter_state: None,
            superbatch: 2,
            teachers: vec![TeacherState {
                id: "primary".into(),
                files: files.clone(),
                position: position.clone(),
            }],
            settings: settings.clone(),
        };
        let json = serde_json::to_value(state).unwrap();
        let decoded: LoaderState = serde_json::from_value(json).unwrap();
        assert_eq!(decoded.validate(&files, &settings, 2).unwrap(), &position);
    }

    #[test]
    fn sidecar_round_trip_checks_teacher_identity_and_version() {
        let dir = tempfile::tempdir().unwrap();
        let teacher = dir.path().join("data.psv");
        std::fs::write(&teacher, [0; 40]).unwrap();
        let checkpoint = dir.path().join("net-3.ckpt");
        let files = LoaderState::identities(&[&teacher]).unwrap();
        let settings = serde_json::json!({"seed": 3});
        let position = TeacherPosition {
            epoch: 5,
            next_record: 12,
            window_index: Some(2),
            window_offset: 7,
            ..Default::default()
        };
        let state = LoaderState {
            version: 1,
            wdl_jitter_state: None,
            superbatch: 3,
            teachers: vec![TeacherState {
                id: "primary".into(),
                files: files.clone(),
                position: position.clone(),
            }],
            settings: settings.clone(),
        };
        state.save(&checkpoint).unwrap();
        let mut loaded = LoaderState::load(&checkpoint).unwrap().unwrap();
        assert_eq!(loaded.validate(&files, &settings, 3).unwrap(), &position);
        assert!(loaded.validate(&files, &settings, 2).is_err());
        assert!(
            loaded
                .validate(&files, &serde_json::json!({"seed":4}), 3)
                .is_err()
        );
        let other = dir.path().join("other.psv");
        std::fs::write(&other, [0; 40]).unwrap();
        assert!(
            loaded
                .validate(&LoaderState::identities(&[&other]).unwrap(), &settings, 3)
                .is_err()
        );
        std::fs::write(&teacher, [0; 80]).unwrap();
        assert!(
            loaded
                .validate(&LoaderState::identities(&[&teacher]).unwrap(), &settings, 3)
                .is_err()
        );
        for version in [0, 2, 3, 99] {
            loaded.version = version;
            assert!(loaded.validate(&files, &settings, 3).is_err());
            loaded.save(&checkpoint).unwrap();
            assert!(LoaderState::load(&checkpoint).is_err());
        }
    }

    #[test]
    fn mixed_sidecar_checks_each_teacher_and_selection_counter() {
        let dir = tempfile::tempdir().unwrap();
        let files = ["main.psv", "mix.psv", "other.psv"].map(|name| {
            let path = dir.path().join(name);
            std::fs::write(&path, [0; 80]).unwrap();
            LoaderState::identities(&[path]).unwrap()
        });
        let mix_position = TeacherPosition {
            epoch: 3,
            next_record: 1,
            ..Default::default()
        };
        let position = TeacherPosition {
            ordinal: 123,
            mixes: vec![mix_position.clone()],
            ..Default::default()
        };
        let settings = serde_json::json!({"mix_rates": [0.2], "mix_seed": 9});
        let state = LoaderState {
            version: 1,
            wdl_jitter_state: None,
            superbatch: 7,
            settings: settings.clone(),
            teachers: vec![
                TeacherState {
                    id: "primary".into(),
                    files: files[0].clone(),
                    position: position.clone(),
                },
                TeacherState {
                    id: "mix-1".into(),
                    files: files[1].clone(),
                    position: mix_position,
                },
            ],
        };
        let checkpoint = dir.path().join("net.ckpt");
        state.save(&checkpoint).unwrap();
        let loaded = LoaderState::load(&checkpoint).unwrap().unwrap();
        assert_eq!(
            loaded
                .validate_mixed(&files[0], &files[1..2], &settings, 7)
                .unwrap(),
            &position
        );
        assert!(
            loaded
                .validate_mixed(&files[0], &files[2..3], &settings, 7)
                .is_err()
        );
        let mut resized = files[1..2].to_vec();
        resized[0][0].bytes += 40;
        assert!(
            loaded
                .validate_mixed(&files[0], &resized, &settings, 7)
                .is_err()
        );
        assert!(
            loaded
                .validate_mixed(
                    &files[0],
                    &files[1..2],
                    &serde_json::json!({"mix_rates": [0.3], "mix_seed": 9}),
                    7
                )
                .is_err()
        );
        assert!(loaded.validate(&files[0], &settings, 7).is_err());
    }

    #[test]
    fn relocated_teachers_and_superbatch_batch_count_preserve_positions() {
        let files = vec![TeacherFileIdentity {
            path: "/old/data.psv".into(),
            bytes: 80,
        }];
        let state = LoaderState {
            version: 1,
            wdl_jitter_state: None,
            superbatch: 1,
            teachers: vec![TeacherState {
                id: "primary".into(),
                files: files.clone(),
                position: TeacherPosition::default(),
            }],
            settings: serde_json::json!({"batches_per_superbatch": 5}),
        };
        let relocated = vec![TeacherFileIdentity {
            path: "/new/data.psv".into(),
            bytes: 80,
        }];
        state
            .validate(
                &relocated,
                &serde_json::json!({"batches_per_superbatch": 9}),
                1,
            )
            .unwrap();
        let resized = vec![TeacherFileIdentity {
            path: "/new/data.psv".into(),
            bytes: 120,
        }];
        let error = state
            .validate(&resized, &serde_json::json!({}), 1)
            .unwrap_err();
        assert!(error.to_string().contains("--resume-input restart"));
    }

    #[test]
    fn bucket_inputs_check_mode_count_and_coefficient_size_and_contents() {
        let dir = tempfile::tempdir().unwrap();
        let coeff = dir.path().join("progress.bin");
        std::fs::write(&coeff, b"abcd").unwrap();
        let settings = serde_json::json!({"bucket_inputs": {
            "mode": "progresskpabs", "count": 8,
            "progress_coeff": artifact_identity(&coeff).unwrap(),
        }});
        let state = LoaderState {
            version: 1,
            wdl_jitter_state: None,
            superbatch: 1,
            teachers: vec![TeacherState {
                id: "primary".into(),
                files: vec![],
                position: TeacherPosition::default(),
            }],
            settings: settings.clone(),
        };
        state.validate(&[], &settings, 1).unwrap();
        for (key, value) in [
            ("mode", serde_json::json!("progresskpabsq16")),
            ("count", serde_json::json!(9)),
        ] {
            let mut changed = settings.clone();
            changed["bucket_inputs"][key] = value;
            assert!(state.validate(&[], &changed, 1).is_err());
        }
        for bytes in [b"abce".as_slice(), b"abcde".as_slice()] {
            std::fs::write(&coeff, bytes).unwrap();
            let mut changed = settings.clone();
            changed["bucket_inputs"]["progress_coeff"] = artifact_identity(&coeff).unwrap();
            assert!(state.validate(&[], &changed, 1).is_err());
        }
    }

    #[test]
    fn old_checkpoint_without_sidecar_still_loads() {
        let dir = tempfile::tempdir().unwrap();
        let checkpoint = dir.path().join("old.ckpt");
        std::fs::write(&checkpoint, b"old checkpoint").unwrap();
        assert!(LoaderState::load(&checkpoint).unwrap().is_none());
        std::fs::write(checkpoint.with_extension("loader.json"), b"broken").unwrap();
        assert!(LoaderState::load(&checkpoint).is_err());
    }
}
