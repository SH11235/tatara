//! Versioned loader metadata stored alongside raw optimizer checkpoints.
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::dataloader::TeacherPosition;

pub fn artifact_identity(path: &Path) -> io::Result<serde_json::Value> {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(path)?;
    let digest = Sha256::digest(&bytes);
    let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    Ok(serde_json::json!({"bytes": bytes.len(), "sha256": hex}))
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
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map(Some)
                .map_err(io::Error::other),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                eprintln!(
                    "[train] warning: checkpoint has no loader state; restarting teacher input from the beginning"
                );
                Ok(None)
            }
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
        let mut saved_settings = self.settings.clone();
        if self.version == 1
            && settings.get("sampling").is_some()
            && let Some(object) = saved_settings.as_object_mut()
        {
            object.entry("sampling").or_insert(
                serde_json::to_value(crate::sampling::Sampling::default())
                    .map_err(io::Error::other)?,
            );
            object.entry("mix_rates").or_insert(serde_json::json!([]));
            object.entry("mix_seed").or_insert(serde_json::json!(0));
        }
        let mut expected_settings = settings.clone();
        if mixes.is_empty() && expected_settings.get("mix_seed").is_some() {
            expected_settings["mix_seed"] = serde_json::json!(0);
        }
        for value in [&mut saved_settings, &mut expected_settings] {
            if let Some(object) = value.as_object_mut() {
                object.remove("batches_per_superbatch");
                if object
                    .get("bucket_inputs")
                    .is_some_and(serde_json::Value::is_null)
                {
                    object.remove("bucket_inputs");
                }
            }
        }
        if !matches!(self.version, 1 | 2)
            || (self.version == 1 && !mixes.is_empty())
            || self.superbatch != superbatch
            || self.teachers.len() != mixes.len() + 1
            || self.teachers[0].id != "primary"
            || !same_files(&self.teachers[0].files, files)
            || saved_settings != expected_settings
            || self
                .teachers
                .iter()
                .skip(1)
                .zip(mixes)
                .enumerate()
                .any(|(i, (teacher, files))| {
                    teacher.id != format!("mix-{}", i + 1) || !same_files(&teacher.files, files)
                })
            || self.teachers[0].position.mixes.len() != mixes.len()
            || self
                .teachers
                .iter()
                .skip(1)
                .zip(&self.teachers[0].position.mixes)
                .any(|(teacher, cursor)| &teacher.position != cursor)
        {
            return Err(io::Error::other(
                "loader state does not match checkpoint, teacher basenames/sizes, or input settings; deleting .loader.json resumes from the beginning of the teachers without restoring input positions (legacy resume)",
            ));
        }
        Ok(&self.teachers[0].position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        loaded.version = 99;
        assert!(loaded.validate(&files, &settings, 3).is_err());
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
            version: 2,
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
    fn version_one_defaults_to_unmixed_unsampled_input() {
        let state: LoaderState = serde_json::from_value(serde_json::json!({
            "version": 1, "superbatch": 2, "settings": {"seed": 3},
            "teachers": [{"id": "primary", "files": [], "position": {"epoch": 1, "next_record": 4, "window_index": null, "window_offset": 0}}]
        })).unwrap();
        let mut settings = serde_json::json!({"seed": 3, "sampling": crate::sampling::Sampling::default(), "mix_rates": [], "mix_seed": 0});
        assert_eq!(state.validate(&[], &settings, 2).unwrap().next_record, 4);
        settings["sampling"]["keep_rate"] = serde_json::json!(0.5);
        assert!(state.validate(&[], &settings, 2).is_err());
    }

    #[test]
    fn relocated_teachers_and_superbatch_batch_count_preserve_positions() {
        let files = vec![TeacherFileIdentity {
            path: "/old/data.psv".into(),
            bytes: 80,
        }];
        let state = LoaderState {
            version: 2,
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
        assert!(error.to_string().contains("deleting .loader.json"));
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
            version: 2,
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
