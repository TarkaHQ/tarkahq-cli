# Training Request Template

Use this for the first customer intake. Keep it plain text until two or
more real workflows prove the stable fields.

## Customer

```text
org_slug:
company:
contact_name:
contact_email:
timezone:
```

## Workload

```text
workload_type: LoRA/QLoRA fine-tune | full fine-tune | pre-training | other
preferred_tool: Axolotl | Unsloth | nanochat | custom
base_model:
model_size:
sequence_length:
precision:
expected_runtime:
training_window: hours | overnight | multi-day | recurring
success_criterion:
```

## Code

```text
repo_url:
repo_visibility: public | private
commit_or_tag:
config_path:
entrypoint_command:
custom_docker_image:
dependency_files: requirements.txt | pyproject.toml | uv.lock | environment.yml | other
```

## Data

```text
data_source_type: Hugging Face | S3 | R2 | GCS | rsync/scp | local upload | other
data_source:
dataset_size:
sample_size_for_smoke:
text_column_or_format:
license_or_access_constraints:
```

Data must be staged onto the GPU node before training starts. Tarka
should not stream training data over WAN during the run.

## Secrets

```text
required_secrets: HF_TOKEN | WANDB_API_KEY | AWS_* | RCLONE_CONFIG | other
secret_owner: customer | Tarka operator
secret_expiry:
```

Do not paste secrets into this document. Use a separate agreed secret
handoff path.

## Outputs

```text
expected_checkpoint_size:
expected_artifacts:
artifact_handoff: tarball | rsync | object-store push | HF model repo | other
retention_window:
```

## Feasibility Notes

```text
single_gpu_fit:
requires_inference_pause:
multi_day_approval:
custom_image_review:
operator_notes:
```
