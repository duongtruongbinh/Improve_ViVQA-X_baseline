# Project Backlog

## Refactoring & Cleanup
- [ ] Refactor custom scripts (e.g., `auto_metric_vqa_rad_rights_alloction.py`) into modular pipeline components.
- [ ] Centralize path and config management to avoid hardcoded values.
- [ ] Standardize error handling and logging across all scripts and modules.
- [ ] Remove or archive legacy/experimental scripts from the root directory.
- [ ] Increase code documentation and add type hints where missing.
- [ ] Unify evaluation logic under the main pipeline where possible.

## Testing & Validation
- [ ] Add/expand unit tests for processors, tasks, and models.
- [ ] Validate all YAML configs for completeness and correctness.

## Documentation
- [ ] Expand README with more usage examples and troubleshooting.
- [ ] Add more docstrings and inline comments to core modules.
- [ ] Document all new rules and conventions in `.cursor/rules/`.

## Extensibility
- [ ] Add templates for new model, processor, and task registration.
- [ ] Review and update onboarding guide as the project evolves. 