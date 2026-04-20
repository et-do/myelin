# Changelog

## [0.2.4](https://github.com/et-do/myelin/compare/v0.2.3...v0.2.4) (2026-04-20)


### Bug Fixes

* export __version__ in package __all__ ([#50](https://github.com/et-do/myelin/issues/50)) ([d650e48](https://github.com/et-do/myelin/commit/d650e4895e14cff4b29df9f874c076d597280a1b))

## [0.2.3](https://github.com/et-do/myelin/compare/v0.2.2...v0.2.3) (2026-04-19)


### Documentation

* document branch naming convention to match commit types ([#45](https://github.com/et-do/myelin/issues/45)) ([44fd048](https://github.com/et-do/myelin/commit/44fd04838af3daf4b5226a908de8c25dd644a81f))

## [0.2.2](https://github.com/et-do/myelin/compare/v0.2.1...v0.2.2) (2026-04-19)


### Bug Fixes

* create benchmarks/perf/ before writing benchmark JSON ([#40](https://github.com/et-do/myelin/issues/40)) ([fe07af7](https://github.com/et-do/myelin/commit/fe07af78bb1106ddb0d8e62d5b8e89b8874f281a))
* publish on tag ([#41](https://github.com/et-do/myelin/issues/41)) ([97f3105](https://github.com/et-do/myelin/commit/97f31055e235feb2bf263891d36c1c5ac859747a))

## [0.2.1](https://github.com/et-do/myelin/compare/v0.2.0...v0.2.1) (2026-04-19)


### Bug Fixes

* remove unevaluated inputs expression from regression job name ([#38](https://github.com/et-do/myelin/issues/38)) ([7595c4a](https://github.com/et-do/myelin/commit/7595c4a62377aac4cb3248c141f187579ae9521c))

## [0.2.0](https://github.com/et-do/myelin/compare/v0.1.2...v0.2.0) (2026-04-19)


### Features

* add explicit relationship injection at store time ([#31](https://github.com/et-do/myelin/issues/31)) ([d0284d3](https://github.com/et-do/myelin/commit/d0284d37e53a1adeda4df1ab677eabb1c842bec4))
* add export-md and import-md CLI commands ([#33](https://github.com/et-do/myelin/issues/33)) ([334e7ec](https://github.com/et-do/myelin/commit/334e7ec8300e8f4c4fa6383dcf76514490fe7ee5))
* add ingest tool for bulk memory loading from files and directories ([#32](https://github.com/et-do/myelin/issues/32)) ([0ba598b](https://github.com/et-do/myelin/commit/0ba598b203ecd33e80e9ef5492a9dccb24f1e9b5))
* background worker for async consolidation and scheduled decay ([#24](https://github.com/et-do/myelin/issues/24)) ([30126d7](https://github.com/et-do/myelin/commit/30126d71d691cf9f1c8256acd41b45fa4e5450f6))
* debug-recall CLI command with full pipeline breakdown ([#25](https://github.com/et-do/myelin/issues/25)) ([d952a19](https://github.com/et-do/myelin/commit/d952a191843c42b109caecbd892b5c581c183f4d))
* multi agent namespace ([#34](https://github.com/et-do/myelin/issues/34)) ([b7a67c9](https://github.com/et-do/myelin/commit/b7a67c9cae44b31eb2296e6445142dd3f72f6149))
* respond to MCP initialize immediately, warm up models in backgr… ([#15](https://github.com/et-do/myelin/issues/15)) ([36a721b](https://github.com/et-do/myelin/commit/36a721b93f2ff53347b001cf3d303a84ad3e6ff3))
* storage cap with LRU eviction and auto-decay background timer ([#26](https://github.com/et-do/myelin/issues/26)) ([a27b46a](https://github.com/et-do/myelin/commit/a27b46ac15698d59b329d63a9cd74d377511fec6))
* test coverage ([#19](https://github.com/et-do/myelin/issues/19)) ([5372199](https://github.com/et-do/myelin/commit/5372199eee04f0b00eeb751b556aae0842298964))


### Bug Fixes

* add LoCoMo data download step to regression CI ([#29](https://github.com/et-do/myelin/issues/29)) ([7e9fb9c](https://github.com/et-do/myelin/commit/7e9fb9c33311337349877c40e2cee242a8411837))
* allow multiple MCP agent processes on the same data directory ([#20](https://github.com/et-do/myelin/issues/20)) ([cccd777](https://github.com/et-do/myelin/commit/cccd7776290f8c0fd5c4b2890f4a34496a13157e))
* create benchmarks/perf/ before writing benchmark JSON ([#37](https://github.com/et-do/myelin/issues/37)) ([9424535](https://github.com/et-do/myelin/commit/942453501c6de7744143cdb8d80f9b7b360048a1))
* harden decay against missing metadata and normalise store respon… ([#30](https://github.com/et-do/myelin/issues/30)) ([b1e6adc](https://github.com/et-do/myelin/commit/b1e6adcc7d9195dbd8338a2ed73472a7fe6478d4))


### Performance Improvements

* latency benchmarks ([#23](https://github.com/et-do/myelin/issues/23)) ([ddd4803](https://github.com/et-do/myelin/commit/ddd48031a0fc878a7c5b66c829763b9bb24c247d))


### Documentation

* update CONTRIBUTING versioning table and releases section ([ddd4803](https://github.com/et-do/myelin/commit/ddd48031a0fc878a7c5b66c829763b9bb24c247d))
* update CONTRIBUTING versioning table and releases section ([3aa1aac](https://github.com/et-do/myelin/commit/3aa1aac2e4f51adabe3e38fd13c9d54738ab1742))

## [0.1.2](https://github.com/et-do/myelin/compare/v0.1.1...v0.1.2) (2026-04-16)


### Features

* add memory upsert via store(overwrite=True) ([1a3712f](https://github.com/et-do/myelin/commit/1a3712f94440ba47aad6221636297efc1095bed0))
* add memory upsert via store(overwrite=True) ([49b146a](https://github.com/et-do/myelin/commit/49b146a4190e9258133d3d8eb2e157bcb6d8660d))
* add PyPI publishing workflow and polish install docs ([d66faa8](https://github.com/et-do/myelin/commit/d66faa8f33cb277c2346da8585664b43bab6b8da))
* add PyPI publishing workflow and polish install docs ([3d200b7](https://github.com/et-do/myelin/commit/3d200b7c4a39cc1e831387088e23d13de31693f3))


### Bug Fixes

* exclusive process lock on data directory to prevent concurrent c… ([2a9a675](https://github.com/et-do/myelin/commit/2a9a675b9e2ba215ce49a1fc8b467d9eb65f35d4))
* exclusive process lock on data directory to prevent concurrent corruption ([6ab8930](https://github.com/et-do/myelin/commit/6ab89302612ba68d56a75b290edeca30893df170))
* improve logging — centralize noise suppression, add operation co… ([dcc0ba9](https://github.com/et-do/myelin/commit/dcc0ba9efdaa3470a49d41dfff789413fa45fc50))
* improve logging — centralize noise suppression, add operation context, promote failures ([dfb62fa](https://github.com/et-do/myelin/commit/dfb62facd0c7c5e20cddcab89c7cf366a3e0f3fc))
* rename PyPI package to myelin-mcp to resolve name conflict ([#14](https://github.com/et-do/myelin/issues/14)) ([b9b0a23](https://github.com/et-do/myelin/commit/b9b0a234c0ebf1e14ca162eea4044341f1351d15))
* update stale DataDirLocked docstring references to DataDirLockedError ([639ef52](https://github.com/et-do/myelin/commit/639ef52d5b95e6eab1ea68c5267c4b9fb564cf3b))


### Documentation

* document store overwrite parameter in MCP tools table ([d89aa82](https://github.com/et-do/myelin/commit/d89aa829d2d2625c265d9a9472d1ecd887788681))
* require squash and merge in contributing workflow ([#13](https://github.com/et-do/myelin/issues/13)) ([ae8fea0](https://github.com/et-do/myelin/commit/ae8fea0a0c8e55b1dbc273f79de4306c844f5e41))

## [0.1.1](https://github.com/et-do/myelin/compare/v0.1.0...v0.1.1) (2026-04-15)


### Features

* add release-please for automated releases ([32cd7cd](https://github.com/et-do/myelin/commit/32cd7cd008a2fa96393bc8a1e53bb2de39c65b78))
* add release-please for automated releases ([c69c503](https://github.com/et-do/myelin/commit/c69c503b9fed4257b8e0cfc4fce9cf0a63f80d82))
* add release-please for automated releases ([456d311](https://github.com/et-do/myelin/commit/456d311081b0680ccbf41b460e88892f503e9d12))
* session evidence aggregation boost for multi-evidence recall ([03a8b19](https://github.com/et-do/myelin/commit/03a8b19f48f69b2a7cd7041d493d6bd22b2ff629))


### Documentation

* add CONTRIBUTING.md and slim down README ([9ecf1b2](https://github.com/et-do/myelin/commit/9ecf1b21994b0c61d92975cab929393df8a684d9))
* clarify recommended venv-based install for Myelin ([2ed1f47](https://github.com/et-do/myelin/commit/2ed1f4713a0eea315527ceade3e9c1852d84689c))
* prettify README header with centered layout and badges ([f52c3c9](https://github.com/et-do/myelin/commit/f52c3c9f1f33361176f974b9e3aa59b3a6862cb9))
* prettify README header with centered layout and badges ([f6ee006](https://github.com/et-do/myelin/commit/f6ee0066b1480d56530894033376a34c4472bae6))
* restructure pipeline walkthrough, add consolidation guidance an… ([344f1e2](https://github.com/et-do/myelin/commit/344f1e26b0a5c458225dd8c78a81b474d2d45ad2))
* restructure pipeline walkthrough, add consolidation guidance and no-admin install option ([05714c4](https://github.com/et-do/myelin/commit/05714c4860c45c6e19aa4a7f2891441f574b76b2))
* rewrite install guide with clearer steps, screenshots, and data inspection ([8ec8ba5](https://github.com/et-do/myelin/commit/8ec8ba597f61238a80d547060f2a3b11fb14bc85))
* update LoCoMo benchmark numbers after session aggregation boost ([df9b6fc](https://github.com/et-do/myelin/commit/df9b6fca73a8f9a834e5b29dd02d18ea32e1b5c4))
