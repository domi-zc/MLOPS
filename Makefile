.PHONY: feature-pipeline
feature-pipeline:
	@echo "Starting Feature Pipeline..."
	uv run --extra pipeline python -m ml_pipeline.feature_pipeline.load

.PHONY: training-pipeline
training-pipeline:
	@echo "Starting Training Pipeline..."
	uv run --extra pipeline python -m ml_pipeline.training_pipeline.train $(ARGS)

.PHONY: select-champion
select-champion:
	@echo "Starting Champion Selection..."
	uv run --extra pipeline python -m ml_pipeline.training_pipeline.select_champion

.PHONY: inference-pipeline
inference-pipeline:
	@echo "Starting Inference Test..."
	uv run python -m ml_pipeline.inference_pipeline.predictor

.PHONY: monitoring-pipeline
monitoring-pipeline:
	@echo "Starting Monitoring Pipeline..."
	uv run python -m ml_pipeline.monitoring_pipeline.monitor