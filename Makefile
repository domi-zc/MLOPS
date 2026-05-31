.PHONY: feature-pipeline
feature-pipeline:
	@echo "Starting Daily Feature Pipeline (30 Days)..."
	uv run --extra pipeline python -m ml_pipeline.feature_pipeline.load

.PHONY: backfill
backfill:
	@echo "Starting Historical Backfill..."
	uv run --extra pipeline python -m ml_pipeline.feature_pipeline.load --backfill

.PHONY: sweep-training
sweep-training:
	@echo "Starting Hyperparameter Sweep..."
	uv run --extra pipeline python -m ml_pipeline.training_pipeline.sweep $(ARGS)

.PHONY: training-pipeline
training-pipeline:
	@echo "Starting Final Production Training..."
	uv run --extra pipeline python -m ml_pipeline.training_pipeline.train

.PHONY: select-champion
select-champion:
	@echo "Starting Champion Selection..."
	uv run --extra pipeline python -m ml_pipeline.training_pipeline.select_champion

.PHONY: inference-pipeline
inference-pipeline:
	@echo "Starting Inference Test..."
	uv run python -m ml_pipeline.inference_pipeline.predictor

.PHONY: backfill-predictions
backfill-predictions:
	@echo "Starting Prediction Backfill..."
	uv run python -m ml_pipeline.inference_pipeline.backfill_predictions

.PHONY: monitoring-pipeline
monitoring-pipeline:
	@echo "Starting Monitoring Pipeline..."
	uv run python -m ml_pipeline.monitoring_pipeline.monitor