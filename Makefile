PYTHON ?= python3
RUN_NAME ?= make_run
SEARCH_MODE ?= grid
SEARCH_EPOCHS ?= 10
SEARCH_MAX_TRIALS ?= 36
SEARCH_NAME ?= $(RUN_NAME)_search
SEARCH_TOP_K ?= 10

COARSE_LEARNING_RATES ?= 0.03,0.01,0.003
COARSE_HIDDEN_DIMS ?= 128,256,512
COARSE_WEIGHT_DECAYS ?= 0,1e-4
COARSE_ACTIVATIONS ?= relu,tanh

IMAGE_SIZE ?= 32
BATCH_SIZE ?= 128
EPOCHS ?= 30
HIDDEN_DIM1 ?= 256
HIDDEN_DIM2 ?= 128
ACTIVATION ?= relu
LEARNING_RATE ?= 0.01
LR_DECAY ?= 0.95
WEIGHT_DECAY ?= 0.0001

CHECKPOINT = results/checkpoints/$(RUN_NAME)_best.npz
HISTORY = results/curves/$(RUN_NAME)_history.json
TEST_SUMMARY = results/errors/$(RUN_NAME)_test_summary.json

.PHONY: install train test search visualize all

install:
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) code/train.py \
		--image-size $(IMAGE_SIZE) \
		--batch-size $(BATCH_SIZE) \
		--epochs $(EPOCHS) \
		--hidden-dim1 $(HIDDEN_DIM1) \
		--hidden-dim2 $(HIDDEN_DIM2) \
		--activation $(ACTIVATION) \
		--learning-rate $(LEARNING_RATE) \
		--lr-decay $(LR_DECAY) \
		--weight-decay $(WEIGHT_DECAY) \
		--run-name $(RUN_NAME)

test:
	$(PYTHON) code/test.py --checkpoint $(CHECKPOINT) --run-name $(RUN_NAME)_test

search:
	$(PYTHON) code/search.py \
		--mode $(SEARCH_MODE) \
		--epochs $(SEARCH_EPOCHS) \
		--image-size $(IMAGE_SIZE) \
		--batch-size $(BATCH_SIZE) \
		--max-trials $(SEARCH_MAX_TRIALS) \
		--search-name $(SEARCH_NAME) \
		--learning-rates $(COARSE_LEARNING_RATES) \
		--hidden-dims $(COARSE_HIDDEN_DIMS) \
		--weight-decays $(COARSE_WEIGHT_DECAYS) \
		--activations $(COARSE_ACTIVATIONS) \
		--top-k $(SEARCH_TOP_K)

visualize:
	$(PYTHON) code/visualize.py \
		--history $(HISTORY) \
		--checkpoint $(CHECKPOINT) \
		--test-summary $(TEST_SUMMARY) \
		--run-name $(RUN_NAME)_viz

all: install search train test visualize
