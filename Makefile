# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Makefile                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: iberegsz <iberegsz@student.42vienna.com>   +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/18 12:18:14 by iberegsz          #+#    #+#              #
#    Updated: 2025/04/01 00:39:58 by iberegsz         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

DATA_DIR = data
SCRIPTS_DIR = scripts
MODELS_DIR = models
RESULTS_DIR = results
PYTHON = python3

all: describe histogram scatter_plot pair_plot train predict

bonus: describe_bonus train_bonus

describe:
	@echo "Calculating dataset statistics..."
	@$(PYTHON) $(SCRIPTS_DIR)/describe.py
	@echo "Statistics calculation complete. Results printed to terminal."

describe_bonus:
	@echo "Calculating extended dataset statistics (bonus)..."
	@$(PYTHON) $(SCRIPTS_DIR)/describe_bonus.py
	@echo "Extended statistics calculation complete. Results printed to terminal."

histogram:
	@echo "Generating histograms..."
	@$(PYTHON) $(SCRIPTS_DIR)/histogram.py
	@echo "Histograms saved to $(RESULTS_DIR)/"

scatter_plot:
	@echo "Generating scatter plot..."
	@$(PYTHON) $(SCRIPTS_DIR)/scatter_plot.py
	@echo "Scatter plot saved to $(RESULTS_DIR)/"

pair_plot:
	@echo "Generating pair plot..."
	@$(PYTHON) $(SCRIPTS_DIR)/pair_plot.py
	@echo "Pair plot saved to $(RESULTS_DIR)/"

train:
	@echo "Training logistic regression model..."
	@$(PYTHON) $(SCRIPTS_DIR)/logreg_train.py
	@echo "Model training complete. Weights saved to $(MODELS_DIR)/weights.txt"

train_bonus:
	@echo "Training logistic regression model with bonus features..."
	@$(PYTHON) $(SCRIPTS_DIR)/logreg_train_bonus.py
	@echo "Bonus model training complete. Weights saved to $(MODELS_DIR)/weights.txt"

predict:
	@echo "Making predictions..."
	@$(PYTHON) $(SCRIPTS_DIR)/logreg_predict.py
	@echo "Predictions saved to $(RESULTS_DIR)/houses.csv"

clean:
	@echo "Cleaning up generated files..."
	@rm -f $(RESULTS_DIR)/*.png $(RESULTS_DIR)/houses.csv $(MODELS_DIR)/weights.txt $(MODELS_DIR)/weights_bonus.txt $(MODELS_DIR)/scaler.pkl $(DATA_DIR)/.~*
	@echo "Cleanup complete."

help:
	@echo "Available targets:"
	@echo "  all           - Run all steps (describe, histogram, scatter_plot, pair_plot, train, predict)"
	@echo "  bonus         - Run extended stats and bonus training (describe_bonus + train_bonus)"
	@echo "  describe      - Calculate dataset statistics"
	@echo "  describe_bonus- Calculate extended statistics (variance, skewness, etc.)"
	@echo "  histogram     - Generate histograms for score distributions"
	@echo "  scatter_plot  - Generate scatter plot for feature relationships"
	@echo "  pair_plot     - Generate pair plot (scatter plot matrix)"
	@echo "  train         - Train the logistic regression model"
	@echo "  train_bonus   - Train the model with bonus features (SGD/mini-batch)"
	@echo "  predict       - Make predictions using the trained model"
	@echo "  clean         - Remove generated files"
	@echo "  help          - Display this help message"

.PHONY: all describe describe_bonus histogram scatter_plot pair_plot train train_bonus predict clean help
