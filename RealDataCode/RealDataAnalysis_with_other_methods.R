# ----------------------------------------------------------------------
# Project: PECA Real Data Analysis: Cross-Site Validation
# Analysis: Distribution Comparison and XGBoost (NACC -> ADNI)
# Author: Henry Bayly
# Date: December 5, 2025
# ----------------------------------------------------------------------

# --- 1. Setup and Library Loading ---
library(dplyr)
library(ggplot2)
library(patchwork)
library(xgboost)
library(pROC)
# Additional necessary packages for advanced methods
library(FNN)  # For k-Nearest Neighbors (LoRe)
library(MASS) # For generalized inverse (ginv) needed for Mahalanobis stability

# Set the working directory to where your output files are saved (if necessary)
# NOTE: The original script used 'setwd("/projectnb/pecaml/baylyh/")', ensure this path is correct if uncommenting.
setwd("/projectnb/pecaml/baylyh/")

# --- 2. Metric Definition Functions ---

# Computes Conditional Misclassification Rate (CMR)
compute_cmr <- function(Y_true, prob_pred) {
  Y_pred <- ifelse(prob_pred > 0.5, 1, 0)
  high_prob_idx <- which(prob_pred > 0.8 | prob_pred < 0.2)
  if (length(high_prob_idx) == 0) {
    message("Warning: No predictions met the high-confidence threshold (Prob < 0.2 or > 0.8). Returning NA.")
    return(NA_real_)
  }
  mean(Y_pred[high_prob_idx] != Y_true[high_prob_idx])
}

# Computes standard Accuracy (ACC)
compute_acc <- function(Y_true, prob_pred) {
  Y_pred <- ifelse(prob_pred > 0.5, 1, 0)
  mean(Y_pred == Y_true)
}

# Computes weighted Accuracy (ACC) - used for KMM-weighted evaluation on training data
compute_weighted_acc <- function(Y_true, prob_pred, weights) {
  Y_pred <- ifelse(prob_pred > 0.5, 1, 0)
  correct <- Y_pred == Y_true
  sum(correct * weights) / sum(weights)
}

# Computes Area Under the ROC Curve (AUC)
compute_auc <- function(Y_true, prob_pred) {
  if (length(unique(Y_true)) < 2) return(NA_real_)
  as.numeric(pROC::roc(Y_true, prob_pred, quiet = TRUE)$auc)
}

# Computes weighted AUC - used for KMM-weighted evaluation on training data
compute_weighted_auc <- function(Y_true, prob_pred, weights) {
  if (length(unique(Y_true)) < 2) return(NA_real_)
  as.numeric(pROC::roc(Y_true, prob_pred, quiet = TRUE, controls = weights)$auc)
}

# Computes Brier Score (BS)
compute_brier <- function(Y_true, prob_pred) {
  mean((prob_pred - Y_true)^2)
}

# Computes weighted Brier Score (BS) - used for KMM-weighted evaluation on training data
compute_weighted_brier <- function(Y_true, prob_pred, weights) {
  sum((prob_pred - Y_true)^2 * weights) / sum(weights)
}

# Computes the number of predictions made with high confidence (Prob > 0.8 or < 0.2)
compute_high_confidence_count <- function(prob_pred) {
  high_prob_idx <- which(prob_pred > 0.8 | prob_pred < 0.2)
  return(length(high_prob_idx))
}

# --- NEW SMOOTHING FUNCTIONS (RBF Local Smoothing) ---

# Local/self-tuning RBF kernel
rbf_local_kernel_vec <- function(X_errors, X_test, k_local = 10) {
  all_X <- rbind(X_errors, X_test)
  n_errors <- nrow(X_errors)
  n_test <- nrow(X_test)
  
  D <- as.matrix(dist(all_X))
  
  get_local_sigma <- function(d_row) {
    sorted <- sort(d_row[d_row > 0])
    if(length(sorted) < k_local) {
      return(sorted[length(sorted)])
    }
    max(sorted[k_local], 1e-6) # Use max(..., 1e-6) for stability
  }
  local_sigmas <- apply(D, 1, get_local_sigma)
  
  sigma_errors <- local_sigmas[1:n_errors]
  sigma_test 	<- local_sigmas[(n_errors + 1):length(local_sigmas)]
  
  D_te <- D[(n_errors + 1):nrow(D), 1:n_errors]^2
  S <- outer(sigma_test, sigma_errors, "*")
  K <- exp(-D_te / S)
  
  K
}

# smooth_local method
smooth_predictions_local <- function(X_error_space, Y_error_space, model_predict_fn,
                                     X_test, probs_test, k = 50, k_local = 10) {
  if (nrow(X_error_space) == 0) {
    message("Warning: Error space is empty. Returning original probabilities.")
    return(probs_test)
  }
  
  probs_error_space <- model_predict_fn(X_error_space)
  pred_error_space <- ifelse(probs_error_space > 0.5, 1, 0)
  error_idx <- which(pred_error_space != Y_error_space)
  
  if (length(error_idx) == 0) {
    message("Warning: Model made no errors in the training set. Returning original probabilities.")
    return(probs_test)
  }
  
  X_errors <- X_error_space[error_idx, , drop = FALSE]
  
  X_means <- colMeans(X_errors)
  X_sds <- apply(X_errors, 2, sd)
  X_sds[X_sds == 0] <- 1
  X_errors_scaled <- scale(X_errors, center = X_means, scale = X_sds)
  X_test_scaled <- scale(X_test, center = X_means, scale = X_sds)
  
  sims <- rbf_local_kernel_vec(X_errors_scaled, X_test_scaled, k_local = k_local)
  
  n_errors_actual <- nrow(X_errors)
  k_use <- min(k, ceiling(0.1 * n_errors_actual))
  if (k_use < 1) k_use <- 1
  if (k_use > n_errors_actual) k_use <- n_errors_actual
  
  sim_scores <- rowMeans(t(apply(sims, 1, function(v) sort(v, decreasing = TRUE)[1:k_use])))
  
  smoothed <- (1 - sim_scores) * probs_test + sim_scores * 0.5
  
  smoothed <- pmin(pmax(smoothed, 0), 1)
  
  return(smoothed)
}


# --- NEW CORE TRANSFER LEARNING METHODS ---

# ------------------------
# LoRe: local histogram recalibration
# ------------------------
apply_lore_recalibration <- function(X_calib, Y_calib, calib_probs, X_test, test_probs,
                                     k_neighbors = 100, n_bins = 10) {
  k <- min(k_neighbors, nrow(X_calib))
  if (k < 1) return(test_probs)
  nn_idx <- FNN::get.knnx(X_calib, X_test, k = k)$nn.index
  calibrated <- numeric(nrow(X_test))
  bin_breaks <- seq(0, 1, length.out = n_bins + 1); bin_breaks[length(bin_breaks)] <- 1.01
  for (i in seq_len(nrow(X_test))) {
    idx <- nn_idx[i, ]
    local_probs <- calib_probs[idx]; local_y <- Y_calib[idx]
    bins <- cut(local_probs, breaks = bin_breaks, include.lowest = TRUE, right = FALSE)
    bin_acc <- tapply(local_y, bins, mean)
    this_bin <- cut(test_probs[i], breaks = bin_breaks, include.lowest = TRUE, right = FALSE)
    val <- bin_acc[as.character(this_bin)]
    if (is.na(val) || is.nan(val)) {
      val <- mean(local_y, na.rm = TRUE)
      if (is.na(val)) val <- 0.5
    }
    calibrated[i] <- val
  }
  pmin(pmax(calibrated, 0), 1)
}

# ------------------------------------------------------------------
# KMM: Kernel Mean Matching
# ------------------------------------------------------------------
kmm_weights_stable <- function(Xs, Xt, sigma = NULL, B = 1000, lambda = 1e-3) {
  n <- nrow(Xs); m <- nrow(Xt)
  if (is.null(sigma)) {
    d <- as.vector(stats::dist(rbind(Xs, Xt)))
    sigma <- median(d[d > 0], na.rm = TRUE)
    if (is.na(sigma) || sigma <= 0) sigma <- 1
  }
  
  # RBF kernel (vectorized for stability)
  rbf <- function(A, B = NULL) {
    if (is.null(B)) B <- A
    A2 <- rowSums(A^2); B2 <- rowSums(B^2)
    K <- exp(-(outer(A2, B2, "+") - 2 * (A %*% t(B))) / (2 * sigma^2))
    K
  }
  
  Kss <- rbf(Xs)
  Kst <- rbf(Xs, Xt)
  
  # This is the correct kappa calculation you already had
  kappa <- rowMeans(Kst) * n
  
  # --- Closed-form solution (replaces quadprog) ---
  res <- tryCatch({
    
    # 1. Create regularized Kss
    #    (lambda is the regularization parameter)
    K_reg <- Kss + diag(lambda, n) 
    
    # 2. Solve the linear system K_reg * w = kappa
    #    This is much more stable than solve.QP
    w <- solve(K_reg, kappa)
    
    # 3. Enforce constraints *after* solving (pragmatic)
    w[w < 0] <- 0
    w[w > B] <- B
    
    # 4. Normalize
    if (sum(w) == 0 || any(is.nan(w))) {
      stop("degenerate: weights are zero or NaN")
    }
    
    w * (n / sum(w))
    
  }, error = function(e) {
    warning(sprintf("KMM closed-form failed: %s. Returning uniform weights", e$message))
    print("KMM closed-form failed: %s. Returning uniform weights")
    rep(1, n)
  })
  
  res
}

# ------------------------
# Mahalanobis OOD Score (used for shrinkage)
# ------------------------
mahalanobis_score <- function(X_ref, X_query, regularize = 1e-6) {
  mu <- colMeans(X_ref)
  S <- cov(X_ref)
  S <- S + diag(regularize, ncol(S))
  invS <- tryCatch(solve(S), error = function(e) MASS::ginv(S))
  d2 <- mahalanobis(X_query, center = mu, cov = S)
  # scale 0-1 (Min-Max normalization of Mahalanobis distance)
  (d2 - min(d2)) / (max(d2) - min(d2) + 1e-12)
}


# --- 3. Data Loading and Preparation ---

# Load the datasets
nacc_data <- read.csv("RealData/NACC/NACC_Future_Impairment_modeling_Jan7.csv")
adni_data <- read.csv("RealData/ADNI/ADNI_Future_Impairment_modeling_Jan7.csv")

# Ensure all character columns are converted to factors (important for consistency)
nacc_data <- nacc_data %>% mutate(across(where(is.character), as.factor))
adni_data <- adni_data %>% mutate(across(where(is.character), as.factor))

# Identify common predictors and target
target_var <- "impaired"
# Drop the index column if present (keeping your preferred syntax for now)
nacc_data <- nacc_data[,c(2:7)]


# --- 4. Predictor Distribution Comparison (NACC vs. ADNI) ---
cat("\n--- 4. Predictor Distribution Comparison ---\n")

# Combine data for plotting, adding a source column
nacc_data$Source <- "NACC (Train)"
adni_data$Source <- "ADNI (Test)"
combined_data <- bind_rows(nacc_data, adni_data)

# FIX: Robustly create data matrix for XGBoost training/testing
combined_data_temp <- combined_data %>% dplyr::select(-Source)

# FIX for select(-all_of(target_var)) error: use base R subsetting
# This removes the target variable from the predictor set
combined_data_no_target <- combined_data_temp[, !names(combined_data_temp) %in% target_var]
combined_data_no_target$Source <- combined_data$Source # Re-add source for matrix split

# --- 5. XGBoost Modeling and Validation ---
cat("\n--- 5. XGBoost Modeling (NACC Train / ADNI Test) ---\n")

# Target variable conversion (1 = Impaired, 0 = Normal)
Y_train <- ifelse(nacc_data[[target_var]] == "Impaired", 1, 0)
Y_test <- ifelse(adni_data[[target_var]] == "Impaired", 1, 0)

# One-hot encoding matrix generation
# The model.matrix call implicitly handles all factor/character variables
full_data_matrix <- model.matrix(~ . - 1 - Source, data = combined_data_no_target)
X_train_matrix <- full_data_matrix[combined_data_no_target$Source == "NACC (Train)", ]
X_test_matrix <- full_data_matrix[combined_data_no_target$Source == "ADNI (Test)", ]

# Create DMatrix objects
dtrain <- xgb.DMatrix(data = X_train_matrix, label = Y_train)
dtest <- xgb.DMatrix(data = X_test_matrix, label = Y_test)

# Define Model Parameters
params <- list(
  objective = "binary:logistic", eval_metric = "logloss",
  eta = 0.05, max_depth = 4, subsample = 0.7, colsample_bytree = 0.7
)

# Train the ORIGINAL model on NACC data
xgb_model <- xgb.train(
  params = params, data = dtrain, nrounds = 100,
  watchlist = list(train = dtrain), verbose = 0
)

# Predict original probabilities
adni_probs <- predict(xgb_model, dtest)
nacc_probs_calib <- predict(xgb_model, dtrain)


# --- 5.1. Apply Smoothing Method (Local RBF) ---
cat("\n--- 5.1. Applying Local RBF Smoothing Method ---\n")
xgb_predict_wrapper <- function(X_matrix) {
  dmatrix <- xgb.DMatrix(data = X_matrix)
  predict(xgb_model, dmatrix)
}
adni_probs_smoothed <- smooth_predictions_local(
  X_error_space = X_train_matrix, Y_error_space = Y_train,
  model_predict_fn = xgb_predict_wrapper, X_test = X_test_matrix,
  probs_test = adni_probs, k = 25, k_local = 10
)


# --- 5.2. Apply LoRe Recalibration ---
cat("\n--- 5.2. Applying LoRe Recalibration ---\n")
adni_probs_lore <- apply_lore_recalibration(
  X_calib = X_train_matrix, Y_calib = Y_train,
  calib_probs = nacc_probs_calib, X_test = X_test_matrix,
  test_probs = adni_probs, k_neighbors = 100, n_bins = 10
)


# --- 5.3. Calculate KMM Weights and Train a KMM-weighted Model ---
cat("\n--- 5.3. KMM Reweighting and Model Training ---\n")
# KMM weights are calculated on the full feature space
#kmm_weights <- kmm_weights_stable(Xs = X_train_matrix, Xt = X_test_matrix)
#cat(sprintf("KMM Weight range (min/max): %.2f / %.2f\n", min(kmm_weights), max(kmm_weights)))

# Train a new XGBoost model using the KMM weights
#dtrain_kmm <- xgb.DMatrix(data = X_train_matrix, label = Y_train, weight = kmm_weights)
#xgb_model_kmm <- xgb.train(
 # params = params, data = dtrain_kmm, nrounds = 100,
  #watchlist = list(train = dtrain_kmm), verbose = 0
#)
#adni_probs_kmm <- predict(xgb_model_kmm, dtest)


# --- 5.4. Mahalanobis Shrinkage (New Method) ---
cat("\n--- 5.4. Applying Mahalanobis Shrinkage ---\n")
# Calculate Mahalanobis OOD score (scaled 0-1)
adni_mah_ood <- mahalanobis_score(X_ref = X_train_matrix, X_query = X_test_matrix)

# Create Mahalanobis-adjusted predictions (Shrinkage)
# mah_ood = 1 (OOD) -> push to 0.5. mah_ood = 0 (In-Dist) -> use adni_probs.
adni_probs_mah <- (1 - adni_mah_ood) * adni_probs + adni_mah_ood * 0.5
mean_ood_score <- mean(adni_mah_ood)
cat(sprintf("Mean Mahalanobis OOD Score (ADNI vs NACC): %.4f\n", mean_ood_score))

# --- 5.5. Visualization of Probability Smoothing ---
cat("\n--- 5.5. Visualization of Probability Smoothing ---\n")

# Combine Original, Smoothed, LoRe, and Mahalanobis probabilities for plotting
prob_data_mah <- data.frame(
  Probability = c(adni_probs, adni_probs_smoothed, adni_probs_lore, adni_probs_mah),
  True_Class = factor(rep(Y_test, 4), levels = c(0, 1), labels = c("Normal (0)", "Impaired (1)")),
  Type = factor(rep(c("Original", "Smoothed", "LoRe", "Mahalanobis"), each = length(adni_probs)))
)

create_density_plot_4 <- function(data_subset, plot_title) {
  ggplot(data_subset, aes(x = Probability, fill = True_Class)) +
    geom_density(alpha = 0.6) +
    geom_vline(xintercept = 0.5, linetype = "dashed", color = "gray50") +
    geom_vline(xintercept = 0.2, linetype = "dotted", color = "black") +
    geom_vline(xintercept = 0.8, linetype = "dotted", color = "black") +
    labs(title = plot_title, x = "Predicted Probability of Impairment (1)", y = "Density") +
    scale_fill_manual(values = c("Normal (0)" = "#0072B2", "Impaired (1)" = "#D55E00"), name = "True Class") +
    theme_minimal()
}

p_mah <- create_density_plot_4(prob_data_mah %>% filter(Type == "Mahalanobis"), "Mahalanobis Shrinkage")

# Since p_original, p_smoothed, p_lore were only defined for the 3-plot layout,
# we need to re-define them using the 4-method data for consistency in the combined plot:
p_original <- create_density_plot_4(prob_data_mah %>% filter(Type == "Original"), "Vanilla")
p_smoothed <- create_density_plot_4(prob_data_mah %>% filter(Type == "Smoothed"), "PECA")
p_lore <- create_density_plot_4(prob_data_mah %>% filter(Type == "LoRe"), "LoRe Recalibrated")
p_combined_final <- p_original + p_smoothed + p_lore + p_mah +
  plot_layout(guides = "collect", ncol = 2) +
  plot_annotation(tag_levels = 'A', title = 'Recalibrated Predicted Probability Distributions on ADNI Test Set')

print(p_combined_final)


# --- 6. Compute and Report Metrics on ADNI Test Data ---
cat("\n--- 6. Performance Metrics on ADNI Test Set ---\n")

# --- Performance Metrics on ADNI Test Data (Unweighted) ---
metrics <- c("ACC", "Brier Score", "CMR", "AUC", "High Confidence Count")

# Original
acc_orig <- compute_acc(Y_test, adni_probs)
brier_orig <- compute_brier(Y_test, adni_probs)
cmr_orig <- compute_cmr(Y_test, adni_probs)
auc_orig <- compute_auc(Y_test, adni_probs)
conf_orig <- compute_high_confidence_count(adni_probs)

# Smoothed
acc_smooth <- compute_acc(Y_test, adni_probs_smoothed)
brier_smooth <- compute_brier(Y_test, adni_probs_smoothed)
cmr_smooth <- compute_cmr(Y_test, adni_probs_smoothed)
auc_smooth <- compute_auc(Y_test, adni_probs_smoothed)
conf_smooth <- compute_high_confidence_count(adni_probs_smoothed)

# LoRe Recalibrated
acc_lore <- compute_acc(Y_test, adni_probs_lore)
brier_lore <- compute_brier(Y_test, adni_probs_lore)
cmr_lore <- compute_cmr(Y_test, adni_probs_lore)
auc_lore <- compute_auc(Y_test, adni_probs_lore)
conf_lore <- compute_high_confidence_count(adni_probs_lore)

# KMM-Weighted Model
#acc_kmm <- compute_acc(Y_test, adni_probs_kmm)
#brier_kmm <- compute_brier(Y_test, adni_probs_kmm)
#cmr_kmm <- compute_cmr(Y_test, adni_probs_kmm)
#auc_kmm <- compute_auc(Y_test, adni_probs_kmm)
#conf_kmm <- compute_high_confidence_count(adni_probs_kmm)


# --- KMM-Weighted Performance on NACC (Transferability estimate) ---
#acc_kmm_w_train <- compute_weighted_acc(Y_train, nacc_probs_calib, kmm_weights)
#brier_kmm_w_train <- compute_weighted_brier(Y_train, nacc_probs_calib, kmm_weights)
#auc_kmm_w_train <- compute_weighted_auc(Y_train, nacc_probs_calib, kmm_weights)


# Mahalanobis Shrinkage
acc_mah <- compute_acc(Y_test, adni_probs_mah)
brier_mah <- compute_brier(Y_test, adni_probs_mah)
cmr_mah <- compute_cmr(Y_test, adni_probs_mah)
auc_mah <- compute_auc(Y_test, adni_probs_mah)
conf_mah <- compute_high_confidence_count(adni_probs_mah)

# Combine results into a formatted data frame
results_adni <- data.frame(
  Metric = metrics,
  Original = c(acc_orig, brier_orig, cmr_orig, auc_orig, conf_orig),
  Smoothed = c(acc_smooth, brier_smooth, cmr_smooth, auc_smooth, conf_smooth),
  LoRe = c(acc_lore, brier_lore, cmr_lore, auc_lore, conf_lore),
  #KMM_Model = c(acc_kmm, brier_kmm, cmr_kmm, auc_kmm, conf_kmm),
  Mahalanobis = c(acc_mah, brier_mah, cmr_mah, auc_mah, conf_mah)
)

#results_kmm_train <- data.frame(
 # Metric = c("ACC (KMM-Weighted Train)", "BS (KMM-Weighted Train)", "AUC (KMM-Weighted Train)"),
  #Score = c(acc_kmm_w_train, brier_kmm_w_train, auc_kmm_w_train)
#)

print(results_adni)
#cat("\n")
#print(results_kmm_train)