setwd("/projectnb/pecaml/baylyh/")

#-----------------------------------------------------------------------
# Sequential Simulation: compare Kernel-smoothing, KMM, Mahalanobis, LoRe
# Computes: CMR, Accuracy, AUC, Brier for each method
#-----------------------------------------------------------------------

# Required packages
required_pkgs <- c("MASS","data.table","ggplot2","e1071","nnet","xgboost",
                   "kernlab","quadprog","FNN","glmnet","pROC","dplyr")
to_install <- required_pkgs[!(required_pkgs %in% installed.packages()[,1])]
if(length(to_install)) install.packages(to_install, repos = "https://cloud.r-project.org")
lapply(required_pkgs, require, character.only = TRUE)

# ------------------------
# Utility / metric fns
# ------------------------
compute_cmr <- function(Y_true, prob_pred) {
  Y_pred <- ifelse(prob_pred > 0.5, 1, 0)
  high_prob_idx <- which(prob_pred > 0.8 | prob_pred < 0.2)
  if (length(high_prob_idx) == 0) return(NA_real_)
  mean(Y_pred[high_prob_idx] != Y_true[high_prob_idx])
}
compute_acc <- function(Y_true, prob_pred) {
  Y_pred <- ifelse(prob_pred > 0.5, 1, 0)
  mean(Y_pred == Y_true)
}
compute_auc <- function(Y_true, prob_pred) {
  if (length(unique(Y_true)) < 2) return(NA_real_)
  as.numeric(pROC::roc(Y_true, prob_pred, quiet = TRUE)$auc)
}
compute_brier <- function(Y_true, prob_pred) {
  mean((prob_pred - Y_true)^2)
}
compute_ccr <- function(Y_true, prob_pred) {
  Y_pred <- ifelse(prob_pred > 0.5, 1, 0)
  high_prob_idx <- which(prob_pred > 0.8 | prob_pred < 0.2)
  if (length(high_prob_idx) == 0) return(NA_real_)
  mean(Y_pred[high_prob_idx] == Y_true[high_prob_idx])
}
# Local Calibration Error (LCE)
compute_lce <- function(X_calib, Y_calib, calib_probs, X_test, k_neighbors = 100) {
  # Find k nearest neighbors in X_calib for each point in X_test
  nn_indices <- get.knnx(X_calib, X_test, k = k_neighbors)$nn.index
  
  lce_values <- sapply(1:nrow(X_test), function(i) {
    local_indices <- nn_indices[i, ]
    mean_pred_local <- mean(calib_probs[local_indices], na.rm = TRUE)
    mean_outcome_local <- mean(Y_calib[local_indices], na.rm = TRUE)
    abs(mean_pred_local - mean_outcome_local)
  })
  
  mean(lce_values, na.rm = TRUE)
}

## <<< ADDED: High Confidence Metrics >>>
# Counts the number of high-confidence predictions
compute_high_conf_n <- function(prob_pred) {
  high_prob_idx <- which(prob_pred > 0.8 | prob_pred < 0.2)
  length(high_prob_idx)
}

# Computes the percentage of high-confidence predictions
compute_high_conf_pct <- function(prob_pred) {
  if (length(prob_pred) == 0) return(NA_real_)
  high_prob_idx <- which(prob_pred > 0.8 | prob_pred < 0.2)
  length(high_prob_idx) / length(prob_pred)
}
## <<< END ADDED >>>


# ------------------------
# Data generation & drift (updated)
# ------------------------

generate_X_mvnorm <- function(n, p, corr_type = "identity", rho = 0.2, block_size = NULL) {
  if (corr_type == "identity") {
    Sigma <- diag(p)
  } else if (corr_type == "toeplitz") {
    Sigma <- toeplitz(rho^(0:(p-1)))
  } else if (corr_type == "block") {
    if (is.null(block_size)) stop("block_size required for block correlation")
    Sigma <- matrix(0, p, p)
    nb <- ceiling(p / block_size)
    for (b in 1:nb) {
      i1 <- (b-1)*block_size + 1
      i2 <- min(p, b*block_size)
      Sigma[i1:i2, i1:i2] <- rho
    }
    diag(Sigma) <- 1
  } else stop("Unknown correlation structure.")
  MASS::mvrnorm(n = n, mu = rep(0, p), Sigma = Sigma)
}

generate_labels <- function(X, beta = NULL, intercept = 0, jitter_sd = 1) {
  n <- nrow(X); p <- ncol(X)
  
  if (is.null(beta)) {
    k <- min(10, p)
    non_zero_betas <- runif(k, min = 0.5, max = 2.0)
    non_zero_betas <- non_zero_betas * sample(c(-1, 1), k, replace = TRUE)
    
    # 3. Combine with the zero (irrelevant) coefficients
    zero_betas <- rep(0, p - k)
    beta <- c(non_zero_betas, zero_betas)
    beta <- sample(beta)
  }
  
  lin <- as.vector(X %*% beta + intercept + rnorm(n, 0, jitter_sd))
  probs <- 1 / (1 + exp(-lin))
  y <- rbinom(n, 1, probs)
  
  list(y = y, probs = probs, beta = beta)
}

# ---- Controlled Covariate Drift ----
apply_covariate_shift <- function(X, strength = 0.5, drift_fraction = 0.2) {
  p <- ncol(X)
  drifted_idx <- sample(seq_len(p), size = ceiling(drift_fraction * p), replace = FALSE)
  shift <- rep(0, p)
  shift[drifted_idx] <- rnorm(length(drifted_idx), mean = strength, sd = strength/2)
  sweep(X, 2, shift, "+")
}

# ---- Controlled Concept Drift ----
apply_concept_drift <- function(beta, strength = 0.5, drift_fraction = 0.2) {
  p <- length(beta)
  drifted_idx <- sample(seq_len(p), size = ceiling(drift_fraction * p), replace = FALSE)
  beta_new <- beta
  beta_new[drifted_idx] <- beta_new[drifted_idx] * (1 + strength * rnorm(length(drifted_idx), -1, 1))
  beta_new
}

# ---- Hybrid Drift ----
apply_hybrid <- function(X, beta, cov_strength = 0.5, concept_strength = 0.5,
                         cov_fraction = 0.2, concept_fraction = 0.2) {
  X_shifted <- apply_covariate_shift(X, cov_strength, drift_fraction = cov_fraction)
  beta_shifted <- apply_concept_drift(beta, concept_strength, drift_fraction = concept_fraction)
  list(X = X_shifted, beta = beta_shifted)
}


# ------------------------
# Models (fit/predict wrappers)
# ------------------------
fit_model <- function(model_type, X, y, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  mt <- tolower(model_type)
  
  if (mt %in% c("logistic","glm")) {
    df <- data.frame(y = y, X)
    mod <- glm(y ~ ., data = df, family = binomial())
    list(model = mod, predict = function(Xnew) predict(mod, newdata = data.frame(Xnew), type = "response"))
    
  } else if (mt == "svm") {
    mod <- e1071::svm(x = X, y = as.factor(y), probability = TRUE)
    list(model = mod, predict = function(Xnew) attr(predict(mod, Xnew, probability = TRUE), "probabilities")[,2])
    
  } else if (mt == "nn") {
    mod <- nnet::nnet(x = X, y = class.ind(as.factor(y)), size = ifelse(ncol(X) < 10, 5, 10),
                      maxit = 200, trace = FALSE)
    list(model = mod, predict = function(Xnew) {
      pr <- predict(mod, Xnew, type = "raw")
      if (is.matrix(pr)) pr[,1] else pr
    })
    
  } else if (mt == "xgboost") {
    dtrain <- xgboost::xgb.DMatrix(data = X, label = y)
    mod <- xgboost::xgboost(data = dtrain, nrounds = 50, objective = "binary:logistic", verbose = 0)
    list(model = mod, predict = function(Xnew) as.numeric(predict(mod, xgboost::xgb.DMatrix(Xnew))))
    
  } else if (mt %in% c("rf", "randomforest")) {
    if (!requireNamespace("randomForest", quietly = TRUE)) install.packages("randomForest")
    mod <- randomForest::randomForest(x = X, y = as.factor(y))
    list(model = mod, predict = function(Xnew) as.numeric(predict(mod, Xnew, type = "prob")[,2]))
    
  } else stop("Unknown model type")
}


# ------------------------
# Kernel smoothing (local/self-tuning)
# ------------------------

# Local/self-tuning RBF kernel
rbf_local_kernel_vec <- function(X_errors, X_test, k_local = 10) {
  # Combine points
  all_X <- rbind(X_errors, X_test)
  n_errors <- nrow(X_errors)
  n_test <- nrow(X_test)
  
  # Compute pairwise Euclidean distances
  D <- as.matrix(dist(all_X))
  
  # Compute local sigma per point based on k-th nearest neighbor
  get_local_sigma <- function(d_row) {
    sorted <- sort(d_row[d_row > 0])
    if(length(sorted) < k_local) return(sorted[length(sorted)]) # fallback
    sorted[k_local]
  }
  local_sigmas <- apply(D, 1, get_local_sigma)
  
  sigma_errors <- local_sigmas[1:n_errors]
  sigma_test   <- local_sigmas[(n_errors + 1):length(local_sigmas)]
  
  # Compute kernel matrix between test and error points
  D_te <- D[(n_errors + 1):nrow(D), 1:n_errors]^2
  S <- outer(sigma_test, sigma_errors, "*") # geometric mean of local sigmas
  K <- exp(-D_te / S)
  
  K
}

# smooth_local method
smooth_predictions_local <- function(X_error_space, Y_error_space, model_predict_fn,
                                     X_test, probs_test, k = 50, k_local = 10) {
  if (nrow(X_error_space) == 0) return(probs_test)
  
  # compute model probabilities on error space
  probs_error_space <- model_predict_fn(X_error_space)
  pred_error_space <- ifelse(probs_error_space > 0.5, 1, 0)
  error_idx <- which(pred_error_space != Y_error_space)
  if (length(error_idx) == 0) return(probs_test)
  
  X_errors <- X_error_space[error_idx, , drop = FALSE]
  
  # Optional: feature scaling for stability
  X_means <- colMeans(X_errors)
  X_sds <- apply(X_errors, 2, sd)
  X_sds[X_sds == 0] <- 1
  X_errors <- scale(X_errors, center = X_means, scale = X_sds)
  X_test <- scale(X_test, center = X_means, scale = X_sds)
  
  # Compute local RBF similarity matrix
  sims <- rbf_local_kernel_vec(X_errors, X_test, k_local = k_local)
  
  # Determine top-k neighbors for smoothing
  k_use <- min(k, ceiling(0.1 * nrow(X_errors)))
  if (k_use == 0) k_use <- 1
  if (k_use > nrow(X_errors)) k_use <- nrow(X_errors)
  
  sim_scores <- rowMeans(t(apply(sims, 1, function(v) sort(v, decreasing = TRUE)[1:k_use])))
  
  # Blend original prediction with 0.5 baseline
  smoothed <- (1 - sim_scores) * probs_test + sim_scores * 0.5
  smoothed
}

# ------------------------
# Kernel smoothing
# ------------------------

rbf_kernel_vec <- function(X_mat, x_vec, sigma) {
  diffs <- sweep(X_mat, 2, x_vec)  # no transpose needed
  sq_dists <- rowSums(diffs^2)
  exp(-sq_dists / (2 * sigma^2))
}


smooth_predictions_method <- function(X_error_space, Y_error_space, model_predict_fn,
                                      X_test, probs_test, k = 50, sigma = 10) {
  
  # Find error points in error space (where model predicted incorrectly)
  if (nrow(X_error_space) == 0) return(probs_test)
  
  # compute model probabilities on error space
  probs_error_space <- model_predict_fn(X_error_space)
  pred_error_space <- ifelse(probs_error_space > 0.5, 1, 0)
  error_idx <- which(pred_error_space != Y_error_space)
  
  if (length(error_idx) == 0) return(probs_test)
  
  X_errors <- X_error_space[error_idx, , drop = FALSE]
  n_test <- nrow(X_test)
  
  if (is.null(sigma)) {
    d_all <- as.vector(dist(rbind(X_errors, X_test)))
    sigma <- median(d_all[d_all > 0], na.rm = TRUE)
    if (is.na(sigma) || sigma <= 0) sigma <- 1
  }
  
  # compute similarity scores
  sims <- matrix(NA_real_, nrow = n_test, ncol = nrow(X_errors))
  for (i in 1:n_test) {
    sims[i, ] <- rbf_kernel_vec(X_errors, X_test[i, ], sigma)
  }
  
  k_use <- min(k, ceiling(0.1 * nrow(X_errors)))  
  
  # Check for edge case where k_use might be 0
  if (k_use == 0) k_use <- 1  
  
  # Check if nrow(X_errors) is too small
  if (k_use > nrow(X_errors)) k_use <- nrow(X_errors)
  
  sim_scores <- rowMeans(t(apply(sims, 1, function(v) sort(v, decreasing = TRUE)[1:k_use])))
  
  smoothed <- (1 - sim_scores) * probs_test + sim_scores * 0.5
  
  smoothed
}

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
    if (all(is.na(bin_acc))) bin_acc[] <- mean(local_y, na.rm = TRUE)
    this_bin <- cut(test_probs[i], breaks = bin_breaks, include.lowest = TRUE, right = FALSE)
    val <- bin_acc[as.character(this_bin)]
    if (is.na(val)) val <- mean(local_y, na.rm = TRUE)
    calibrated[i] <- val
  }
  calibrated
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
# Mahalanobis OOD score
# ------------------------
mahalanobis_score <- function(X_ref, X_query, regularize = 1e-6) {
  mu <- colMeans(X_ref)
  S <- cov(X_ref)
  S <- S + diag(regularize, ncol(S))
  invS <- tryCatch(solve(S), error = function(e) MASS::ginv(S))
  d2 <- mahalanobis(X_query, center = mu, cov = S)
  # scale 0-1 for convenience (min-max)
  (d2 - min(d2)) / (max(d2) - min(d2) + 1e-12)
}

# ------------------------
# Orchestrator: single-run
# ------------------------
run_one <- function(seed = 1,
                    n_train = 1500, n_test = 500, p = 100,
                    corr_type = "identity", rho = 0.2,
                    drift_type = c("none","covariate","concept","hybrid"),
                    drift_strength = 0.5,
                    model_type = "xgboost",
                    k_lore = 100, k_smooth = 50,
                    k_lce = 100, drift_fraction = 0.2) { # <-- 'drift_fraction' is the argument name
  set.seed(seed)
  drift_type <- match.arg(drift_type)
  # generate source (train)
  Xs <- generate_X_mvnorm(n_train, p, corr_type, rho)
  gen_s <- generate_labels(Xs)
  ys <- gen_s$y; beta <- gen_s$beta
  # split source for model fit and calibration/error space
  idx_fit <- sample(seq_len(n_train), size = floor(0.7 * n_train))
  X_fit <- Xs[idx_fit, , drop = FALSE]; y_fit <- ys[idx_fit]
  X_error_space <- Xs[-idx_fit, , drop = FALSE]; y_error_space <- ys[-idx_fit]
  # generate target (test) with drift
  Xt_base <- generate_X_mvnorm(n_test, p, corr_type, rho)
  beta_target <- beta
  if (drift_type == "none") {
    Xt <- Xt_base
  } else if (drift_type == "covariate") {
    Xt <- apply_covariate_shift(Xt_base, drift_strength, drift_fraction) # <-- Uses drift_fraction
  } else if (drift_type == "concept") {
    beta_target <- apply_concept_drift(beta, drift_strength, drift_fraction) # <-- Uses drift_fraction
    Xt <- Xt_base
  } else { # hybrid
    hh <- apply_hybrid(Xt_base, beta, cov_strength = drift_strength, concept_strength = drift_strength, 
                       cov_fraction = drift_fraction, concept_fraction = drift_fraction) # <-- Uses drift_fraction
    Xt <- hh$X; beta_target <- hh$beta
  }
  gen_t <- generate_labels(Xt, beta_target)
  y_test <- gen_t$y
  # fit base model on X_fit
  fitted <- fit_model(model_type, X_fit, y_fit, seed = seed + 100)
  base_predict <- fitted$predict
  probs_orig <- base_predict(Xt)
  probs_fit <- base_predict(X_fit)
  probs_error_space <- base_predict(X_error_space)
  # prepare error points for smoothing
  pred_error_space <- ifelse(probs_error_space > 0.5, 1, 0)
  error_idx <- which(pred_error_space != y_error_space)
  X_errors <- if(length(error_idx)>0) X_error_space[error_idx, , drop = FALSE] else matrix(nrow=0, ncol=ncol(X_fit))
  
  # --- Mahalanobis (FIX 1) ---
  # This is now a method, not just a metric
  mah01 <- mahalanobis_score(X_fit, Xt)
  # Create Mahalanobis-adjusted predictions
  # If mah01=1 (OOD), push to 0.5. If mah01=0 (In-Dist), use probs_orig.
  mah_probs <- (1 - mah01) * probs_orig + mah01 * 0.5
  
  # --- KMM weights and retrained weighted logistic ---
  # Note: This now calls the fast, closed-form version.
  # The lambda parameter (1e-3) is used by default.
  w_kmm <- kmm_weights_stable(X_fit, Xt)
  
  # check for uniform weights
  if (all(abs(w_kmm - mean(w_kmm)) < 1e-8)) {
    message(sprintf("Uniform KMM weights (QP failed or flat) for seed=%d", seed))
    kmm_probs <- probs_orig  # fallback: identical to baseline
  } else {
    # Fit weighted model using same baseline type for fair comparison
    if (model_type %in% c("glm", "logistic")) {
      mod_kmm <- glm(y_fit ~ ., data = data.frame(y_fit = y_fit, X_fit),
                     family = binomial(), weights = w_kmm)
      kmm_probs <- as.numeric(predict(mod_kmm, newdata = data.frame(Xt), type = "response"))
    } else if (model_type == "xgboost") {
      dtrain_w <- xgboost::xgb.DMatrix(data = X_fit, label = y_fit, weight = w_kmm)
      mod_kmm <- xgboost::xgboost(data = dtrain_w, nrounds = 50,
                                  objective = "binary:logistic", verbose = 0)
      kmm_probs <- as.numeric(predict(mod_kmm, xgboost::xgb.DMatrix(Xt)))
    } else {
      # default fallback to glmnet if model_type is not recognized
      cv <- glmnet::cv.glmnet(x = X_fit, y = y_fit,
                              family = "binomial", weights = w_kmm, nfolds = 5)
      kmm_probs <- as.numeric(predict(cv, newx = Xt, s = "lambda.min", type = "response"))
    }
  }
  
  
  # --- LoRe ---
  lore_probs <- apply_lore_recalibration(X_fit, y_fit, probs_fit, Xt, probs_orig, k_neighbors = k_lore, n_bins = 10)
  
  # --- Kernel smoothing (your method) ---
  smooth_probs <- smooth_predictions_method(X_error_space, y_error_space, base_predict, Xt, probs_orig, k = k_smooth)
  
  # --- Kernel smoothing (local/self-tuning) ---
  smooth_local_probs <- smooth_predictions_local(X_error_space, y_error_space, 
                                                 base_predict, Xt, probs_orig, 
                                                 k = k_smooth, k_local = 10)
  
  methods <- list(
    original = probs_orig, 
    smooth = smooth_probs, 
    smooth_local = smooth_local_probs,
    kmm = kmm_probs, 
    lore = lore_probs,
    mah = mah_probs)
  
  out_rows <- list()
  for (mname in names(methods)) {
    pp <- methods[[mname]]
    
    # --- LCE (FIX 3) ---
    # Calculate LCE for *each method's* final predictions on the test set.
    # We measure local calibration *within* the test set.
    lce_val <- compute_lce(X_calib = Xt, 
                           Y_calib = y_test, 
                           calib_probs = pp, 
                           X_test = Xt, 
                           k_neighbors = k_lce)
    
    out_rows[[mname]] <- data.table(
      seed = seed, model = model_type, p = p, n_train = n_train, n_test = n_test,
      corr_type = corr_type, rho = rho, drift_type = drift_type, drift_strength = drift_strength,drift_fraction = drift_fraction,
      method = mname,
      auc = compute_auc(y_test, pp),
      acc = compute_acc(y_test, pp),
      brier = compute_brier(y_test, pp),
      cmr = compute_cmr(y_test, pp),
      ccr = compute_ccr(y_test, pp),
      lce = lce_val, # <-- (FIX 3) Using the per-method LCE value
      high_conf_n = compute_high_conf_n(pp),   ## <<< ADDED >>>
      high_conf_pct = compute_high_conf_pct(pp), ## <<< ADDED >>>
      mah_mean = mean(mah01, na.rm = TRUE) # This is a diagnostic for the *dataset*
    )
  }
  rbindlist(out_rows)
}

# ------------------------
# High-level sequential grid run
# Model Type
# ------------------------
PARAMS <- list(
  seeds = 1:100,
  models = c("xgboost", "glm","randomforest"),
  #models = c("xgboost"),
  p_vals = c(100),
  n_train = 1000,
  n_test = 500,
  corr_types = c("identity"),
  rhos = c(0.3),
  drift_types = c("none", "covariate", "concept", "hybrid"),
  #drift_strengths = c(0, 1.0, 2.0),
  #drift_fraction = c(0.1, 0.5, 0.9)
  drift_strengths = c(1.0),
  drift_fraction = c(0.25)
)

results_all <- list()
ctr <- 1
total_runs <- length(PARAMS$seeds) * length(PARAMS$models) * length(PARAMS$p_vals) *
  length(PARAMS$corr_types) * length(PARAMS$rhos) * length(PARAMS$drift_types) * length(PARAMS$drift_strengths) * length(PARAMS$drift_fraction)

# Total runs is per-method-group, so 5 methods will be generated per run
message("Total runs (method groups): ", total_runs, " -> total method rows = ", total_runs * 5)  

for (seed in PARAMS$seeds) {
  for (model in PARAMS$models) {
    for (p in PARAMS$p_vals) {
      for (corr in PARAMS$corr_types) {
        for (rho in PARAMS$rhos) {
          for (drift in PARAMS$drift_types) {
            for (dstr in PARAMS$drift_strengths) {
              for(frac in PARAMS$drift_fraction){
                
                # --- Minor Bug Fixes Here ---
                # 1. Corrected 'ds=V' to 'ds=%.2f'
                # 2. Matched 'frac' to its '%.2f' format specifier
                message(sprintf("[%d/%d] seed=%d model=%s p=%d corr=%s rho=%.2f drift=%s ds=%.2f frac=%.2f",
                                ctr, total_runs, seed, model, p, corr, rho, drift, dstr, frac))
                
                res_row <- run_one(seed = seed,
                                   n_train = PARAMS$n_train, n_test = PARAMS$n_test,
                                   p = p, corr_type = corr, rho = rho,
                                   drift_type = drift, drift_strength = dstr,
                                   model_type = model, 
                                   drift_fraction = frac) # <-- (FIX 2) Changed 'drift_frac' to 'drift_fraction'
                
                results_all[[length(results_all) + 1]] <- res_row
                ctr <- ctr + 1
              }
            }
          }
        }
      }
    }
  }
}

results_dt <- rbindlist(results_all, use.names = TRUE, fill = TRUE)

# --- Minor Bug Fix Here ---
# 3. Matched the message to the actual filename being saved.
# save results
saveRDS(results_dt, "/restricted/projectnb/ptsd/hbayly/Temp_Thesis/SimulationResults/Model_Type_ProjectOne_Simulation.rds")
