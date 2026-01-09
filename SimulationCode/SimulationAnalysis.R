# Analysis of Simulation Results: Project One
# Henry Bayly
# November 22 2025

################################################################################


setwd("/projectnb/pecaml/baylyh/SimulationResults/")
list.files()
library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)
########################################################################################

# Drift Type: None, Covariate, Concept, and Covariate

dat <- readRDS("Drift_Type_ProjectOne_Simulation.rds")
dat <- dat %>%
  mutate(drift_type = str_to_title(drift_type))


a <- dat %>% filter(method != "smooth") %>% group_by(method, drift_type) %>% summarise(avg_cmr = mean(cmr,na.rm=TRUE),
                                                                                       avg_acc = mean(acc,na.rm=TRUE),
                                                                                       avg_auc = mean(auc,na.rm=TRUE),
                                                                                       avg_brier = mean(brier,na.rm=TRUE),
                                                                                       avg_n = mean(high_conf_n,na.rm=TRUE))
print(a,n=nrow(a))
df_plot <- dat %>%
  filter(method != "smooth") %>%
  select(method, drift_type, cmr, acc, auc, brier) %>%
  pivot_longer(
    cols = c(cmr, acc, auc, brier),
    names_to = "metric",
    values_to = "value"
  ) %>%
  # Rename metrics
  mutate(metric = recode(metric,
                         cmr = "CMR",
                         acc = "Accuracy",
                         auc = "AUC",
                         brier = "Brier"),
         # Rename methods
         method = recode(method,
                         kmm = "KMM",
                         lore = "LoRe",
                         mah_ood = "MAH_OOD",
                         original = "Original",
                         smooth_local = "PECA"))


# Boxplot panel

ggplot(df_plot, aes(x = method, y = value, fill = drift_type)) +
  geom_boxplot(outlier.alpha = 0.3) +
  facet_wrap(~ metric, scales = "free_y") +
  theme_minimal(base_size = 13) +
  labs(x = "Method", y = "Value", fill = "Drift Type",
       title = "Performance Metrics by Method and Drift Type") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

################################################################################

# Drift Fraction: 0.1, 0.5, and 0.9

dat <- readRDS("Drift_Fraction_ProjectOne_Simulation.rds")
a <- dat %>% filter(method != "smooth") %>% group_by(method, drift_fraction) %>% summarise(avg_cmr = mean(cmr,na.rm=TRUE),
                                                                                           avg_acc = mean(acc,na.rm=TRUE),
                                                                                           avg_auc = mean(auc,na.rm=TRUE),
                                                                                           avg_brier = mean(brier,na.rm=TRUE),
                                                                                           avg_n = mean(high_conf_n,na.rm=TRUE))
print(a,n=nrow(a))
df_plot <- dat %>%
  filter(method != "smooth") %>%
  # Convert drift_fraction to a factor here
  mutate(drift_fraction = as.factor(drift_fraction)) %>%
  select(method, drift_fraction, cmr, acc, auc, brier) %>%
  pivot_longer(cols = c(cmr, acc, auc, brier),
               names_to = "metric", values_to = "value")
# Boxplot panel
ggplot(df_plot, aes(x = method, y = value, fill = drift_fraction)) +
  geom_boxplot(outlier.alpha = 0.3) +
  facet_wrap(~ metric, scales = "free_y") +
  theme_minimal(base_size = 13) +
  labs(x = "Method", y = "Value", fill = "Drift Fraction",
       title = "Performance Metrics by Method and Drift Fraction") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


################################################################################

# Drift Stength: 0, 1, and 2

dat <- readRDS("Drift_Strength_ProjectOne_Simulation.rds")
a <- dat %>% filter(method != "smooth") %>% group_by(method, drift_strength) %>% summarise(avg_cmr = mean(cmr,na.rm=TRUE),
                                                                                           avg_acc = mean(acc,na.rm=TRUE),
                                                                                           avg_auc = mean(auc,na.rm=TRUE),
                                                                                           avg_brier = mean(brier,na.rm=TRUE),
                                                                                           avg_n = mean(high_conf_n,na.rm=TRUE))
print(a,n=nrow(a))
df_plot <- dat %>%
  filter(method != "smooth") %>%
  # Convert drift_strength to a factor here
  mutate(drift_strength = as.factor(drift_strength)) %>%
  select(method, drift_strength, cmr, acc, auc, brier) %>%
  pivot_longer(cols = c(cmr, acc, auc, brier),
               names_to = "metric", values_to = "value")
# Boxplot panel
ggplot(df_plot, aes(x = method, y = value, fill = drift_strength)) +
  geom_boxplot(outlier.alpha = 0.3) +
  facet_wrap(~ metric, scales = "free_y") +
  theme_minimal(base_size = 13) +
  labs(x = "Method", y = "Value", fill = "Drift Strength",
       title = "Performance Metrics by Method and Drift Strength") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


################################################################################

# Correlation Type: Identity, Toeplitz, and Block (size = 10)

dat <- readRDS("Correlation_Type_ProjectOne_Simulation.rds")
a <- dat %>% filter(method != "smooth") %>% group_by(method, corr_type) %>% summarise(avg_cmr = mean(cmr,na.rm=TRUE),
                                                                                      avg_acc = mean(acc,na.rm=TRUE),
                                                                                      avg_auc = mean(auc,na.rm=TRUE),
                                                                                      avg_brier = mean(brier,na.rm=TRUE),
                                                                                      avg_n = mean(high_conf_n,na.rm=TRUE))
print(a,n=nrow(a))
df_plot <- dat %>%
  filter(method != "smooth") %>%
  # Convert corr_type to a factor here
  mutate(corr_type = as.factor(corr_type)) %>%
  select(method, corr_type, cmr, acc, auc, brier) %>%
  pivot_longer(cols = c(cmr, acc, auc, brier),
               names_to = "metric", values_to = "value")
# Boxplot panel
ggplot(df_plot, aes(x = method, y = value, fill = corr_type)) +
  geom_boxplot(outlier.alpha = 0.3) +
  facet_wrap(~ metric, scales = "free_y") +
  theme_minimal(base_size = 13) +
  labs(x = "Method", y = "Value", fill = "Correlation Type",
       title = "Performance Metrics by Method and Correlation Type") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


################################################################################

# Feature Count: 50,100,500,1000

dat <- readRDS("Feature_Count_ProjectOne_Simulation.rds")
a <- dat %>% filter(method != "smooth") %>% group_by(method, p) %>% summarise(avg_cmr = mean(cmr,na.rm=TRUE),
                                                                              avg_acc = mean(acc,na.rm=TRUE),
                                                                              avg_auc = mean(auc,na.rm=TRUE),
                                                                              avg_brier = mean(brier,na.rm=TRUE),
                                                                              avg_n = mean(high_conf_n,na.rm=TRUE))
print(a,n=nrow(a))
df_plot <- dat %>%
  filter(method != "smooth") %>%
  # Convert p to a factor here
  mutate(p = as.factor(p)) %>%
  select(method, p, cmr, acc, auc, brier) %>%
  pivot_longer(cols = c(cmr, acc, auc, brier),
               names_to = "metric", values_to = "value")
# Boxplot panel
ggplot(df_plot, aes(x = method, y = value, fill = p)) +
  geom_boxplot(outlier.alpha = 0.3) +
  facet_wrap(~ metric, scales = "free_y") +
  theme_minimal(base_size = 13) +
  labs(x = "Method", y = "Value", fill = "Feature Count",
       title = "Performance Metrics by Method and Feature Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

################################################################################

# Model Type: Logistic Reg, and XGBoost (need to add RF)

dat <- readRDS("Model_Type_ProjectOne_Simulation.rds")
a <- dat %>% filter(method != "smooth") %>% group_by(method, model) %>% summarise(avg_cmr = mean(cmr,na.rm=TRUE),
                                                                                  avg_acc = mean(acc,na.rm=TRUE),
                                                                                  avg_auc = mean(auc,na.rm=TRUE),
                                                                                  avg_brier = mean(brier,na.rm=TRUE),
                                                                                  avg_n = mean(high_conf_n,na.rm=TRUE))
print(a,n=nrow(a))
df_plot <- dat %>%
  filter(method != "smooth") %>%
  # Convert model to a factor here
  mutate(model = as.factor(model)) %>%
  select(method, model, cmr, acc, auc, brier) %>%
  pivot_longer(cols = c(cmr, acc, auc, brier),
               names_to = "metric", values_to = "value")
# Boxplot panel
ggplot(df_plot, aes(x = method, y = value, fill = model)) +
  geom_boxplot(outlier.alpha = 0.3) +
  facet_wrap(~ metric, scales = "free_y") +
  theme_minimal(base_size = 13) +
  labs(x = "Method", y = "Value", fill = "Model Type",
       title = "Performance Metrics by Method and Model Type") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

