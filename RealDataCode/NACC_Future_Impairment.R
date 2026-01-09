# ----------------------------------------------------------------------
# Project: PECA Real Data Analysis: MCI/AD Prediction Across ADRCs
# Analysis: NACC Data Cleaning and Preparation for Modeling (Future Impairment)
# Author: Henry Bayly 
# Date: December 5, 2025
# ----------------------------------------------------------------------

# --- 1. Setup and Library Loading ---
setwd("/projectnb/pecaml/baylyh/") # Keep this if running locally

library(dplyr)
library(readr) # Added readr explicit call for read_csv
library(tidyr)

# --- 2. Data Loading ---
data_path <- "RealData/NACC/nacc_11_05_2025.csv"
nacc_data <- read_csv(data_path)

# Define predictors needed from Baseline (Visit 1)
predictors_list <- c("NACCID", "NACCVNUM", "NACCAGEB", "SEX", "EDUC",
                     "NACCMOCA", "NACCMMSE", "NACCNE4S")

# --- 3. Split Data into Baseline (Predictors) and Follow-up (Target) ---

# 3.1 Prepare Baseline Data (Visit 1)
baseline_data <- nacc_data %>%
  filter(NACCVNUM == 1) %>%
  select(all_of(predictors_list)) %>%
  as.data.frame()

# 3.2 Prepare Follow-up Data (Visit 2) to get the Future Outcome
# We only need ID and the diagnosis/CDR variable from the next visit
followup_data <- nacc_data %>%
  filter(NACCVNUM == 2) %>%
  select(NACCID, Future_UDSD = NACCUDSD) 
# NOTE: If you specifically want the CDR Global score, change 'NACCUDSD' 
# to 'CDRGLOB' (or your specific CDR variable name) in the line above.

# 3.3 Join Future Outcome to Baseline Data
clean_data <- left_join(baseline_data, followup_data, by = "NACCID")

# --- 4. Data Cleaning and Transformation (On Baseline Predictors) ---

# 4.1. Handle EDUCATION (EDUC) outlier
clean_data <- clean_data %>%
  filter(EDUC < 99)

# 4.2. Handle MMSE Transformation
mmse_map <- c('6' = 0, '9' = 1, '10' = 2, '11' = 3, '12' = 4.5,
              '13' = 6, '14' = 7, '15' = 8.5, '16' = 10, '17' = 11,
              '18' = 12, '19' = 13, '20' = 14, '21' = 15, '22' = 16, '23' = 17,
              '24' = 18, '25' = 19, '26' = 20, '27' = 21, '28' = 22.5,'29'=25,'30'=28.5)

clean_data$NACCMMSE_TRN <- as.numeric(
  mmse_map[as.character(clean_data$NACCMMSE)]
)

clean_data$NACCMMSE_TRN <- ifelse(clean_data$NACCMMSE_TRN == -4, NA, clean_data$NACCMMSE_TRN)
clean_data$NACCMOCA <- ifelse(clean_data$NACCMOCA == -4, NA, clean_data$NACCMOCA)

# 4.3. Combine MOCA and transformed MMSE into one variable (MOCA_new)
clean_data <- clean_data %>%
  mutate(
    MOCA_new = coalesce(NACCMOCA, NACCMMSE_TRN)
  )

# 4.4. Handle APOE allele count (9 is missing)
clean_data <- clean_data[which(clean_data$NACCNE4S != 9),]

# --- 5. Create the Future TARGET variable ---

# Target is now based on 'Future_UDSD' (Visit 2 status)
# 1 = Normal, 2 = MCI, 3 = Dementia
# Impaired (1) = Future MCI or Dementia
# Normal (0) = Future Normal
clean_data <- clean_data %>%
  mutate(
    impaired = ifelse(Future_UDSD == 1, 0, 1)
  )

# --- 6. Final Selection and Output ---

# Select final model variables
# Note: Rows with no Visit 2 data (NA in 'impaired') will be removed here by na.omit()
final_data <- clean_data %>%
  select(SEX, EDUC, MOCA_new, NACCAGEB, NACCNE4S, impaired) %>%
  na.omit() 

# Filter MOCA outliers if necessary
final_data <- final_data[which(final_data$MOCA_new < 88),]

# Convert target to factor
final_data$impaired <- factor(final_data$impaired, levels = c(0, 1), labels = c("Normal", "Impaired"))

# Check the distribution
print(table(final_data$impaired))

# Save
write.csv(final_data, "RealData/NACC/NACC_Future_Impairment_modeling_Jan7.csv")
