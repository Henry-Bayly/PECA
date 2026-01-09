# ----------------------------------------------------------------------
# Project: PECA Real Data Analysis: MCI/AD Prediction Across ADRCs
# Analysis: ADNI Data Cleaning and Preparation for Modeling (Future Impairment)
# Author: Henry Bayly
# Date: December 5, 2025
# ----------------------------------------------------------------------

# --- 1. Setup and Library Loading ---
setwd("/projectnb/pecaml/baylyh/RealData/ADNI/") # Keep this if running locally

library(readr)
library(dplyr)
library(tidyr)
library(lubridate)
library(stringr)

# --- 2. Data Loading and Feature Extraction ---

# 2.1. APOE (Genotype)
APOERES <- read_csv("All_Subjects_APOERES_08Sep2025.csv") %>%
  mutate(E4s = stringr::str_count(GENOTYPE, "4")) %>%
  select(PTID, VISCODE, E4s) %>%
  mutate(VISCODE2 = VISCODE) %>%
  select(-VISCODE)

# 2.2. LABDATA (B12)
LABDATA <- read_csv("All_Subjects_LABDATA_08Sep2025.csv") %>%
  mutate(
    B12_deficiency = if_else(BAT126 < 200, 1, 0)
  ) %>% select(PTID, VISCODE2, B12_deficiency)

# 2.3. CDR (Diagnosis Score)
CDR <- read_csv("All_Subjects_CDR_08Sep2025.csv") %>%
  select(PTID, VISCODE2, CDGLOBAL) %>%
  mutate(CDGLOBAL = ifelse(CDGLOBAL %in% c(-1), NA, CDGLOBAL))

# 2.4. MMSE & MOCA (Cognitive Scores)
MMSE <- read_csv("All_Subjects_MMSE_08Sep2025.csv") %>%
  select(PTID, VISCODE2, MMSCORE) %>%
  mutate(MMSCORE = ifelse(MMSCORE %in% c(-1, -4), NA, MMSCORE))

MOCA <- read_csv("All_Subjects_MOCA_08Sep2025.csv") %>%
  select(PTID, VISCODE2, MOCA) %>%
  mutate(MOCA = ifelse(MOCA %in% c(-1, -4), NA, MOCA))

# 2.5. PTDEMOG (Age, Sex, Education, Race)
PTDEMOG <- read_csv("All_Subjects_PTDEMOG_08Sep2025.csv") %>%
  select(PTID, VISCODE2, PTGENDER, PTEDUCAT, PTRACCAT) %>%
  mutate(PTGENDER = ifelse(PTGENDER %in% c(-4), NA, PTGENDER),
         PTEDUCAT = ifelse(PTEDUCAT %in% c(-1,-4), NA, PTEDUCAT),
         PTRACCAT = ifelse(PTRACCAT %in% c(-1, -4, 6, 7), NA, PTRACCAT),
         PTRACCAT = ifelse(str_detect(PTRACCAT, "\\|"), "6", PTRACCAT),
         Race = recode(PTRACCAT,
                       `5` = 1, `4` = 2, `1` = 3, `3` = 4, `2` = 5)) %>%
  select(-PTRACCAT)

Study_Entry <- read_csv("All_Subjects_Study_Entry_08Sep2025.csv") %>%
  mutate(PTID = subject_id, VISCODE2 = entry_visit) %>%
  select(PTID, VISCODE2, entry_age)

# --- 3. Cognitive Score Transformation and Combination ---

# 3.1. MMSE -> MOCA Transformation
data_combined <- full_join(MMSE, MOCA, by = c("PTID", "VISCODE2"))

# Define the MMSE-to-MOCA mapping rule
mapping_rule <- c('6' = 0, '9' = 1, '10' = 2, '11' = 3, '12' = 4.5,
                  '13' = 6, '14' = 7, '15' = 8.5, '16' = 10, '17' = 11,
                  '18' = 12, '19' = 13, '20' = 14, '21' = 15, '22' = 16, '23' = 17,
                  '24' = 18, '25' = 19, '26' = 20, '27' = 21, '28' = 22.5,'29'=25,'30'=28.5)

# Apply the mapping rule
for (old_value in names(mapping_rule)) {
  data_combined$MMSCORE[data_combined$MMSCORE == as.numeric(old_value)] <- mapping_rule[old_value]
}

# 3.2. Combine MOCA and transformed MMSE (Priority: MOCA > Transformed MMSE)
data_combined$MOCA_new <- ifelse(!is.na(data_combined$MOCA),
                                 data_combined$MOCA,
                                 ifelse(is.na(data_combined$MOCA), data_combined$MMSCORE, NA))

data_combined <- data_combined %>% select(PTID, VISCODE2, MOCA_new) %>% unique()

# --- 4. Merge All Visits Data First ---

# Merge all available variables across all timepoints
all_visits_data <- list(
  APOERES, LABDATA, CDR,
  data_combined, PTDEMOG, Study_Entry
) %>%
  Reduce(function(x, y) full_join(x, y, by = c("PTID", "VISCODE2")), .)

# Fill in baseline demographics (up/down) so they exist at all timepoints
all_visits_data <- all_visits_data %>%
  group_by(PTID) %>%
  fill(c("E4s", "PTGENDER", "PTEDUCAT", "Race", "entry_age"), .direction = "downup") %>%
  ungroup()

# --- 5. Split into Baseline (Predictors) and Follow-up (Target) ---

# 5.1 Define Baseline Data (Visit 1 / 'bl' or 'sc')
baseline_data <- all_visits_data %>%
  filter(VISCODE2 %in% c("bl", "sc")) %>%
  distinct(PTID, .keep_all = TRUE) # If both bl and sc exist, take first

# 5.2 Define Follow-up Data (Visit 2 / 'm12')
# NOTE: Using 'm12' (Month 12) to match NACC's annual visit structure.
# If you want 6-month prediction, change "m12" to "m06".
followup_data <- all_visits_data %>%
  filter(VISCODE2 == "m12") %>%
  select(PTID, Future_CDGLOBAL = CDGLOBAL)

# 5.3 Join Future Outcome to Baseline Data
clean_data <- left_join(baseline_data, followup_data, by = "PTID")

# --- 6. Target Creation and Final Cleaning ---

# 6.1. Create the Future TARGET variable based on 'Future_CDGLOBAL'
# CDGLOBAL: 0 = Normal, 0.5 = MCI, 1/2/3 = Dementia
# Impaired (1) = Future MCI or Dementia (CDGLOBAL >= 0.5)
# Normal (0) = Future Normal (CDGLOBAL == 0)
clean_data <- clean_data %>%
  mutate(
    impaired = ifelse(Future_CDGLOBAL == 0, 0,
                      ifelse(Future_CDGLOBAL >= 0.5, 1, NA_real_))
  )

# 6.2. Final Selection of Model Variables
final_data <- clean_data %>%
  select(
    SEX = PTGENDER,
    EDUC = PTEDUCAT,
    MOCA_new,
    NACCAGEB = entry_age,
    NACCNE4S = E4s,
    impaired
  ) %>%
  na.omit() # Remove rows with missing predictors OR missing future outcome

# Convert target to factor for classification
final_data$impaired <- factor(final_data$impaired, levels = c(0, 1), labels = c("Normal", "Impaired"))

# Check stats
print(table(final_data$impaired))
write.csv(final_data, "ADNI_Future_Impairment_modeling_Jan7.csv", row.names = FALSE)
