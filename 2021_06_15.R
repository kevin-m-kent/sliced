
# Loading Libraries -------------------------------------------------------

library(tidymodels)
library(usemodels)
library(tidyverse)
library(GGally)
library(skimr)
library(naniar)
library(plotly)
library(visdat)
library(finetune)
source("step_isofor.R")

doParallel::registerDoParallel(cores = 6)

# EDA ---------------------------------------------------------------------

raw_data <- read_csv(here::here("Raw_Data", "train.csv")) 
holdout <- read_csv(here::here("Raw_Data", "test.csv"))

split <- initial_split(raw_data)
train <- training(split)
test <- testing(split)

folds <- vfold_cv(train, v = 10)

# score to beat:
sd(raw_data$profit)
#222.338

train %>% 
  ggplot(aes(discount, profit, col = region)) + geom_point() 

train %>%
  select(-id) %>%
  select_if(is.numeric) %>%
  ggpairs()

# Feature Engineering -----------------------------------------------------

metrics <- metric_set(rmse)


# Model Setup -------------------------------------------------------------

lreg_spec  <- recipe(profit ~ ., data = train) %>%
  update_role(id, new_role = "id") %>%
  step_select(-country) %>%
  step_novel(all_nominal_predictors()) %>%
  step_other(city, state, sub_category, postal_code) %>%
  step_naomit(all_predictors())

rf_spec <- recipe(profit ~ ., data = train) %>%
  update_role(id, new_role = "id") %>%
  step_select(-country) %>%
  step_novel(all_nominal_predictors(), new_level = "new") %>%
  step_other(city, state, sub_category, postal_code) 

xgb_spec <- recipe(profit ~ ., data = train) %>%
  update_role(id, new_role = "id") %>%
  step_select(-country) %>%
  step_novel(all_nominal_predictors()) %>%
  step_other(city, state, sub_category, postal_code) %>%
  step_dummy(all_nominal_predictors())

linear_reg <- linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("regression")

rf_mod <- rand_forest(mtry = tune(), trees = 1000) %>%
  set_engine("ranger") %>%
  set_mode("regression")

xgb_mod <- 
  boost_tree(
    trees = 1000, 
    tree_depth = tune(), min_n = tune(), 
    loss_reduction = tune(),                     ## first three: model complexity
    sample_size = tune(), mtry = tune(),         ## randomness
    learn_rate = tune(),                         ## step size
  ) %>% 
  set_engine("xgboost") %>% 
  set_mode("regression")

wf_set <-
  workflow_set(
    preproc = list(rf_spec = rf_spec, xgb_spec = xgb_spec),
    models = list(rf = rf_mod,
                   xgb = xgb_mod), cross = FALSE
  )

rf_param <- 
  rf_mod %>%
  parameters() %>%
  update(mtry = mtry(c(1, 10)))

wf_set <-
  wf_set %>%
  option_add(param = rf_param, id = "rf_spec_rf")

# Model Training ----------------------------------------------------------

grid_ctrl <-
  control_grid(
    save_pred = TRUE,
    parallel_over = "everything",
    save_workflow = TRUE
  )

grid_results <-
  wf_set %>%
  workflow_map(
    seed = 1503,
    resamples = folds,
    grid = 10,
    control = grid_ctrl,
    metrics = metrics,
    verbose = TRUE
  )

# Final Fits --------------------------------------------------------------

best_results <- 
  grid_results %>% 
  pull_workflow_set_result("rf_spec_rf") %>% 
  select_best(metric = "rmse")

boosting_test_results <- 
  grid_results %>% 
  pull_workflow("rf_spec_rf") %>% 
  finalize_workflow(best_results) %>% 
  last_fit(split = split) 

hopreds <- predict(boosting_test_results$.workflow[[1]], holdout)

hopreds %>%
  write_csv(here::here("01_pred_2021_06_15.csv"))


