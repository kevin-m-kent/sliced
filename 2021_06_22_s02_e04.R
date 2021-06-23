library(tidymodels)
library(tidyverse)
library(here)
library(GGally)
library(naniar)
library(modeltime)
library(timetk) 
library(stringi)

doParallel::registerDoParallel()

# Read in Data ------------------------------------------------------------

folder <- "sliced-s01e04-knyna9"

geo_codes <- read_csv(here::here("Raw_Data", "au.csv")) %>%
  group_by(city) %>%
  dplyr::slice(1:1)

raw_data <- read_csv(here::here("Raw_Data", folder, "train.csv")) %>%
  rowwise() %>%
  mutate(location = stri_replace_all(location,  regex = "(?<=[a-z])([A-Z])", replacement = " $1")) %>%
  mutate(location = str_remove_all(location, "Airport")) %>%
  ungroup() %>%
  left_join(geo_codes, by = c("location" = "city"))

holdout <- read_csv(here::here("Raw_Data", folder, "test.csv")) %>%
  rowwise() %>%
  mutate(location = stri_replace_all(location,  regex = "(?<=[a-z])([A-Z])", replacement = " $1")) %>%
  mutate(location = str_remove_all(location, "Airport")) %>%
  ungroup() %>%
  left_join(geo_codes, by = c("location" = "city"))

# EDA ---------------------------------------------------------------------

splits <- initial_split(raw_data)

train <- training(splits)
test <- testing(splits)

folds <- vfold_cv(train, 5, strata = rain_tomorrow)

train %>%
  slice_sample(n = 5000) %>%
  select_if(is.numeric) %>%
  ggcorr()

# Modeling ----------------------------------------------------------------

init_recp <- recipe(rain_tomorrow ~ ., data = train) %>%
  update_role(id, new_role = "id") %>%
  step_novel(all_nominal_predictors()) %>%
  step_rm(country, iso2, admin_name, capital) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_mutate(rain_tomorrow = as.factor(rain_tomorrow), skip = TRUE) %>%
  step_mutate(evaporation = as.numeric(evaporation), sunshine = as.numeric(sunshine)) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_timeseries_signature(date) %>%
  step_rm(location) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_rm(date) 
  
xg_model <- boost_tree(mtry = tune(), trees = 1000, min_n = tune(), tree_depth = tune(),
                    learn_rate = tune(), loss_reduction = tune()) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xg_wf <- workflow() %>%
  add_model(xg_model) %>%
  add_recipe(init_recp)

final_xg_param <- xg_wf %>%
  parameters() %>%
  update(mtry = mtry(c(1, 20)))

grid_control <- control_grid(verbose = TRUE)

metric <- metric_set(mn_log_loss)

grid_results <- tune_grid(
  xg_wf,
  param_info = final_xg_param,
  resamples = folds,
  grid = 15,
  metrics = metric,
  control = grid_control
)

grid_results %>%
  autoplot()

# finalize ----------------------------------------------------------------

best_mod <- select_best(grid_results, metric = "mn_log_loss")

final_wf <- finalize_workflow(xg_wf, best_mod) 

lastf <- last_fit(final_wf, splits, metrics = metric)

predictions <- predict(lastf$.workflow[[1]], holdout, type = "prob")  %>%
  mutate(rain_tomorrow = .pred_1)

submission <- holdout %>%
  mutate(rain_tomorrow = predictions$rain_tomorrow) %>%
  select(id, rain_tomorrow)

submission %>% 
  write_csv("sub_4.csv")
