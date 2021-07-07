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
library(tidytext)
library(textrecipes)
library(here)
#source("step_isofor.R")

doParallel::registerDoParallel(cores = 6)


# Read Data ---------------------------------------------------------------

folder <- "sliced-s01e05-WXx7h8"

raw_data <- read_csv(here::here("Raw_Data", folder, "train.csv"))

splits <- initial_split(raw_data)
train <- training(splits)
test <- testing(splits)

folds <- validation_split(train)

holdout <- read_csv(here::here("Raw_Data", folder, "test.csv"))

# EDA ---------------------------------------------------------------------

#' last_review and reviews per month are often missing together, but these are usually the only columns missing
#' some cool stuff to do with the name of the listing with nlp. I've seen words like 'sunny' a lot in listings before in craigslist. I wonder if some sentiment analysis will be helpful here
#' probably also want to review punctuation from the review 
#' maybe some lat/lon clusters for a feature
#' distance from center of city - approximate with center of dataset city?

naniar::gg_miss_upset(raw_data)

raw_data %>%
  separate_rows(name, sep = " ") %>%
  anti_join(stop_words, by = c("name" = "word")) %>%
  count(name, sort = TRUE)
  
raw_data %>%
  count(room_type, sort = TRUE)

raw_data %>%
  select_if(is.numeric) %>%
  ggpairs()

tssq_lat <- 40.7580

tssq_lon <- -73.9855

# Modeling ----------------------------------------------------------------

metrics <- metric_set(rmse)

xbg_recip <- recipe(price ~ ., data = train) %>%
  update_role(id, new_role = "id") %>%
  step_tokenize(name) %>%
  step_stopwords(name) %>%
  step_tokenfilter(name) %>%
  step_tf(name) %>%
  step_dummy(neighbourhood_group) %>%
  step_mutate(date_since_today = as.numeric(Sys.Date() - last_review)) %>%
  step_rm(last_review) %>%
  step_mutate(distance_ctr = sqrt((tssq_lat - latitude)^2 + (tssq_lon - longitude)^2)) %>%
  step_rm(all_nominal_predictors()) %>%
  step_impute_mean(all_numeric_predictors())
  
xgb <- boost_tree(mtry = tune(),
                  trees = 2000,
                  min_n = tune(),
                  tree_depth = tune(),
                  learn_rate = tune()) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

xgb_param <- 
  xgb %>% 
  parameters() %>% 
  update(mtry = mtry(c(1, 20)))

wfset <- workflow_set(
  preproc = list(xgb_recp = xbg_recip),
  models = list(xgb_mod = xgb),
  cross = FALSE
)

wkflow <- workflow() %>%
  add_recipe(xbg_recip) %>%
  add_model(xgb)

tuned_results <- wkflow %>%
  tune_grid(
    folds,
    grid = 15,
    param_info = xgb_param
  )

tuned_results %>% 
  autoplot()

ctrl_sa <- control_sim_anneal(verbose = TRUE, no_improve = 10L)

set.seed(1234)
xbg_sa <-
  wkflow %>%
  tune_sim_anneal(
    resamples = folds,
    metrics = metrics,
    initial = tuned_results,
    param_info = xgb_param,
    iter = 10,
    control = ctrl_sa)
    


# Final Fit and Predict ---------------------------------------------------

xbg_sa %>%
  autoplot()

best_mod <- tuned_results %>%
  select_best("rmse")

final_work <- wkflow %>%
  finalize_workflow(best_mod)

final_fit <- last_fit(final_work, best_mod, split = splits)

hopreds <- predict(final_fit$.workflow[[1]], holdout)

holdout %>%
  select(id) %>%
  mutate(price = hopreds$.pred) %>%
  write_csv(here::here("04_pred_2021_06_29.csv"))

                      