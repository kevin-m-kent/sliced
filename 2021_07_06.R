library(tidymodels)
library(tidyverse)
library(here)
library(naniar)
library(ggally)
library(textrecipes)

doParallel::registerDoParallel(cores = 6)

# Read Data ---------------------------------------------------------------
#' feature eng - numeric values for titles
#' 

get_split_title <- function(df) {
  
  df %>%
    separate(Game, into = c("base_title", "subtitle"), sep = ":", remove = FALSE)

}

folder <- "sliced-s01e06-2ld97c"

raw <- read_csv(here::here("Raw_Data", folder,  "train.csv")) %>%
  get_split_title()

splits <- initial_split(raw)
train <- training(splits) 
test <- testing(splits)
folds <- vfold_cv(train)

holdout <- read_csv(here::here("Raw_Data", folder,  "test.csv")) %>%
  get_split_title()

# Preproc and Modeling ----------------------------------------------------

basic_recip <- recipe(Hours_watched ~ Month + Year + Game  + base_title + subtitle + Hours_Streamed + Peak_viewers + 
        Peak_channels + Streamers + Avg_viewer_ratio, data = train) %>%
        step_mutate(numeric_name = as.numeric(str_extract(base_title, "[0-9]+$"))) %>%
        step_mutate(has_subtitle = case_when(!is.na(subtitle) ~ TRUE, TRUE ~ FALSE)) %>%
        step_mutate(numeric_name = case_when(is.na(numeric_name) ~ 0, TRUE ~ numeric_name)) %>%
        step_mutate(base_title = str_remove_all(base_title, "[0-9]")) %>%
        step_mutate(roman_str = str_extract(base_title, " [IVX]+")) %>%
        step_mutate(roman_str = as.numeric(roman2int((roman_str)))) %>%
        step_mutate(numeric_name = case_when(numeric_name == 0 & !is.na(roman_str) ~ roman_str, TRUE ~ numeric_name)) %>%
        step_mutate(base_title  = as.factor(str_trim(base_title))) %>%
        step_unknown(base_title, Game) %>%
        step_novel(base_title, Game) %>%
        step_other(base_title, Game, threshold = tune()) %>%
        step_dummy(base_title) %>%
        step_rm(Game, roman_str, subtitle) 

xg_mod <- boost_tree(mtry = tune(), trees = 1000, learn_rate = tune()) %>%
  set_mode("regression") %>%
  set_engine("xgboost")

xg_wflow <- workflow() %>%
  add_model(xg_mod) %>%
  add_recipe(basic_recip)

final_params <- xg_wflow %>%
  parameters() %>%
  update(mtry = mtry(c(1, 100)))

ctrl <- control_grid(save_pred = TRUE, save_workflow = TRUE)

tuned <- tune_grid(
    xg_wflow,
    param_info = final_params,
    resamples = folds,
    grid = 15
)

best_mod <- select_best(tuned)

final_wflw <- finalize_workflow(xg_wflow, best_mod)

final_mod <- last_fit(final_wflw, splits)

hopreds <- predict(final_mod$.workflow[[1]], holdout)

final_mod %>% 
  pluck(".workflow", 1) %>%   
  pull_workflow_fit() %>% 
  vip(num_features = 20)


holdout %>%
  select(Game) %>%
  mutate(.pred = hopreds$.pred) %>%
  mutate(Rank = rank(-.pred)) %>%
  select(-.pred) %>%
  write_csv(here::here("2021-07-06-pred_2.csv"))
