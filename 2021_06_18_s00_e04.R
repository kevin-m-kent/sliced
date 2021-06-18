
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
library(glue)
library(textrecipes)
source("step_isofor.R")

doParallel::registerDoParallel(cores = 6)

# Reading Data ---------------------------------------------------------------------

episode <- "s00e04"

raw_data <- read_csv(here::here("Raw_Data", glue("sliced-{episode}"),  glue("{episode}-sliced_data.csv"))) 
holdout <- read_csv(here::here("Raw_Data", glue("sliced-{episode}"),  glue("{episode}-holdout-data.csv"))) 


split <- initial_split(raw_data)
train <- training(split)
test <- testing(split)

folds <- validation_split(train)


# EDA ---------------------------------------------------------------------
#' notes
#'Total views  definitely log normal
#'All columns complete except subtitle, which is ~ 65% complete
#'downloads, kernels are especially correlated (~ .70)
#'might be able to do some cool nlp stuff with the title and subtitle - tokenization, remove stop words, title length/complexity, topics

train %>%
  ggplot(aes(TotalViews)) + geom_histogram() + scale_x_log10(labels = scales::comma_format())
  
train %>% 
  select_if(is.numeric) %>%
  select(-Id) %>%
  ggpairs()

## lets take a look at the n-grams here

train %>%
  mutate(Title = tolower(Title), subtitle = tolower(Subtitle), Name = tolower(Name)) %>%
  separate_rows(Name, sep = " ") %>%
  count(Name, sort = TRUE)

## maybe the length of the title matters? 

train %>%
  mutate(Title_len = nchar(Title)) %>%
  ggplot(aes(Title_len, TotalViews)) + geom_point() + scale_y_log10()
  

# Feature Engineering -----------------------------------------------------

metrics <- metric_set(rmse)


# Model Setup -------------------------------------------------------------

recp_1g <- recipe(TotalViews ~ ., data = train) %>%
  update_role(Id, new_role = "Id") %>%
  step_mutate(Title = tolower(Title), Subtitle = tolower(Subtitle), Name = tolower(Name)) %>%
  step_tokenize(Title, Subtitle, Name, token = "ngrams", options = list(n = 1, 
                                                                        ngram_delim = "_")) %>%
  step_stopwords(Title, Subtitle, Name) %>%
  step_tokenfilter(Title, Subtitle, Name) %>%
  step_lda(Title, Subtitle, Name) 


recp_2g <- recipe(TotalViews ~ ., data = train) %>%
  update_role(Id, new_role = "Id") %>%
  step_mutate(Title = tolower(Title), Subtitle = tolower(Subtitle), Name = tolower(Name)) %>%
  step_tokenize(Title, Subtitle, Name, token = "ngrams", options = list(n = 2, 
                                                       ngram_delim = "_")) %>%
  step_stopwords(Title, Subtitle, Name) %>%
  step_tokenfilter(Title, Subtitle, Name) %>%
  step_lda(Title, Subtitle, Name) 

randf <- rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>%
  set_engine("ranger") %>%
  set_mode("regression")

rf_param <- 
  randf %>%
  parameters() %>%
  update(mtry = mtry(c(1, 30)))

wfset <-  workflow_set(preproc = list(one_gram = recp_1g, two_gram = recp_2g), models = list(randf = randf))

wfset <-
  wfset %>%
  option_add(param = rf_param, id = "one_gram_randf") %>%
  option_add(param = rf_param, id = "two_gram_randf") 
  

# Model Training ----------------------------------------------------------

grid_ctrl <-
  control_grid(
    save_pred = TRUE,
    parallel_over = "everything",
    save_workflow = TRUE
  )

grid_results <-
  wfset %>%
  workflow_map(
    seed = 1503,
    resamples = folds,
    grid = 15,
    control = grid_ctrl,
    metrics = metrics,
    verbose = TRUE
  )


# Final Fits --------------------------------------------------------------

grid_results %>%
  autoplot()

best_result <- grid_results %>%
  pull_workflow_set_result("one_gram_randf") %>%
  select_best("rmse")

final_mod <- 
  grid_results %>% 
  pull_workflow("one_gram_randf") %>% 
  finalize_workflow(best_result) %>% 
  last_fit(split = split) 

hopreds <- predict(final_mod$.workflow[[1]], holdout)

holdout %>%
  select(id) %>%
  mutate(profit = hopreds$.pred) %>%
  write_csv(here::here(glue("01_pred_{episode}.csv")))


