library(tidymodels)
library(solitude)

step_isofor <- function(
  recipe, 
  ..., 
  role = NA, 
  trained = FALSE, 
  iso_mod = NULL,
  sample_size = 256,
  num_trees = 100,
  max_depth = 10, 
  options = list(nproc = 1, replace = FALSE, seed = 101, respect_unordered_factors = "partition"),
  skip = FALSE,
  id = rand_id("isofor")
) {
  
  ## The variable selectors are not immediately evaluated by using
  ##  the `quos()` function in `rlang`. `ellipse_check()` captures 
  ##  the values and also checks to make sure that they are not empty.  
  terms <- ellipse_check(...) 
  
  add_step(
    recipe, 
    step_isofor_new(
      terms = terms, 
      trained = trained,
      role = role, 
      iso_mod = iso_mod,
      sample_size = sample_size,
      num_trees = num_trees,
      max_depth = max_depth, 
      options = options,
      skip = skip,
      id = id
    )
  )
}

step_isofor_new <- 
  function(terms, trained, role, iso_mod, sample_size, num_trees, max_depth, options, skip, id) {
    step(
      subclass = "isofor", 
      terms = terms,
      trained = trained,
      role = role,
      iso_mod = iso_mod,
      sample_size = sample_size,
      num_trees = num_trees,
      max_depth = max_depth,
      options = options,
      skip = skip,
      id = id
    )
  }

prep.step_isofor <- function(x, training, info = NULL, ...) {

  col_names <- terms_select(terms = x$terms, info = info) 
  
  iso_mod <- solitude::isolationForest$new(sample_size = x$sample_size, num_trees = x$num_trees, max_depth = x$max_depth, 
                                     nproc = x$options$nproc, replace = x$options$replace, seed = x$options$seed,
                                 respect_unordered_factors = x$options$respect_unordered_factors)

  iso_mod$fit(training[, col_names])
  
  step_isofor_new(
    terms = x$terms,
    role = x$role,
    trained = TRUE,
    iso_mod = iso_mod,
    sample_size = x$sample_size,
    num_trees = x$num_trees,
    max_depth = x$max_depth,
    options = x$options, 
    skip = x$skip,
    id = x$id
  )
}

bake.step_isofor <- function(object, new_data, ...) {
  
  iso_mod <- object$iso_mod
  
  new_data$if_score <- iso_mod$predict(new_data)$anomaly_score
 
  ## Always convert to tibbles on the way out
  tibble::as_tibble(new_data)
}

tunable.step_isofor <- function (x, ...) {
  tibble::tibble(
    name = c("sample_size", "max_depth"),
    call_info = list(list(pkg = "dials", fun = "sample_size"), list(pkg = "dials", fun = "tree_depth")),
    source = "recipe",
    component = "step_isofor",
    component_id = x$id
  )
}


