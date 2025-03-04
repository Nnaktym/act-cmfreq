# Functions for collaborative matrix factorization.

load_bravehins <- function(cas_dataset_path) {
    #' Load CAS dataset
    #'
    #' @param cas_dataset_path path to CAS dataset
    #' @return brvehins
    # ! git clone https://github.com/dutangc/CASdatasets.git
    load(paste(cas_dataset_path, "brvehins1a.rda", sep = "/"))
    load(paste(cas_dataset_path, "brvehins1b.rda", sep = "/"))
    load(paste(cas_dataset_path, "brvehins1c.rda", sep = "/"))
    load(paste(cas_dataset_path, "brvehins1d.rda", sep = "/"))
    load(paste(cas_dataset_path, "brvehins1e.rda", sep = "/"))
    brvehins <- rbind(brvehins1a, brvehins1b, brvehins1c, brvehins1d, brvehins1e)
    return(brvehins)
}

train_test_split <- function(X, ratio = 0.75, seed = 123) {
    #' Split a matrix into training and testing sets.
    #'
    #' This function splits a matrix into training and testing sets based on a specified
    #' ratio. It handles potential NA values in the matrix.
    #'
    #' @param X A matrix (dense or sparse).
    #' @param ratio The proportion of data to include in the training set (default: 0.75).
    #' @param seed An optional seed for the random number generator.
    #'
    #' @return A list containing two matrices: X_train and X_test.
    set.seed(seed)
    if (!is.matrix(X) && !inherits(X, "sparseMatrix")) {
        stop("X must be a matrix or a sparseMatrix")
    }
    valid_indices <- which(!is.na(X))
    n_valid <- length(valid_indices)
    n_train <- floor(ratio * n_valid)
    train_indices <- sample(valid_indices, n_train, replace = FALSE)
    X_train <- X
    X_test <- X
    X_train[setdiff(valid_indices, train_indices)] <- NA
    X_test[train_indices] <- NA
    return(list(X_train = X_train, X_test = X_test))
}

k_fold_split <- function(X, k = 4, seed = 123) {
    #' Split a matrix into k folds for cross-validation.
    #'
    #' @param X A matrix (dense or sparse).
    #' @param k The number of folds.
    #' @param seed An optional seed for the random number generator.
    #' @return A list containing the folds and the original matrix.
    if (!is.matrix(X) && !inherits(X, "sparseMatrix")) {
        stop("X must be a matrix or a sparseMatrix")
    }
    valid_indices <- which(!is.na(X))
    n_valid <- length(valid_indices)
    set.seed(seed)
    shuffled_indices <- sample(valid_indices)
    fold_size <- floor(n_valid / k)
    folds <- list()
    start_index <- 1
    for (i in 1:k) {
        end_index <- ifelse(i == k, n_valid, start_index + fold_size - 1)
        fold_indices <- shuffled_indices[start_index:end_index]
        validation_indices <- fold_indices
        training_indices <- setdiff(valid_indices, validation_indices)
        X_train <- X
        X_val <- X
        X_train[validation_indices] <- NA
        X_val[training_indices] <- NA
        folds[[i]] <- list(train = X_train, val = X_val)
        start_index <- end_index + 1
    }
    return(list(folds = folds, X = X)) # Return the folds and the original matrix
}

fill_with_na <- function(df, threshold) {
    #' Fill values lower than a threshold with NA
    #'
    #' @param df The input dataframe.
    #' @param threshold The threshold.
    #' @return The dataframe with values lower than the threshold converted to NA.
    df[df < threshold] <- NA
    return(df)
}

wide_to_long_format <- function(wide_format_data, value_names = c("var1", "var2", "value"), na_omit = TRUE) {
    #' Convert wide format data to long format
    #'
    #' @param wide_format_data The input data in wide format.
    #' @param value_names The names of the columns in the long format.
    #' @param na_omit Whether to omit NA values.
    long_format_data <- melt(wide_format_data)
    if (na_omit) {
        long_format_data <- na.omit(long_format_data)
    }
    colnames(long_format_data) <- value_names
    return(long_format_data)
}

get_total <- function(data, category_to_analyze, aggregate_col, threshold = NA) {
    #' Get total of aggregate_col by category_to_analyze
    #'
    #' @param data The input data.
    #' @param category_to_analyze The category to analyze.
    #' @param aggregate_col The column to aggregate.
    message("aggregate_col: ", aggregate_col, "   group_cols: ", category_to_analyze)
    data[, category_to_analyze] <- lapply(data[, category_to_analyze], as.character)
    total_data <- tapply(data[, aggregate_col], list(data[, category_to_analyze[1]], data[, category_to_analyze[2]]), sum)
    if (!is.na(threshold)) {
        total_data <- fill_with_na(total_data, threshold)
    }
    print(dim(total_data))
    return(total_data)
}

get_prediction <- function(model, X) {
    #' Get prediction for a matrix
    #'
    #' @param model The trained model.
    #' @param X The matrix to predict.
    #' @return The predicted matrix.
    non_na_indices <- which(!is.na(X), arr.ind = TRUE)
    pred <- predict(model, user = non_na_indices[, 1], item = non_na_indices[, 2])
    X_pred <- X
    X_pred[non_na_indices] <- pred
    return(X_pred)
}

calc_rmse <- function(pred, act, show = TRUE) {
    #' Print the RMSE
    #'
    #' @param pred The predicted values.
    #' @param act The actual values.
    #' @return The RMSE.
    rmse <- sqrt(mean((na.omit(as.vector(pred)) - na.omit(as.vector(act)))^2))
    if (show) {
        cat(sprintf("RMSE : %.4f\n", rmse))
    }
    return(rmse)
}

optimize_params <- function(X, n_folds, k_values, lambda_values, random_seed = 123) {
    #' Optimize the hyper parameters for the CMF model using cross-validation.
    #'
    #' @param X The input matrix.
    #' @param n_folds The number of folds for cross-validation.
    #' @param k_values The list of k values to try.
    #' @param lambda_values The list of lambda values to try.
    set.seed(random_seed)
    cv_split <- k_fold_split(X, k = n_folds)
    cv_result <- NULL
    for (k in k_values) {
        for (lambda in lambda_values) {
            cv_score <- 0
            for (i in 1:n_folds) {
                X_train <- cv_split$folds[[i]]$train
                X_val <- cv_split$folds[[i]]$val
                cmf_args <- list(X = X_train, k = k, lambda = lambda, niter = 30, nonneg = TRUE, verbose = FALSE)
                model <- do.call(CMF, cmf_args)
                pred <- get_prediction(model, X_val)
                cv_score <- cv_score + calc_rmse(pred, X_val, show = FALSE) / n_folds
            }
            print(paste("k:", k, "lambda:", lambda, "CV RMSE:", cv_score))
            cv_result <- rbind(cv_result, c(k, lambda, cv_score))
        }
    }
    cv_result <- as.data.frame(cv_result)
    colnames(cv_result) <- c("k", "lambda", "cv_score")
    best_params <- cv_result[which.min(cv_result$cv_score), ]
    return(best_params)
}

visualize_heatmap <- function(data) {
    #' Visualize a heatmap of the data
    #'
    #' @param data A matrix of data to visualize
    #' @return A ggplot2 object
    data_matrix <- as.matrix(data)
    p <- ggplot(data = melt(data_matrix), aes(x = Var2, y = Var1, fill = value)) +
        geom_tile() +
        scale_fill_viridis(na.value = "white", limits = c(0, 10000)) +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
        labs(x = "Category", y = "Model", fill = "Pure Premium")

    return(p)
}
