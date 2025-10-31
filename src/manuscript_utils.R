# Figure & Statistics Functions
# Anna Maria Langmüller
# 2025 

# Read global Sobol sensitivity analysis data
read_SA_data <- function(path, params) {
  sa_data <- read.delim(path, header = TRUE, sep = "\t")
  
  # Ensure factors are ordered according to parameters
  sa_data$var1 <- factor(sa_data$var1, levels = params, ordered = TRUE)
  sa_data$var2 <- factor(sa_data$var2, levels = params, ordered = TRUE)
  
  return(sa_data)
}


# Format ggplot aesthetics
format_plot <- function(g) {
  g <- g +
    theme_minimal(base_size = 10, base_family = "Arial") +
    theme(
      legend.position = "top",
      legend.direction = "horizontal",
      legend.title = element_text(hjust = 0.5, vjust = 1),
      legend.text  = element_text(hjust = 0.5),
      legend.key.width  = unit(0.125, "in"),
      legend.key.height = unit(0.125, "in"),
      axis.title = element_text(size = 10),
      axis.text  = element_text(size = 8),
      panel.grid.major = element_line(linewidth = 0.3),
      panel.grid.minor = element_blank(),
      plot.tag = element_text(face = "bold", size = 10, family = "Arial"),
      plot.tag.position = c(0, 1.05),
      plot.margin = margin(t = 10, r = 15, b = 5, l = 5)
    )
  return(g)
}


# Helper to load data and prepare for plotting
prepare_test_data <- function(path) {
  df <- read.delim(path, header = TRUE)
  df$mean <- df$pred  # for compatibility with plotting functions
  return(df)
}


# Helper to process training snapshot
prepare_train_snapshot <- function(train_path) {
  train_data <- read.delim(train_path, header = TRUE)
  snapshot <- train_data %>%
    dplyr::group_by(Round) %>%
    dplyr::filter(ValidationRMSE == min(ValidationRMSE)) %>%
    dplyr::ungroup()
  snapshot$N <- c(5000, seq(6000, 20000, 1000))
  return(snapshot)
}


# Plot observed vs predicted GP values
plot_GP_pred <- function(test_data, resp, sample_size = 500, 
                         plot_color, label_x, label_y, seed = 42) {
  
  set.seed(seed)  # reproducibility
  
  # Subsample test data
  test_data <- test_data[sample(nrow(test_data), sample_size), ]
  
  # Create scatter plot
  g <- ggplot(test_data, aes(x = .data[[resp]], y = pred)) +
    geom_point(size = 0.75) +
    geom_abline(slope = 1, intercept = 0, color = plot_color, size = 1) +
    xlab(label_x) +
    ylab(label_y)
  g <- format_plot(g)
  return(g)
}


# Plot RMSE validation for GP
plot_rmse <- function(train_data, test_data, resp, plot_color, y_label) {
  
  # Compute RMSE on test set
  rmse <- sqrt(sum((test_data[[resp]] - test_data$mean)^2) / nrow(test_data))
  test_rmse <- data.frame(N = max(train_data$N), rmse = rmse)
  print(test_rmse)
  
  # Plot
  g <- ggplot() +
    geom_line(data = train_data, aes(x = N, y = ValidationRMSE)) +
    geom_point(data = train_data, aes(x = N, y = ValidationRMSE)) +
    geom_point(data = test_rmse, aes(x = N, y = rmse),
               col = plot_color, size = 2, shape = 15) +
    scale_x_continuous(labels = scales::comma) +
    xlab(bquote("N"[training])) +
    ylab(y_label)
  g <- format_plot(g)
  return(g)
}


# Clip predicted values to specified range
clip_pred <- function(df, limits) {
  df$mean[df$mean < min(limits)] <- min(limits)
  df$mean[df$mean > max(limits)] <- max(limits)
  return(df)
}


#Plot first order & total effect Sobol Indices for all model parameters
plot_SA_bar <- function(df, my_params_labels, my_plot_colors) {
  # Subset only first-order and total effects
  df_subset <- subset(df, effect %in% c("Total Effects", "First Order"))
  df_subset$var2 <- NULL  # not needed for bar plot
  
  # Order factor levels
  df_subset$effect <- factor(df_subset$effect, labels = c("First Order", "Total Effects"), ordered = TRUE)
  names(my_plot_colors) <- levels(df_subset$effect)
  
  g <- ggplot(df_subset, aes(x = var1, y = sobol, fill = effect)) +
    geom_bar(stat = "identity", position = position_dodge(0.9)) +
    geom_errorbar(aes(ymin = sobol - ci, ymax = sobol + ci),
                  width = 0.4, position = position_dodge(0.9)) +
    geom_text(aes(label = round(sobol, 2)), vjust = -0.5, color = "black",
              position = position_dodge(0.9), size = 8/.pt) +
    scale_y_continuous(breaks = seq(0, 1, 0.25), limits = c(0, 1), expand = c(0, 0)) +
    scale_x_discrete(labels = my_params_labels) +
    scale_fill_manual("", values = my_plot_colors,
                      labels = c("First order effect", "Total effect")) +
    ylab("Sensitivity index") +
    xlab("") +
    coord_cartesian(ylim = c(0, 1)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) 
  g <- format_plot(g)
  g <- g + theme(axis.text.x = element_text(angle = 45, hjust=1))
  return(g)
}


#Plot second order Sobol Indices for all model parameters
plot_SA_second <- function(df, my_params_labels, my_plot_colors, my_plot_limits = NULL) {
  # Subset second-order effects
  df_second <- subset(df, effect == "Second Order")
  df_second$sobol[df_second$sobol < 0] <- 0
  
  input_params <- levels(df_second$var1)
  helper_df <- data.frame(var1 = character(),
                          var2 = character(),
                          sobol = numeric(),
                          ci = numeric())
  
  # Build full matrix of 2nd order interactions
  for (i in seq_len(length(input_params) - 1)) {
    for (j in (i + 1):length(input_params)) {
      id <- which(df_second$var1 == input_params[i] & df_second$var2 == input_params[j])
      if (length(id) == 0)  # try reversed pair
        id <- which(df_second$var1 == input_params[j] & df_second$var2 == input_params[i])
      
      temp <- data.frame(var1 = input_params[i],
                         var2 = input_params[j],
                         sobol = df_second$sobol[id],
                         ci = df_second$ci[id])
      helper_df <- rbind(helper_df, temp)
    }
  }
  
  # Factor ordering
  helper_df$var1 <- factor(helper_df$var1, levels = input_params, ordered = TRUE)
  helper_df$var2 <- factor(helper_df$var2, levels = input_params, ordered = TRUE)
  
  # Highlight significant interactions
  sig_id <- which(helper_df$sobol - helper_df$ci > 0)
  helper_df_sig <- helper_df[sig_id, ]
  helper_df_sig$lw <- 0.5
  helper_df_sig$lw[which.max(helper_df_sig$sobol)] <- 1
  
  # Plot heatmap
  g <- ggplot(helper_df, aes(x = var1, y = var2, fill = sobol)) +
    geom_tile(color = "black") +
    geom_tile(data = helper_df_sig, aes(x = var1, y = var2),
              fill = NA, color = my_plot_colors[3],
              linewidth = helper_df_sig$lw)
  
  # Color scale
  if (is.null(my_plot_limits)) {
    g <- g + scale_fill_gradient("Sensitivity index",
                                 low = my_plot_colors[1],
                                 high = my_plot_colors[2])
  } else {
    g <- g + scale_fill_gradient("Sensitivity index",
                                 low = my_plot_colors[1],
                                 high = my_plot_colors[2],
                                 limits = my_plot_limits)
  }
  
  g <- format_plot(g) +
    theme(legend.key.width = unit(0.25, "in"),
          axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(x = "", y = "", fill = "Sobol Index") +
    geom_text(aes(label = format(sobol, scientific = TRUE, digits = 1)),
              color = "black", size = 8/.pt) +
    scale_x_discrete(drop = FALSE, labels = my_params_labels) +
    scale_y_discrete(drop = FALSE, labels = my_params_labels)
  
  return(g)
}


#Plot GP predictions
plot_pred <- function(df, my_vars, my_params_labels, my_plot_colors, model_type) {
  names(my_plot_colors) <- NULL
  
  # Set legend label and limits based on model type
  plot_limits <- switch(model_type,
                        "imax" = c(0, 1),
                        "duration" = c(min(df$mean), max(df$mean)),
                        "establishment" = c(0, 1))
  
  legend_label <- switch(model_type,
                         "imax" = bquote(italic(i)[italic(max)]),
                         "duration" = bquote(log[10](duration)),
                         "establishment" = "outbreak probability")
  
  g <- ggplot(df, aes(x = .data[[my_vars[1]]], y = .data[[my_vars[2]]], fill = mean)) +
    geom_raster() +
    scale_fill_gradient(legend_label,
                        low = my_plot_colors[1],
                        high = my_plot_colors[2],
                        limits = plot_limits) +
    xlab(my_params_labels[1]) +
    ylab(my_params_labels[2])
  
  g <- format_plot(g) +
    theme(legend.key.width = unit(0.25, "in"))
  
  return(g)
}


#Plot Sobol Indices for parameter subdomains
plot_sa_dynamic <- function(df, my_vars, my_params_labels, effect_type,
                            focal_param, my_plot_colors) {
  # Filter data by effect type (e.g., "First Order") and focal parameter
  df <- subset(df, effect == effect_type & var1 == focal_param)
  
  # Base heatmap of Sobol sensitivity
  g <- ggplot(data = df, aes(x = .data[[my_vars[1]]], y = .data[[my_vars[2]]])) +
    geom_raster(aes(fill = sobol)) +
    scale_fill_gradient(
      name = "Sensitivity index",
      low = my_plot_colors[1],
      high = my_plot_colors[2],
      limits = c(0, 1)
    )
  
  # Apply consistent theme and labeling
  g <- format_plot(g)
  g <- g +
    theme(legend.key.width = unit(0.25, "in")) +
    xlab(my_params_labels[1]) +
    ylab(my_params_labels[2])
  
  return(g)
}


#Determine municipalities with high average infectivity estimates
format_alphaRest_distr <- function(path_alphas, path_params, outlier_thres) {
  # Load fitted alphaRest and model parameters
  path_alphas <- FIT_ALPHAREST_PATH
  path_params <- FIT_PARAMS_PATH
  alphas <- read.csv(path_alphas, sep = "\t", header = TRUE)
  params <- read.csv(path_params, sep = "\t", header = TRUE)
  params$rank <- seq_len(nrow(params))
  
  # Combine and adjust alphaRest values
  fit_params <- merge(alphas, params, by = "rank")
  fit_params$alphaRest_adj <- fit_params$alphaRest * fit_params$correctionFactor
  
  # Median adjusted alphaRest per municipality
  alpha_median <- fit_params %>%
    group_by(municipality) %>%
    summarise(m = median(alphaRest_adj, na.rm = TRUE), .groups = "drop")
  
  # Identify high-alphaRest outliers (top X%)
  cutoff <- quantile(alpha_median$m, probs = (1 - outlier_thres))
  alpha_median$outlier <- alpha_median$m >= cutoff
  
  # Merge outlier classification back into data
  res <- merge(fit_params, alpha_median, by = "municipality")
  res$municipality <- factor(
    res$municipality,
    levels = alpha_median$municipality[order(alpha_median$m)]
  )
  
  return(res)
}


#Plot average infectivity estimate distributions per municipality
plot_alpha_dist <- function(my_df, my_plot_colors, my_param_labels) {
  names(my_plot_colors) <- c("FALSE", "TRUE")
  my_df$outlier <- factor(my_df$outlier, levels = c("FALSE", "TRUE"), ordered = T)
  
  g <- ggplot(data = my_df, aes(x = municipality, y = alphaRest_adj))
  g <- g + geom_boxplot(aes(col = outlier), outlier.size = 0.25,
                        linewidth = 0.25)
  g <- format_plot(g)
  g <- g + scale_color_manual("", values = my_plot_colors)
  g <- g  + theme(axis.text.x = element_blank(),
                  axis.ticks.x = element_blank(),
                  legend.position = "none",
                  panel.grid.major = element_blank(),
                  panel.grid.minor = element_blank(),
                  panel.background = element_blank(),
                  axis.line = element_line(color = "black", linewidth = 0.15))
  g <- g + xlab(my_param_labels[1])
  g <- g + ylab(my_param_labels[2])
  
  return(g)
}


# Plot economic comparison for municipalities with high average infectivity estimates
plot_GCP <- function(my_df, test_var, my_plot_colors, my_x_label, my_y_label) {
  names(my_plot_colors) <- c("FALSE", "TRUE")
  
  # Keep one row per municipality
  my_df <- my_df[!duplicated(my_df$municipality), ]
  my_df$outlier <- factor(my_df$outlier, levels = c("FALSE", "TRUE"))
  
  # Statistical comparison (H1: outlier municipalities have larger GCP)
  x <- my_df[[test_var]][my_df$outlier == "TRUE"]
  y <- my_df[[test_var]][my_df$outlier == "FALSE"]
  test_result <- wilcox.test(x, y, alternative = "greater")
  print(test_result)
  
  # Plot log10-transformed GCP
  g <- ggplot(my_df, aes(x = outlier, y = log10(.data[[test_var]]), col = outlier)) +
    geom_boxplot(outliers = FALSE) +
    geom_jitter(size = 0.5) +
    scale_x_discrete(
      name = my_x_label,
      labels = c("FALSE" = "remaining 95%", "TRUE" = "top 5%")
    ) +
    scale_color_manual("", values = my_plot_colors) +
    ylab(my_y_label)
  g <- format_plot(g) + theme(legend.position = "none")
  
  return(g)
}


# Observed vs. predicted correlation plot
plot_obs_pred <- function(my_df, my_var, my_plot_colors, my_label_x, my_label_y) {
  names(my_plot_colors) <- NULL
  
  # Compute Spearman correlation
  corr_test <- cor.test(my_df[[my_var]], my_df[["pred"]], method = "spearman")
  corr_label <- paste0("Spearman’s ρ = ", round(corr_test$estimate, 3))
  
  g <- ggplot(my_df, aes(x = .data[[my_var]], y = pred)) +
    geom_point(size = 0.75) +
    geom_abline(intercept = 0, slope = 1, col = my_plot_colors, linewidth = 1) +
    annotate(
      "text",
      x = Inf, y = Inf,
      label = corr_label,
      hjust = 1.1, vjust = 5,
      size = 8 / .pt
    ) +
    xlab(my_label_x) +
    ylab(my_label_y)
  
  g <- format_plot(g)
  return(g)
}


# Permutation test visualization
plot_permutations <- function(permut_df, pred, permut_type, my_plot_colors) {
  names(my_plot_colors) <- NULL
  
  # Select relevant permutation data
  permut_df <- subset(permut_df, select = permut_type)
  colnames(permut_df) <- "value"
  
  # Compute observed Spearman correlation
  pred_corr <- cor.test(pred$max_incidence, pred$pred, method = "spearman")$estimate
  
  # Calculate percentile of observed correlation within permutation distribution
  corr_percentile <- mean(permut_df$value < pred_corr) * 100
  percentile_label <- paste0("Percentile: ", round(corr_percentile, 2), "%")
  
  g <- ggplot(permut_df, aes(x = value)) +
    geom_histogram(bins = 30, fill = NA, col = "black", linewidth = 0.3) +
    geom_vline(xintercept = pred_corr, col = my_plot_colors, linewidth = 1) +
    annotate(
      "text",
      x = max(permut_df$value) * 0.5,
      y = nrow(permut_df) / 7,
      label = percentile_label,
      hjust = -0.1,
      vjust = 2,
      size = 8 / .pt
    ) +
    xlab("Spearman’s ρ") +
    ylab("Count")
  
  g <- format_plot(g)
  return(g)
}