rm(list = ls())
wd <- "./"
setwd(wd)

#Packages----
library(ggplot2)
library(dplyr)
library(plyr)
library(cowplot)
library(ggridges)

#Functions----
plot_training_procedure <- function(stats_path) {
  df <- read.table(stats_path, header = T) #read data 
  df$step <- c(0:(nrow(df)-1)) #training iteration number 
  #snap = best model snap shot (based on validation data)
  snap <- df %>% group_by(ModelType, Round) %>% filter(ValidationRMSE == min(ValidationRMSE)) %>% ungroup()
  
  g <- ggplot(data = df, aes(x = step, y = TrainingLoss)) #training loss 
  g <- g + geom_line() + geom_point()
  g <- g + theme_minimal()
  g <- g + geom_point(data = snap, col = "darkred", size = 2)
  g <- g + xlab("iteration") + ylab("loss")
  g <- g + geom_vline(xintercept = which(df$Snapshot == 0), col = "darkgrey", linetype = "dashed")
  g <- g + theme(text = element_text(size = 10))
  
  g1 <- ggplot(data = df)
  g1 <- g1 + geom_line(aes(x = step, y = ValidationRMSE), linetype = "solid")
  g1 <- g1 + geom_point(aes(x = step, y = ValidationRMSE))
  g1 <- g1 + geom_point(data = snap, aes(x = step, y = ValidationRMSE), col = "darkred", size = 2)
  g1 <- g1 + theme_minimal()
  g1 <- g1 + xlab("iteration") + ylab("RMSE val")
  g1 <- g1 + geom_vline(xintercept = which(df$Snapshot == 0), col = "darkgrey", linetype = "dashed")
  g1 <- g1 + theme(text =  element_text(size = 10))
  
  g2 <- ggplot(data = df)
  g2 <- g2 + geom_line(aes(x = step, y = TrainingRMSE), linetype = "solid")
  g2 <- g2 + geom_point(aes(x = step, y = TrainingRMSE))
  g2 <- g2 + geom_point(data = snap, aes(x = step, y = TrainingRMSE), col = "darkred", size = 2)
  g2 <- g2 + theme_minimal()
  g2 <- g2 + xlab("iteration") + ylab("RMSE train")
  g2 <- g2 + geom_vline(xintercept = which(df$Snapshot == 0), col = "darkgrey", linetype = "dashed")
  g2 <- g2 + theme(text = element_text(size = 10))
  
  g3 <- ggplot(data = snap, aes(x = Round, y = ValidationRMSE)) #best model RMSE
  g3 <- g3 + geom_bar(stat = "identity", width = 0.5, fill = "darkred", col = "transparent")
  g3 <- g3 + geom_text(aes(x = Round, label = round(ValidationRMSE, 3)), 
                       nudge_y = 0.005, col = "darkred",
                       size = 2.5)
  g3 <- g3 + theme_minimal()
  g3 <- g3 + ylab("RMSE") + xlab("round")
  g3 <- g3 + scale_x_continuous(breaks = seq(min(snap$Round), max(snap$Round)))
  g3 <- g3 + theme_minimal()
  g3 <- g3 + theme(text = element_text(size = 10))
  
  g_all <- plot_grid(g, g2, g1, g3, align = "hv", ncol = 1) #align plots
  return(g_all) #return 
}

plot_sampling_procedure <- function(training_path, resp) {
  df <- read.table(training_path, header = T)
  min_y = floor(min(df[[resp]])) #calculate lower threshold 
  max_y = ceiling(max(df[[resp]])) #calculate upper threshold 
  
  g <- ggplot(data = df, aes(x = .data[[resp]], y = simRound, group = simRound, fill = after_stat(x)))
  g <- g + geom_density_ridges_gradient(scale = 0.9, quantile_lines = TRUE, from = min_y , to = max_y)
  g <- g + scale_fill_viridis_c(name = "" , option = "C")
  g <- g + theme_minimal() + ylab("training round")
  g <- g + ggtitle(label = "sampled points")
  return(g)
}

plot_test_performance <- function(test_path, resp, n_sample = NULL) {
  df <- read.table(test_path, header = T) #read data
  df$CI.width <- df$upper - df$lower #calculate CI width
  df$ovlp <- 0 #observed value inside CI? 
  df$ovlp[which(df$upper < df[[resp]] | df$lower > df[[resp]])] <- c(-1)
  df$ovlp <- factor(df$ovlp, levels = c(0, -1), ordered = T)
  
  if(!is.null(n_sample)) { #check if sub sampling is requested 
    id <- sample(c(1:nrow(df)), size = n_sample, replace = F)
    df <- df[id,]
  }
  
  rmse <- sqrt( sum( (df[[resp]]  - df[["pred"]])^2 )/ nrow(df) ) #RMSE
  
  df$pred.mod <- df$pred
  if (resp == "duration") {
    if (any(df$pred < 1 ))
      df$pred.mod[which(df$pred < 1)] <- 1
    if (any(df$pred > 3))
      df$pred.mod[which(df$pred.mod > 3)] <- 3
  } else {
  df$pred.mod[which(df$pred.mod > 1 )] <- 1
  df$pred.mod[which(df$pred.mod < 0 )] <- 0
  }
  
  rmse.mod <- sqrt( sum ( (df[[resp]] - df[["pred.mod"]])^2) / nrow(df) ) #RMSE mod 
  avg.CI.width <- mean(df$CI.width) #average CI width
  my_label <- paste("RMSE: ", round(rmse, 3),"/", round(rmse.mod, 3),"\nN: ",nrow(df), sep = "") #for annotation
  my_breaks = round(seq(min(df[[resp]]), max(df[[resp]]), length.out = 5), 2) #for scale breaks 
  my_label_CI <- paste("overlap: ", round(sum(df$ovlp == 0)/nrow(df)*100, 2), "%\navg. width: ", round(avg.CI.width,2), sep = "")
  
  g <- ggplot(data = df)
  g <- g + geom_point(aes(x = .data[[resp]], y = pred))
  g <- g + geom_abline(intercept = 0 , slope = 1, col = "darkgreen", 
                       linetype = "solid", size = 1.25)
  g <- g + theme_minimal()
  
  g <- g + ylab(paste("predicted", resp)) + xlab(paste("observed", resp))
  
  g <- g + scale_x_continuous(breaks = my_breaks)
  g <- g + scale_y_continuous(breaks = my_breaks)
  g <- g + coord_cartesian(xlim = c(my_breaks[1], my_breaks[length(my_breaks)]),
                           ylim = c(my_breaks[1], my_breaks[length(my_breaks)]))
  g <- g + ggtitle(label = my_label)
  
  g1 <- ggplot(data = df)
  g1 <- g1 + geom_histogram(aes(x = CI.width), 
                            bins = 100, 
                            col = "black", fill = "transparent")
  g1 <- g1 + geom_vline(xintercept = avg.CI.width, col = "darkred", size = 1.25)
  g1 <- g1 + theme_minimal()
  g1 <- g1 + xlab("CI width") + ylab("count")
  g1 <- g1 + ggtitle(label = my_label_CI)
  
  g_all <- plot_grid(g, g1, ncol = 1, 
                     rel_heights = c(0.65, 0.35),
                     align = "hv", axis = "b") #align plots
  return(g_all) 
}

#Global Variables----
STATS_PATH <-"../GPs/duration/stats-duration.txt"
TRAINING_PATH <- "../GPs/duration/sim-training-duration-round15.txt"
TEST_PATH <- "../GPs/duration/duration-round15-snap7-pred.txt"
MY_RESP <- "duration"

#EDA----
plot_training_procedure(STATS_PATH)
plot_sampling_procedure(TRAINING_PATH, resp = MY_RESP)
plot_test_performance(TEST_PATH, resp = MY_RESP)
plot_test_performance(TEST_PATH, resp = MY_RESP, n_sample = 250)


