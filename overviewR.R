# set up environment
rm(list = ls())
if(!is.null(dev.list())) dev.off()

library(corrplot)
library(dplyr)
library(ggplot2)
library(corrgram)

# load and view data
path_personal <- "C:\\Users\\janal\\OneDrive\\Dokumente\\GitHub\\"
path_shared <- "cs699_project_classification\\data\\redfin_2020-02-10-15-33-53.csv"
file <- paste0(path_personal,path_shared)
data <- read.csv(file, header = TRUE)


View(data)
print(colnames(data), row.names = FALSE) # get relevant variables
str(data)
data.rel <- na.omit(data[c(8,9,10,12,13,14,15,17,26,27)])

data.goal <- data[3]
distinct = unique(data.goal)
distinct
       
# create overview of new dataset
str(data.rel)
cor(data.rel)
corrgram(data.rel)
pairs(data.rel)
