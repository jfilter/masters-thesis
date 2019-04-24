library(plyr)
library(tidyr)
library(ggplot2)

# http://colorbrewer2.org/#type=qualitative&scheme=Set2&n=3

# df <- data.frame(Dataset=c("Negative", "Neutral", "Mixed", "Positive"),
#                  Number=c(5020, 2311, 1146, 591))
# 
# df$Dataset <- factor(df$Dataset, c('Negative', 'Neutral', 'Mixed', 'Positive'))
# 
# p<-ggplot(data=df, aes(Dataset, Number)) +
#   geom_bar(position = "dodge", stat="identity", fill="#8da0cb") + 
#   geom_text(aes(label=round(Number, digits = 4)), position=position_dodge(width=0.9), vjust=-0.25) +
#   theme_bw(base_size = 16) + 
#   #  coord_cartesian(ylim = c(0.5, 0.75)) +
#   theme(panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(), axis.title.x=element_blank()) +
#   labs(y = "Number of Samples")
# 
# p
# 
# ggsave(filename="class_dist_ynacc_sentiment.pdf", plot=p, width=5, height=5)

# ggplot(data=df.facet_data, aes(x=month,y=sales, group=1)) +
#   geom_line() +
#   facet_grid(region ~ .)

df1 <- data.frame(losses=c(4.372642, 4.027451, 3.951885, 3.899897, 3.848145, 3.905525, 3.87816, 3.904202, 3.941682, 3.918227, 4.177016, 3.94559, 3.878929, 3.862061, 3.864187, 3.867463, 3.881567, 3.891063, 3.905075, 3.921756), kinds=c("train_loss", "train_loss", "train_loss", "train_loss", "train_loss", "train_loss", "train_loss", "train_loss", "train_loss", "train_loss", "valid_loss", "valid_loss", "valid_loss", "valid_loss", "valid_loss", "valid_loss", "valid_loss", "valid_loss", "valid_loss", "valid_loss"), model=c("Ner No Over", "Ner No Over", "Ner No Over", "Ner No Over", "Ner No Over", "Ner No Over", "Ner No Over", "Ner No Over", "Ner No Over", "Ner No Over", "Ner No Over", "Ner No Over", "Ner No Over", "Ner No Over", "Ner No Over", "Ner No Over", "Ner No Over", "Ner No Over", "Ner No Over", "Ner No Over"), epochs=c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

df2 <- data.frame(losses=c(4.339731, 4.023022, 3.962061, 3.91848, 3.890093, 3.797395, 3.75537, 3.699048, 3.548948, 3.509849, 3.440444, 3.372857, 4.128653, 3.934242, 3.885226, 3.869871, 3.850875, 3.822968, 3.791306, 3.764314, 3.739092, 3.723798, 3.718909, 3.721872), kinds=c("train_loss", "train_loss", "train_loss", "train_loss", "train_loss", "train_loss", "train_loss", "train_loss", "train_loss", "train_loss", "train_loss", "train_loss", "valid_loss", "valid_loss", "valid_loss", "valid_loss", "valid_loss", "valid_loss", "valid_loss", "valid_loss", "valid_loss", "valid_loss", "valid_loss", "valid_loss"), model=c("Ner Best", "Ner Best", "Ner Best", "Ner Best", "Ner Best", "Ner Best", "Ner Best", "Ner Best", "Ner Best", "Ner Best", "Ner Best", "Ner Best", "Ner Best", "Ner Best", "Ner Best", "Ner Best", "Ner Best", "Ner Best", "Ner Best", "Ner Best", "Ner Best", "Ner Best", "Ner Best", "Ner Best"), epochs=c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))

df3 <- data.frame(losses=c(4.310607, 4.020872, 3.946503, 3.875124, 3.837373, 3.842546, 3.861492, 3.845368, 3.871381, 3.850032, 4.152143, 3.938158, 3.877094, 3.855972, 3.85763, 3.865411, 3.872628, 3.886682, 3.884652, 3.866189), kinds=c("train_loss", "train_loss", "train_loss", "train_loss", "train_loss", "train_loss", "train_loss", "train_loss", "train_loss", "train_loss", "valid_loss", "valid_loss", "valid_loss", "valid_loss", "valid_loss", "valid_loss", "valid_loss", "valid_loss", "valid_loss", "valid_loss"), model=c("Normal No Over", "Normal No Over", "Normal No Over", "Normal No Over", "Normal No Over", "Normal No Over", "Normal No Over", "Normal No Over", "Normal No Over", "Normal No Over", "Normal No Over", "Normal No Over", "Normal No Over", "Normal No Over", "Normal No Over", "Normal No Over", "Normal No Over", "Normal No Over", "Normal No Over", "Normal No Over"), epochs=c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

df4 <- data.frame(losses=c(4.452074, 4.105614, 4.00929, 3.965018, 3.945313, 3.899778, 3.818363, 3.753621, 3.653286, 3.58189, 3.523214, 3.460139, 4.167853, 3.954849, 3.908577, 3.896338, 3.874606, 3.846656, 3.81153, 3.776657, 3.74667, 3.724043, 3.713683, 3.714541), kinds=c("train_loss", "train_loss", "train_loss", "train_loss", "train_loss", "train_loss", "train_loss", "train_loss", "train_loss", "train_loss", "train_loss", "train_loss", "valid_loss", "valid_loss", "valid_loss", "valid_loss", "valid_loss", "valid_loss", "valid_loss", "valid_loss", "valid_loss", "valid_loss", "valid_loss", "valid_loss"), model=c("Normal Best", "Normal Best", "Normal Best", "Normal Best", "Normal Best", "Normal Best", "Normal Best", "Normal Best", "Normal Best", "Normal Best", "Normal Best", "Normal Best", "Normal Best", "Normal Best", "Normal Best", "Normal Best", "Normal Best", "Normal Best", "Normal Best", "Normal Best", "Normal Best", "Normal Best", "Normal Best", "Normal Best"), epochs=c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))

df_all <- rbind.fill(df1, df2, df3, df4)

# Basic line plot with points
ggplot(data=df_all, aes(y=losses, x=epochs, color=kinds)) +
  facet_wrap(~model) +
  geom_line() 

