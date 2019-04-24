library(ggplot2)
library(RColorBrewer)


# http://colorbrewer2.org/#type=qualitative&scheme=Set2&n=3

df <- data.frame(Dataset=c("Negative", "Neutral", "Mixed", "Positive"),
                 Number=c(5020, 2311, 1146, 591))

df$Dataset <- factor(df$Dataset, c('Negative', 'Neutral', 'Mixed', 'Positive'))

p<-ggplot(data=df, aes(Dataset, Number)) +
  geom_bar(position = "dodge", stat="identity", fill=brewer.pal(7,"Blues")[4]) + 
  # geom_text(aes(label=round(Number, digits = 4)), position=position_dodge(width=0.9), vjust=-0.25) +
  theme_bw(base_size = 18) + 
  #  coord_cartesian(ylim = c(0.5, 0.75)) +
  theme(panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(), axis.title.x=element_blank()) +
  labs(y = "Count")

p

ggsave(filename="class_dist_ynacc_sentiment.pdf", plot=p, width=5, height=5)



df <- data.frame(Dataset=c("Negative", "Neutral", "Mixed", "Positive", "No Maj."),
                 Number=c(4366, 1764, 665, 395, 1969))

df$Dataset <- factor(df$Dataset, c('Negative', 'Neutral', 'Mixed', 'Positive', "No Maj."))

p<-ggplot(data=df, aes(Dataset, Number)) +
  geom_bar(position = "dodge", stat="identity", fill=brewer.pal(7,"Blues")[4]) + 
  # geom_text(aes(label=round(Number, digits = 4)), position=position_dodge(width=0.9), vjust=-0.25) +
  theme_bw(base_size = 18) + 
  #  coord_cartesian(ylim = c(0.5, 0.75)) +
  theme(panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(), axis.title.x=element_blank()) +
  labs(y = "Count")

p

ggsave(filename="class_dist_ynacc_sentiment_no_maj.pdf", plot=p, width=5, height=5)
