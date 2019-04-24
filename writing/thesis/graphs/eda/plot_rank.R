library(tidyverse)
library(RColorBrewer)

# for all language model data

df <- read_csv('rank_com_index_lm.csv')

df$num_replies <- factor(df$rank, levels = df$rank)

# Basic barplot
p<-ggplot(data=df, aes(x=num_replies, y=count)) +
  geom_bar(stat="identity", fill=brewer.pal(7,"Blues")[4]) + theme_bw(base_size = 16) + 
  theme(panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) +
  ylab("Count") + xlab("Comments per Rank")
  
p

ggsave('comments_per_rank_lm.pdf', plot = last_plot(),
       scale = 1, width = 6, height = 3,
       dpi = 300)

# for classification data

df <- read_csv('rank_com_index_cl.csv')

df$num_replies <- factor(df$rank, levels = df$rank)

# Basic barplot
p<-ggplot(data=df, aes(x=num_replies, y=count)) +
  geom_bar(stat="identity", fill=brewer.pal(7,"Blues")[4]) + theme_bw(base_size = 16) + 
  theme(panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) +
  ylab("Count") + xlab("Comments per Rank")

p

ggsave('comments_per_rank_cl.pdf', plot = last_plot(),
       scale = 1, width = 6, height = 3,
       dpi = 300)

