library(tidyverse)
library(RColorBrewer)

# for all language model data

df <- read_csv('threads_per_article_lm.csv')

df$num_replies <- factor(df$num_thr, levels = df$num_thr)

# Basic barplot
p<-ggplot(data=df, aes(x=num_replies, y=count)) +
  geom_bar(stat="identity", fill=brewer.pal(7,"Blues")[4]) + theme_bw(base_size = 16) + 
  theme(panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) +
  ylab("Count") + xlab("Number of Sub-dialogues per Article")
  
p

ggsave('threads_per_article_lm.pdf', plot = last_plot(),
       scale = 1, width = 6, height = 3,
       dpi = 300)

# for classification data

df <- read_csv('threads_per_article_cl.csv')

df$num_replies <- factor(df$num_thr, levels = df$num_thr)

# Basic barplot
p<-ggplot(data=df, aes(x=num_replies, y=count)) +
  geom_bar(stat="identity", fill=brewer.pal(7,"Blues")[4]) + theme_bw(base_size = 16) + 
  theme(panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) +
  ylab("Count") + xlab("Number of Sub-dialogues per Article")

p

ggsave('threads_per_article_cl.pdf', plot = last_plot(),
       scale = 1, width = 6, height = 3,
       dpi = 300)

