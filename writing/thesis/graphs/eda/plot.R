library(tidyverse)
library(RColorBrewer)

# for all language model data

df <- read_csv('lm_num_replies.csv')

df$num_replies <- factor(df$num_replies, levels = df$num_replies)

# Basic barplot
p<-ggplot(data=df, aes(x=num_replies, y=count)) +
  geom_bar(stat="identity", fill=brewer.pal(7,"Blues")[4]) + theme_bw(base_size = 16) + 
  theme(panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) +
  ylab("Count") + xlab("Number of Replies")
  
p

ggsave('lm_num_replies.pdf', plot = last_plot(),
       scale = 1, width = 6, height = 3,
       dpi = 300)

# for classification data

df <- read_csv('cl_num_replies.csv')

df$num_replies <- factor(df$num_replies, levels = df$num_replies)

# Basic barplot
p<-ggplot(data=df, aes(x=num_replies, y=count)) +
  geom_bar(stat="identity", fill=brewer.pal(7,"Blues")[4]) + theme_bw(base_size = 16) + 
  theme(panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) +
  ylab("Count") + xlab("Number of Replies") 

p

ggsave('cl_num_replies.pdf', plot = last_plot(),
       scale = 1, width = 6, height = 3,
       dpi = 300)

