library(tidyverse)
library(RColorBrewer)


df <- read_csv('word_freq.csv')

df$token <- factor(df$token, levels = rev(df$token))

# Basic barplot
p<- ggplot(df, aes(x=token, y=count)) +
  geom_bar(stat="identity", fill=brewer.pal(7,"Blues")[4]) + theme_bw(base_size = 16) + 
  theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank()) +
  ylab("Count") + xlab("") + coord_flip()
  
p

ggsave('word_freq.pdf', plot = last_plot(),
       scale = 1, width = 6, height = 8,
       dpi = 300)

