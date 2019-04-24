library(tidyverse)
library(RColorBrewer)

# for all language model data

df<- read.table(text = "
characters accuracy type
20-40 0.0  Context-agnostic
40-60 0.44  Context-agnostic
                60-80 0.7142857142857143  Context-agnostic
                80-100 0.6226415094339622  Context-agnostic
                100-120 0.7209302325581395  Context-agnostic
                120-140 0.6956521739130435  Context-agnostic
                140-160 0.6829268292682927  Context-agnostic
                160-180 0.4857142857142857  Context-agnostic
                180-200 0.6176470588235294  Context-agnostic
                200-220 0.7272727272727273  Context-agnostic
                220-240 0.4  Context-agnostic
                240-260 0.6666666666666666  Context-agnostic
                260-280 0.8666666666666667  Context-agnostic
                280-300 0.7647058823529411  Context-agnostic
                300-320 0.8888888888888888  Context-agnostic
                320-340 0.9230769230769231  Context-agnostic
                340-360 0.7692307692307693  Context-agnostic
                360-380 0.8  Context-agnostic
                380-400 0.6363636363636364  Context-agnostic
                20-40 0.75  Context-aware
40-60 0.7307692307692307  Context-aware
                60-80 0.7962962962962963  Context-aware
                80-100 0.6739130434782609  Context-aware
                100-120 0.825  Context-aware
                120-140 0.775  Context-aware
                140-160 0.7428571428571429  Context-aware
                160-180 0.6666666666666666  Context-aware
                180-200 0.7142857142857143  Context-aware
                200-220 0.8235294117647058  Context-aware
                220-240 0.8571428571428571  Context-aware
                240-260 0.7272727272727273  Context-aware
                260-280 0.8  Context-aware
                280-300 1.0  Context-aware
                300-320 0.6666666666666666  Context-aware
                320-340 0.6363636363636364  Context-aware
                340-360 0.8333333333333334  Context-aware
                360-380 0.6  Context-aware
                380-400 0.75  Context-aware
                
                
                
                ", header = TRUE, sep = "")
df$characters <- factor(df$characters, levels = head(df$characters, 19))

# Basic barplot
p<-ggplot(data=df, aes(x=characters, y=accuracy)) +
  geom_bar(stat="identity", fill=brewer.pal(7,"Blues")[4]) + theme_bw(base_size = 16) + 
  theme(panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(), axis.text.x=element_text(size=9)) +
  ylab("Accuracy") + xlab("Number of Characters") + facet_wrap( ~ type, ncol=1)

  
p

ggsave('length_acc.pdf', plot = last_plot(),
       scale = 1, width = 12, height = 6,
       dpi = 300)

# for classification data

df <- read_csv('cl_num_replies.csv')

df$num_replies <- factor(df$num_replies, levels = df$num_replies)

# Basic barplot
p<-ggplot(data=df, aes(x=num_replies, y=count)) +
  geom_bar(stat="identity", fill=brewer.pal(7,"Blues")[4]) + theme_bw(base_size = 16) + 
  theme(panel.grid.major.x = element_blank(), panel.grid.minor = element_blank()) +
  ylab("Count") + xlab("Number of Replies") + facet_wrap( ~ type, ncol=1)


p

ggsave('cl_num_replies.pdf', plot = last_plot(),
       scale = 1, width = 6, height = 3,
       dpi = 300)

