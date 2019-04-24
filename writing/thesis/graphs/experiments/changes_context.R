library(tidyverse)
library(RColorBrewer)

# for all language model data

df<- read.table(text = "
per cat type
-0.34 Persuasive f1micro
3.74 Persuasive f1macro
12.46 Persuasive kappa
-4.57 Audience f1micro
-3.98 Audience f1macro
-9.37 Audience kappa
0.88 Agreement f1micro
6.12 Agreement f1macro
19.14 Agreement kappa
-0.81 Informative f1micro
1.46 Informative f1macro
3.70 Informative kappa
0.6 Mean f1micro
-0.4 Mean f1macro
-0.79 Mean kappa
4.67 Controversial f1micro
2.07 Controversial f1macro
7.65 Controversial kappa
2.05 Disagreement f1micro
1.81 Disagreement f1macro
4.76 Disagreement kappa
2.8 Off-topic f1micro
10.45 Off-topic f1macro
36.26 Off-topic kappa
3.03 Sentiment f1micro
-8.92 Sentiment f1macro
12.11 Sentiment kappa
1.53 Average f1micro
3.08 Average f1macro
11.33 Average kappa

                
                ", header = TRUE, sep = "")
df$type <- factor(df$type, levels = c('f1micro', 'f1macro', 'kappa'))
df$cat <- factor(df$cat, levels = c('Persuasive', 'Audience', 'Agreement', 'Informative', 'Mean', 'Controversial', 'Disagreement', 'Off-topic', 'Sentiment', 'Average'))

levels(df$type) <- c(expression('F1 MICRO'), "F1 MACRO", "KAPPA")

# Basic barplot
p<-ggplot(data=df, aes(x=cat, y=per)) +
  geom_bar(stat="identity", fill=brewer.pal(7,"Blues")[4]) + theme_bw(base_size = 16) + 
  theme(panel.spacing = unit(2, "lines"), panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(), axis.text.x=element_text(size=9)) +
  ylab("Change in %") + xlab("") + facet_wrap( ~ type, ncol=1) +
  # geom_text(aes(label=per), vjust=-0.3, size=3.5)
  geom_text(aes(label = paste(per, "%"),
              vjust = ifelse(per >= 0, -0.1, 1.1))) 

p

ggsave('changes.pdf', plot = last_plot(),
       scale = 1, width = 9, height = 12,
       dpi = 300)

