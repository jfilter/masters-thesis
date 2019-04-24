library(tidyverse)

library(RColorBrewer)


# http://colorbrewer2.org/#type=qualitative&scheme=Set2&n=3


df<- read.table(text = "Dataset;Number;Classification
0;Persuasive;1601;pos.
1;Persuasive;7559;neg.
2;Audience (Reply);6415;pos.
3;Audience (Reply);2679;neg.
4;Agreement;906;pos.
5;Agreement;8254;neg.
6;Disagreement;4090;pos.
7;Disagreement;5070;neg.
8;Informative;1534;pos.
9;Informative;7626;neg.
10;Mean;1982;pos.
11;Mean;7178;neg.
12;Controversial;3501;pos.
13;Controversial;5659;neg.
14;Off-topic;6244;neg.
15;Off-topic;2916;pos.", header = TRUE, sep = ";")

df$Classification <- factor(df$Classification, levels = c('pos.', 'neg.'))
df$Dataset <- factor(df$Dataset, levels = c('Persuasive', 'Audience (Reply)', 'Agreement', 'Informative', 'Mean', 'Controversial', 'Disagreement', 'Off-topic'))


p<-ggplot(data=df, aes(Dataset, Number, fill=Classification)) +
  geom_bar(position = "dodge", stat="identity") + 
  scale_fill_manual(values=brewer.pal(3,"Paired")[0:2], labels=c("positive","negative")) +
  # geom_text(aes(label=round(Number, digits = 4)), position=position_dodge(width=0.9), vjust=-0.25) +
  theme_bw(base_size = 18) + 
#  coord_cartesian(ylim = c(0.5, 0.75)) +
  theme(panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(), axis.title.x=element_blank()) +
  labs(y = "Count")

p

ggsave(filename="class_dist_ynacc_bin.pdf", plot=p, width=14, height=5)



df <- read_csv('class_dist_no_maj.csv')

df$value <- factor(df$value, levels = c(1, 0, -1))
df$col <- factor(df$col, levels = c('Persuasive', 'Audience (Reply)', 'Agreement', 'Informative', 'Mean', 'Controversial', 'Disagreement', 'Off-topic'))

p<-ggplot(data=df, aes(col, count, fill=value)) +
  geom_bar(position = "dodge", stat="identity") + 
  scale_fill_manual(name="Classification", values=brewer.pal(3,"Paired"), labels=c("positive","negative", 'no majority')) +
  # geom_text(aes(label=round(Number, digits = 4)), position=position_dodge(width=0.9), vjust=-0.25) +
  theme_bw(base_size = 18) + 
  #  coord_cartesian(ylim = c(0.5, 0.75)) +
  theme(panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(), axis.title.x=element_blank()) +
  labs(y = "Count")

p

ggsave(filename="class_dist_ynacc_bin_no_maj.pdf", plot=p, width=14, height=5)


