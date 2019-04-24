library(tidyr)
library(scales)
library(ggplot2)
library(RColorBrewer)

# https://www.medienanstalt-nrw.de/fileadmin/user_upload/lfm-nrw/Foerderung/Forschung/Dateien_Forschung/forsaHate_Speech_2018_Ergebnisbericht_LFM_NRW.PDF

df <- read.csv("data.csv")

df %>%
  gather(answer, occ, veryoften, often, notoften, never, dontknow) %>%
  ggplot() + geom_bar(aes(y = occ / 100, x = year, fill = answer), stat="identity") +
  scale_fill_manual(guide=guide_legend(reverse=T), values = brewer.pal(5, "Blues"), labels=c("no answer", "never", "not often", "often", "very often")) + theme_bw() +
  theme(legend.title=element_blank(), panel.grid.major.y = element_blank(), panel.grid.minor.y = element_blank(), axis.title=element_blank(), axis.ticks.y=element_blank()) +
  scale_y_continuous(position = "right", labels = scales::percent) + coord_flip()
  # labs(title='"Have you witnessed hate speech or hate posts on the Internet?"', caption = "Source: Landesanstalt Medien NRW")


ggsave('hate_paper.pdf', plot = last_plot(),
       scale = 1, width = 6, height = 2,
       dpi = 300)
