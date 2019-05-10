 library(tidyr)
library(scales)
library(ggplot2)


df <- read.csv("german_lm_2.csv")

df %>%
  gather(Results, Value, Training, Validation) %>%
  ggplot(aes(x=Epoch, y=Value, colour=Results)) +
  theme(panel.grid.minor.y = element_blank()) +
  geom_line() + scale_color_brewer(palette="Paired") + theme_bw() +
  ylab("Loss") 
  # scale_x_continuous(breaks = c(1956, 1970, 1980, 1990, 2000, 2012)) +

ggsave('training_curves.jpg', plot = last_plot(),
       scale = 1, width = 6, height = 2,
       dpi = 300)

df %>%
  gather(Revenue, Dollar, Advertising, Circulation) %>%
  ggplot(aes(x=Year, y=Dollar / 1000000000, colour=Revenue)) +
  geom_line(size=2) + scale_color_brewer(palette="Paired") + theme_bw(base_size = 16) +
  ylab("Dollars in Billion") +
  theme(panel.grid.minor = element_blank()) +
  scale_y_continuous(labels = dollar_format()) + 
  scale_x_continuous(breaks = c(1956, 1970, 1980, 1990, 2000, 2012))

ggsave('rev_paper.pdf', plot = last_plot(),
       scale = 1, width = 6, height = 3,
       dpi = 300)


df2 <- read.csv("employers.csv")

df2 %>%
  ggplot(aes(x=Year, y=Total / 1000)) +
  geom_line(size=2 ,color=brewer.pal(7,"Blues")[4]) + scale_color_brewer(palette="Paired") + theme_bw() +
  ylab(NULL) +
  scale_y_continuous(labels = unit_format(unit = "K")) +
  scale_x_continuous(breaks = c(2004, 2008, 2012, 2017)) +
  # labs(title='Total Number of Employers in U.S. Newspapers Sector', caption = "Source: Pew Research Center analysis of Bureau of Labor Statistics Occupational Employment Statistics data")
  labs(title='Total Number of Employees in the U.S. Newspaper Sector', caption = "Source: Pew Research Center analysis of U.S. Bureau of Labor Statistics")


ggsave('emp.jpg', plot = last_plot(),
       scale = 1, width = 6, height = 4,
       dpi = 300)

df2 %>%
  ggplot(aes(x=Year, y=Total / 1000)) +
  geom_line(size=2 ,color=brewer.pal(7,"Blues")[4]) + scale_color_brewer(palette="Paired") + theme_bw(base_size = 16) +
  ylab(NULL) +
  theme(panel.grid.minor = element_blank()) +
  scale_y_continuous(labels = unit_format(unit = "K")) +
  scale_x_continuous(breaks = c(2004, 2008, 2012, 2017))

ggsave('emp_paper.pdf', plot = last_plot(),
       scale = 1, width = 6, height = 3,
       dpi = 300)
