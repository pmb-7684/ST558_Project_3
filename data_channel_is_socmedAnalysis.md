Project 3: Predictive Model for `data_channel_is_socmed` Analysis
================
Smitali Paknaik and Paula Bailey
2022-11-16

<style type="text/css">

h1.title {
  font-size: 38px;
  text-align: center;
}
h4.author {
    font-size: 25px;
  font-family: "Times New Roman", Times, serif;
  text-align: center;
}
h4.date { 
  font-size: 25px;
  font-family: "Times New Roman", Times, serif;
  text-align: center;
}
</style>

# Introduction

Social and Online media is now perfused into our society and by getting
to know how people think and what they like, we get valuable information
on how new/information spreads through out a small society or globally
and how it affects the members of a community, their thinking and their
actions. This information is very useful for product based companies
trying to sell products, advertising companies, writers, influencers
etc. Not only that, we also know that it also has lot of impact at
political and economic changes occurring within a state or country. What
is advertised well can make good best sales, going by that theory
companies have now started encashing machine learning methods to know
about consumer responses from historical data and use it for expanding
markets.

The main goal of this project is to analyze a similar
OnlineNewsPopularity data using Exploratory Data Analysis (EDA) and
apply regression and ensemble methods to predict the response variable
‘\~share’ using a set of predictor variables optimized through variable
selection methods. That is, we will try to find out what parameters
impact number of online shares of information and whether they can help
us predict the number of shares from these parameters.

This OnlineNewsPopularity data is divided into some genre or channel and
each individual channel has its characteristics that need to be looked
at separately. And some generalization of analysis is necessary as it
helps speed up the process and automate the entire analysis at a mouse
click.

Our first step is to shape the data to get the various
data_channel_groups together in such a format that each channel can be
looked into separately . This is needed in order to get proper filtering
of the required channels in the automation script. The EDA is performed
on each channel using graphs, tables and heatmaps.

The next step includes creating train and test data for train fit and
test prediction. Prior to model fitting, variable
selection/pre-processing of data (if needed) is done. Two linear
regression models are the first step of performing model fit and
predictions. The data may be easily explained by a set of linear
equations and may not require complex algorithms that can consume time,
cost and may require computation speed. Hence, regression models often
are a great start to run predictions.

The other method evaluated is two types of ensemble methods- that are
Random Forest and Boosted Tree method. Both are types of ensemble
methods that use tree based model to evaluate future values. These
methods do not have any specified methodology in model fit. They work on
the best statistic obtained while constructing the model fit.
Theoretically ensemble methods are said to have a good prediction, and
using this data it will be evaluated on what categories how these models
perform.

At the end for each channel analysis, all 4 models are compared and best
model that suits this category of channel is noted.

#### About Data and Data Preparation

There are 48 parameters in the dataset. The channel parameter is
actually a categorical variable which will be used for filtering data in
this project and form our analysis into different set of channel files.
The categories in this are :
‘lifestyle’,‘entertainment’,‘bus’,‘socmed’,‘tech’ and ‘world’.
Additional information about this data can be accessed
[here](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity).

The csv file is imported read using `read_csv()`. As we have automated
the analysis, we need each channel to be fed into the R code we have and
get the analysis and predictions as a different file. The channel
categories are in column (or wide) format so they have been converted
into long format using `pivot_longer`. The required category is then
easily filtered by passing the channel variables into the filter code.

Irrelevant columns are removed like URL and timedelta as they do not
have any significance in our predictions models.

The Exploratory analysis and Modelling was carried on the remaining set
of select variables.

## Package Imports

-   The following packages are required for creating predictive models.

1.  `caret` To run the Regression and ensemble methods with Train/Split
    and cross validation.
2.  `dplyr` A part of the `tidyverse` used for manipulating data.
3.  `GGally` To create ggcorr and ggpairs correlation plots .
4.  `glmnet` To access best subset selection.
5.  `ggplot2` A part of the `tidyverse` used for creating graphics.
6.  `gridextra` To plot with multiple grid objects.
7.  `gt` To test a low-dimensional nullhypothesis against
    high-dimensional alternative models.
8.  `knitr` To get nice table printing formats, mainly for the
    contingency tables.
9.  `leaps` To identify different best models of different sizes.
10. `markdown` To render several output formats.
11. `MASS` To access forward and backward selection algorithms
12. `randomforest` To access random forest algorithms
13. `tidyr` A part of the `tidyverse` used for data cleaning

# Load data and check for NAs

``` r
data <- read.csv("OnlineNewsPopularity.csv") %>% 
                              rename(
                                    Monday      = `weekday_is_monday`,
                                    Tuesday     = `weekday_is_tuesday`,
                                    Wednesday   = `weekday_is_wednesday`,
                                    Thursday    = `weekday_is_thursday`,
                                    Friday      = `weekday_is_friday`,
                                    Saturday    = `weekday_is_saturday`,
                                    Sunday      = `weekday_is_sunday`) %>% 
                                    dplyr::select(-url, -timedelta)

#check for missing values
anyNA(data)  
```

    ## [1] FALSE

The UCI site mentioned that `url` and `timedelta` are non-predictive
variables, so we will remove them from our data set during the importing
process. Afterwards, we checked to validate the data set contained no
missing values. anyNA(data) returned FALSE, so the file has no missing
data.

For the automation process, it will be easier if all the channels (ie
data_channel_is\_\*) are in one column. We used `pivot_longer()` to
pivot columns: data_channel_is_lifestyle, data_channel_is_entertainment,
data_channel_is_bus, data_channel_is_socmed, data_channel_is_tech, and
data_channel_is_world from wide to long format.

``` r
dataPivot <- data %>% pivot_longer(cols = c("data_channel_is_lifestyle", "data_channel_is_entertainment", "data_channel_is_bus","data_channel_is_socmed","data_channel_is_tech", "data_channel_is_world"),names_to = "channel",values_to = "Temp") 


newData <- dataPivot %>% filter(Temp != 0) %>% dplyr::select(-Temp)
```

Now, the individual channel columns are combined into one column named
channel. The variable Temp represents if an article exists for that
particular channel. For the final data set, we will remove any values
with 0. We performed the same pivot_longer() process on the days of the
week.

``` r
dataPivot <- newData %>% pivot_longer(c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"), names_to = "weekday",values_to = "Temp1") 

newData <- dataPivot %>% filter(Temp1 != 0) %>% dplyr::select(-Temp1)
```

# Create Training and Testing Sets

When we run `Render_Code.R`, this code chunk filters data for each
channel type to complete the analysis.

``` r
selectChannel <- newData %>% filter(channel == params$channel)
```

To make the data reproducible, we set the seed to 21 and created the
training and testing set with a 70% split.

``` r
set.seed(21)
trainIndex <- createDataPartition(selectChannel$shares, p = 0.7, list = FALSE)
selectTrain <- selectChannel[trainIndex, ]
selectTest <- selectChannel[-trainIndex, ]
```

# Exploratory Data Analysis

The data has been analysed in this sections. This includes basic
statistics, summary tables, contingency tables and useful plots. We have
tried to capture as much information as we can in this.

## Training Data Summary

``` r
str(selectTrain)
```

    ## tibble [1,628 x 48] (S3: tbl_df/tbl/data.frame)
    ##  $ n_tokens_title              : num [1:1628] 8 8 9 9 7 6 11 10 10 8 ...
    ##  $ n_tokens_content            : num [1:1628] 257 218 1226 168 518 ...
    ##  $ n_unique_tokens             : num [1:1628] 0.568 0.663 0.41 0.778 0.486 ...
    ##  $ n_non_stop_words            : num [1:1628] 1 1 1 1 1 ...
    ##  $ n_non_stop_unique_tokens    : num [1:1628] 0.671 0.688 0.617 0.865 0.654 ...
    ##  $ num_hrefs                   : num [1:1628] 9 14 10 6 11 24 5 19 3 8 ...
    ##  $ num_self_hrefs              : num [1:1628] 7 3 10 4 1 6 1 2 1 8 ...
    ##  $ num_imgs                    : num [1:1628] 0 11 1 11 1 1 0 6 1 12 ...
    ##  $ num_videos                  : num [1:1628] 1 0 1 0 0 0 0 0 0 0 ...
    ##  $ average_token_length        : num [1:1628] 4.64 4.44 4.39 4.68 4.79 ...
    ##  $ num_keywords                : num [1:1628] 9 10 7 9 7 8 9 7 5 7 ...
    ##  $ kw_min_min                  : num [1:1628] 0 0 0 217 217 217 217 217 217 217 ...
    ##  $ kw_max_min                  : num [1:1628] 0 0 0 690 4800 737 1900 775 690 2600 ...
    ##  $ kw_avg_min                  : num [1:1628] 0 0 0 572 1480 ...
    ##  $ kw_min_max                  : num [1:1628] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_max_max                  : num [1:1628] 0 0 0 17100 28000 28000 28000 37400 37400 37400 ...
    ##  $ kw_avg_max                  : num [1:1628] 0 0 0 3110 6814 ...
    ##  $ kw_min_avg                  : num [1:1628] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_max_avg                  : num [1:1628] 0 0 0 2322 4800 ...
    ##  $ kw_avg_avg                  : num [1:1628] 0 0 0 832 1862 ...
    ##  $ self_reference_min_shares   : num [1:1628] 1300 3900 992 6600 3500 1600 1500 39200 0 3300 ...
    ##  $ self_reference_max_shares   : num [1:1628] 2500 3900 4700 6600 3500 1600 1500 39200 0 5200 ...
    ##  $ self_reference_avg_sharess  : num [1:1628] 1775 3900 2858 6600 3500 ...
    ##  $ is_weekend                  : num [1:1628] 0 0 0 0 0 0 0 1 0 0 ...
    ##  $ LDA_00                      : num [1:1628] 0.4392 0.1993 0.0298 0.0231 0.509 ...
    ##  $ LDA_01                      : num [1:1628] 0.0225 0.2477 0.1939 0.0223 0.0287 ...
    ##  $ LDA_02                      : num [1:1628] 0.0224 0.0201 0.0288 0.0224 0.0299 ...
    ##  $ LDA_03                      : num [1:1628] 0.0233 0.5127 0.7181 0.9096 0.4038 ...
    ##  $ LDA_04                      : num [1:1628] 0.4926 0.0202 0.0293 0.0226 0.0286 ...
    ##  $ global_subjectivity         : num [1:1628] 0.4 0.522 0.408 0.638 0.599 ...
    ##  $ global_sentiment_polarity   : num [1:1628] 0.00741 0.29912 0.10661 0.08798 0.28797 ...
    ##  $ global_rate_positive_words  : num [1:1628] 0.0311 0.055 0.0228 0.0714 0.0676 ...
    ##  $ global_rate_negative_words  : num [1:1628] 0.0272 0.0183 0.0114 0.0476 0.0135 ...
    ##  $ rate_positive_words         : num [1:1628] 0.533 0.75 0.667 0.6 0.833 ...
    ##  $ rate_negative_words         : num [1:1628] 0.467 0.25 0.333 0.4 0.167 ...
    ##  $ avg_positive_polarity       : num [1:1628] 0.36 0.536 0.395 0.492 0.49 ...
    ##  $ min_positive_polarity       : num [1:1628] 0.0333 0.1 0.0625 0.1 0.1 ...
    ##  $ max_positive_polarity       : num [1:1628] 0.6 1 1 1 1 0.8 0.5 1 0.8 1 ...
    ##  $ avg_negative_polarity       : num [1:1628] -0.393 -0.237 -0.258 -0.502 -0.35 ...
    ##  $ min_negative_polarity       : num [1:1628] -0.5 -0.25 -1 -1 -1 -0.5 -0.4 -0.6 -0.125 -0.5 ...
    ##  $ max_negative_polarity       : num [1:1628] -0.125 -0.2 -0.1 -0.15 -0.05 ...
    ##  $ title_subjectivity          : num [1:1628] 0.667 0.5 0 1 0 ...
    ##  $ title_sentiment_polarity    : num [1:1628] -0.5 0.5 0 -1 0 0.25 0 0.7 0 0 ...
    ##  $ abs_title_subjectivity      : num [1:1628] 0.167 0 0.5 0.5 0.5 ...
    ##  $ abs_title_sentiment_polarity: num [1:1628] 0.5 0.5 0 1 0 0.25 0 0.7 0 0 ...
    ##  $ shares                      : int [1:1628] 2600 690 4800 4800 775 1600 9000 14300 992 4900 ...
    ##  $ channel                     : chr [1:1628] "data_channel_is_socmed" "data_channel_is_socmed" "data_channel_is_socmed" "data_channel_is_socmed" ...
    ##  $ weekday                     : chr [1:1628] "Monday" "Monday" "Monday" "Wednesday" ...

`str()` allows us to view the structure of the data. We see each
variable has an appropriate data type.

``` r
selectTrain %>% dplyr::select(shares, starts_with("rate")) %>% summary()
```

    ##      shares      rate_positive_words rate_negative_words
    ##  Min.   :    5   Min.   :0.0000      Min.   :0.0000     
    ##  1st Qu.: 1400   1st Qu.:0.6667      1st Qu.:0.1667     
    ##  Median : 2100   Median :0.7506      Median :0.2446     
    ##  Mean   : 3527   Mean   :0.7430      Mean   :0.2514     
    ##  3rd Qu.: 3800   3rd Qu.:0.8333      3rd Qu.:0.3293     
    ##  Max.   :59000   Max.   :1.0000      Max.   :1.0000

This `summary()` provides us with information about the distribution
(shape) of shares (response variable), rate_positive_words, and
rate_negative_word.

    If mean is greater than median, then Right-skewed with outliers towards the right tail.  
    If mean is less than median, then Left-skewed with outliers towards the left tail.  
    If mean equals to median, then Normal distribtuion.  

## Training Data Visualizations

Note: Due to the extreme minimize and maximize values in our response
variable, shares, we transformed the data by applying log(). It allows
us to address the differences in magnitude throughout the share
variable. The results visually are much better.

``` r
g <- ggplot(selectTrain, aes(x=rate_positive_words, y = log(shares)))+geom_point() 
g + geom_jitter(aes(color = as.factor(weekday))) + 
        labs(x = "positive words", y = "shares",
        title = "Rate of Positive Words") +
  scale_fill_discrete(breaks=c("Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"))+
      scale_colour_discrete("") 
```

<img src="data_channel_is_socmedAnalysis_files/figure-gfm/scatter share-1.png" style="display: block; margin: auto;" />

We can inspect the trend of shares as a function of the positive word
rate. If the points show an upward trend, then articles with more
positive words tend to be shared more often. If we see a negative trend
then articles with more positive words tend to be shared less often.

``` r
g <- ggplot(selectTrain, aes(x = shares))
g + geom_histogram(bins = sqrt(nrow(selectTrain)))  + 
          labs(title = "Histogram of Shares")
```

<img src="data_channel_is_socmedAnalysis_files/figure-gfm/histogram shares-1.png" style="display: block; margin: auto;" />
To view the true shape of the response variable, shares, we did not
apply log() to transform the data.

We can inspect the shape of the response variable, shares. If the share
of the histogram is symmetrical or bell-shaped, then shares has a normal
distribution with the shares evenly spread throughout. The mean is equal
to the median. If the shape is left skewed or right leaning, then the
tail which contains a most of the outliers will be extended to the left.
The mean is less than the median. If the shape is right skewed or left
leaning, then the tail which contains a most of the outliers will be
extended to the right. The mean is greater than the median.

``` r
g <- ggplot(selectTrain) 

 g + aes(x = weekday, y = log(shares)) +
  geom_boxplot(varwidth = TRUE) + 
  geom_jitter(alpha = 0.25, width = 0.2,aes(color = as.factor(weekday)))  + 
     labs(title = "Box Plot of Shares Per Day") +
   scale_colour_discrete("") +
   scale_x_discrete(limits = c("Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday")) +
   theme(legend.position = "none")
```

<img src="data_channel_is_socmedAnalysis_files/figure-gfm/boxplot shares-1.png" style="display: block; margin: auto;" />
We can inspect the distribution of shares as a function of the day of
the week. If the points are within the body of the box, then articles
shared are within the 1st and 3rd quartile. If we see black points
outside of the whiskers of the boxplot, it represents outliers in the
data. These outliers contribute to the shape of the distribute.

## Contingency Tables

``` r
kable_if(table(selectTrain$channel, selectTrain$weekday))
```

    ##                         
    ##                          Friday Monday Saturday Sunday Thursday Tuesday Wednesday
    ##   data_channel_is_socmed    235    244      119    101      339     299       291

Articles per Channel vs Weekday - We can inspect the distribution of
articles shared as a function of the day of the week. We can easily see
which days articles are more likely shared.

``` r
kable_if(table(selectTrain$weekday, selectTrain$num_keywords))
```

    ##            
    ##              1  2  3  4  5  6  7  8  9 10
    ##   Friday     7  2 14 22 47 40 29 27 18 29
    ##   Monday     0  5 14 25 34 37 35 38 28 28
    ##   Saturday   1  0  3  8 13 16 15 13 13 37
    ##   Sunday     5  0  2  3 12 18 18 13 11 19
    ##   Thursday   8  1 20 39 45 59 53 50 32 32
    ##   Tuesday    4  3 16 37 35 48 59 39 24 34
    ##   Wednesday  7  2 12 36 51 38 58 39 26 22

Day of the Week vs Number of Keywords - We can inspect the distribution
of number of keywords as a function of the day of the week. Across the
top are the unique number of keywords in the channel. Articles shared
are divided by number of keywords and the day the article is shared.
It’s highly likely the more keywords, the more likely an article will be
shared with others.

## Correlations

We want to remove any predictor variables that are highly correlated to
the response variable which can cause multicollinearity. If variables
have this characteristic it can lead to skewed or misleading results. We
created groupings of predictor variables to look at the relationship
with our response variable, share and to each other.

``` r
ggcorr(selectTrain%>% dplyr::select(n_tokens_title, n_tokens_content, n_unique_tokens, n_non_stop_words, n_non_stop_unique_tokens,  shares),label_round = 2, label = "FALSE", label_size=1)
```

<img src="data_channel_is_socmedAnalysis_files/figure-gfm/tab3-1.png" style="display: block; margin: auto;" />
The correlation chart above does not show any strong relationships
between shares and the variables displayed. We can see many other
relationships which is indicated by the darker orange and blue colors.
For instance, n_token_title and n_non_stop_unique_tokens has a negative
relationship.

``` r
ggcorr(selectTrain%>% dplyr::select( average_token_length, num_keywords,num_hrefs, num_self_hrefs, num_imgs, num_videos, shares),label_round = 2, label_size=3, label = "FALSE")
```

<img src="data_channel_is_socmedAnalysis_files/figure-gfm/tab4-1.png" style="display: block; margin: auto;" />
The correlation chart above does not show any strong relationships
between shares and the variables displayed. We can see many other
relationships which is indicated by the darker orange and blue colors.

``` r
ggcorr(selectTrain%>% dplyr::select(kw_min_min, kw_max_min, kw_avg_min, kw_min_max, kw_max_max,  kw_avg_max, kw_min_avg, kw_max_avg, kw_avg_avg, shares),label_round = 2, label = "FALSE", label_size=3)
```

<img src="data_channel_is_socmedAnalysis_files/figure-gfm/tab5-1.png" style="display: block; margin: auto;" />
The correlation chart above does not show any strong relationships
between shares and the variables displayed. We can see many other
relationships which is indicated by the darker orange and blue colors.
We would expect to use this level of correlation for this grouping of
variables.

``` r
ggcorr(selectTrain%>% dplyr::select(self_reference_min_shares, self_reference_max_shares, self_reference_avg_sharess, global_subjectivity, global_sentiment_polarity, global_rate_positive_words, global_rate_negative_words, shares),label_round = 2, label = "FALSE", label_size=3)
```

<img src="data_channel_is_socmedAnalysis_files/figure-gfm/tab6-1.png" style="display: block; margin: auto;" />
The correlation chart above does not show any strong relationships
between shares and the variables displayed. We can see many other
relationships which is indicated by the darker orange and blue colors.

``` r
ggcorr(selectTrain%>% dplyr::select(LDA_00, LDA_01, LDA_02, LDA_03, LDA_04,  shares),label_round = 2, label = "FALSE", label_size=3)
```

<img src="data_channel_is_socmedAnalysis_files/figure-gfm/tab7-1.png" style="display: block; margin: auto;" />
The correlation chart above does not show any strong relationships
between shares and the variables displayed. We can see many other
variables above have a mild relationship to one another.

``` r
ggcorr(selectTrain%>% dplyr::select(rate_positive_words, rate_negative_words, avg_positive_polarity, min_positive_polarity, max_positive_polarity,  avg_negative_polarity, min_negative_polarity, max_negative_polarity, shares),label_round = 2, label = "FALSE", label_size=3)
```

<img src="data_channel_is_socmedAnalysis_files/figure-gfm/tab8-1.png" style="display: block; margin: auto;" />
The correlation chart above does not show any strong relationships
between shares and the variables displayed. We can see many other
relationships which is indicated by the darker orange and blue colors.

``` r
ggcorr(selectTrain%>% dplyr::select( title_subjectivity, title_sentiment_polarity, abs_title_subjectivity, abs_title_sentiment_polarity, shares),label_round = 2, label = "FALSE")
```

<img src="data_channel_is_socmedAnalysis_files/figure-gfm/tab9-1.png" style="display: block; margin: auto;" />
The correlation chart above does not show any strong relationships
between shares and the variables displayed. We can see many other
relationships which is indicated by the darker orange and blue colors.

Looking at the overall results from the `ggcorr()`, we do not see any
highly correlated relationships(positive or negative) with our response
variable, share. If there was a relationship, it would be orange for
highly positive correlated or blue for highly negative correlated. We do
notice that many of the variables are highly correlated with one
another.

For the custom limitations of correlation maps above, the decision was
made not to include the actual correlation value. Even with the attempt
to shorten the label for the variable, the maps were cluttered and busy.

## More EDA.

The above observations have been simplified further by categorizing the
response variable shares into ‘Poor’and ’Good’. This gives information,
how data is split between the quantiles of the training data. For
`params$channel` , the shares below Q2 are grouped Poor and above Q2 as
Good The summary statistics can be seen for each category easily and
deciphered at more easily here.

``` r
q<-quantile(selectTrain$shares,0.5)
cat_share<-selectTrain%>% 
  mutate(Rating=ifelse(shares<q,"Poor","Good"))

head(cat_share,5)
```

    ## # A tibble: 5 x 49
    ##   n_tokens_ti~1 n_tok~2 n_uni~3 n_non~4 n_non~5 num_h~6 num_s~7 num_i~8 num_v~9 avera~* num_k~* kw_mi~* kw_ma~* kw_av~*
    ##           <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>
    ## 1             8     257   0.568    1.00   0.671       9       7       0       1    4.64       9       0       0       0
    ## 2             8     218   0.663    1.00   0.688      14       3      11       0    4.44      10       0       0       0
    ## 3             9    1226   0.410    1.00   0.617      10      10       1       1    4.39       7       0       0       0
    ## 4             9     168   0.778    1.00   0.865       6       4      11       0    4.68       9     217     690     572
    ## 5             7     518   0.486    1.00   0.654      11       1       1       0    4.79       7     217    4800    1480
    ## # ... with 35 more variables: kw_min_max <dbl>, kw_max_max <dbl>, kw_avg_max <dbl>, kw_min_avg <dbl>,
    ## #   kw_max_avg <dbl>, kw_avg_avg <dbl>, self_reference_min_shares <dbl>, self_reference_max_shares <dbl>,
    ## #   self_reference_avg_sharess <dbl>, is_weekend <dbl>, LDA_00 <dbl>, LDA_01 <dbl>, LDA_02 <dbl>, LDA_03 <dbl>,
    ## #   LDA_04 <dbl>, global_subjectivity <dbl>, global_sentiment_polarity <dbl>, global_rate_positive_words <dbl>,
    ## #   global_rate_negative_words <dbl>, rate_positive_words <dbl>, rate_negative_words <dbl>,
    ## #   avg_positive_polarity <dbl>, min_positive_polarity <dbl>, max_positive_polarity <dbl>,
    ## #   avg_negative_polarity <dbl>, min_negative_polarity <dbl>, max_negative_polarity <dbl>, ...

#### Summary Statistics.

The most basic statistic is assessed here, which is mean across the 2
categories. Not all variables can be informative but the effect of each
parameter on the Rating can be seen here. This depicts how each
parameter on average has contributed to the shares for each category.

``` r
library(gt)

cat_share$Rating=as.factor(cat_share$Rating)

means1<-aggregate(cat_share,by=list(cat_share$Rating),mean)
means2<-cat_share %>% group_by(Rating,is_weekend) %>% summarise_all('mean')

means1
```

    ##   Group.1 n_tokens_title n_tokens_content n_unique_tokens n_non_stop_words n_non_stop_unique_tokens num_hrefs
    ## 1    Good       9.549296         640.6021       0.5157885        0.9964789                0.6648345  12.93075
    ## 2    Poor       9.684278         564.5168       0.5593035        0.9935567                0.7051679  12.81186
    ##   num_self_hrefs num_imgs num_videos average_token_length num_keywords kw_min_min kw_max_min kw_avg_min kw_min_max
    ## 1       4.868545 3.900235   1.051643             4.634658     6.755869   38.87559   1293.461   406.6289   16291.81
    ## 2       4.398196 4.631443   1.371134             4.629494     6.372423   33.41624   1121.447   376.7867   35046.64
    ##   kw_max_max kw_avg_max kw_min_avg kw_max_avg kw_avg_avg self_reference_min_shares self_reference_max_shares
    ## 1   712438.1   220996.6   1366.050   5848.025   3344.346                  6587.901                 20773.004
    ## 2   730302.1   233653.5   1285.544   5058.638   3104.697                  3567.133                  9935.586
    ##   self_reference_avg_sharess is_weekend    LDA_00     LDA_01    LDA_02    LDA_03    LDA_04 global_subjectivity
    ## 1                  11476.774  0.1584507 0.4653380 0.06433426 0.1668529 0.1540292 0.1494457           0.4517152
    ## 2                   5603.816  0.1095361 0.3171477 0.09442593 0.2087137 0.2155877 0.1641250           0.4698735
    ##   global_sentiment_polarity global_rate_positive_words global_rate_negative_words rate_positive_words
    ## 1                 0.1403779                 0.04726337                 0.01558535           0.7466636
    ## 2                 0.1496080                 0.04527639                 0.01576863           0.7390241
    ##   rate_negative_words avg_positive_polarity min_positive_polarity max_positive_polarity avg_negative_polarity
    ## 1           0.2498153             0.3475203            0.06540418             0.7822219            -0.2610838
    ## 2           0.2532440             0.3701800            0.09388684             0.7862046            -0.2544711
    ##   min_negative_polarity max_negative_polarity title_subjectivity title_sentiment_polarity abs_title_subjectivity
    ## 1            -0.5372029            -0.1104216          0.2633455               0.09468802              0.3576558
    ## 2            -0.5031378            -0.1152319          0.2555671               0.10055084              0.3459604
    ##   abs_title_sentiment_polarity   shares channel weekday Rating
    ## 1                    0.1537448 5524.883      NA      NA     NA
    ## 2                    0.1559767 1334.210      NA      NA     NA

``` r
means2
```

    ## # A tibble: 4 x 49
    ## # Groups:   Rating [2]
    ##   Rating is_weekend n_tokens_~1 n_tok~2 n_uni~3 n_non~4 n_non~5 num_h~6 num_s~7 num_i~8 num_v~9 avera~* num_k~* kw_mi~*
    ##   <fct>       <dbl>       <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>
    ## 1 Good            0        9.53    615.   0.519   0.997   0.667    11.7    4.30    3.63   1.10     4.63    6.61    36.6
    ## 2 Good            1        9.67    775.   0.499   0.993   0.655    19.3    7.89    5.32   0.793    4.64    7.56    51.1
    ## 3 Poor            0        9.70    560.   0.561   0.993   0.706    12.6    4.05    4.54   1.43     4.62    6.30    31.4
    ## 4 Poor            1        9.56    602.   0.546   1.00    0.700    14.2    7.2     5.35   0.894    4.69    6.93    49.8
    ## # ... with 35 more variables: kw_max_min <dbl>, kw_avg_min <dbl>, kw_min_max <dbl>, kw_max_max <dbl>,
    ## #   kw_avg_max <dbl>, kw_min_avg <dbl>, kw_max_avg <dbl>, kw_avg_avg <dbl>, self_reference_min_shares <dbl>,
    ## #   self_reference_max_shares <dbl>, self_reference_avg_sharess <dbl>, LDA_00 <dbl>, LDA_01 <dbl>, LDA_02 <dbl>,
    ## #   LDA_03 <dbl>, LDA_04 <dbl>, global_subjectivity <dbl>, global_sentiment_polarity <dbl>,
    ## #   global_rate_positive_words <dbl>, global_rate_negative_words <dbl>, rate_positive_words <dbl>,
    ## #   rate_negative_words <dbl>, avg_positive_polarity <dbl>, min_positive_polarity <dbl>, max_positive_polarity <dbl>,
    ## #   avg_negative_polarity <dbl>, min_negative_polarity <dbl>, max_negative_polarity <dbl>, ...

And scatter plots are generated to get an idea on the direction of the
predictor variables and to see what is the direction and extent of
linearity or non-linearity of the predictors.

``` r
q<-quantile(cat_share$shares,c(0.25,0.75))

name<-names(cat_share)
predictors1<-(name)[30:35]
predictors2<-(name) [36:41]
predictors3<-(name) [42:46]
response <- "Rating"

par(mfrow = c(3, 1))

cat_share %>% 
  select(c(response, predictors1)) %>% 
    select_if(~ !any(is.na(.))) %>%   
      ggpairs(aes(colour = Rating,alpha=0.5),
              upper = list(continuous = wrap("cor", title="")))
```

<img src="data_channel_is_socmedAnalysis_files/figure-gfm/cat3-1.png" style="display: block; margin: auto;" />

``` r
cat_share %>% 
  select(c(response, predictors2)) %>% 
    select_if(~ !any(is.na(.))) %>%   
      ggpairs(aes(colour = Rating,alpha=0.5),
              upper = list(continuous = wrap("cor", title="")))
```

<img src="data_channel_is_socmedAnalysis_files/figure-gfm/cat3-2.png" style="display: block; margin: auto;" />

``` r
cat_share %>% 
  select(c(response, predictors3)) %>% 
    select_if(~ !any(is.na(.))) %>%   
      ggpairs(aes(colour = Rating,alpha=0.5),
              upper = list(continuous = wrap("cor", title="")))
```

<img src="data_channel_is_socmedAnalysis_files/figure-gfm/cat3-3.png" style="display: block; margin: auto;" />

The next aspect is to check the shape of some predictor variables w.r.t
response variable. These mostly are parameters like number of keywords,
videos, pictures etc. This specifies if the channel shares are skewed by
some these factors and at what extent.For example,shares can be maximum,
if the channel has more kewywords and hence, we see a right skewed bar
plot.

``` r
bar_dat<-cat_share %>% select(c('n_tokens_title','n_tokens_content','shares',
                                'num_self_hrefs','num_imgs','num_videos','num_keywords',
                                'Rating'))

ggplot(bar_dat,aes())+
geom_density(aes(x = shares,fill=Rating,alpha=0.5))
```

<img src="data_channel_is_socmedAnalysis_files/figure-gfm/cat4-1.png" style="display: block; margin: auto;" />

``` r
g1<-ggplot(bar_dat,aes())+
  geom_bar(aes(x = n_tokens_title, y = shares, fill = Rating),stat = 'identity', position = "dodge")
g2<-ggplot(bar_dat,aes())+
  geom_bar(aes(x = num_self_hrefs, y = shares, fill = Rating),stat = 'identity', position = "dodge")
g3<-ggplot(bar_dat,aes())+
  geom_bar(aes(x = num_imgs, y = shares, fill = Rating),stat = 'identity', position = "dodge")
g4<-ggplot(bar_dat,aes())+
  geom_bar(aes(x = num_videos, y = shares, fill = Rating),stat = 'identity', position = "dodge")
g5<-ggplot(bar_dat,aes())+
  geom_bar(aes(x = num_keywords, y = shares, fill = Rating),stat = 'identity', position = "dodge")
g6<-ggplot(bar_dat,aes())+
  geom_boxplot(aes(y = shares))+coord_flip()
grid.arrange(g1, g2, g3,g4,g5,g6, ncol = 2, nrow = 3)
```

<img src="data_channel_is_socmedAnalysis_files/figure-gfm/cat4-2.png" style="display: block; margin: auto;" />

#### Exploratory Data Analysis (EDA) with Principal Component Analysis (PCA).

A different aspect of EDA that is tried here. Although, we are uncertain
if this will be a good exercise or not. But we do see the data set has
too many predictor variables and in any industry cost of computations
and labor counts.

In order to find more efficient way to reduce size of data set and
understand variance between variables,we are using Principal Component
Analysis on training data to see, if the the predictors variables can be
used together to determine some relationship with the response variable.
Principal Component is a method that has many usages for datasets with
too many predictors. It is used in dimensionality reduction, and can
also help in understanding the relationship among different variables
and their impact on the response variable in terms of variance and also
an effective method to remove collinearity from the dataset by removing
highly correlated predictors.

Here we are using `prcomp` on our train data to find our principal
components. Some select variables have been discarded as their magnitude
was very less and may /may not have impact into the model but to keep
the analysis simple , some critical ones have been considered to
demonstrate dimensionality reduction.

The PCA plot is plot for variance, another statistic that is critical in
understanding relationship between variables. These plots can easily
tell how much each variable contributes towards the response variable.
PC1 holds the maximum variance , PC2 then the next. We have removed some
irrelevant variables and tested the model here..

``` r
PC_Train<-selectTrain %>% select(- c('shares','channel','LDA_00','LDA_01','LDA_02','LDA_03','LDA_04','n_tokens_content','n_non_stop_unique_tokens','weekday'))

PC_fit <- prcomp(PC_Train,  scale = TRUE)
summary(PC_fit)
```

    ## Importance of components:
    ##                           PC1     PC2     PC3     PC4    PC5     PC6     PC7    PC8     PC9    PC10    PC11    PC12
    ## Standard deviation     2.0242 1.90383 1.79916 1.73317 1.5932 1.56105 1.47377 1.2901 1.27178 1.23855 1.10694 1.05316
    ## Proportion of Variance 0.1078 0.09538 0.08518 0.07905 0.0668 0.06413 0.05716 0.0438 0.04256 0.04037 0.03224 0.02919
    ## Cumulative Proportion  0.1078 0.20321 0.28839 0.36744 0.4342 0.49837 0.55553 0.5993 0.64189 0.68226 0.71450 0.74369
    ##                           PC13    PC14   PC15    PC16   PC17    PC18    PC19    PC20    PC21    PC22    PC23    PC24
    ## Standard deviation     1.01450 0.96852 0.9328 0.90428 0.8340 0.77991 0.75979 0.75314 0.72244 0.71317 0.65611 0.57321
    ## Proportion of Variance 0.02708 0.02468 0.0229 0.02152 0.0183 0.01601 0.01519 0.01493 0.01373 0.01338 0.01133 0.00865
    ## Cumulative Proportion  0.77078 0.79546 0.8184 0.83988 0.8582 0.87419 0.88938 0.90431 0.91804 0.93143 0.94276 0.95140
    ##                           PC25    PC26    PC27    PC28    PC29    PC30    PC31    PC32    PC33    PC34    PC35    PC36
    ## Standard deviation     0.52513 0.52271 0.48362 0.44930 0.44303 0.42147 0.35475 0.33154 0.27759 0.24949 0.22307 0.18741
    ## Proportion of Variance 0.00726 0.00719 0.00615 0.00531 0.00517 0.00467 0.00331 0.00289 0.00203 0.00164 0.00131 0.00092
    ## Cumulative Proportion  0.95866 0.96585 0.97200 0.97732 0.98248 0.98716 0.99047 0.99336 0.99539 0.99703 0.99834 0.99926
    ##                           PC37    PC38
    ## Standard deviation     0.12199 0.11509
    ## Proportion of Variance 0.00039 0.00035
    ## Cumulative Proportion  0.99965 1.00000

``` r
par(mfrow = c(4, 1))

 #screeplot
screeplot(PC_fit, type = "bar")

#Proportion of Variance
plot(PC_fit$sdev^2/sum(PC_fit$sdev^2), xlab = "Principal Component",
ylab = "Proportion of Variance Explained", ylim = c(0, 1), type = 'b')

# Cumulative proportion of variance.
plot(cumsum(PC_fit$sdev^2/sum(PC_fit$sdev^2)), xlab = "Principal Component",
ylab = "Cum. Prop of Variance Explained", ylim = c(0, 1), type = 'b')
```

<img src="data_channel_is_socmedAnalysis_files/figure-gfm/pca1-1.png" style="display: block; margin: auto;" />

# Variable Selection by Correlation Results

Before using any reduction algorithm to determine which variables to be
in the model, we used the correlation tables above to reduce the
predictors. If the predictors are correlated, it’s not necessary to use
both predictors.

We removed any variable that contained min and max, if the predictor
also contained age:

    min_positive_polarity   
    max_positive_polarity   
    min_negative_polarity   
    max_negative_polarity   
    self_reference_min_shares   
    self_reference_max_shares   
    kw_min_avg   
    kw_max_avg   
    kw_min_max   
    kw_max_max   
    kw_min_min  
    kw_max_min  

We removed all of the “LDA\_”. When you view the correlation chart
above, they seem to be correlated. In addition, I am not sure what LDA
means. When googled, the results referenced modeling and Latent
Dirichlet Allocation.

    LDA_00: Closeness to LDA topic 0
    LDA_01: Closeness to LDA topic 1
    LDA_02: Closeness to LDA topic 2
    LDA_03: Closeness to LDA topic 3
    LDA_04: Closeness to LDA topic 4

We removed the following variables, because the data contains the
absolute value of the same information:

    title_subjectivity
    title_sentiment_polarity

We removed the following variables, because of their strong relationship
between global_rate_positive_words and global_rate_negative_words:

    global_subjectivity
    global_sentiment_polarity

We removed the following variables, because of their strong relationship
between n_tokens_content:

    n_unique_tokens
    n_non_stop_unique_tokens
    n_non_stop_words

The final data set will contain the following columns (features):

    n_tokens_title
    n_tokens_content
    num_hrefs
    num_self_hrefs
    num_imgs
    num_videos
    average_token_length
    num_keywords
    kw_avg_min
    kw_avg_max
    kw_avg_avg
    self_reference_avg_sharess
    global_rate_positive_words
    global_rate_negative_words
    rate_positive_words
    rate_negative_words
    avg_positive_polarity
    avg_negative_polarity
    abs_title_subjectivity
    abs_title_sentiment_polarity

Out of the original 61 variables, we will be looking at using up to 20
variables. There is a relationship between num_hrefs and num_self_hrefs.
One of those may be removed later in the analysis. We decided to use
best stepwise to further reduce the number of variables in the model.

We will also run a separate linear regression model using the variables
selected from visually correlation results (above).

# Linear Regression

Linear regression is a method to understand the relationship between a
response variables Y and one or more predictor variables x, x1, x2, etc.
This method creates a line that best fits the data called “least known
regression line”.

For a Simple Linear Regression, we have one response variable and one
predictor variables. It uses the formula y = b0 + b1x, where

    y:  response variable
    b0: intercept (baseline value, if all other values are zero)
    b1: regression coefficient
    x:  independent variable

For Multiple Linear Regression, we have one response variable and any
number of predictor variables. It uses the formula Y = B0 + B1X1 +
B2X2 + … + BpXp + E, where

    Y: The response variable
    Xp: The pth predictor variable
    Bp: The average effect on Y of a one unit increase in Xp, holding all other predictors fixed
    E: The error term

For the results of a linear regression model to be valid and reliable,
the following must be true

    1. Linear relationship: There exists a linear relationship between the independent variable, x, and the dependent variable, y. 
    2. Independence: The residuals are independent. 
    3. Homoscedasticity: The residuals have constant variance at every level of x.
    4. Normality: The residuals of the model are normally distributed.

## Linear Regression using Best Stepwise

The Best Stepwise method combines the backward and forward step-wise
methods to find the best result. We start with no predictor then
sequentially add the predictors that contribute the most (Forward).
However after adding each new variables, it will remove any variables
that no longer improve the fit of the model.

After running the Best Stepwise method, we applied the results to the
test data to determine how well the model performed.

``` r
set.seed(21)

lmFit<- train(shares ~ ., data = selectTrain%>% dplyr::select(shares, n_tokens_title,
n_tokens_content,
num_hrefs,
num_self_hrefs,
num_imgs,
num_videos,
average_token_length,
num_keywords,
kw_avg_min,
kw_avg_max,
kw_avg_avg,
self_reference_avg_sharess,
global_rate_positive_words,
global_rate_negative_words,
rate_positive_words,
rate_negative_words,
avg_positive_polarity,
avg_negative_polarity,
abs_title_subjectivity,
abs_title_sentiment_polarity),
method = 'leapSeq',
preProcess = c("center", "scale"),
tuneGrid  = data.frame(nvmax = 1:20),
trControl = trainControl(method = "cv", number = 5)
)

lmFit$results
```

    ##    nvmax     RMSE    Rsquared      MAE   RMSESD  RsquaredSD    MAESD
    ## 1      1 4900.521 0.009470799 2513.782 610.5379 0.008684540 143.5417
    ## 2      2 4862.734 0.032178236 2485.034 615.4258 0.034234769 153.9710
    ## 3      3 4894.243 0.018407380 2499.764 637.3418 0.013361279 148.5270
    ## 4      4 4902.206 0.014474113 2511.203 636.2796 0.009469377 154.2800
    ## 5      5 4907.344 0.016329842 2505.766 660.9088 0.010465191 173.6946
    ## 6      6 4887.841 0.011190208 2521.482 632.0651 0.011333754 159.7106
    ## 7      7 4909.349 0.014538826 2510.741 643.5759 0.007900605 148.1696
    ## 8      8 4880.588 0.021036437 2494.646 656.5868 0.017125282 177.1553
    ## 9      9 4916.379 0.014216227 2509.825 668.5570 0.009355317 146.5803
    ## 10    10 4902.282 0.018451114 2493.541 685.5217 0.017713079 185.3806
    ## 11    11 4883.695 0.024908367 2481.286 684.1737 0.016442039 188.6472
    ## 12    12 4894.270 0.023596591 2484.322 697.1670 0.017558918 200.6780
    ## 13    13 4886.319 0.025661649 2480.873 694.4023 0.018318116 197.1002
    ## 14    14 4874.746 0.027503762 2472.487 685.1046 0.018012723 188.4650
    ## 15    15 4878.319 0.027504429 2462.946 688.7328 0.017281277 173.5550
    ## 16    16 4887.559 0.025338866 2478.618 698.1525 0.017711957 187.3610
    ## 17    17 4886.130 0.026488449 2474.127 701.0749 0.018103061 189.4969
    ## 18    18 4891.055 0.024573880 2471.223 701.3404 0.017133979 189.9564
    ## 19    19 4890.441 0.025998125 2480.904 705.3105 0.019328787 185.2135
    ## 20    20 4886.878 0.026581277 2474.104 702.7556 0.018027662 190.8529

``` r
lmFit$bestTune
```

    ##   nvmax
    ## 2     2

``` r
set.seed(21)

btTrain <- selectTrain%>% dplyr::select(num_videos, n_tokens_content,kw_avg_max, kw_avg_avg, rate_positive_words,shares)
                                 
btTest <- selectTest%>% dplyr::select(num_videos, n_tokens_content,kw_avg_max, kw_avg_avg, rate_positive_words,shares)

lmFit2<- train(shares ~ ., data = btTrain,
method = 'lm',
trControl = trainControl(method = "cv", number = 5)
)

lmFit2$results
```

    ##   intercept     RMSE   Rsquared      MAE   RMSESD RsquaredSD    MAESD
    ## 1      TRUE 4861.552 0.02648816 2509.586 635.8092 0.02590999 146.8222

``` r
predBest <- predict(lmFit2, newdata = btTest)  %>% as_tibble()
LinearRegression_1 <- postResample(predBest, obs =btTest$shares)
```

## Linear Regression on Original Variable Selection

As a comparison, we also included a model that includes the variables
selected from correlation heat maps.

``` r
set.seed(21)

stTrain <- selectTrain%>% dplyr::select(shares, n_tokens_title,
n_tokens_content,
num_hrefs,
num_self_hrefs,
num_imgs,
num_videos,
average_token_length,
num_keywords,
kw_avg_min,
kw_avg_max,
kw_avg_avg,
self_reference_avg_sharess,
global_rate_positive_words,
global_rate_negative_words,
rate_positive_words,
rate_negative_words,
avg_positive_polarity,
avg_negative_polarity,
abs_title_subjectivity,
abs_title_sentiment_polarity)

                                 
stTest <- selectTest%>% dplyr::select(shares, n_tokens_title,
n_tokens_content,
num_hrefs,
num_self_hrefs,
num_imgs,
num_videos,
average_token_length,
num_keywords,
kw_avg_min,
kw_avg_max,
kw_avg_avg,
self_reference_avg_sharess,
global_rate_positive_words,
global_rate_negative_words,
rate_positive_words,
rate_negative_words,
avg_positive_polarity,
avg_negative_polarity,
abs_title_subjectivity,
abs_title_sentiment_polarity)

lmFit3<- train(shares ~ ., data = stTrain,
method = 'lm',
trControl = trainControl(method = "cv", number = 5)
)

lmFit3$results
```

    ##   intercept     RMSE   Rsquared      MAE   RMSESD RsquaredSD    MAESD
    ## 1      TRUE 4886.878 0.02658128 2474.104 702.7556 0.01802766 190.8529

``` r
OrgBest <- predict(lmFit3, newdata = stTest)  %>% as_tibble()
LinearRegression_3 <- postResample(predBest, obs = stTest$shares)
```

## Principal Component selections and Linear Regression with PCA.

This chunk of code below selects the principal components from the above
PCA code for EDA based on the desired cumulative variance threshold. As
there are too many predictor variables, we are going ahead with a
threshold of 0.80 i.e. we are taking all predictors that contribute to
total 80% of the data. The number of principal components selected can
be seen below after the code is implemented.

These principal components selected will be used below in the linear
regression model after feeding our PCA fit into the training and test
dataset.

``` r
var_count<-cumsum(PC_fit$sdev^2/sum(PC_fit$sdev^2))

count=0

for (i in (1:length(var_count)))
{
  if (var_count[i]<0.80)
  { per=var_count[i]
    count=count+1
    }
  else if (var_count[i]>0.80)
  {
    break}
}

print(paste("Number of Principal Components optimized are",count ,"for cumuative variance of 0.80"))
```

    ## [1] "Number of Principal Components optimized are 14 for cumuative variance of 0.80"

The aim here is to test if PCA helps us reduce the dimensions of our
data and utilize this on our regression model. The PCA is very efficient
method of feature extraction and can help get maximum information at
minimum cost. We will check out a multinomial logistic regression with
PCA here. The principal components obtained in the above section are
used here on train and test data to get PCA fits and then apply linear
regression fit on train data and get predictions for test data. It is
not expected this may give a very efficient result however, it does help
get idea about the predictor variables whether they can be fit enough
for a linear model or not.

``` r
new_PC <- predict(PC_fit, newdata = PC_Train) %>% as_tibble() 
res<-selectTrain$shares
PC_Vars<-data.frame(new_PC[1:count],res)
head(PC_Vars,5)
```

    ##           PC1        PC2      PC3        PC4       PC5        PC6          PC7         PC8         PC9       PC10
    ## 1 -2.25831535  0.1443198 3.087125 -1.3344084 2.2693291 -0.3494869 -1.519287493  0.43772792 -0.08976523  0.8555012
    ## 2  2.94227708 -1.0172897 2.648682 -0.2648289 2.1949395  0.6063529 -1.846356426  0.84397039  0.71347974 -0.4656442
    ## 3 -0.83517181 -0.4324806 2.469767 -2.4843448 0.3959718 -1.4162355 -0.003274301 -0.02502814  0.52196264 -1.3074378
    ## 4  0.06496609  2.5092579 4.391824  0.2938474 3.5065721 -1.5023884 -1.822506013 -0.34710824  0.31851636 -1.0942802
    ## 5  2.32640469 -0.1443404 1.303032 -1.5137287 2.0027051 -2.6247896  0.084454572 -1.27289531  0.72359671 -2.2355303
    ##        PC11        PC12       PC13         PC14  res
    ## 1 0.2592595  0.21139613 -0.1892965  0.075421497 2600
    ## 2 0.1613024 -0.56998976 -0.3458020  0.554146332  690
    ## 3 0.7063446 -0.46298921 -0.2302499 -0.084786430 4800
    ## 4 2.0632708  0.44093802 -0.4244635  0.001576401 4800
    ## 5 2.1480909  0.08359695 -0.9122579  0.797064965  775

``` r
fit <- train(res~ .,method='lm', data=PC_Vars,metric='RMSE')

#apply PCA fit to test data
PC_Test<-selectTest %>% select(- c('shares','channel','LDA_00','LDA_01',
                                   'LDA_02','LDA_03','LDA_04','n_tokens_content',
                                   'n_non_stop_unique_tokens','weekday'))

test_PC<-predict(PC_fit,newdata=PC_Test) %>% as_tibble()
test_res<-selectTest$shares
PC_Vars_Test<-data.frame(test_PC[1:count],test_res)
head(PC_Vars_Test,5)
```

    ##          PC1        PC2        PC3       PC4        PC5        PC6         PC7        PC8        PC9       PC10
    ## 1  0.8829171 -0.7463250  2.3159532 -1.919057 0.04312915 -1.9166453  0.11882567 -0.9235891 -0.4403478 -0.8621198
    ## 2  0.8005027 -4.8749409 -0.5803859 -2.411211 2.36610355  0.2886239 -2.01838603 -2.0917744  0.2849453  2.3279490
    ## 3  1.1563712 -0.7457301  1.6278077 -3.107106 1.02403948 -1.1932993  0.03863872 -1.7375924  0.9491143 -0.7922814
    ## 4 -1.8775547 -0.8014410  0.7908008 -2.388952 2.05060986 -1.5561443 -0.88284130 -1.8455138  0.8522756 -0.8168667
    ## 5  2.7401982 -1.9196280  0.3074097 -1.586252 3.25131473  0.9607719 -2.13794739 -1.3391392  0.1822325  0.5158230
    ##        PC11       PC12       PC13        PC14 test_res
    ## 1 1.0928159 -0.2916770 -0.3063492 -0.46620093      851
    ## 2 0.4335492  0.7272938 -0.3946410 -0.47455916     9200
    ## 3 1.3306845 -0.2647594 -0.2695200 -0.78764467     1600
    ## 4 0.9352038  0.2357450 -0.9907427 -0.08705472    18200
    ## 5 0.2582666 -0.1402293  2.9706055 -0.09122512     1700

``` r
lm_predict<-predict(fit,newdata=PC_Vars_Test)%>% as_tibble()
LinearRegression_2<-postResample(lm_predict, obs = PC_Vars_Test$test_res)
```

# Ensemble Methods

Ensemble methods is a machine learning models that combines several base
models in order to produce one optimal predictive model. A step further
to the decision trees, Ensemble methods do not rely on one Decision Tree
and hope to right decision at each split, they take a sample of Decision
Trees, calculate metrics at each split, and make a final predictor based
on the aggregated results of the sampled Decision Trees.

## Random Forest

Random forest is a decision tree method. It takes a desired number of
bootstrapped samples from the data and builds a tree for each sample.
While building the trees, if a split occurs, then only a random sample
of trees are averaged. The average of predictions are used to determine
the final model.

The random forest model using R is created using `rf` as the method and
`mtry` for the tuning grid. The tuning grid will provide results using 1
variable up to nine variables in the model.

``` r
trainCtrl <- trainControl(method = 'cv', number = 5)
#trainCtrl <- trainControl(method = 'repeatedcv', number = 3, repeats =  1)

rfTrain <- selectTrain %>% dplyr::select(num_videos, num_imgs,num_keywords,
                                   global_subjectivity,global_sentiment_polarity,
                                   global_rate_positive_words,is_weekend,
                                   avg_positive_polarity,title_sentiment_polarity,shares)

rfTest <- selectTest %>% dplyr::select(num_videos, num_imgs,num_keywords,
                                   global_subjectivity,global_sentiment_polarity,
                                   global_rate_positive_words,is_weekend,
                                   avg_positive_polarity,title_sentiment_polarity,shares)

rfFit <- train(shares ~. , data = rfTrain,
method = 'rf',
trControl = trainCtrl,
preProcess = c("center", "scale"),
tuneGrid  = data.frame(mtry = 1:9)    
)


rfPredict<-predict(rfFit,newdata=rfTest) %>% as_tibble()
Random_Forest<-postResample(rfPredict, obs = rfTest$shares)
```

## Boosted Trees

Boosting involves adding ensemble members in sequential manner so that
corrections are applied to the predictions made by prior models and
outputs a weighted average of these predictions. It is different to
random forest in implementation of decision tree split .That is, in this
case the decision trees are built in additive manner not in parallel
manner as in random forest. It is based on weak learners which is high
bias and low variance. Boosting looks mainly into reducing bias (and
eventually variance, by aggregating the output metric from many models.
We have used some cross validation here to tune the parameters. The
tuning parameters have been kept small for the `tuneGrid` due to limited
processing speed. The method `caret` uses to fit the train data is`gbm`.
The fit is plotted using `plot()` here to show the results of cross
validation which may get complex to interpret, but we are not worrying
about it as `caret` does the work of selecting the best parameters and
allocating them to our model fit that can be used directly into
predictions.

``` r
BT_Train<-selectTrain %>% select(c('num_videos', 'num_imgs','num_keywords',
                                   'global_subjectivity','global_sentiment_polarity',
                                   'global_rate_positive_words','is_weekend',
                                   'avg_positive_polarity','title_sentiment_polarity','shares'))

BT_Test<-selectTest %>% select(c('num_videos', 'num_imgs','num_keywords',
                                   'global_subjectivity','global_sentiment_polarity',
                                   'global_rate_positive_words','is_weekend',
                                   'avg_positive_polarity','title_sentiment_polarity','shares'))

boostfit <- train(shares ~., data = BT_Train, method = "gbm",
trControl = trainControl(method = "repeatedcv", number = 5,
                         repeats = 3,verboseIter = FALSE,allowParallel = TRUE),
preProcess = c("scale"),
tuneGrid = expand.grid(n.trees = c(50,100,200),
           interaction.depth = 1:5,
           shrinkage = c(0.001,0.1),
           n.minobsinnode = c(3,5)),
           verbose = FALSE)

plot(boostfit)
```

<img src="data_channel_is_socmedAnalysis_files/figure-gfm/boost-1.png" style="display: block; margin: auto;" />

``` r
boostfit$bestTune
```

    ##    n.trees interaction.depth shrinkage n.minobsinnode
    ## 32     100                 1       0.1              3

``` r
boostpredict<-predict(boostfit,newdata=BT_Test) %>% as_tibble()
Boosted_Trees<-postResample(boostpredict, obs = BT_Test$shares)
```

# Model Comparisons.

Linear Regression and Ensemble method results are now put together here
for comparison.

Some metrics come up as we see the `postResample` results after doing
our fit on testdata that takes not of these metrics after comparison
with actual test data values.

The **RMSE** or Root Mean Square Error is the square root of the
variance of the residual error. It tells about the absolute fit of the
model to the data–how close the true data values are to the model’s
predicted values.

**R-squared** is the proportion of variance in the response variable
that can be explained by the predictor variables. In other words, this
is a measurement of goodness of fit. A high R-squared means our model
has good fit over the training /test data

**MAE** or Mean Absolute Error gives the absolute distance of the
observation to the predictions and averaging it over all the given
observations.

``` r
# Some nomenclature things for sanity.

LinearRegression_Stepwise_Sel<-LinearRegression_1  
LinearRegression_Orignal_Var_Sel<-LinearRegression_3   
Linear_Regression_PCA<-LinearRegression_2 

# Getting out best model in terms of RMSE
Metrics_df<-data.frame(rbind(LinearRegression_Stepwise_Sel,
                             LinearRegression_Orignal_Var_Sel,
                             Linear_Regression_PCA, Boosted_Trees, 
                             Random_Forest))
Metrics_df
```

    ##                                      RMSE    Rsquared      MAE
    ## LinearRegression_Stepwise_Sel    6800.929 0.009716788 2830.137
    ## LinearRegression_Orignal_Var_Sel 6800.929 0.009716788 2830.137
    ## Linear_Regression_PCA            6805.452 0.009628168 2821.958
    ## Boosted_Trees                    6806.565 0.009257887 2759.865
    ## Random_Forest                    6840.866 0.001450501 2791.482

``` r
best_model<-Metrics_df[which.min(Metrics_df$RMSE), ]
rmse_min<-best_model$RMSE
r2max<-best_model$Rsquared
```

The best model for predicting the number of shares for dataset called
data_channel_is_socmed in terms of RMSE is LinearRegression_Stepwise_Sel
with RMSE of 6800.93. This model also showed an R-Squared of about
0.0097168, which shows that the predictors of this model do not have
significant impact on the response variable.

The prediction spread of all the models is also depicted through the box
plot with comparison to the actual test values. As we the test actual
values have a outlier going very high. To see more improvements in the
Linear Regression models we can also remove the outliers in the train
and test data and repeat our regression steps. Linear regression may
show more bias due to outliers in the data and hence, data cleaning is
the concept that comes up often while working with complex dataset as
part of pre-processing. However, this needs to cautiously used as
valuable information may be lost in this exercise which may change the
outcomes of our prediction. However, this box plot is simply to
summarize the test predict values as its too hard to print so much of
data!.

``` r
box_df<-data.frame(Actual_Response=selectTest$shares,
                  LinearRegression_Stepwise=predBest$value,
                  LinearRegression_Orignal_Variable=OrgBest$value,
                  LinearRegression_PCA=lm_predict$value,
                  Random_Forest = rfPredict$value,
                  Boosted_Trees=boostpredict$value)

ggplot(stack(box_df), aes(x = ind, y = values)) +
  stat_boxplot(geom = "errorbar") +
  labs(x="Models", y="Response Values") +
  geom_boxplot(fill = "white", colour = "red") +coord_flip()
```

<img src="data_channel_is_socmedAnalysis_files/figure-gfm/plots-1.png" style="display: block; margin: auto;" />

# Conclusion

Different genre of news and information has significant importance in
every industry and forecasting user sentiments is one of the critical
requirements on which today’s news and information content is shaped. To
capture the dynamics of every type of population, understanding of the
parameters and their affect of online shares is very important. We hope
that this analysis becomes useful to anybody to wants to perform step by
step evaluation of such large dataset and extract critical information.
The codes meant in this analysis are meant to capture both conceptual
aspects of the analysis being done along with results. This set of
algorithm can also be applied directly to any data set that has similar
attributes and results can be quickly obtained. We hope this analysis is
helpful going forward to anybody who wants to build similar model
comparisons.

# Automation for Building Six Reports

``` r
selectID <- unique(newData$channel)  

output_file <- paste0(selectID, "Analysis.md")  

params = lapply(selectID, FUN = function(x){list(channel = x)})

reports <- tibble(output_file, params)

library(rmarkdown)

apply(reports, MARGIN = 1,
      FUN = function(x){
        render(input = "./Project_3.Rmd",
               output_format = "github_document", 
               output_file = x[[1]], 
               params = x[[2]])
      })
```
