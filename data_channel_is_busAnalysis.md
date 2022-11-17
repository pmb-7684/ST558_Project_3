Project 3: Predictive Model for `data_channel_is_bus` Analysis
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

    ## tibble [4,382 x 48] (S3: tbl_df/tbl/data.frame)
    ##  $ n_tokens_title              : num [1:4382] 9 8 13 11 10 12 6 13 9 9 ...
    ##  $ n_tokens_content            : num [1:4382] 255 397 244 723 142 444 109 306 759 233 ...
    ##  $ n_unique_tokens             : num [1:4382] 0.605 0.625 0.56 0.491 0.655 ...
    ##  $ n_non_stop_words            : num [1:4382] 1 1 1 1 1 ...
    ##  $ n_non_stop_unique_tokens    : num [1:4382] 0.792 0.806 0.68 0.642 0.792 ...
    ##  $ num_hrefs                   : num [1:4382] 3 11 3 18 2 9 3 3 17 5 ...
    ##  $ num_self_hrefs              : num [1:4382] 1 0 2 1 1 8 2 2 9 1 ...
    ##  $ num_imgs                    : num [1:4382] 1 1 1 1 1 23 1 1 1 1 ...
    ##  $ num_videos                  : num [1:4382] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ average_token_length        : num [1:4382] 4.91 5.45 4.42 5.23 4.27 ...
    ##  $ num_keywords                : num [1:4382] 4 6 4 6 5 10 6 10 6 4 ...
    ##  $ kw_min_min                  : num [1:4382] 0 0 0 0 0 0 0 217 217 217 ...
    ##  $ kw_max_min                  : num [1:4382] 0 0 0 0 0 0 0 5700 690 425 ...
    ##  $ kw_avg_min                  : num [1:4382] 0 0 0 0 0 ...
    ##  $ kw_min_max                  : num [1:4382] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_max_max                  : num [1:4382] 0 0 0 0 0 0 0 17100 17100 17100 ...
    ##  $ kw_avg_max                  : num [1:4382] 0 0 0 0 0 ...
    ##  $ kw_min_avg                  : num [1:4382] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_max_avg                  : num [1:4382] 0 0 0 0 0 ...
    ##  $ kw_avg_avg                  : num [1:4382] 0 0 0 0 0 ...
    ##  $ self_reference_min_shares   : num [1:4382] 0 0 2800 0 0 585 821 0 767 2000 ...
    ##  $ self_reference_max_shares   : num [1:4382] 0 0 2800 0 0 1600 821 0 6800 2000 ...
    ##  $ self_reference_avg_sharess  : num [1:4382] 0 0 2800 0 0 ...
    ##  $ is_weekend                  : num [1:4382] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ LDA_00                      : num [1:4382] 0.8 0.867 0.3 0.867 0.441 ...
    ##  $ LDA_01                      : num [1:4382] 0.05 0.0333 0.05 0.0333 0.04 ...
    ##  $ LDA_02                      : num [1:4382] 0.0501 0.0333 0.05 0.0333 0.2393 ...
    ##  $ LDA_03                      : num [1:4382] 0.0501 0.0333 0.05 0.0333 0.24 ...
    ##  $ LDA_04                      : num [1:4382] 0.05 0.0333 0.5497 0.0333 0.04 ...
    ##  $ global_subjectivity         : num [1:4382] 0.341 0.374 0.332 0.375 0.443 ...
    ##  $ global_sentiment_polarity   : num [1:4382] 0.1489 0.2125 -0.0923 0.1827 0.1143 ...
    ##  $ global_rate_positive_words  : num [1:4382] 0.0431 0.0655 0.0164 0.0636 0.0211 ...
    ##  $ global_rate_negative_words  : num [1:4382] 0.01569 0.01008 0.02459 0.0083 0.00704 ...
    ##  $ rate_positive_words         : num [1:4382] 0.733 0.867 0.4 0.885 0.75 ...
    ##  $ rate_negative_words         : num [1:4382] 0.267 0.133 0.6 0.115 0.25 ...
    ##  $ avg_positive_polarity       : num [1:4382] 0.287 0.382 0.292 0.341 0.367 ...
    ##  $ min_positive_polarity       : num [1:4382] 0.0333 0.0333 0.1364 0.0333 0.2 ...
    ##  $ max_positive_polarity       : num [1:4382] 0.7 1 0.433 1 0.5 ...
    ##  $ avg_negative_polarity       : num [1:4382] -0.119 -0.145 -0.456 -0.214 -0.3 ...
    ##  $ min_negative_polarity       : num [1:4382] -0.125 -0.2 -1 -0.6 -0.3 ...
    ##  $ max_negative_polarity       : num [1:4382] -0.1 -0.1 -0.125 -0.1 -0.3 0 -0.1 0 -0.2 -0.1 ...
    ##  $ title_subjectivity          : num [1:4382] 0 0 0.7 0.5 0 ...
    ##  $ title_sentiment_polarity    : num [1:4382] 0 0 -0.4 0.5 0 ...
    ##  $ abs_title_subjectivity      : num [1:4382] 0.5 0.5 0.2 0 0.5 ...
    ##  $ abs_title_sentiment_polarity: num [1:4382] 0 0 0.4 0.5 0 ...
    ##  $ shares                      : int [1:4382] 711 3100 852 425 575 819 732 1200 459 2000 ...
    ##  $ channel                     : chr [1:4382] "data_channel_is_bus" "data_channel_is_bus" "data_channel_is_bus" "data_channel_is_bus" ...
    ##  $ weekday                     : chr [1:4382] "Monday" "Monday" "Monday" "Monday" ...

`str()` allows us to view the structure of the data. We see each
variable has an appropriate data type.

``` r
selectTrain %>% dplyr::select(shares, starts_with("rate")) %>% summary()
```

    ##      shares         rate_positive_words rate_negative_words
    ##  Min.   :    28.0   Min.   :0.0000      Min.   :0.0000     
    ##  1st Qu.:   952.2   1st Qu.:0.6667      1st Qu.:0.1667     
    ##  Median :  1400.0   Median :0.7500      Median :0.2500     
    ##  Mean   :  3175.4   Mean   :0.7354      Mean   :0.2605     
    ##  3rd Qu.:  2500.0   3rd Qu.:0.8305      3rd Qu.:0.3333     
    ##  Max.   :690400.0   Max.   :1.0000      Max.   :1.0000

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

<img src="data_channel_is_busAnalysis_files/figure-gfm/scatter share-1.png" style="display: block; margin: auto;" />

We can inspect the trend of shares as a function of the positive word
rate. If the points show an upward trend, then articles with more
positive words tend to be shared more often. If we see a negative trend
then articles with more positive words tend to be shared less often.

``` r
g <- ggplot(selectTrain, aes(x = shares))
g + geom_histogram(bins = sqrt(nrow(selectTrain)))  + 
          labs(title = "Histogram of Shares")
```

<img src="data_channel_is_busAnalysis_files/figure-gfm/histogram shares-1.png" style="display: block; margin: auto;" />
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

<img src="data_channel_is_busAnalysis_files/figure-gfm/boxplot shares-1.png" style="display: block; margin: auto;" />
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
    ##                       Friday Monday Saturday Sunday Thursday Tuesday Wednesday
    ##   data_channel_is_bus    607    800      176    224      833     831       911

Articles per Channel vs Weekday - We can inspect the distribution of
articles shared as a function of the day of the week. We can easily see
which days articles are more likely shared.

``` r
kable_if(table(selectTrain$weekday, selectTrain$num_keywords))
```

    ##            
    ##               2   3   4   5   6   7   8   9  10
    ##   Friday      1  19  67 115 114  88  83  56  64
    ##   Monday      3  24  90 153 169 129  89  61  82
    ##   Saturday    0   2  15  37  44  24  15  13  26
    ##   Sunday      0  16  10  35  44  36  22  12  49
    ##   Thursday    2  37 100 166 139 137  85  58 109
    ##   Tuesday     3  16 112 160 180 112  92  70  86
    ##   Wednesday   5  40 119 184 177 131  91  59 105

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

<img src="data_channel_is_busAnalysis_files/figure-gfm/tab3-1.png" style="display: block; margin: auto;" />
The correlation chart above does not show any strong relationships
between shares and the variables displayed. We can see many other
relationships which is indicated by the darker orange and blue colors.
For instance, n_token_title and n_non_stop_unique_tokens has a negative
relationship.

``` r
ggcorr(selectTrain%>% dplyr::select( average_token_length, num_keywords,num_hrefs, num_self_hrefs, num_imgs, num_videos, shares),label_round = 2, label_size=3, label = "FALSE")
```

<img src="data_channel_is_busAnalysis_files/figure-gfm/tab4-1.png" style="display: block; margin: auto;" />
The correlation chart above does not show any strong relationships
between shares and the variables displayed. We can see many other
relationships which is indicated by the darker orange and blue colors.

``` r
ggcorr(selectTrain%>% dplyr::select(kw_min_min, kw_max_min, kw_avg_min, kw_min_max, kw_max_max,  kw_avg_max, kw_min_avg, kw_max_avg, kw_avg_avg, shares),label_round = 2, label = "FALSE", label_size=3)
```

<img src="data_channel_is_busAnalysis_files/figure-gfm/tab5-1.png" style="display: block; margin: auto;" />
The correlation chart above does not show any strong relationships
between shares and the variables displayed. We can see many other
relationships which is indicated by the darker orange and blue colors.
We would expect to use this level of correlation for this grouping of
variables.

``` r
ggcorr(selectTrain%>% dplyr::select(self_reference_min_shares, self_reference_max_shares, self_reference_avg_sharess, global_subjectivity, global_sentiment_polarity, global_rate_positive_words, global_rate_negative_words, shares),label_round = 2, label = "FALSE", label_size=3)
```

<img src="data_channel_is_busAnalysis_files/figure-gfm/tab6-1.png" style="display: block; margin: auto;" />
The correlation chart above does not show any strong relationships
between shares and the variables displayed. We can see many other
relationships which is indicated by the darker orange and blue colors.

``` r
ggcorr(selectTrain%>% dplyr::select(LDA_00, LDA_01, LDA_02, LDA_03, LDA_04,  shares),label_round = 2, label = "FALSE", label_size=3)
```

<img src="data_channel_is_busAnalysis_files/figure-gfm/tab7-1.png" style="display: block; margin: auto;" />
The correlation chart above does not show any strong relationships
between shares and the variables displayed. We can see many other
variables above have a mild relationship to one another.

``` r
ggcorr(selectTrain%>% dplyr::select(rate_positive_words, rate_negative_words, avg_positive_polarity, min_positive_polarity, max_positive_polarity,  avg_negative_polarity, min_negative_polarity, max_negative_polarity, shares),label_round = 2, label = "FALSE", label_size=3)
```

<img src="data_channel_is_busAnalysis_files/figure-gfm/tab8-1.png" style="display: block; margin: auto;" />
The correlation chart above does not show any strong relationships
between shares and the variables displayed. We can see many other
relationships which is indicated by the darker orange and blue colors.

``` r
ggcorr(selectTrain%>% dplyr::select( title_subjectivity, title_sentiment_polarity, abs_title_subjectivity, abs_title_sentiment_polarity, shares),label_round = 2, label = "FALSE")
```

<img src="data_channel_is_busAnalysis_files/figure-gfm/tab9-1.png" style="display: block; margin: auto;" />
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
    ## 1             9     255   0.605    1.00   0.792       3       1       1       0    4.91       4       0       0       0
    ## 2             8     397   0.625    1.00   0.806      11       0       1       0    5.45       6       0       0       0
    ## 3            13     244   0.560    1.00   0.680       3       2       1       0    4.42       4       0       0       0
    ## 4            11     723   0.491    1.00   0.642      18       1       1       0    5.23       6       0       0       0
    ## 5            10     142   0.655    1.00   0.792       2       1       1       0    4.27       5       0       0       0
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
    ## 1    Good       10.20181         622.3109        0.530193        0.9943941                0.6910154 10.476930
    ## 2    Poor       10.33786         445.6500        0.564426        0.9980611                0.7176491  7.774115
    ##   num_self_hrefs num_imgs num_videos average_token_length num_keywords kw_min_min kw_max_min kw_avg_min kw_min_max
    ## 1       2.845623 2.009487  0.9133247             4.667077     6.687796   31.67745  1286.2604   361.7348   22298.19
    ## 2       2.748425 1.569074  0.3994183             4.704523     6.335919   29.39457   908.7955   290.7235   19679.31
    ##   kw_max_max kw_avg_max kw_min_avg kw_max_avg kw_avg_avg self_reference_min_shares self_reference_max_shares
    ## 1   741895.6   316640.3  1203.6594   5777.077   3158.954                  4943.085                  13525.24
    ## 2   733775.2   307311.5   964.3731   4554.979   2685.057                  2494.867                   7391.16
    ##   self_reference_avg_sharess is_weekend    LDA_00     LDA_01     LDA_02     LDA_03    LDA_04 global_subjectivity
    ## 1                   8193.254 0.14445882 0.6718638 0.07294225 0.07779052 0.06847687 0.1089265           0.4427410
    ## 2                   4423.772 0.03150751 0.6277259 0.08251620 0.08718800 0.06312747 0.1394425           0.4273173
    ##   global_sentiment_polarity global_rate_positive_words global_rate_negative_words rate_positive_words
    ## 1                 0.1380934                 0.04462196                 0.01526295           0.7374673
    ## 2                 0.1308072                 0.04163606                 0.01446735           0.7331332
    ##   rate_negative_words avg_positive_polarity min_positive_polarity max_positive_polarity avg_negative_polarity
    ## 1           0.2569268             0.3539556            0.07990358             0.7844654            -0.2484085
    ## 2           0.2644432             0.3503668            0.09486929             0.7460106            -0.2376595
    ##   min_negative_polarity max_negative_polarity title_subjectivity title_sentiment_polarity abs_title_subjectivity
    ## 1            -0.5134448            -0.1027177          0.2519449               0.08465277              0.3457129
    ## 2            -0.4462116            -0.1158537          0.2461066               0.07389471              0.3313322
    ##   abs_title_sentiment_polarity    shares channel weekday Rating
    ## 1                    0.1416757 5187.2359      NA      NA     NA
    ## 2                    0.1334356  913.9908      NA      NA     NA

``` r
means2
```

    ## # A tibble: 4 x 49
    ## # Groups:   Rating [2]
    ##   Rating is_weekend n_tokens_~1 n_tok~2 n_uni~3 n_non~4 n_non~5 num_h~6 num_s~7 num_i~8 num_v~9 avera~* num_k~* kw_mi~*
    ##   <fct>       <dbl>       <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>
    ## 1 Good            0        10.2    591.   0.537   0.995   0.696    9.97    2.90    1.99   0.946    4.66    6.65    31.3
    ## 2 Good            1        10.2    809.   0.487   0.991   0.660   13.5     2.55    2.15   0.722    4.68    6.93    33.7
    ## 3 Poor            0        10.3    438.   0.566   0.998   0.719    7.67    2.76    1.56   0.394    4.70    6.34    29.9
    ## 4 Poor            1        10.6    673.   0.504   1.00    0.672   11.0     2.51    1.95   0.554    4.80    6.35    14.4
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

<img src="data_channel_is_busAnalysis_files/figure-gfm/cat3-1.png" style="display: block; margin: auto;" />

``` r
cat_share %>% 
  select(c(response, predictors2)) %>% 
    select_if(~ !any(is.na(.))) %>%   
      ggpairs(aes(colour = Rating,alpha=0.5),
              upper = list(continuous = wrap("cor", title="")))
```

<img src="data_channel_is_busAnalysis_files/figure-gfm/cat3-2.png" style="display: block; margin: auto;" />

``` r
cat_share %>% 
  select(c(response, predictors3)) %>% 
    select_if(~ !any(is.na(.))) %>%   
      ggpairs(aes(colour = Rating,alpha=0.5),
              upper = list(continuous = wrap("cor", title="")))
```

<img src="data_channel_is_busAnalysis_files/figure-gfm/cat3-3.png" style="display: block; margin: auto;" />

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

<img src="data_channel_is_busAnalysis_files/figure-gfm/cat4-1.png" style="display: block; margin: auto;" />

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

<img src="data_channel_is_busAnalysis_files/figure-gfm/cat4-2.png" style="display: block; margin: auto;" />

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
    ##                           PC1     PC2     PC3     PC4     PC5     PC6     PC7     PC8     PC9    PC10    PC11    PC12
    ## Standard deviation     2.0574 1.92562 1.74101 1.67420 1.59253 1.56447 1.49927 1.25604 1.22923 1.18916 1.12782 1.01893
    ## Proportion of Variance 0.1114 0.09758 0.07977 0.07376 0.06674 0.06441 0.05915 0.04152 0.03976 0.03721 0.03347 0.02732
    ## Cumulative Proportion  0.1114 0.20897 0.28874 0.36250 0.42924 0.49365 0.55280 0.59432 0.63408 0.67130 0.70477 0.73209
    ##                           PC13    PC14    PC15    PC16   PC17    PC18    PC19    PC20    PC21    PC22   PC23    PC24
    ## Standard deviation     0.99133 0.97870 0.90850 0.88429 0.8520 0.81933 0.80956 0.77551 0.73380 0.70707 0.6893 0.65672
    ## Proportion of Variance 0.02586 0.02521 0.02172 0.02058 0.0191 0.01767 0.01725 0.01583 0.01417 0.01316 0.0125 0.01135
    ## Cumulative Proportion  0.75795 0.78316 0.80488 0.82546 0.8446 0.86223 0.87948 0.89530 0.90947 0.92263 0.9351 0.94648
    ##                           PC25   PC26    PC27    PC28    PC29    PC30    PC31    PC32    PC33    PC34    PC35    PC36
    ## Standard deviation     0.65332 0.5514 0.49122 0.46808 0.45911 0.39880 0.36394 0.32476 0.28664 0.25498 0.21900 0.14922
    ## Proportion of Variance 0.01123 0.0080 0.00635 0.00577 0.00555 0.00419 0.00349 0.00278 0.00216 0.00171 0.00126 0.00059
    ## Cumulative Proportion  0.95771 0.9657 0.97206 0.97783 0.98338 0.98756 0.99105 0.99382 0.99598 0.99769 0.99896 0.99954
    ##                           PC37    PC38
    ## Standard deviation     0.11134 0.07057
    ## Proportion of Variance 0.00033 0.00013
    ## Cumulative Proportion  0.99987 1.00000

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

<img src="data_channel_is_busAnalysis_files/figure-gfm/pca1-1.png" style="display: block; margin: auto;" />

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
    ## 1      1 14580.15 0.001814906 2915.960 9754.437 0.002279148 453.3134
    ## 2      2 14631.24 0.003531910 2950.277 9699.430 0.001345455 435.8948
    ## 3      3 14607.56 0.005389277 2940.204 9725.750 0.002429026 487.0818
    ## 4      4 14637.78 0.006280172 2957.599 9689.388 0.003693275 496.2659
    ## 5      5 14638.66 0.006745166 2951.670 9707.265 0.005337106 490.9332
    ## 6      6 14663.39 0.007584945 2966.155 9688.052 0.005928633 492.7084
    ## 7      7 14672.97 0.007095007 2979.936 9691.291 0.006137920 483.9072
    ## 8      8 14666.32 0.008193147 3024.822 9677.449 0.005672490 503.8208
    ## 9      9 14691.96 0.007155929 2999.404 9677.962 0.006768726 458.4169
    ## 10    10 14686.05 0.007785607 3031.719 9643.431 0.004766027 468.2667
    ## 11    11 14617.27 0.010603574 3037.113 9766.757 0.013348923 525.7388
    ## 12    12 14705.08 0.007262999 3063.496 9651.682 0.005364517 477.9655
    ## 13    13 14699.18 0.007881879 3067.539 9644.920 0.005821882 477.9703
    ## 14    14 14700.99 0.007826310 3070.336 9646.497 0.005858577 474.2323
    ## 15    15 14672.28 0.008635112 3044.399 9676.336 0.007095622 509.3101
    ## 16    16 14684.97 0.008692343 3048.945 9678.770 0.008236910 500.9637
    ## 17    17 14686.81 0.009113668 3053.075 9654.324 0.007465843 453.6104
    ## 18    18 14692.05 0.008899649 3082.382 9663.475 0.008135573 479.6573
    ## 19    19 14695.69 0.008514184 3087.055 9661.875 0.007296193 474.2630
    ## 20    20 14694.36 0.008920229 3087.769 9662.411 0.007689914 473.8974

``` r
lmFit$bestTune
```

    ##   nvmax
    ## 1     1

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

    ##   intercept     RMSE   Rsquared      MAE   RMSESD RsquaredSD   MAESD
    ## 1      TRUE 14524.86 0.01376807 2881.522 9772.875 0.01188474 490.297

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

    ##   intercept     RMSE    Rsquared      MAE   RMSESD  RsquaredSD    MAESD
    ## 1      TRUE 14694.36 0.008920229 3087.769 9662.411 0.007689914 473.8974

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

    ##          PC1      PC2         PC3      PC4       PC5        PC6        PC7          PC8        PC9       PC10
    ## 1 -0.3353067 3.739356 -0.93445918 1.509244 -1.021557 -0.1869505 -0.5450764  0.127513704 -0.6906040  1.0682828
    ## 2 -2.9544398 3.054296  0.03589795 1.309889 -1.937007  0.1188220 -0.4139527  0.229777174 -0.6209824  0.6481593
    ## 3  4.7245394 2.431915  2.25396535 2.387283  1.135924 -2.9190127 -0.9087885 -0.990953666  0.1167943  0.8319042
    ## 4 -3.6154479 1.801195  1.57034619 1.160830  1.896376 -1.6127953 -0.8927083  0.004922892  0.3133193  1.2142783
    ## 5  0.5327564 3.477317 -0.88671955 1.581541 -1.847834 -1.7627167 -0.4079432 -1.759702341 -0.2538222 -0.8168413
    ##          PC11        PC12        PC13       PC14  res
    ## 1  1.19555625 -0.09698631  0.10774210 -0.5787766  711
    ## 2  1.23384729 -0.08682241  0.60814508 -0.6477056 3100
    ## 3  0.04005855  0.02569868 -1.31628333  0.3046667  852
    ## 4  0.76943258 -0.43733625  0.20626018 -0.3626292  425
    ## 5 -1.78627259 -0.56423273 -0.07232882  0.2093666  575

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

    ##          PC1      PC2        PC3      PC4        PC5         PC6        PC7        PC8        PC9       PC10
    ## 1 -2.9162807 1.471541  2.4233103 1.389583 -2.4368074 -0.50993208 -0.5028330 -2.0521969 -1.8753134 -2.1727613
    ## 2 -1.1271164 2.473086  1.1681608 1.897367 -1.0611718  0.68865742 -1.2577312 -0.2172985 -0.9447033  0.1087318
    ## 3 -2.4974550 4.017898 -3.0361322 2.341030 -1.2621238  0.07100826 -0.9312115  0.2728872 -2.1297572  1.7263394
    ## 4 -0.6341183 2.576254  0.3453688 3.219850  0.5298874 -1.96310384 -1.0379893 -1.0457835 -0.3728597  1.3067079
    ## 5  0.6663634 1.500609  1.8945554 3.304806  0.8655273 -2.61684878 -0.3988232 -0.1172751 -1.5735080  0.5946899
    ##          PC11       PC12       PC13       PC14 test_res
    ## 1 -0.59021968 -0.3180525 -0.7759876 -1.2250231     1500
    ## 2  0.43120413  0.1724015  0.1465553 -1.0294685     3200
    ## 3  0.08863063 -0.7842791  0.6113669 -1.1833613     2500
    ## 4 -1.03538197 -0.7163523  0.5038267 -0.2415182      648
    ## 5 -0.20457474  0.5285465 -1.8428699  0.8567122     1200

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

<img src="data_channel_is_busAnalysis_files/figure-gfm/boost-1.png" style="display: block; margin: auto;" />

``` r
boostfit$bestTune
```

    ##   n.trees interaction.depth shrinkage n.minobsinnode
    ## 3     200                 1     0.001              3

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

We will be comparing our models based on minimum RMSE.

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
    ## LinearRegression_Stepwise_Sel    9086.279 0.009954345 2453.048
    ## LinearRegression_Orignal_Var_Sel 9086.279 0.009954345 2453.048
    ## Linear_Regression_PCA            9163.818 0.009201406 2493.819
    ## Boosted_Trees                    9104.501 0.000838805 2549.136
    ## Random_Forest                    9257.141 0.004849775 2527.242

``` r
best_model<-Metrics_df[which.min(Metrics_df$RMSE), ]
rmse_min<-best_model$RMSE
r2max<-best_model$Rsquared
```

The best model for predicting the number of shares for dataset called
data_channel_is_bus in terms of RMSE is LinearRegression_Stepwise_Sel
with RMSE of 9086.28. This model also showed an R-Squared of about
0.0099543, which shows that the predictors of this model do not have
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

<img src="data_channel_is_busAnalysis_files/figure-gfm/plots-1.png" style="display: block; margin: auto;" />

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
