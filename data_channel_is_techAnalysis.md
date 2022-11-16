Project 3: Predictive Model for `data_channel_is_tech` Analysis
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

    ## tibble [5,145 x 48] (S3: tbl_df/tbl/data.frame)
    ##  $ n_tokens_title              : num [1:5145] 13 10 12 11 8 11 8 8 12 14 ...
    ##  $ n_tokens_content            : num [1:5145] 1072 370 989 97 1207 ...
    ##  $ n_unique_tokens             : num [1:5145] 0.416 0.56 0.434 0.67 0.411 ...
    ##  $ n_non_stop_words            : num [1:5145] 1 1 1 1 1 ...
    ##  $ n_non_stop_unique_tokens    : num [1:5145] 0.541 0.698 0.572 0.837 0.549 ...
    ##  $ num_hrefs                   : num [1:5145] 19 2 20 2 24 20 5 5 22 0 ...
    ##  $ num_self_hrefs              : num [1:5145] 19 2 20 0 24 20 2 3 22 0 ...
    ##  $ num_imgs                    : num [1:5145] 20 0 20 0 42 20 1 1 28 14 ...
    ##  $ num_videos                  : num [1:5145] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ average_token_length        : num [1:5145] 4.68 4.36 4.62 4.86 4.72 ...
    ##  $ num_keywords                : num [1:5145] 7 9 9 7 8 7 10 9 9 9 ...
    ##  $ kw_min_min                  : num [1:5145] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_max_min                  : num [1:5145] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_avg_min                  : num [1:5145] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_min_max                  : num [1:5145] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_max_max                  : num [1:5145] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_avg_max                  : num [1:5145] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_min_avg                  : num [1:5145] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_max_avg                  : num [1:5145] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_avg_avg                  : num [1:5145] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ self_reference_min_shares   : num [1:5145] 545 8500 545 0 545 545 924 2500 545 0 ...
    ##  $ self_reference_max_shares   : num [1:5145] 16000 8500 16000 0 16000 16000 924 2500 16000 0 ...
    ##  $ self_reference_avg_sharess  : num [1:5145] 3151 8500 3151 0 2830 ...
    ##  $ is_weekend                  : num [1:5145] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ LDA_00                      : num [1:5145] 0.0286 0.0222 0.0222 0.4583 0.025 ...
    ##  $ LDA_01                      : num [1:5145] 0.0288 0.3067 0.1507 0.029 0.0252 ...
    ##  $ LDA_02                      : num [1:5145] 0.0286 0.0222 0.2434 0.0287 0.025 ...
    ##  $ LDA_03                      : num [1:5145] 0.0286 0.0222 0.0222 0.0297 0.025 ...
    ##  $ LDA_04                      : num [1:5145] 0.885 0.627 0.561 0.454 0.9 ...
    ##  $ global_subjectivity         : num [1:5145] 0.514 0.437 0.543 0.539 0.539 ...
    ##  $ global_sentiment_polarity   : num [1:5145] 0.281 0.0712 0.2986 0.1611 0.2883 ...
    ##  $ global_rate_positive_words  : num [1:5145] 0.0746 0.0297 0.0839 0.0309 0.0696 ...
    ##  $ global_rate_negative_words  : num [1:5145] 0.0121 0.027 0.0152 0.0206 0.0116 ...
    ##  $ rate_positive_words         : num [1:5145] 0.86 0.524 0.847 0.6 0.857 ...
    ##  $ rate_negative_words         : num [1:5145] 0.14 0.476 0.153 0.4 0.143 ...
    ##  $ avg_positive_polarity       : num [1:5145] 0.411 0.351 0.428 0.567 0.427 ...
    ##  $ min_positive_polarity       : num [1:5145] 0.0333 0.1364 0.1 0.4 0.1 ...
    ##  $ max_positive_polarity       : num [1:5145] 1 0.6 1 0.8 1 1 0.35 1 1 1 ...
    ##  $ avg_negative_polarity       : num [1:5145] -0.22 -0.195 -0.243 -0.125 -0.227 ...
    ##  $ min_negative_polarity       : num [1:5145] -0.5 -0.4 -0.5 -0.125 -0.5 -0.5 -0.2 -0.5 -0.5 -0.4 ...
    ##  $ max_negative_polarity       : num [1:5145] -0.05 -0.1 -0.05 -0.125 -0.05 ...
    ##  $ title_subjectivity          : num [1:5145] 0.455 0.643 1 0.125 0.5 ...
    ##  $ title_sentiment_polarity    : num [1:5145] 0.136 0.214 0.5 0 0 ...
    ##  $ abs_title_subjectivity      : num [1:5145] 0.0455 0.1429 0.5 0.375 0 ...
    ##  $ abs_title_sentiment_polarity: num [1:5145] 0.136 0.214 0.5 0 0 ...
    ##  $ shares                      : int [1:5145] 505 855 891 3600 17100 445 783 1500 1800 480 ...
    ##  $ channel                     : chr [1:5145] "data_channel_is_tech" "data_channel_is_tech" "data_channel_is_tech" "data_channel_is_tech" ...
    ##  $ weekday                     : chr [1:5145] "Monday" "Monday" "Monday" "Monday" ...

`str()` allows us to view the structure of the data. We see each
variable has an appropriate data type.

``` r
selectTrain %>% dplyr::select(shares, starts_with("rate")) %>% summary()
```

    ##      shares       rate_positive_words rate_negative_words
    ##  Min.   :    36   Min.   :0.0000      Min.   :0.0000     
    ##  1st Qu.:  1100   1st Qu.:0.6724      1st Qu.:0.1667     
    ##  Median :  1700   Median :0.7532      Median :0.2444     
    ##  Mean   :  3124   Mean   :0.7474      Mean   :0.2497     
    ##  3rd Qu.:  3000   3rd Qu.:0.8333      3rd Qu.:0.3247     
    ##  Max.   :663600   Max.   :1.0000      Max.   :1.0000

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

<img src="data_channel_is_techAnalysis_files/figure-gfm/scatter share-1.png" style="display: block; margin: auto;" />

We can inspect the trend of shares as a function of the positive word
rate. If the points show an upward trend, then articles with more
positive words tend to be shared more often. If we see a negative trend
then articles with more positive words tend to be shared less often.

``` r
g <- ggplot(selectTrain, aes(x = shares))
g + geom_histogram(bins = sqrt(nrow(selectTrain)))  + 
          labs(title = "Histogram of Shares")
```

<img src="data_channel_is_techAnalysis_files/figure-gfm/histogram shares-1.png" style="display: block; margin: auto;" />
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

<img src="data_channel_is_techAnalysis_files/figure-gfm/boxplot shares-1.png" style="display: block; margin: auto;" />
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
    ##                        Friday Monday Saturday Sunday Thursday Tuesday Wednesday
    ##   data_channel_is_tech    693    870      366    283      905    1042       986

Articles per Channel vs Weekday - We can inspect the distribution of
articles shared as a function of the day of the week. We can easily see
which days articles are more likely shared.

``` r
kable_if(table(selectTrain$weekday, selectTrain$num_keywords))
```

    ##            
    ##               2   3   4   5   6   7   8   9  10
    ##   Friday      0   4  15  57 114 137 129  83 154
    ##   Monday      1   3  12  71 141 156 152 126 208
    ##   Saturday    0   3   4  17  26  65  75  65 111
    ##   Sunday      0   3   3  21  23  44  49  51  89
    ##   Thursday    0   5  22  80 130 186 164 140 178
    ##   Tuesday     0   5  27  62 148 213 179 169 239
    ##   Wednesday   1   4  29  69 141 222 160 153 207

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

<img src="data_channel_is_techAnalysis_files/figure-gfm/tab3-1.png" style="display: block; margin: auto;" />
The correlation chart above does not show any strong relationships
between shares and the variables displayed. We can see many other
relationships which is indicated by the darker orange and blue colors.
For instance, n_token_title and n_non_stop_unique_tokens has a negative
relationship.

``` r
ggcorr(selectTrain%>% dplyr::select( average_token_length, num_keywords,num_hrefs, num_self_hrefs, num_imgs, num_videos, shares),label_round = 2, label_size=3, label = "FALSE")
```

<img src="data_channel_is_techAnalysis_files/figure-gfm/tab4-1.png" style="display: block; margin: auto;" />
The correlation chart above does not show any strong relationships
between shares and the variables displayed. We can see many other
relationships which is indicated by the darker orange and blue colors.

``` r
ggcorr(selectTrain%>% dplyr::select(kw_min_min, kw_max_min, kw_avg_min, kw_min_max, kw_max_max,  kw_avg_max, kw_min_avg, kw_max_avg, kw_avg_avg, shares),label_round = 2, label = "FALSE", label_size=3)
```

<img src="data_channel_is_techAnalysis_files/figure-gfm/tab5-1.png" style="display: block; margin: auto;" />
The correlation chart above does not show any strong relationships
between shares and the variables displayed. We can see many other
relationships which is indicated by the darker orange and blue colors.
We would expect to use this level of correlation for this grouping of
variables.

``` r
ggcorr(selectTrain%>% dplyr::select(self_reference_min_shares, self_reference_max_shares, self_reference_avg_sharess, global_subjectivity, global_sentiment_polarity, global_rate_positive_words, global_rate_negative_words, shares),label_round = 2, label = "FALSE", label_size=3)
```

<img src="data_channel_is_techAnalysis_files/figure-gfm/tab6-1.png" style="display: block; margin: auto;" />
The correlation chart above does not show any strong relationships
between shares and the variables displayed. We can see many other
relationships which is indicated by the darker orange and blue colors.

``` r
ggcorr(selectTrain%>% dplyr::select(LDA_00, LDA_01, LDA_02, LDA_03, LDA_04,  shares),label_round = 2, label = "FALSE", label_size=3)
```

<img src="data_channel_is_techAnalysis_files/figure-gfm/tab7-1.png" style="display: block; margin: auto;" />
The correlation chart above does not show any strong relationships
between shares and the variables displayed. We can see many other
variables above have a mild relationship to one another.

``` r
ggcorr(selectTrain%>% dplyr::select(rate_positive_words, rate_negative_words, avg_positive_polarity, min_positive_polarity, max_positive_polarity,  avg_negative_polarity, min_negative_polarity, max_negative_polarity, shares),label_round = 2, label = "FALSE", label_size=3)
```

<img src="data_channel_is_techAnalysis_files/figure-gfm/tab8-1.png" style="display: block; margin: auto;" />
The correlation chart above does not show any strong relationships
between shares and the variables displayed. We can see many other
relationships which is indicated by the darker orange and blue colors.

``` r
ggcorr(selectTrain%>% dplyr::select( title_subjectivity, title_sentiment_polarity, abs_title_subjectivity, abs_title_sentiment_polarity, shares),label_round = 2, label = "FALSE")
```

<img src="data_channel_is_techAnalysis_files/figure-gfm/tab9-1.png" style="display: block; margin: auto;" />
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
    ## 1            13    1072   0.416    1.00   0.541      19      19      20       0    4.68       7       0       0       0
    ## 2            10     370   0.560    1.00   0.698       2       2       0       0    4.36       9       0       0       0
    ## 3            12     989   0.434    1.00   0.572      20      20      20       0    4.62       9       0       0       0
    ## 4            11      97   0.670    1.00   0.837       2       0       0       0    4.86       7       0       0       0
    ## 5             8    1207   0.411    1.00   0.549      24      24      42       0    4.72       8       0       0       0
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
    ## 1    Good       10.12467         626.7307       0.5203812        0.9954802                0.6753190 10.245574
    ## 2    Poor       10.29317         510.3165       0.5435746        0.9987952                0.6919566  8.438153
    ##   num_self_hrefs num_imgs num_videos average_token_length num_keywords kw_min_min kw_max_min kw_avg_min kw_min_max
    ## 1       4.582674 4.651977  0.4836158             4.576454     7.795104   33.41431   1227.564   332.9432   7698.508
    ## 2       4.597992 4.140964  0.4112450             4.592866     7.802410   25.91566   1072.685   302.9073   6374.589
    ##   kw_max_max kw_avg_max kw_min_avg kw_max_avg kw_avg_avg self_reference_min_shares self_reference_max_shares
    ## 1   732819.3   214771.9  1079.5246   4821.949   2853.856                  4605.119                  12897.95
    ## 2   739322.2   208611.7   897.5905   4302.417   2619.538                  3384.145                  10217.56
    ##   self_reference_avg_sharess is_weekend     LDA_00     LDA_01    LDA_02     LDA_03    LDA_04 global_subjectivity
    ## 1                   7652.432 0.17212806 0.07407841 0.05981041 0.1104253 0.06288843 0.6927974           0.4583858
    ## 2                   5933.493 0.07710843 0.07527138 0.06999202 0.1133103 0.06171170 0.6797146           0.4537724
    ##   global_sentiment_polarity global_rate_positive_words global_rate_negative_words rate_positive_words
    ## 1                 0.1417335                 0.04252166                 0.01480342           0.7405381
    ## 2                 0.1498561                 0.04330786                 0.01395902           0.7547299
    ##   rate_negative_words avg_positive_polarity min_positive_polarity max_positive_polarity avg_negative_polarity
    ## 1           0.2549421             0.3561750             0.0951579             0.7826641            -0.2328669
    ## 2           0.2440652             0.3564062             0.1025385             0.7593671            -0.2223164
    ##   min_negative_polarity max_negative_polarity title_subjectivity title_sentiment_polarity abs_title_subjectivity
    ## 1            -0.4720216           -0.09828627          0.2669736               0.10030755              0.3427370
    ## 2            -0.4273531           -0.10585417          0.2489727               0.07747673              0.3445966
    ##   abs_title_sentiment_polarity   shares channel weekday Rating
    ## 1                    0.1502520 5022.976      NA      NA     NA
    ## 2                    0.1270862 1098.072      NA      NA     NA

``` r
means2
```

    ## # A tibble: 4 x 49
    ## # Groups:   Rating [2]
    ##   Rating is_weekend n_tokens_~1 n_tok~2 n_uni~3 n_non~4 n_non~5 num_h~6 num_s~7 num_i~8 num_v~9 avera~* num_k~* kw_mi~*
    ##   <fct>       <dbl>       <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>
    ## 1 Good            0        10.1    615.   0.521   0.995   0.675    9.87    4.42    4.62   0.501    4.57    7.70    32.9
    ## 2 Good            1        10.1    683.   0.519   1.00    0.677   12.0     5.34    4.81   0.398    4.61    8.26    35.9
    ## 3 Poor            0        10.3    503.   0.545   0.999   0.693    8.20    4.48    4.12   0.401    4.59    7.78    25.4
    ## 4 Poor            1        10.3    597.   0.530   1.00    0.683   11.3     6.06    4.36   0.531    4.63    8.10    32.6
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

<img src="data_channel_is_techAnalysis_files/figure-gfm/cat3-1.png" style="display: block; margin: auto;" />

``` r
cat_share %>% 
  select(c(response, predictors2)) %>% 
    select_if(~ !any(is.na(.))) %>%   
      ggpairs(aes(colour = Rating,alpha=0.5),
              upper = list(continuous = wrap("cor", title="")))
```

<img src="data_channel_is_techAnalysis_files/figure-gfm/cat3-2.png" style="display: block; margin: auto;" />

``` r
cat_share %>% 
  select(c(response, predictors3)) %>% 
    select_if(~ !any(is.na(.))) %>%   
      ggpairs(aes(colour = Rating,alpha=0.5),
              upper = list(continuous = wrap("cor", title="")))
```

<img src="data_channel_is_techAnalysis_files/figure-gfm/cat3-3.png" style="display: block; margin: auto;" />

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

<img src="data_channel_is_techAnalysis_files/figure-gfm/cat4-1.png" style="display: block; margin: auto;" />

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

<img src="data_channel_is_techAnalysis_files/figure-gfm/cat4-2.png" style="display: block; margin: auto;" />

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
    ##                           PC1     PC2     PC3    PC4    PC5     PC6     PC7     PC8     PC9    PC10    PC11    PC12
    ## Standard deviation     1.9960 1.89310 1.74526 1.6472 1.5885 1.53959 1.50135 1.28083 1.22061 1.18594 1.10061 1.02628
    ## Proportion of Variance 0.1048 0.09431 0.08016 0.0714 0.0664 0.06238 0.05932 0.04317 0.03921 0.03701 0.03188 0.02772
    ## Cumulative Proportion  0.1048 0.19915 0.27931 0.3507 0.4171 0.47949 0.53880 0.58198 0.62118 0.65820 0.69007 0.71779
    ##                          PC13   PC14    PC15   PC16    PC17    PC18    PC19    PC20    PC21    PC22    PC23    PC24
    ## Standard deviation     1.0035 0.9939 0.89996 0.8847 0.86699 0.83016 0.81433 0.78066 0.75229 0.73232 0.69514 0.66855
    ## Proportion of Variance 0.0265 0.0260 0.02131 0.0206 0.01978 0.01814 0.01745 0.01604 0.01489 0.01411 0.01272 0.01176
    ## Cumulative Proportion  0.7443 0.7703 0.79160 0.8122 0.83198 0.85012 0.86757 0.88361 0.89850 0.91261 0.92533 0.93709
    ##                           PC25    PC26    PC27    PC28    PC29    PC30    PC31    PC32    PC33    PC34    PC35    PC36
    ## Standard deviation     0.63610 0.59106 0.54564 0.50953 0.50006 0.47048 0.41177 0.32469 0.31534 0.28612 0.26320 0.22494
    ## Proportion of Variance 0.01065 0.00919 0.00783 0.00683 0.00658 0.00583 0.00446 0.00277 0.00262 0.00215 0.00182 0.00133
    ## Cumulative Proportion  0.94774 0.95693 0.96477 0.97160 0.97818 0.98400 0.98847 0.99124 0.99386 0.99601 0.99784 0.99917
    ##                           PC37      PC38
    ## Standard deviation     0.17793 8.986e-09
    ## Proportion of Variance 0.00083 0.000e+00
    ## Cumulative Proportion  1.00000 1.000e+00

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

<img src="data_channel_is_techAnalysis_files/figure-gfm/pca1-1.png" style="display: block; margin: auto;" />

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
    ## 1      1 8171.234 0.006105018 2440.750 7141.797 0.002893703 246.3496
    ## 2      2 8179.941 0.008388835 2432.245 7133.939 0.005494212 243.9121
    ## 3      3 8190.666 0.010180155 2445.107 7113.574 0.002735053 232.4214
    ## 4      4 8222.009 0.009927879 2458.906 7091.488 0.002321399 223.3428
    ## 5      5 8200.811 0.016319636 2450.229 7097.634 0.012510725 222.0339
    ## 6      6 8202.975 0.017896973 2469.372 7095.028 0.014931432 212.9023
    ## 7      7 8199.329 0.018637351 2469.125 7107.875 0.014787613 220.1123
    ## 8      8 8186.633 0.020708083 2459.122 7099.203 0.014770631 219.2688
    ## 9      9 8188.561 0.020486721 2466.269 7100.472 0.011735524 220.5207
    ## 10    10 8186.979 0.019487519 2467.289 7100.529 0.013694206 221.3014
    ## 11    11 8174.059 0.021209711 2437.615 7107.143 0.011388059 237.5651
    ## 12    12 8180.926 0.022731365 2472.545 7101.997 0.011753108 212.8487
    ## 13    13 8177.783 0.022625141 2464.060 7102.076 0.011893492 220.9713
    ## 14    14 8181.862 0.022357519 2469.553 7100.037 0.011621268 216.8929
    ## 15    15 8187.602 0.021761703 2475.639 7101.443 0.011420565 209.5488
    ## 16    16 8184.006 0.021909497 2471.778 7100.787 0.011117123 217.0650
    ## 17    17 8187.322 0.021752920 2473.589 7101.441 0.011268912 211.2034
    ## 18    18 8188.349 0.021571007 2474.104 7100.994 0.011067797 211.7362
    ## 19    19 8188.544 0.021637662 2478.156 7097.883 0.010447878 211.1618
    ## 20    20 8189.119 0.021516715 2478.696 7098.577 0.010533110 212.0992

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

    ##   intercept    RMSE   Rsquared      MAE   RMSESD  RsquaredSD   MAESD
    ## 1      TRUE 8170.75 0.01317215 2419.985 7138.265 0.003794956 237.741

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
    ## 1      TRUE 8189.119 0.02151671 2478.696 7098.577 0.01053311 212.0992

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

    ## [1] "Number of Principal Components optimized are 15 for cumuative variance of 0.80"

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

    ##          PC1        PC2       PC3        PC4          PC5       PC6       PC7        PC8        PC9         PC10
    ## 1 -4.3628842  1.6372770 -3.398218 -2.9861062 -0.008850196  1.330718 -1.387416 -0.3896073 -0.8091655  0.008928743
    ## 2  1.9642486 -0.4166969 -4.155421 -2.2514113 -0.695598064 -2.030663 -2.123254 -1.0484139  0.4545709 -0.358526824
    ## 3 -5.0737381  2.5528661 -3.368466 -3.1790240 -0.192134316  0.238771 -1.731547 -0.7161574 -0.2553140  0.096120675
    ## 4  0.4594227 -2.5102508 -3.885399 -0.8519933 -3.994433486 -1.412783 -0.789889 -2.9766651  2.2396870  1.690004255
    ## 5 -4.8545781  2.5785447 -3.821617 -2.9337662  0.165780995  2.627806 -1.250721 -1.0977761 -1.6921144  0.331423779
    ##          PC11       PC12       PC13       PC14        PC15   res
    ## 1  0.45812273 -2.1234311 -1.0330792 1.23365057  0.13575185   505
    ## 2 -0.02745095 -0.2913891  0.1708709 0.06577931 -0.05305938   855
    ## 3 -0.52347396 -1.3958347 -0.3779574 0.50244245  0.38527403   891
    ## 4 -2.39089249 -0.7981007 -0.6064898 0.40423350 -0.23351217  3600
    ## 5 -1.60620692 -2.0877416 -0.2181974 0.49560202  0.27998950 17100

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

    ##         PC1        PC2       PC3        PC4        PC5      PC6        PC7        PC8          PC9         PC10
    ## 1 -3.936881  1.0269303 -3.894128 -2.0431432 -0.3416997 3.385908 -0.7137948 -0.9791078  0.002647391  0.004147827
    ## 2 -1.139062 -0.6771225 -4.075513 -1.4667281 -1.5741920 1.662185 -0.7955351  1.2522215 -0.710649225 -0.302909581
    ## 3 -4.106870  1.3078053 -3.934040 -1.9859307 -0.6456274 3.500528 -0.7146189 -0.7655442 -0.583906964  0.363418926
    ## 4 -2.383905 -4.7739834 -3.394997 -0.4951824 -1.4412918 0.687892 -1.0026843 -1.5878508 -0.505614985  0.483151615
    ## 5 -3.935711  1.7694217 -3.954268 -2.6073749  0.1345809 4.049825 -0.6764892 -1.1211577 -1.253349742  0.472038094
    ##          PC11       PC12       PC13       PC14         PC15 test_res
    ## 1 -0.43762657 -1.3987472 -1.2987548  1.0773893  0.525388263     2800
    ## 2  0.36445877 -1.3081810  0.4155056  0.1863508 -1.020557135     3900
    ## 3 -0.05812291 -1.0322722 -0.2868704  0.5657776  0.448241882     1100
    ## 4 -0.51989332  0.4456486  1.1076063 -0.9824010 -0.841003525      401
    ## 5 -1.51671220 -2.9189818 -1.7669177  1.4737833 -0.001564336     4200

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

<img src="data_channel_is_techAnalysis_files/figure-gfm/boost-1.png" style="display: block; margin: auto;" />

``` r
boostfit$bestTune
```

    ##    n.trees interaction.depth shrinkage n.minobsinnode
    ## 31      50                 1       0.1              3

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
    ## LinearRegression_Stepwise_Sel    4505.446 0.006651132 2273.529
    ## LinearRegression_Orignal_Var_Sel 4505.446 0.006651132 2273.529
    ## Linear_Regression_PCA            4485.873 0.007960910 2266.673
    ## Boosted_Trees                    4489.731 0.003977623 2286.755
    ## Random_Forest                    4509.622 0.003118459 2282.283

``` r
best_model<-Metrics_df[which.min(Metrics_df$RMSE), ]
rmse_min<-best_model$RMSE
r2max<-best_model$Rsquared
```

The best model for predicting the number of shares for dataset called
data_channel_is_tech in terms of RMSE is Linear_Regression_PCA with RMSE
of 4485.87. This model also showed an R-Squared of about 0.0079609,
which shows that the predictors of this model do not have significant
impact on the response variable.

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

<img src="data_channel_is_techAnalysis_files/figure-gfm/plots-1.png" style="display: block; margin: auto;" />

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
