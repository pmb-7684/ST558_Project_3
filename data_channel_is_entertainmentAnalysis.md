Project 3: Predictive Model for `data_channel_is_entertainment` Analysis
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

    ## tibble [4,941 x 48] (S3: tbl_df/tbl/data.frame)
    ##  $ n_tokens_title              : num [1:4941] 12 9 12 11 5 11 10 10 6 7 ...
    ##  $ n_tokens_content            : num [1:4941] 219 531 161 454 356 281 909 413 241 376 ...
    ##  $ n_unique_tokens             : num [1:4941] 0.664 0.504 0.669 0.566 0.618 ...
    ##  $ n_non_stop_words            : num [1:4941] 1 1 1 1 1 ...
    ##  $ n_non_stop_unique_tokens    : num [1:4941] 0.815 0.666 0.752 0.755 0.766 ...
    ##  $ num_hrefs                   : num [1:4941] 4 9 5 5 3 5 3 6 5 3 ...
    ##  $ num_self_hrefs              : num [1:4941] 2 0 4 3 3 4 2 1 5 2 ...
    ##  $ num_imgs                    : num [1:4941] 1 1 0 1 12 1 1 13 1 0 ...
    ##  $ num_videos                  : num [1:4941] 0 0 6 0 1 0 1 0 0 11 ...
    ##  $ average_token_length        : num [1:4941] 4.68 4.4 4.45 4.89 4.47 ...
    ##  $ num_keywords                : num [1:4941] 5 7 10 6 10 4 5 6 5 9 ...
    ##  $ kw_min_min                  : num [1:4941] 0 0 0 0 0 217 217 217 217 217 ...
    ##  $ kw_max_min                  : num [1:4941] 0 0 0 0 0 593 593 598 598 1200 ...
    ##  $ kw_avg_min                  : num [1:4941] 0 0 0 0 0 ...
    ##  $ kw_min_max                  : num [1:4941] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_max_max                  : num [1:4941] 0 0 0 0 0 17100 17100 17100 17100 17100 ...
    ##  $ kw_avg_max                  : num [1:4941] 0 0 0 0 0 ...
    ##  $ kw_min_avg                  : num [1:4941] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_max_avg                  : num [1:4941] 0 0 0 0 0 ...
    ##  $ kw_avg_avg                  : num [1:4941] 0 0 0 0 0 ...
    ##  $ self_reference_min_shares   : num [1:4941] 496 0 638 0 1700 951 20900 527 475 1100 ...
    ##  $ self_reference_max_shares   : num [1:4941] 496 0 29200 0 2500 951 20900 527 4400 1100 ...
    ##  $ self_reference_avg_sharess  : num [1:4941] 496 0 8261 0 2100 ...
    ##  $ is_weekend                  : num [1:4941] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ LDA_00                      : num [1:4941] 0.5003 0.0286 0.1258 0.2003 0.4565 ...
    ##  $ LDA_01                      : num [1:4941] 0.3783 0.4193 0.0203 0.3399 0.4822 ...
    ##  $ LDA_02                      : num [1:4941] 0.04 0.4947 0.02 0.0333 0.02 ...
    ##  $ LDA_03                      : num [1:4941] 0.0413 0.0289 0.8139 0.3931 0.0213 ...
    ##  $ LDA_04                      : num [1:4941] 0.0401 0.0286 0.02 0.0333 0.02 ...
    ##  $ global_subjectivity         : num [1:4941] 0.522 0.43 0.572 0.467 0.436 ...
    ##  $ global_sentiment_polarity   : num [1:4941] 0.0926 0.1007 0.1662 0.1255 0.1767 ...
    ##  $ global_rate_positive_words  : num [1:4941] 0.0457 0.0414 0.0497 0.0441 0.0618 ...
    ##  $ global_rate_negative_words  : num [1:4941] 0.0137 0.0207 0.0186 0.0132 0.014 ...
    ##  $ rate_positive_words         : num [1:4941] 0.769 0.667 0.727 0.769 0.815 ...
    ##  $ rate_negative_words         : num [1:4941] 0.231 0.333 0.273 0.231 0.185 ...
    ##  $ avg_positive_polarity       : num [1:4941] 0.379 0.386 0.427 0.363 0.359 ...
    ##  $ min_positive_polarity       : num [1:4941] 0.1 0.1364 0.1 0.1 0.0333 ...
    ##  $ max_positive_polarity       : num [1:4941] 0.7 0.8 0.85 1 1 0.5 1 0.8 0.5 0.8 ...
    ##  $ avg_negative_polarity       : num [1:4941] -0.35 -0.37 -0.364 -0.215 -0.373 ...
    ##  $ min_negative_polarity       : num [1:4941] -0.6 -0.6 -0.8 -0.5 -0.7 ...
    ##  $ max_negative_polarity       : num [1:4941] -0.2 -0.1667 -0.125 -0.1 -0.0714 ...
    ##  $ title_subjectivity          : num [1:4941] 0.5 0 0.583 0.427 0.455 ...
    ##  $ title_sentiment_polarity    : num [1:4941] -0.188 0 0.25 0.168 0.136 ...
    ##  $ abs_title_subjectivity      : num [1:4941] 0 0.5 0.0833 0.0727 0.0455 ...
    ##  $ abs_title_sentiment_polarity: num [1:4941] 0.188 0 0.25 0.168 0.136 ...
    ##  $ shares                      : int [1:4941] 593 1200 1200 4600 631 1300 1700 455 6400 1900 ...
    ##  $ channel                     : chr [1:4941] "data_channel_is_entertainment" "data_channel_is_entertainment" "data_channel_is_entertainment" "data_channel_is_entertainment" ...
    ##  $ weekday                     : chr [1:4941] "Monday" "Monday" "Monday" "Monday" ...

`str()` allows us to view the structure of the data. We see each
variable has an appropriate data type.

``` r
selectTrain %>% dplyr::select(shares, starts_with("rate")) %>% summary()
```

    ##      shares       rate_positive_words rate_negative_words
    ##  Min.   :    49   Min.   :0.0000      Min.   :0.0000     
    ##  1st Qu.:   832   1st Qu.:0.5789      1st Qu.:0.2000     
    ##  Median :  1200   Median :0.6875      Median :0.3000     
    ##  Mean   :  3027   Mean   :0.6655      Mean   :0.3052     
    ##  3rd Qu.:  2100   3rd Qu.:0.7818      3rd Qu.:0.4037     
    ##  Max.   :210300   Max.   :1.0000      Max.   :1.0000

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

<img src="data_channel_is_entertainmentAnalysis_files/figure-gfm/scatter share-1.png" style="display: block; margin: auto;" />

We can inspect the trend of shares as a function of the positive word
rate. If the points show an upward trend, then articles with more
positive words tend to be shared more often. If we see a negative trend
then articles with more positive words tend to be shared less often.

``` r
g <- ggplot(selectTrain, aes(x = shares))
g + geom_histogram(bins = sqrt(nrow(selectTrain)))  + 
          labs(title = "Histogram of Shares")
```

<img src="data_channel_is_entertainmentAnalysis_files/figure-gfm/histogram shares-1.png" style="display: block; margin: auto;" />
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

<img src="data_channel_is_entertainmentAnalysis_files/figure-gfm/boxplot shares-1.png" style="display: block; margin: auto;" />
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
    ##                                 Friday Monday Saturday Sunday Thursday Tuesday Wednesday
    ##   data_channel_is_entertainment    679    952      255    382      851     911       911

Articles per Channel vs Weekday - We can inspect the distribution of
articles shared as a function of the day of the week. We can easily see
which days articles are more likely shared.

``` r
kable_if(table(selectTrain$weekday, selectTrain$num_keywords))
```

    ##            
    ##               2   3   4   5   6   7   8   9  10
    ##   Friday      1  11  62 117 113 135  88  72  80
    ##   Monday      0  10  81 156 190 177 128  92 118
    ##   Saturday    0   3  23  38  41  44  27  37  42
    ##   Sunday      0   3  17  44  63  50  70  46  89
    ##   Thursday    0   6  74 132 171 163 111  74 120
    ##   Tuesday     0  11  91 146 163 160 129  95 116
    ##   Wednesday   0  16  92 147 174 144 143  65 130

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

<img src="data_channel_is_entertainmentAnalysis_files/figure-gfm/tab3-1.png" style="display: block; margin: auto;" />
The correlation chart above does not show any strong relationships
between shares and the variables displayed. We can see many other
relationships which is indicated by the darker orange and blue colors.
For instance, n_token_title and n_non_stop_unique_tokens has a negative
relationship.

``` r
ggcorr(selectTrain%>% dplyr::select( average_token_length, num_keywords,num_hrefs, num_self_hrefs, num_imgs, num_videos, shares),label_round = 2, label_size=3, label = "FALSE")
```

<img src="data_channel_is_entertainmentAnalysis_files/figure-gfm/tab4-1.png" style="display: block; margin: auto;" />
The correlation chart above does not show any strong relationships
between shares and the variables displayed. We can see many other
relationships which is indicated by the darker orange and blue colors.

``` r
ggcorr(selectTrain%>% dplyr::select(kw_min_min, kw_max_min, kw_avg_min, kw_min_max, kw_max_max,  kw_avg_max, kw_min_avg, kw_max_avg, kw_avg_avg, shares),label_round = 2, label = "FALSE", label_size=3)
```

<img src="data_channel_is_entertainmentAnalysis_files/figure-gfm/tab5-1.png" style="display: block; margin: auto;" />
The correlation chart above does not show any strong relationships
between shares and the variables displayed. We can see many other
relationships which is indicated by the darker orange and blue colors.
We would expect to use this level of correlation for this grouping of
variables.

``` r
ggcorr(selectTrain%>% dplyr::select(self_reference_min_shares, self_reference_max_shares, self_reference_avg_sharess, global_subjectivity, global_sentiment_polarity, global_rate_positive_words, global_rate_negative_words, shares),label_round = 2, label = "FALSE", label_size=3)
```

<img src="data_channel_is_entertainmentAnalysis_files/figure-gfm/tab6-1.png" style="display: block; margin: auto;" />
The correlation chart above does not show any strong relationships
between shares and the variables displayed. We can see many other
relationships which is indicated by the darker orange and blue colors.

``` r
ggcorr(selectTrain%>% dplyr::select(LDA_00, LDA_01, LDA_02, LDA_03, LDA_04,  shares),label_round = 2, label = "FALSE", label_size=3)
```

<img src="data_channel_is_entertainmentAnalysis_files/figure-gfm/tab7-1.png" style="display: block; margin: auto;" />
The correlation chart above does not show any strong relationships
between shares and the variables displayed. We can see many other
variables above have a mild relationship to one another.

``` r
ggcorr(selectTrain%>% dplyr::select(rate_positive_words, rate_negative_words, avg_positive_polarity, min_positive_polarity, max_positive_polarity,  avg_negative_polarity, min_negative_polarity, max_negative_polarity, shares),label_round = 2, label = "FALSE", label_size=3)
```

<img src="data_channel_is_entertainmentAnalysis_files/figure-gfm/tab8-1.png" style="display: block; margin: auto;" />
The correlation chart above does not show any strong relationships
between shares and the variables displayed. We can see many other
relationships which is indicated by the darker orange and blue colors.

``` r
ggcorr(selectTrain%>% dplyr::select( title_subjectivity, title_sentiment_polarity, abs_title_subjectivity, abs_title_sentiment_polarity, shares),label_round = 2, label = "FALSE")
```

<img src="data_channel_is_entertainmentAnalysis_files/figure-gfm/tab9-1.png" style="display: block; margin: auto;" />
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
    ## 1            12     219   0.664    1.00   0.815       4       2       1       0    4.68       5       0       0       0
    ## 2             9     531   0.504    1.00   0.666       9       0       1       0    4.40       7       0       0       0
    ## 3            12     161   0.669    1.00   0.752       5       4       0       6    4.45      10       0       0       0
    ## 4            11     454   0.566    1.00   0.755       5       3       1       0    4.89       6       0       0       0
    ## 5             5     356   0.618    1.00   0.766       3       3      12       1    4.47      10       0       0       0
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
    ## 1    Good       10.93121         611.5284       0.8084015        1.3825050                0.9240822 11.545527
    ## 2    Poor       11.02514         590.5519       0.5358806        0.9732069                0.6762772  9.699505
    ##   num_self_hrefs num_imgs num_videos average_token_length num_keywords kw_min_min kw_max_min kw_avg_min kw_min_max
    ## 1       3.528032 6.710537   2.429026             4.478706     7.106958   24.76342   1133.718   301.9810   12543.10
    ## 2       3.458780 6.132317   2.358203             4.476010     6.766694   17.85284   1007.265   276.0858   14072.12
    ##   kw_max_max kw_avg_max kw_min_avg kw_max_avg kw_avg_avg self_reference_min_shares self_reference_max_shares
    ## 1   756193.1   240961.5   1180.784   5901.366   3271.921                  3083.251                  9763.969
    ## 2   774400.4   246533.0   1059.794   5333.398   3020.036                  2222.451                  7482.466
    ##   self_reference_avg_sharess is_weekend     LDA_00    LDA_01     LDA_02    LDA_03     LDA_04 global_subjectivity
    ## 1                   5679.406 0.19443340 0.06735912 0.4221998 0.08687207 0.3581465 0.06502495           0.4540505
    ## 2                   4141.112 0.06100577 0.06428977 0.4256666 0.09460356 0.3530542 0.06238594           0.4489850
    ##   global_sentiment_polarity global_rate_positive_words global_rate_negative_words rate_positive_words
    ## 1                  0.113462                 0.03995909                 0.01847165           0.6684780
    ## 2                  0.108851                 0.04053175                 0.01941977           0.6624031
    ##   rate_negative_words avg_positive_polarity min_positive_polarity max_positive_polarity avg_negative_polarity
    ## 1           0.2997129             0.3664590            0.09354050             0.7980472            -0.2934460
    ## 2           0.3108039             0.3649256            0.09521202             0.7915279            -0.2944704
    ##   min_negative_polarity max_negative_polarity title_subjectivity title_sentiment_polarity abs_title_subjectivity
    ## 1            -0.5853050            -0.1127268          0.3220161               0.06646644              0.3213594
    ## 2            -0.5852528            -0.1122461          0.3140612               0.06192588              0.3247682
    ##   abs_title_sentiment_polarity    shares channel weekday Rating
    ## 1                    0.1774378 5168.1109      NA      NA     NA
    ## 2                    0.1685820  807.9711      NA      NA     NA

``` r
means2
```

    ## # A tibble: 4 x 49
    ## # Groups:   Rating [2]
    ##   Rating is_weekend n_tokens_~1 n_tok~2 n_uni~3 n_non~4 n_non~5 num_h~6 num_s~7 num_i~8 num_v~9 avera~* num_k~* kw_mi~*
    ##   <fct>       <dbl>       <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>
    ## 1 Good            0        11.0    604.   0.880   1.48    0.992   11.3     3.53    6.49    2.40    4.48    7.03    25.9
    ## 2 Good            1        10.8    641.   0.513   0.959   0.643   12.6     3.53    7.63    2.55    4.47    7.42    20.1
    ## 3 Poor            0        11.0    584.   0.538   0.974   0.679    9.71    3.45    5.92    2.38    4.48    6.73    17.8
    ## 4 Poor            1        11.0    688.   0.503   0.959   0.635    9.51    3.67    9.39    2.09    4.44    7.26    18.1
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

<img src="data_channel_is_entertainmentAnalysis_files/figure-gfm/cat3-1.png" style="display: block; margin: auto;" />

``` r
cat_share %>% 
  select(c(response, predictors2)) %>% 
    select_if(~ !any(is.na(.))) %>%   
      ggpairs(aes(colour = Rating,alpha=0.5),
              upper = list(continuous = wrap("cor", title="")))
```

<img src="data_channel_is_entertainmentAnalysis_files/figure-gfm/cat3-2.png" style="display: block; margin: auto;" />

``` r
cat_share %>% 
  select(c(response, predictors3)) %>% 
    select_if(~ !any(is.na(.))) %>%   
      ggpairs(aes(colour = Rating,alpha=0.5),
              upper = list(continuous = wrap("cor", title="")))
```

<img src="data_channel_is_entertainmentAnalysis_files/figure-gfm/cat3-3.png" style="display: block; margin: auto;" />

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

<img src="data_channel_is_entertainmentAnalysis_files/figure-gfm/cat4-1.png" style="display: block; margin: auto;" />

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

<img src="data_channel_is_entertainmentAnalysis_files/figure-gfm/cat4-2.png" style="display: block; margin: auto;" />

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
    ##                           PC1     PC2     PC3     PC4     PC5     PC6     PC7    PC8     PC9    PC10    PC11    PC12
    ## Standard deviation     2.1169 1.94205 1.82600 1.63183 1.48706 1.44416 1.42111 1.3478 1.28250 1.16601 1.15679 1.01544
    ## Proportion of Variance 0.1179 0.09925 0.08774 0.07008 0.05819 0.05488 0.05315 0.0478 0.04328 0.03578 0.03521 0.02713
    ## Cumulative Proportion  0.1179 0.21718 0.30493 0.37500 0.43320 0.48808 0.54123 0.5890 0.63231 0.66809 0.70331 0.73044
    ##                           PC13    PC14    PC15    PC16    PC17    PC18    PC19    PC20   PC21    PC22    PC23    PC24
    ## Standard deviation     1.00060 0.97748 0.91059 0.87994 0.85229 0.83414 0.80787 0.77859 0.7474 0.69815 0.67182 0.67051
    ## Proportion of Variance 0.02635 0.02514 0.02182 0.02038 0.01912 0.01831 0.01717 0.01595 0.0147 0.01283 0.01188 0.01183
    ## Cumulative Proportion  0.75679 0.78193 0.80375 0.82413 0.84325 0.86156 0.87873 0.89468 0.9094 0.92221 0.93409 0.94592
    ##                           PC25    PC26    PC27    PC28    PC29    PC30    PC31    PC32    PC33    PC34    PC35    PC36
    ## Standard deviation     0.66366 0.57327 0.52853 0.52462 0.44573 0.34479 0.32269 0.27952 0.27646 0.23982 0.21218 0.16442
    ## Proportion of Variance 0.01159 0.00865 0.00735 0.00724 0.00523 0.00313 0.00274 0.00206 0.00201 0.00151 0.00118 0.00071
    ## Cumulative Proportion  0.95751 0.96616 0.97351 0.98075 0.98598 0.98911 0.99185 0.99391 0.99592 0.99743 0.99862 0.99933
    ##                           PC37     PC38
    ## Standard deviation     0.15960 0.005237
    ## Proportion of Variance 0.00067 0.000000
    ## Cumulative Proportion  1.00000 1.000000

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

<img src="data_channel_is_entertainmentAnalysis_files/figure-gfm/pca1-1.png" style="display: block; margin: auto;" />

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

    ##    nvmax     RMSE   Rsquared      MAE   RMSESD RsquaredSD    MAESD
    ## 1      1 8691.528 0.01317443 3047.799 1404.920 0.01521344 240.4556
    ## 2      2 8663.686 0.03104806 3037.205 1408.140 0.04340353 251.1840
    ## 3      3 8687.443 0.01789690 3019.944 1454.459 0.01510084 257.6219
    ## 4      4 8695.803 0.01710375 3031.769 1466.932 0.01418705 275.4545
    ## 5      5 8717.373 0.01474396 3041.916 1476.013 0.01422552 277.9917
    ## 6      6 8724.324 0.01337191 3042.407 1473.543 0.01357449 272.5588
    ## 7      7 8724.431 0.01315512 3046.903 1466.807 0.01269102 271.5935
    ## 8      8 8733.472 0.01336679 3045.953 1472.548 0.01257292 272.7083
    ## 9      9 8736.249 0.01321570 3046.264 1477.577 0.01269465 269.9529
    ## 10    10 8741.898 0.01292561 3042.131 1479.939 0.01241773 267.8469
    ## 11    11 8752.945 0.01055014 3050.838 1456.388 0.01053229 264.3318
    ## 12    12 8738.087 0.01299714 3046.723 1469.307 0.01222898 260.2246
    ## 13    13 8739.594 0.01292161 3053.133 1470.649 0.01194432 263.0806
    ## 14    14 8738.625 0.01280733 3054.817 1469.724 0.01196256 265.3539
    ## 15    15 8741.327 0.01303178 3048.407 1476.151 0.01227460 270.1190
    ## 16    16 8737.866 0.01407029 3052.564 1485.954 0.01334514 263.6619
    ## 17    17 8742.090 0.01307932 3054.131 1475.513 0.01221078 265.7675
    ## 18    18 8742.494 0.01304095 3053.837 1474.310 0.01217293 267.7958
    ## 19    19 8741.100 0.01332990 3055.955 1477.701 0.01252078 266.6986
    ## 20    20 8741.723 0.01322997 3054.771 1476.220 0.01239190 267.1839

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

    ##   intercept     RMSE   Rsquared      MAE   RMSESD RsquaredSD   MAESD
    ## 1      TRUE 8312.948 0.04162089 3028.061 1171.275 0.06872489 222.674

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

    ##   intercept     RMSE   Rsquared      MAE  RMSESD RsquaredSD    MAESD
    ## 1      TRUE 8741.723 0.01322997 3054.771 1476.22  0.0123919 267.1839

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

    ##         PC1       PC2      PC3       PC4        PC5         PC6        PC7         PC8       PC9        PC10
    ## 1 0.7701039 -2.718143 2.778664 -1.806332 -0.9930002  0.47090564 -0.1413708 -1.10911273 0.8735269  0.88254413
    ## 2 0.2991286 -2.651044 3.000036 -2.423090  1.2981953 -0.07814979 -0.2619246 -0.78429597 0.7756287  0.01849173
    ## 3 1.7363815 -2.654221 2.215671 -1.998729 -1.4380806  1.90602365 -0.1001790  0.15468592 0.2450467  0.47668955
    ## 4 0.6406039 -3.387213 2.526563 -1.341653 -0.9752363  0.59714632  0.1004311 -0.07014842 0.9035593 -0.28156661
    ## 5 1.1183910 -3.748250 2.331680 -1.777817 -0.8682650  1.07699934  0.3507416  0.80899540 0.8537205  0.79998534
    ##          PC11        PC12        PC13       PC14  res
    ## 1  0.51816243 -1.39456950 -0.01850893  1.4027068  593
    ## 2  0.21341906 -0.07211077  0.11047881 -0.4830124 1200
    ## 3 -0.09135378 -0.13734842  0.79214988  0.5987597 1200
    ## 4 -0.13085528 -1.10210871 -0.13708059  0.6503949 4600
    ## 5  0.02348973  0.49245277 -1.40547751 -1.8334525  631

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

    ##          PC1       PC2        PC3         PC4        PC5        PC6         PC7         PC8       PC9        PC10
    ## 1  0.6044592 -6.137403  0.8586422  0.71705212 -0.7300889  1.2751267 -0.09585045  0.42619642 0.8516867 -1.30328831
    ## 2  1.4768978 -4.887558  1.6767083 -0.71490316 -2.6449481  1.2396468  0.16668273  0.11393523 0.4164997  0.07074497
    ## 3 -0.4795112 -3.535795  0.4113324 -3.63412388 -2.2447241  1.4449558  0.13254606 -0.09783769 1.8038558 -0.43693979
    ## 4 -0.1784589 -5.206086  1.2331465 -1.63923386  1.2629670 -0.1587659  0.36912268  0.74518931 3.4870436 -0.07526952
    ## 5  3.3258514 -9.178730 -1.6437238  0.01654149  0.5032508 -0.6356186  0.79032895  1.83257276 3.0685082 -0.69433733
    ##          PC11       PC12       PC13       PC14 test_res
    ## 1 -1.47886947 -1.7656068  0.1322932  2.0146754     2100
    ## 2 -0.44928498 -0.7631542 -0.2834098  0.3961791     1200
    ## 3  1.21990152  0.1265284  1.0388441 -0.2206698    19400
    ## 4 -0.08927557 -0.6959685  0.4159557  0.3578151     2700
    ## 5 -2.98667037  0.4848775 -0.9397003 -0.5164115      904

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

<img src="data_channel_is_entertainmentAnalysis_files/figure-gfm/boost-1.png" style="display: block; margin: auto;" />

``` r
boostfit$bestTune
```

    ##    n.trees interaction.depth shrinkage n.minobsinnode
    ## 33     200                 1       0.1              3

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
    ## LinearRegression_Stepwise_Sel    6494.540 0.023154900 2810.239
    ## LinearRegression_Orignal_Var_Sel 6494.540 0.023154900 2810.239
    ## Linear_Regression_PCA            6531.368 0.017059407 2779.967
    ## Boosted_Trees                    6516.248 0.014908315 2807.884
    ## Random_Forest                    6557.816 0.006200649 2818.416

``` r
best_model<-Metrics_df[which.min(Metrics_df$RMSE), ]
rmse_min<-best_model$RMSE
r2max<-best_model$Rsquared
```

The best model for predicting the number of shares for dataset called
data_channel_is_entertainment in terms of RMSE is
LinearRegression_Stepwise_Sel with RMSE of 6494.54. This model also
showed an R-Squared of about 0.0231549, which shows that the predictors
of this model do not have significant impact on the response variable.

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

<img src="data_channel_is_entertainmentAnalysis_files/figure-gfm/plots-1.png" style="display: block; margin: auto;" />

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
