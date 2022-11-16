## Predictive Models for Online New Popularity Data Set 

The purpose of this repository is to create predictive models and automating R Markdown reports. Analysis are completed on the Online News Popularity Data Set from UCI. Additional information about this data can be accessed [here](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity).

#### The data contains the following variables:

0. url: URL of the article (non-predictive)
1. timedelta: Days between the article publication and the dataset acquisition (non-predictive)
2. n_tokens_title: Number of words in the title
3. n_tokens_content: Number of words in the content
4. n_unique_tokens: Rate of unique words in the content
5. n_non_stop_words: Rate of non-stop words in the content
6. n_non_stop_unique_tokens: Rate of unique non-stop words in the content
7. num_hrefs: Number of links
8. num_self_hrefs: Number of links to other articles published by Mashable
9. num_imgs: Number of images
10. num_videos: Number of videos
11. average_token_length: Average length of the words in the content
12. num_keywords: Number of keywords in the metadata
13. data_channel_is_lifestyle: Is data channel 'Lifestyle'?
14. data_channel_is_entertainment: Is data channel 'Entertainment'?
15. data_channel_is_bus: Is data channel 'Business'?
16. data_channel_is_socmed: Is data channel 'Social Media'?
17. data_channel_is_tech: Is data channel 'Tech'?
18. data_channel_is_world: Is data channel 'World'?
19. kw_min_min: Worst keyword (min. shares)
20. kw_max_min: Worst keyword (max. shares)
21. kw_avg_min: Worst keyword (avg. shares)
22. kw_min_max: Best keyword (min. shares)
23. kw_max_max: Best keyword (max. shares)
24. kw_avg_max: Best keyword (avg. shares)
25. kw_min_avg: Avg. keyword (min. shares)
26. kw_max_avg: Avg. keyword (max. shares)
27. kw_avg_avg: Avg. keyword (avg. shares)
28. self_reference_min_shares: Min. shares of referenced articles in Mashable
29. self_reference_max_shares: Max. shares of referenced articles in Mashable
30. self_reference_avg_sharess: Avg. shares of referenced articles in Mashable
31. weekday_is_monday: Was the article published on a Monday?
32. weekday_is_tuesday: Was the article published on a Tuesday?
33. weekday_is_wednesday: Was the article published on a Wednesday?
34. weekday_is_thursday: Was the article published on a Thursday?
35. weekday_is_friday: Was the article published on a Friday?
36. weekday_is_saturday: Was the article published on a Saturday?
37. weekday_is_sunday: Was the article published on a Sunday?
38. is_weekend: Was the article published on the weekend?
39. LDA_00: Closeness to LDA topic 0
40. LDA_01: Closeness to LDA topic 1
41. LDA_02: Closeness to LDA topic 2
42. LDA_03: Closeness to LDA topic 3
43. LDA_04: Closeness to LDA topic 4
44. global_subjectivity: Text subjectivity
45. global_sentiment_polarity: Text sentiment polarity
46. global_rate_positive_words: Rate of positive words in the content
47. global_rate_negative_words: Rate of negative words in the content
48. rate_positive_words: Rate of positive words among non-neutral tokens
49. rate_negative_words: Rate of negative words among non-neutral tokens
50. avg_positive_polarity: Avg. polarity of positive words
51. min_positive_polarity: Min. polarity of positive words
52. max_positive_polarity: Max. polarity of positive words
53. avg_negative_polarity: Avg. polarity of negative words
54. min_negative_polarity: Min. polarity of negative words
55. max_negative_polarity: Max. polarity of negative words
56. title_subjectivity: Title subjectivity
57. title_sentiment_polarity: Title polarity
58. abs_title_subjectivity: Absolute subjectivity level
59. abs_title_sentiment_polarity: Absolute polarity level
60. shares: Number of shares (target)

In this project, subsets by data_channel_is_* were produced for automating R Markdown reports.  Predictive models used include linear regression models,  random forest model, and boosted tree. These models were constructed on training data set and than tested on testing data set. The best model was selected based on lowest RMSE.

#### List of packages used:

1. `caret`         To run the Regression and ensemble methods with Train/Split and cross validation.
2. `dplyr`         A part of the `tidyverse` used for manipulating data.
3. `GGally`        To create ggcorr() and ggpairs() correlation plots .
4. `glmnet`        To access best subset selection.
5. `ggplot2`       A part of the `tidyverse` used for creating graphics.
6. `gridextra`     To plot with multiple grid objects.
7. `gt`            To test a  low-dimensional null hypothesis against high-dimensional alternative models.
8. `knitr`         To get nice table printing formats, mainly for the contingency tables.
9. `leaps`         To identify different best models of different sizes.
10. `markdown`     To render several output formats.
11. `MASS`         To access forward and backward selection algorithms
12. `randomforest` To access random forest algorithms
13. `tidyr`        A part of the `tidyverse` used for data cleaning

#### Links to the view results of analysis.

  The analysis for [Lifestyle articles is available here.](https://pmb-7684.github.io/ST558_Project_3/data_channel_is_lifestyleAnalysis.html)   
  The analysis for [Entertainment articles is available here.](https://pmb-7684.github.io/ST558_Project_3/data_channel_is_entertainmentAnalysis.html)   
  The analysis for [Business articles is available here.](https://pmb-7684.github.io/ST558_Project_3/data_channel_is_busAnalysis.html)     
  The analysis for [Social media articles is available here.](https://pmb-7684.github.io/ST558_Project_3/data_channel_is_socmedAnalysis.html)   
  The analysis for [Tech articles is available here.](https://pmb-7684.github.io/ST558_Project_3/data_channel_is_techAnalysis.html)   
  The analysis for [World articles is available here.](https://pmb-7684.github.io/ST558_Project_3/data_channel_is_worldAnalysis.html)   

#### Code used to create the analysis.
```
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

