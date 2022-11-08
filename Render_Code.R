selectID <- unique(newData$channel)  # from notes this should work

#create filename
output_file <- paste0(selectID, "Analysis.html")   #should be .md for the project
#create a group for each team with just the team name parameter
params = lapply(selectID, FUN = function(x){list(channel = x)})

#put into a data frame
reports <- tibble(output_file, params)

library(rmarkdown)

apply(reports, MARGIN = 1,
      FUN = function(x){
        render(input = "./TestOne.Rmd",
               output_format = "github_document", 
               output_file = x[[1]], 
               params = x[[2]])
      })