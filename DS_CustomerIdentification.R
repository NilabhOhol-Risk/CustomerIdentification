setwd("C:/Users/imse6/Desktop/PhD_Industrial/Uptake/")

response <- read.table("zip_cleaned.txt", header = TRUE)
zip.mkt <- read.table("zipCodeMarketingCosts.txt", header = TRUE)

for(i in 1:nrow(response))
{
  for(j in 1:nrow(zip.mkt))
  {
    if(response[i,1]==zip.mkt[j,2])
      response[i,1]=zip.mkt[j,1]
    else
      response[i,1]=response[i,1]
  }
}
write.csv(response,paste('response','csv',sep = '.'))
