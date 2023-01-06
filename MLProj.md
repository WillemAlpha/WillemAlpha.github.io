---
title: "ML Proj"
author: "Willem Abrie"
date: "2023-01-06"
output: 
  html_document: 
    keep_md: yes
---



## Abstract
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

## Executive Summary
Variables with little or no data in them were excluded from the analysis. A validation set was set aside using 25% of the testing data.

A decision tree was used for this classification problem. A CART model was computationally efficient but resulted in low accuracy (about 47%). Hence a boosted model was used which resulted in 92% accuracy.


## System Info

The following version of RStudion was used: RStudion 2022.07.2 Build 576.
For more info on packages see the session info in the Appendix.


## Data Gathering

Data was downloaded directly from the links that were provided with the assignment.


```r
rm(list = ls())
training <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
testing <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
```

```{r`exploration}

```

## Data Wrangling

The columns with little or no information were dropped. The classe and user name variables were converted to factors.

```r
training1 <- training[, c(2, 8:11, 37:49, 60:68, 84:86, 160)]
training1$user_name <- factor(training1$user_name)
training1$classe <- factor(training1$classe)
#summary(training1)

test <- testing[, c(2, 8:11, 37:49, 60:68, 84:86, 160)]
test$user_name <- factor(test$user_name)
```

## Model Training

A validation set was split off from the training data. We trained a boosted model on the remaining training data. It was quite slow.

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 4.2.2
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 4.2.2
```

```
## Loading required package: lattice
```

```r
set.seed(100)
part <- createDataPartition(training1$classe, p = 3/4, list = FALSE)
train <- training1[part, ]
validate <- training1[-part, ]

start.time <- Sys.time()
model1 <- train(classe ~ ., method = "gbm", data = train, verbose = FALSE)
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken
```

```
## Time difference of 29.66611 mins
```

```r
print(model1)
```

```
## Stochastic Gradient Boosting 
## 
## 14718 samples
##    30 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 14718, 14718, 14718, 14718, 14718, 14718, ... 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa    
##   1                   50      0.6585643  0.5648746
##   1                  100      0.7132887  0.6361029
##   1                  150      0.7486451  0.6815850
##   2                   50      0.7737696  0.7134316
##   2                  100      0.8373509  0.7942030
##   2                  150      0.8715264  0.8374470
##   3                   50      0.8326957  0.7882694
##   3                  100      0.8898414  0.8606355
##   3                  150      0.9159570  0.8937034
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were n.trees = 150, interaction.depth =
##  3, shrinkage = 0.1 and n.minobsinnode = 10.
```

```r
plot(model1)
```

![](MLProj_files/figure-html/ModelFit-1.png)<!-- -->

## Validation
The model was validated by predicting on the validation set and comparing to the reference values in the classe variable. The outcome was favourable with a 92% accuracy. It was decided that this is adequate, even though no preprocessing like transformations or PCA's were performed.


```r
predClasse <- predict(object = model1, newdata = validate)
confusionMatrix(validate$classe, predClasse)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1307   29   18   27   14
##          B   53  844   37    7    8
##          C   15   51  765   19    5
##          D   13    5   39  746    1
##          E   16   12   10    4  859
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9219         
##                  95% CI : (0.914, 0.9293)
##     No Information Rate : 0.2863         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9012         
##                                          
##  Mcnemar's Test P-Value : 0.003668       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9309   0.8969   0.8803   0.9290   0.9684
## Specificity            0.9749   0.9735   0.9777   0.9859   0.9895
## Pos Pred Value         0.9369   0.8894   0.8947   0.9279   0.9534
## Neg Pred Value         0.9724   0.9755   0.9743   0.9861   0.9930
## Prevalence             0.2863   0.1919   0.1772   0.1637   0.1809
## Detection Rate         0.2665   0.1721   0.1560   0.1521   0.1752
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9529   0.9352   0.9290   0.9574   0.9790
```
## Prediction of the 20 test set values


```r
predTest <- predict(object = model1, newdata = test)
predTest
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

## Conclusion

The boosted decision tree was computationally slow but had a good outcome. All 20 test values proved to be correct, resulting in a 100% score when submitted on the Coursera assessment form.

# Appendix

## Session Info


```r
sessionInfo()
```

```
## R version 4.2.0 (2022-04-22 ucrt)
## Platform: x86_64-w64-mingw32/x64 (64-bit)
## Running under: Windows 10 x64 (build 19044)
## 
## Matrix products: default
## 
## locale:
## [1] LC_COLLATE=English_Australia.utf8  LC_CTYPE=English_Australia.utf8   
## [3] LC_MONETARY=English_Australia.utf8 LC_NUMERIC=C                      
## [5] LC_TIME=English_Australia.utf8    
## 
## attached base packages:
## [1] stats     graphics  grDevices utils     datasets  methods   base     
## 
## other attached packages:
## [1] caret_6.0-93    lattice_0.20-45 ggplot2_3.4.0  
## 
## loaded via a namespace (and not attached):
##  [1] Rcpp_1.0.9           lubridate_1.9.0      listenv_0.8.0       
##  [4] class_7.3-20         assertthat_0.2.1     digest_0.6.29       
##  [7] ipred_0.9-13         foreach_1.5.2        utf8_1.2.2          
## [10] parallelly_1.32.1    R6_2.5.1             plyr_1.8.8          
## [13] stats4_4.2.0         hardhat_1.2.0        e1071_1.7-12        
## [16] evaluate_0.18        highr_0.9            pillar_1.8.1        
## [19] rlang_1.0.6          data.table_1.14.4    rstudioapi_0.14     
## [22] jquerylib_0.1.4      rpart_4.1.16         Matrix_1.5-3        
## [25] rmarkdown_2.18       splines_4.2.0        gower_1.0.0         
## [28] stringr_1.4.1        munsell_0.5.0        proxy_0.4-27        
## [31] compiler_4.2.0       xfun_0.32            pkgconfig_2.0.3     
## [34] gbm_2.1.8.1          globals_0.16.2       htmltools_0.5.3     
## [37] nnet_7.3-17          tidyselect_1.2.0     tibble_3.1.8        
## [40] prodlim_2019.11.13   codetools_0.2-18     fansi_1.0.3         
## [43] future_1.29.0        dplyr_1.0.10         withr_2.5.0         
## [46] ModelMetrics_1.2.2.2 MASS_7.3-56          recipes_1.0.3       
## [49] grid_4.2.0           nlme_3.1-157         jsonlite_1.8.3      
## [52] gtable_0.3.1         lifecycle_1.0.3      DBI_1.1.3           
## [55] magrittr_2.0.3       pROC_1.18.0          scales_1.2.1        
## [58] future.apply_1.10.0  cli_3.3.0            stringi_1.7.8       
## [61] cachem_1.0.6         reshape2_1.4.4       timeDate_4021.106   
## [64] bslib_0.4.1          generics_0.1.3       vctrs_0.5.0         
## [67] lava_1.7.0           iterators_1.0.14     tools_4.2.0         
## [70] glue_1.6.2           purrr_0.3.5          parallel_4.2.0      
## [73] fastmap_1.1.0        survival_3.3-1       yaml_2.3.6          
## [76] timechange_0.1.1     colorspace_2.0-3     knitr_1.40          
## [79] sass_0.4.2
```



