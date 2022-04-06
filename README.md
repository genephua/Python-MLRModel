# Python-MLRModel
## Transformation of Variables in Multi Linear Regression Models to Solve Multi-Collinearity of Predictors, Non-Normality and Non-Homoscedasticity of Residuals

### Project Details
The project aims to solve the Multi-Collinearity, Non-Normality and Non-Homoscedasticity present in a multi-linear regression model by transforming the variables and using Principal Component Analysis.

### Input features of the dataset
The dataset contains data about a student's achievement score which was measured by using an index constructed from some standardized test scores, as well as the possible predictors of this achievement score.

**Response Variable:** Achievement

**Predictor Variables:** School, Family, Peer

(a)	Model 1: Additive MLR Model consisting of all 3 predictors:
Achieve ~ School + Family + Peer

Visual Analysis

  


 

Preliminary analysis on the predictors in Model 1 through visual charts such as the heat map and pair plot show that there is high multi collinearity between the predictors. The heat map shows that between each pair of predictors, the strength of the correlation is extremely close to one. The scatter plots in the pair plot for each pair of predictors also show a very strong linear relation which signals high collinearity.

Numerical Analysis
 

The severe multicollinearity issue present between predictors is further shown by the high VIF values of the 3 predictors all of which are over 10. Having a VIF figure greater than 10 points to high multicollinearity being present.

Checking on the normality assumption of the residuals for the MLR model, it can be seen that the Omnibus and Jarque-Bera P values are < 0.05 which suggests that the residuals are not normal.

  


 

Checking on the homoscedastic assumption of the residuals for the MLR model, by calculating the Breusch pagan p value, the p value of 0.3019 suggests that the residuals are homoscedastic. However, plotting a residual graph shows that the residuals are not actually homoscedastic and suggests that the fit of the model can be improved.

In summary, Model 1 has the following issues: 1) residuals are not normal, 2) residuals are not homoscedastic and 3) there is high multicollinearity between the predictors. Therefore, to first solve the two issues on the residuals, we can look at transforming the response variable which leads us to Model 2.

Model 2: Investigate log-transformation of response variable Achieve:
log(Achieve) ~ School + Family + Peer

Checking Skewness of Distribution of ‘Achieve’ and of the Residuals

 

Looking at the histogram of both the Achieve response variable and of the residuals, it can be seen that they are both of a right skewed distribution. To reduce the right skewness present, the logarithm of the response variable Achieve could be taken to transform it

  

Analysis
Looking at the Omnibus and Jarque-Bera P values of 0.756 and 0.578 respectively, it can be seen that the residuals are now normal (in comparison to figures in model 1) as a result of transforming the response variable.

  
Running the Breusch Pagan test, we also see that the P value is now 0.944. This is an increase in the figure as compared to model 1, which suggests that the homoscedasticity of the residuals has increased. Cross checking this by looking at the residual plot, it can be seen that the fit of the model has improved as compared to model 1, as the residuals now are more evenly located around the centre of the plot. However, the fit can still be improved. 

Now that the issues surrounding the normality and homoscedasticity of the model has been resolved, we can next solve the multicollinearity issue surrounding the predictors. To do this, we can use PCA to transform the predictors.

Model 3: Achieve ~ PC1
 
 

To use the PCA method, we first have to use the PCA function from the sklearn.decomposition to create the principal components. After that, we will then be able to extract the eigen values, eigen vectors and proportion of the variance that each PC is responsible for. As can be seen from above, PC1 accounts for almost all of the variance with it accounting for 98.4% of the variance for the data.

We next have to use the pca.transform function to transform the original 3 predictors into 3 principal components. 	

 
  
Plotting the scatterplot of the 3 predictors PC1, PC2 and PC3, it can be seen that there is practically no linear correlation between each principal component. This is expected, as by design, principal components are not to be correlated with each other.  
     
 
Plotting the scatter plots between the principal components and the predictors, it can be seen that only PC1 has a strong linear relation with each of the other predictors. This is to be expected as PC1 accounts for 98.4% of the variance in the data. As such only PC1 should be selected and is sufficient to represent the model.
 
The VIF of PC1 is 1, and this proves that the multicollinearity issue that existed before, before using PC1 in place of the 3 other predictors has now been resolved.
(b)	Comparison of Model Fit and Multicollinearity of Model 1, Model 2 and Model 3
	Model 1	Model 2	Model 3
R2	0.043	0.206	0.183
R2 Adj	-0.000	0.170	0.171
AIC	642.2	304.4	302.4
BIC	651.2	313.4	306.9
VIF of Family	37.581	37.581	-
VIF of Peer	30.212	30.212	-
VIF of School	83.155	83.155	-
VIF of PC1	-	-	1

Model Fit
Model 3 has the best fit followed by model 2 and then model 3. The values of R2, R2 Adj , AIC and BIC have decreased from Model 1 to Model 2, and have also continued to decrease from Model 2 to Model 3 albeit on a smaller subsequent magnitude. Model 2 accounted for the largest increase in fit from Model 1 as the response factor was transformed and its log was taken to improve the fit of its residuals. This transformation helped to fix the non-normality and non-constant variance of the residuals which were present in Model 1.

Multicollinearity
Model 3 has the least multicollinearity as compared to Model 1 and Model 2. PC1 was created to solve the multicollinearity issue by replacing the three predictors in Model 3 with it, as it accounts for most (98.45) of the variance in the data. PC1 has a VIF of 1 which is much lower than the VIF of the 3 predictors for Model 1 and Model 2. Both Model 1 and Model 2 have the exact same high VIF figures which shows that both Model 1 and Model 2 have severe multicollinearity amongst their predictors. It is not surprising that the Model 2 has the same VIF figures as Model 1, as nothing was changed in Model 2 from Model 1 to address the multicollinearity issue. 
