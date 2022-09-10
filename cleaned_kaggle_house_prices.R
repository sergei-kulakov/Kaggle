setwd("C:\\pythonProject\\My Website\\djangoenv\\mywebsite\\Data R und Python")


library(glmnet)
library(xgboost)
library(randomForest)

train<- read.csv("train.csv")
test <- read.csv("test.csv")

#combining the data for cleaning the outliers
data = rbind(train,cbind(test, SalePrice=NA), deparse.level = 1)

#remvoing the Utilities column because the matrix in the test sample will not be full rank
data = data[,-10] 
datadim = dim(data)

#defining the function for the lasso/ridge regression 
LAMBDA <- 2^seq(1,-8, length.out=40)
RIDGE <-  function(XREG,YREG, alpha_coef){
		model<- glmnet(XREG, YREG, alpha=alpha_coef, standardize=TRUE, lambda=LAMBDA)
		#Selecting the optimal cofficient
		BETA<- as.matrix(model$beta) 
		allRES<- as.numeric(YREG) - (apply(XREG,2,as.numeric) %*% as.matrix(BETA))
		RSS<-colSums(allRES*allRES) # residual sums of squares
		IND<- apply(BETA!=0,2, which)
		DF<- sapply(IND, length) ## number of parameters ()
		nobs<- dim(XREG)[1] ## number of parameters 
		kappa<- 2 #AIC criterion, the more relaxed one 
		IC<-  log(RSS) + kappa*DF/nobs # HQC
		tmpind<- which.min(IC) ## index of minimal IC
		BETA.opt<- as.matrix(BETA[ , tmpind ]) ## IC-chosen beta
		return(rbind(model$a0[tmpind], BETA.opt))	
		}
	
#removing Xs from data colnames	
first_X_letters = which(substr(colnames(data),1,1) == "X")
colnames(data)[first_X_letters] = sub(".","",colnames(data)[first_X_letters]) #removing first X character



###
###
###
###
#SECTION WITH MANUAL FIXES	
	
#Mapping of quality descirptions into numbers
initial_mapping_manual = c(NA, "Fa", "Po", "TA", "Gd", "Ex")
manual_columns = c("ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual", 
	"FireplaceQu", "GarageQual", "GarageCond", "PoolQC")
for (column in manual_columns){
	data[,column] = match(data[,column], initial_mapping_manual)-1 #sequence is always fixed 
	}

#checking for NAs and zeroing them out in some cases; from here we see items where there is too many NAs
apply(data, 2, function(x){length(which(is.na(x)))} )

#PoolQC fix
data[which(data[,"PoolArea"]==0),"PoolQC"] = 0 

#CentralAir fix
data[,"CentralAir"] = match(data[,"CentralAir"], c("Y", "N"))-1

#Zero NAs fix
additional_paramteres = c("Alley", "Fence", "MiscFeature", "GarageCars", "GarageArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF")
for (column in additional_paramteres){
	current_NA_index = which(is.na(data[,column])==TRUE)
	data[current_NA_index,column] = 0 
	}

#GarageType/GarageYrBuilt fix
additional_paramteres_garage = c("GarageType", "GarageYrBlt", "GarageFinish")
for (column in additional_paramteres_garage){
	current_NA_index = which(data[,"GarageQual"]==0)
	data[current_NA_index,column] = 0 
	}

#
additional_paramteres_bsmt = c("BsmtExposure", "BsmtFinType1", "BsmtFinType2")
for (column in additional_paramteres_bsmt){
	current_NA_index = which(data[,"BsmtCond"]==0)
	data[current_NA_index,column] = 0 
	}


#MasVnrType
NA_MasVnrType_entries = which(is.na(data[,"MasVnrType"]))
data[NA_MasVnrType_entries,"MasVnrType"] = "brkface"

#false indexing
index_false_nones = which(data[,"MasVnrArea"] == 0 & data[,"MasVnrType"] != "none")
data[index_false_nones, "MasVnrType"] = "none"

index_false_area = which(data[,"MasVnrArea"] != 0 & data[,"MasVnrType"] == "none")
data[index_false_nones, "MasVnrType"] = "brkface"

#Negative Fix Mas VnrType
data[which(data[,"MasVnrArea"]<0),"MasVnrArea"] = 3


###Outlier/missings process in the quantitative part 
numeric_conversion = apply(data[,-datadim[2]],2, as.numeric)
quantitative_variables = which(apply(numeric_conversion,2,sum, na.rm=T)>0)
quantitative_nas = apply(data[,quantitative_variables],2,function(x){which(is.na(x))}) 
quantitative_nas_cleaned = quantitative_nas[sapply(quantitative_nas,length)>0]
quantitative_nas_index = which(is.na(match(colnames(data), names(quantitative_nas_cleaned)))==FALSE)


#setting the NA values to -1 to be able to create a sparse matrix
for(column in names(quantitative_nas_cleaned)){
	data[quantitative_nas_cleaned[[column]],column]=  -1
	}

#Preparing the creating of the sparse matrix
test_index = (dim(train)[1]+1):dim(data)[1]
data[test_index,datadim[2]]=test_index #dummy values for the SalePrice in the test sample
colnames(data) = paste("X.",colnames(data), sep="") 

#Remaining ones are simply replaced with median values
#Certainly not the most optimal approach - solution with lm() or cor() would be better - but this would do the job for now
remaining_columns = which(apply(data, 2, function(x){length(which(is.na(x)))} )>0)
for (column in remaining_columns){
	index = which(is.na(data[,column]))
	data[index,column] = names(sort(table(data[,column]), decreasing=T))[1]
	}

#creating sparse matrix and hten cleaning up NAs in the matrix
sparse_matrix <- sparse.model.matrix(X.SalePrice~.-1, data = data)
NA_columns_initial = apply(sparse_matrix, 2, function(x){length(which(x==-1))})
NA_columns = names(NA_columns_initial[which(NA_columns_initial!=0)])

#Using Lasso regresion to fill in the outliers
for (current_NA_column in NA_columns){
	current_column = which(colnames(sparse_matrix)==current_NA_column)
	current_NA_index = which(sparse_matrix[,current_column]==-1)
	current_NA_column_index = which(!apply(sparse_matrix[current_NA_index,]==-1, 2, any))
	regrindex = which( !apply(sparse_matrix==-1,1, any)) 
	XREG = sparse_matrix[regrindex,current_NA_column_index]
	YREG = sparse_matrix[regrindex,current_column]
	model_output = RIDGE (XREG, YREG, 0) #seems to work better with ridge 
	sparse_matrix[current_NA_index, current_column] = round(as.numeric(
								model_output[1,1]+ 
								t(model_output[2:dim(model_output)[1],]) %*% 
								t(sparse_matrix[current_NA_index,current_NA_column_index])
								))
		}
	


output_vector= data[, datadim[2]]

#Estimating the betas / making the predictions 

xgboost_model <- xgboost(data = sparse_matrix[-test_index,], label=output_vector[-test_index], max.depth = 2, eta = 1, nthread = 2, nrounds = 15000, verbose = 0)
xgboost_prediction = predict(xgboost_model, sparse_matrix[test_index,])


RIDGE_model= RIDGE(sparse_matrix[-test_index,], output_vector[-test_index],0)
RIDGE_prediction = t(RIDGE_model[1,1] + t(RIDGE_model[-1,]) %*% t(sparse_matrix[test_index,]))

LASSO_model= RIDGE(sparse_matrix[-test_index,], output_vector[-test_index],1)
LASSO_prediction = t(LASSO_model[1,1] + t(LASSO_model[-1,]) %*% t(sparse_matrix[test_index,]))


RF_model = randomForest(x = as.matrix(sparse_matrix[-test_index,]), y=as.matrix(output_vector[-test_index]))
RF_prediction = predict(RF_model, sparse_matrix[test_index,])


#Combining models 
forecast = 0.1*xgboost_prediction + 
		0.1* as.vector(RIDGE_prediction) +
		0.3* as.vector(LASSO_prediction) +
		0.5* as.vector(RF_prediction)
 
submit_mat = cbind(1461:(1461+1458), forecast)
colnames(submit_mat)=c("Id","SalePrice") 
#Manual fix for expensive ones ;-) 
submit_mat[which(submit_mat[,2]>440000),2] = submit_mat[which(submit_mat[,2]>440000),2]+140000


write.csv(as.matrix(submit_mat), "new_submission.csv", row.names=FALSE)
