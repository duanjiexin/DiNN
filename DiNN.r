###############################################################
# Statistical Guarantees of Distributed Nearest Neighbor Classification 
# DiNN, NeurIPS 2020
# By Jiexin Duan  
# Department of Statistics, Purdue University   10/15/2020
###############################################################

##Cleaning up the workspace
rm(list=ls())

library(snn)  # for nearest neighbor classification
library(MASS) # for mvrnorm function in data generation

# setup of parameters
## parameter of simulation
rep_sim = 1000 # replication times for simulation

##parameters of dataset
d = 8  # dimension of dataset
mu= 2/(d^0.5) # mu of class2; mu of class1 (mu1=0)
portion = 1/3   # portion of first class (mu1=0)
ntest=1000 # number of observation in test data

NO = 27000  # original size of oracle dataset

gamma = 0.3   # input the gamma vector for number of machines; gamma=(log(s))/(log(N)), i.e. s=N^gamma; gamma=0 for oracle kNN
if(((floor(NO^gamma))%%2)==1) s=floor(NO^gamma) else s=ceiling(NO^gamma) #number of machines/subsets, nearest odd number
if(NO-s*floor(NO/s)<s*ceiling(NO/s)-NO)   n=floor(NO/s)  else n=ceiling(NO/s) #find floor(N/s),ceiling(N/s) which is nearer to N

N=s*n   # size for the oracle big data (adjusted after splitting)
alpha =0.5 # voting threshold for the voting scheme of DiNN
k_O = N^0.7  # k for oracle kNN in oracle dataset
if((floor(k_O)%%2)==1) k_O=floor(k_O) else k_O=ceiling(k_O)
if(gamma==0){ 
  k_m=k_O/s     # i.e. gamma=0 <=> oracle kNN
}
if(gamma!=0){ 
  k_m=((pi/2)^(d/(d+4)))*k_O/s # k_m for knn (each subset) in M-DiNN
}
if((floor(k_m)%%2)==1) k_m=floor(k_m) else k_m=ceiling(k_m)
k_w=k_O/s   # k_w for knn (each subset) in W-DiNN
if((floor(k_w)%%2)==1) k_w=floor(k_w) else k_w=ceiling(k_w)


#######################################################################


###########################################################
#  Calculation of Bayes Risk by Monte Carlo Simulation
###########################################################
library(mnormt)   # the multivariate normal and t distributions
NN=round(1000000/d)  # simulation times
prop = portion
n1=floor(prop*NN)   # number of first class
n2=NN-n1        # number of second class
myx = rbind(mvrnorm(floor(n1/2), rep(0,d), diag(d)),mvrnorm(n1 - floor(n1/2), rep(3,d), 2*diag(d)),mvrnorm(floor(n2/2), rep(1.5,d), diag(d)),mvrnorm(n2 - floor(n2/2), rep(4.5,d), 2*diag(d)))
eta=function(x){    
  f1=log(0.5*dmnorm(x,rep(0,d), diag(d),log=FALSE)+0.5*dmnorm(x,rep(3,d), 2*diag(d),log=FALSE)) 
  f2=log(0.5*dmnorm(x,rep(1.5,d), diag(d),log=FALSE)+0.5*dmnorm(x,rep(4.5,d), 2*diag(d),log=FALSE)) 
  1/(1+((1-prop)/prop)*exp(f2-f1))
}
obj = function(x){
  ifelse(eta(x)>1/2,1-eta(x),eta(x))
}
risk_Bayes = mean(obj(myx)) 
risk_Bayes = round(risk_Bayes, digits=6); risk_Bayes
xx1 <-0; xx2 <-0; myx <- 0  # remove myx to save space

#######################################################################

##############################################################################
# Two forms of wnn and knn, output is Local Average Regression and predicted class.
# Generate results for M-DiNN and W-DiNN
############################################################################

mycwnn = function (train, test, weight) 
{
  train = as.matrix(train)
  n = dim(train)[1]
  d = dim(train)[2] - 1
  X = as.matrix(train[, 1:d])
  Y = train[, d + 1]
  Ysort = rep(0, n)
  dist = function(x) {
    sqrt(t(x - test) %*% (x - test))
  }
  Dis = apply(X, 1, dist)
  Ysort = Y[order(Dis)]
  label = sum(weight[which(Ysort == 1)])
  return(label)
}


mycknn = function (train, test, K) 
{
  n = dim(train)[1]
  weight = rep(0, n)
  weight[1:K] = 1/K
  if (is.vector(test) == TRUE) {
    if (dim(train)[2] - 1 == 1) {
      test.mat = as.matrix(test)
    }
    else {
      test.mat = t(as.matrix(test))
    }
  }
  else {
    test.mat = test
  }
  if (dim(test.mat)[2] != (dim(train)[2] - 1)) 
    stop("training data and test data have different dimensions")
  label = 2- apply(test.mat, 1, function(x) mycwnn(train, x, weight))  
  return(label)
}

#######################################################################

#######################################################################
# data generating function for simulation 2 in numerical analysis section
#######################################################################

mydata_sim2 = function(n, d, portion){
  n1 = floor(n*portion)
  n2 = n - n1	
  X1 = rbind(mvrnorm(floor(n1/2), rep(0,d), diag(d)),mvrnorm(n1 - floor(n1/2), rep(3,d), 2*diag(d)))     				
  X2 = rbind(mvrnorm(floor(n2/2), rep(1.5,d), diag(d)),mvrnorm(n2 - floor(n2/2), rep(4.5,d), 2*diag(d)))     					
  data1 = rbind(X1,X2)
  y = c(rep(1,n1),rep(2,n2))
  DATA=cbind(data1,y)
  DATA
}

#####################################################################


# simulation 
for(i_rep in 1:rep_sim){
  
  ###################################################################################
  # Oracle Training data for classifier,  generate N obs
  # DATA1 is used for calculation of Risk
  # If we have real data, we can read it directly
  ####################################################################################
  
  time0_i = Sys.time()  #start time for simulation i 
  
  # generalized training set as dataframe
  DATA_oracle1 = as.data.frame(mydata_sim2(N, d, portion))  

  # genelize testing dataset
  TEST = cbind(mydata_sim2(ntest, d, portion))
  
  ####################################################
  #  Divide & Conquer Process
  ####################################################
  
  # initialize prediect.sum list to store sum of prediction before mv
  predict1.sum_m = rep(0,ntest)
  predict1.sum_w = rep(0,ntest)
  
  ## Step 1: Divide process
  permIndex_1 = sample(nrow(DATA_oracle1))
  DATA_oracle1 = DATA_oracle1[permIndex_1,]

  # Step 2: run base NN on subsets
  time_divide_m = 0
  time_divide_w = 0
  index = 1
  for(j in 1:s){
    # M-DiNN
    time_temp0_m = Sys.time()
  
    # Training data for classifier \phi
    DATA1 = DATA_oracle1[index:(index+n-1),]

    # get preditive value from trained classfier on testing data
    predict1_m = myknn(DATA1, TEST[,1:d], k_m)

    # sum all predicts for majority voting
    predict1.sum_m = predict1.sum_m + predict1_m

    time_temp1_m = Sys.time()
    time_temp_m = time_temp1_m - time_temp0_m
    time_temp_m = round(as.numeric(time_temp_m, units = "secs"),digits=2) 
    if(time_temp_m >= time_divide_m) time_divide_m = time_temp_m 
    
    # start calculate W-DiNN dividing time 
    time_temp0_w = Sys.time()
    
    # get preditive value from trained classfier on testing data
    predict1_w = mycknn(DATA1, TEST[,1:d], k_w)

    predict1.sum_w = predict1.sum_w + predict1_w

    time_temp1_w = Sys.time()
    time_temp_w = time_temp1_w - time_temp0_w
    time_temp_w = round(as.numeric(time_temp_w, units = "secs"),digits=2) 
    if(time_temp_w >= time_divide_w) time_divide_w = time_temp_w 
  
    # moving index to beginning of next subset
    index = index + n
  }
    
  # Step3: Majority voting process 
  time_wombine0_m = Sys.time()
  
  # Majoriting voting for M-DiNN
  predict1.mv_m = ifelse(s^(-1)*predict1.sum_m > 2-alpha, 2, 1)

  # Store risk value
  risk_big_m = myerror(predict1.mv_m, TEST[,d+1])

  time_wombine1_m = Sys.time()
  time_wombine_m = time_wombine1_m - time_wombine0_m
  time_wombine_m = round(as.numeric(time_wombine_m, units="secs"),digits=2) # display 2 digits by unit secs
  time_m = time_divide_m + time_wombine_m 
  
  time_wombine0_w = Sys.time()
  
  # Weighted voting for W-DiNN
  predict1.mv_w = ifelse(s^(-1)*predict1.sum_w > 2-alpha, 2, 1)

  # Store risk value
  risk_big_w = myerror(predict1.mv_w, TEST[,d+1])
  
  time_wombine1_w = Sys.time()
  time_wombine_w = time_wombine1_w - time_wombine0_w
  time_wombine_w = round(as.numeric(time_wombine_w, units="secs"),digits=2) # display 2 digits by unit secs
  time_w = time_divide_w + time_wombine_w 
  
  # Print results together
  print(paste(NO,gamma,N,s,n,d,portion,mu,k_m,k_w,k_O,risk_big_m,risk_big_w,risk_Bayes,time_m,time_w)) 
  
}


