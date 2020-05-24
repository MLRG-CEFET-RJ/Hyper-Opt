pre_install_libraries <- function()  
{
  install.packages("caret")
  
  install.packages("glmnet")
  install.packages("leaps")
  
  install.packages("rJava")
  install.packages("RWeka")
  install.packages("RWekajars")
  install.packages("FSelector")
  
  install.packages("doBy")
}

pre_load_libraries <- function() 
{
  library(caret)
  
  library(glmnet)
  library(leaps)
  
  library(rJava)
  library(RWeka)
  library(RWekajars)
  library(FSelector)
  
  library(doBy)
}

sample.random <- function(data, perc=0.8)
{
  idx = sample(1:nrow(data),as.integer(perc*nrow(data)))
  sample = data[idx,]
  residual = data[-idx,]
  return (list(sample, residual))
}

sample.stratified <- function(data, clabel, perc=0.8)
{
  predictors_name  = setdiff(colnames(data), clabel)
  
  predictors = data[,predictors_name] 
  predictand = data[,clabel] 
  
  idx = createDataPartition(predictand, p=perc, list=FALSE)  
  sample = data[idx,]
  residual = data[-idx,]
  return (list(sample, residual))
}

sample.random_kfold <- function(data, k=10)
{
  sets = list()
  p = 1.0 / k
  while (k > 1) {
    samples = sample.random(data, p)
    fold = samples[[1]]
    data = samples[[2]]
    sets = append(sets, list(fold))
    k = k - 1
    p = 1.0 / k
  }
  sets = append(sets, list(data))
  return (sets)
}


sample.stratified_kfold <- function(data, clabel, k=10)
{
  sets = list()
  p = 1.0 / k
  while (k > 1) {
    samples = sample.stratified(data, clabel, p)
    fold = samples[[1]]
    data = samples[[2]]
    sets = append(sets, list(fold))
    k = k - 1
    p = 1.0 / k
  }
  sets = append(sets, list(data))
  return (sets)
}

# OUTLIER REMOVAL
outliers.boxplot <- function(data, clabel, alpha = 1.5, recursive = FALSE)
{
  org = nrow(data)
  q = as.data.frame(lapply(data, quantile))
  n = ncol(data)
  for (i in 1:n)
  {
    if (colnames(data[i]) == clabel)
    {
      next
    }
    IQR = q[4,i] - q[2,i]
    lq1 = q[2,i] - alpha*IQR
    hq3 = q[4,i] + alpha*IQR
    cond = data[,i] >= lq1 & data[,i] <= hq3
    data = data[cond,]
  }
  final = nrow(data)
  if ((recursive) & (final != org))
    return (outliers.boxplot(data, alpha, recursive))
  else
    return (data)
}

# NORMALIZACAO MIN-MAX
normalize.minmax <- function(data, norm.set=NULL)
{
  data = data.frame(data)
  if(is.null(norm.set))
  {
    minmax = data.frame(t(sapply(data, max, na.rm=TRUE)))
    minmax = rbind(minmax, t(sapply(data, min, na.rm=TRUE)))
  }
  else {
    minmax = norm.set
  }
  data = rbind(data, minmax)
  normalize_minmax <- function(x)
  {
    maxd = x[length(x)-1]
    mind = x[length(x)]
    return ((x-mind)/(maxd-mind))
  }
  data = data.frame(sapply(data, normalize_minmax))
  data = data[1:(nrow(data)-2),]
  return (list(data, minmax))
}

# NORMALIZACAO Z-SCORE
normalize.zscore <- function(data, norm.set=NULL)
{
  data = data.frame(data)
  if(is.null(norm.set))
  {
    zscore = data.frame(t(sapply(data, mean, na.rm=TRUE)))
    zscore = rbind(zscore, t(sapply(data, sd, na.rm=TRUE)))
  }
  else
  {
    zscore = norm.set
  }
  data = rbind(data, zscore)
  normalize_zscore <- function(x)
  {
    zmean = x[length(x)-1]
    zsd = x[length(x)]
    return ((x-zmean)/zsd)
  }
  data = data.frame(sapply(data, normalize_zscore))
  data = data[1:(nrow(data)-2),]
  return (list(data, zscore))
}


# FEATURE SELECTION: LASSO
fs.lasso <- function(data, clabel)
{
  predictors_name  = setdiff(colnames(data), clabel)
  
  predictors = as.matrix(data[,predictors_name])
  predictand = data[,clabel]
  grid = 10^ seq (10,-2, length = 100)
  cv.out = cv.glmnet (predictors, predictand, alpha = 1)
  bestlam = cv.out$lambda.min
  out = glmnet(predictors, predictand, alpha = 1, lambda = grid)
  lasso.coef = predict (out,type = "coefficients", s = bestlam)
  l = lasso.coef[(lasso.coef[,1]) != 0,0]
  vec = rownames(l)[-1]
  vec = c(vec, clabel)
  data = data[,vec]
  return (list(data, vec))
}

# FEATURE SELECTION: Forward Stepwise Selection

fs.fss <- function(data, clabel)
{
  predictors_name  = setdiff(colnames(data), clabel)
  
  predictors = as.matrix(data[,predictors_name])
  predictand = data[,clabel]
  
  regfit.fwd = regsubsets(predictors, predictand, nvmax=ncol(data)-1, method="forward")  
  summary(regfit.fwd)
  reg.summaryfwd = summary(regfit.fwd)
  b1 = which.max(reg.summaryfwd$adjr2)
  t = coef(regfit.fwd,b1)
  vec = c(names(t)[-1], clabel)
  return (list(data[,vec], vec))
}

# FEATURE SELECTION: Correlation-based Feature Selection (CFS)

fs.cfs <- function(data, clabel)
{
  class_formula = formula(paste(clabel, "  ~ ."))
  subset = cfs(class_formula, data)
  vec = c(subset, clabel)
  return (list(data[,vec], vec))
}

# FEATURE SELECTION: Information Gain

fs.ig <- function(data, clabel)
{
  class_formula = formula(paste(clabel, "  ~ ."))
  weights = information.gain(class_formula, data)
  
  tab=data.frame(weights)
  tab=orderBy(~-attr_importance, data=tab)
  tab$i=row(tab)
  tab$import_acum=cumsum(tab$attr_importance)
  res = curvature.min(tab$i, tab$import_acum)
  tab = tab[tab$import_acum < res$y,]
  vec = c(rownames(tab), clabel)
  return (list(data[,vec], vec))
}

# FEATURE SELECTION: Relief

fs.relief <- function(data, class)
{
  class_formula = formula(paste(class, "  ~ ."))
  weights = relief(class_formula, data)
  
  tab=data.frame(weights)
  tab=orderBy(~-attr_importance, data=tab)
  tab$i=row(tab)
  tab$import_acum=cumsum(tab$attr_importance)
  res = curvature.min(tab$i, tab$import_acum)
  tab = tab[tab$import_acum < res$y,]
  vec = c(rownames(tab), class)
  return (list(data[,vec], vec))
}

# FEATURE SELECTION: PCA

dt.pca <- function(data, clabel, transf = NULL)
{
  predictors_name  = setdiff(colnames(data), clabel)
  
  predictors = as.matrix(data[,predictors_name])
  predictand = data[,clabel]
  
  if (is.null(transf)) {
    pca_res = prcomp(predictors, center=TRUE, scale.=TRUE)
    cumvar = cumsum(pca_res$sdev^2/sum(pca_res$sdev^2))
    res = curvature.min(c(1:(length(cumvar))), cumvar)
    transf = as.matrix(pca_res$rotation[, 1:res$x])
  }
  
  dataset = predictors %*% transf
  dataset = data.frame(dataset, predictand)
  colnames(dataset)[ncol(dataset)] <- clabel
  return (list(dataset, transf))
}

# CURVATURE ANALYSIS

curvature.max <- function(x, y, df=3) {
  smodel = smooth.spline(x, y, df = df)
  curvature = predict(smodel, x = x, deriv = 2)
  yv = max(curvature$y)
  xv = match(yv,curvature$y)
  plot(x, y)
  points(x[xv], y[xv], pch=19)
  res = data.frame(x[xv], y[xv], yv)
  colnames(res) = c("x", "y", "z")
  return (res)
}

curvature.min <- function(x, y, df=3) {
  smodel = smooth.spline(x, y, df = df)
  curvature = predict(smodel, x = x, deriv = 2)
  yv = min(curvature$y)
  xv = match(yv,curvature$y)
  plot(x, y)
  points(x[xv], y[xv], pch=19)
  res = data.frame(x[xv], y[xv], yv)
  colnames(res) = c("x", "y", "z")
  return (res)
}

binning <- function(v, n) {
  p <- seq(from = 0, to = 1, by = 1/n)
  q <- quantile(v, p)
  vp <- cut(v, q, FALSE, include.lowest=TRUE)
  m <- tapply(v, vp, mean)
  vm <- m[vp]
  mse <- mean( (v - vm)^2, na.rm = TRUE)
  return (list(binning=m, bins_factor=vp, bins=vm, mse=mse))
}

binning_opt <- function(v, n=10) {
  z <- data.frame()
  for (i in 1:n)
  {
    t <- binning(v, i)
    newrow <- c(t$mse , i)
    z <- rbind(z,newrow)
  }
  colnames(z)<-c("media","num") 
  #mod <- nls(media ~ exp(a + b * num), data = z, start = list(a = 0, b = 0))
  #zf <- predict(mod)
  ideal = curvature.max(z$num, z$media)
  return (binning(v, ideal$x))
}

