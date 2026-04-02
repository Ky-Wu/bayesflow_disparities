library(rstan)
library(sf)
library(spdep)
library(Rcpp)
library(RcppArmadillo)
library(coda)
Rcpp::sourceCpp(file.path(getwd(), "rcpp", "bym2_mcmc.cpp"))
source(file.path(getwd(), "R", "disp_helpers.R"))

model_fp <- file.path(getwd(), "rstan", "bym2_collapsed.stan")
data_cleaned <- read.csv(file.path(getwd(), "..", "output", "RDA", "data_cleaned.csv"))
shp_fp <- file.path(getwd(), "..", "output", "RDA", "us_mainland_data.shp")
shp <- st_read(shp_fp)
shp$County_FIPS <- as.integer(paste0(shp$STATEFP, shp$COUNTYFP))
data_shp <- merge(shp, data_cleaned, by = "County_FIPS")
pred_cols <- c('total_mean_smoking', 'unemployed_2014', 'SVI_2014',
               'inactivity_2014',
               'uninsured_2012_2016', 'diabetes_2014', 'obesity_2014')
X <- data_cleaned[, pred_cols]
X[] <- lapply(X, scale)
p <- ncol(X)
X <- as.matrix(cbind(1.0, X))
y <- as.vector(scale(data_cleaned$mortality2014))
N <- nrow(X)

county_nbs <- spdep::poly2nb(data_shp, queen = FALSE)
W <- nb2mat(county_nbs, style = "B")
D <- diag(rowSums(W))
alpha <- 0.99
Q <- D - alpha * W
Q_cholR <- chol(Q)
Sigma <- chol2inv(Q_cholR)
scaling_factor <- exp(mean(log(diag(Sigma))))
Q_scaled <- Q * scaling_factor
Sigma_scaled <- Sigma / scaling_factor
Sigma_chol <- t(chol(Sigma_scaled))
N <- nrow(W)

adj_df <- data.frame(
  i = rep(seq_len(N), times = vapply(county_nbs, length, numeric(1))),
  j = unlist(county_nbs)
)
adj_df <- adj_df[adj_df$i < adj_df$j, ]
rownames(adj_df) <- NULL
adj_df$node1_fips = data_shp[adj_df$i,]$County_FIPS
adj_df$node2_fips = data_shp[adj_df$j,]$County_FIPS
qr_res <- qr(X)
Q_x <- qr.Q(qr_res)
R_x <- qr.R(qr_res)
stan_data <- list(Q_x = Q_x, Y = y, p = p, N = N, Sigma_chol = Sigma_chol)

runtime <- system.time({
  fit <- rstan::stan(file = model_fp, data = stan_data,
                     iter = 1200, warmup = 1000, chains = 5)
})

print(fit)
samps <- as.matrix(fit)
theta_samps <- samps[,grep("theta", colnames(samps))]
sigma2_samps <- samps[, 'sigma2']
rho_samps <- samps[, 'rho']
beta_samps <- t(backsolve(R_x, t(theta_samps), upper.tri = TRUE))

samps <- list(beta = beta_samps,
              sigma2 = sigma2_samps,
              rho = rho_samps)

saveRDS(samps, file.path(getwd(), '..', 'output', 'RDA', 'mcmc_samps.rds'))
