library(rstan)
library(sf)
library(spdep)
library(Rcpp)
library(RcppArmadillo)
library(coda)
Rcpp::sourceCpp(file.path(getwd(), "rcpp", "bym2_mcmc.cpp"))
source(file.path(getwd(), "R", "disp_helpers.R"))

data_cleaned <- read.csv(file.path(getwd(), "..", "output", "RDA", "data_cleaned.csv"))
shp_fp <- file.path(getwd(), "..", "output", "RDA", "us_mainland_data.geojson")
data_shp <- st_read(shp_fp)
pred_cols <- c('total_mean_smoking', 'unemployed_2014', 'SVI_2014',
               'inactivity_2014',
               'uninsured_2012_2016', 'diabetes_2014', 'obesity_2014')
X <- st_drop_geometry(data_shp)[, pred_cols]
X[] <- lapply(X, scale)
p <- ncol(X)
X <- as.matrix(cbind(1.0, X))
y <- as.vector(scale(data_shp$mortality2014))
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


# Priors
a_sigma <- 0.1
b_sigma <- 0.1
a_rho <- 0
b_rho <- 0
# limits on possible values of rho
lower_rho <- 0.00
upper_rho <- 1.0
lambda_rho <- 0.001
# initialize sampler


gibbsSampler <- new(BYM2FlatBetaMCMC, X, y, Q_scaled)
gibbsSampler$setPriors(a_sigma, b_sigma, rho_prior_type = "pc",
                       a_rho, b_rho,
                       lower_rho, upper_rho, lambda_rho)
n_chains <- 4

gibbsSampler$initOLS()

print("10000 Burn in:")
for(j in seq_len(100)) {
  print(paste0(j, "/100"))
  print(paste0("current rho: ", gibbsSampler$rho))
  print(paste0("current sigma2: ", gibbsSampler$sigma2))
  gibbsSampler$burnMCMCSample(100)
}

# DRAW POSTERIOR SAMPLES
n_sim <- 10000
system.time({
  samps <- gibbsSampler$MCMCSample(n_sim, 20)
})

saveRDS(samps, file.path(getwd(), '..', 'output', 'RDA', 'mcmc_samps.rds'))
thin_indx = seq(1, n_sim, by = 10)

beta_sim <- samps$beta[thin_indx,]
colMeans(beta_sim)

gamma_sim <- samps$gamma[thin_indx,]
sigma2_sim <- samps$sigma2[thin_indx,]
rho_sim <- samps$rho[thin_indx,]

# check mcmc diagnostics
# ESS
coda::effectiveSize(beta_sim)
coda::effectiveSize(gamma_sim)
coda::effectiveSize(sigma2_sim)
coda::effectiveSize(rho_sim)

phi_sim <- apply(gamma_sim, MARGIN = 2, function(x) {
  x / sqrt(sigma2_sim * rho_sim)
})

phi_diffs <- BYM2_StdDiff(phi_sim, rho_sim, Q_scaled, X, adj_df)

eps_optim <- optim(median(abs(phi_diffs)), function(e) {
  v <- compute_diff_prob(phi_diffs, epsilon = e)
  conditional_entropy_loss(v)
}, method = "Brent", lower = 0.0001, upper = 3.0,
control = list(abstol = 0.001))

optim_e <- eps_optim$par
diff_prob <- compute_diff_prob(phi_diffs, epsilon = optim_e)
adj_df$optim_e <- optim_e
adj_df$diff_prob <- diff_prob

write.csv(adj_df, file.path(getwd(), "..", "output", "RDA", "mcmc_diff_prob.csv"))
