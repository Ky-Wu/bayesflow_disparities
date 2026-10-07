library(INLA)
library(sf)
library(spdep)
rm(list = ls())
set.seed(1130)

source(file.path(getwd(), "R", "disp_helpers.R"))

data_cleaned <- read.csv(file.path(getwd(), "..", "output", "RDA", "data_cleaned.csv"))
shp_fp <- file.path(getwd(), "..", "output", "RDA", "us_mainland_data.shp")
shp <- st_read(shp_fp)
shp$County_FIPS <- as.integer(paste0(shp$STATEFP, shp$COUNTYFP))
data_shp <- merge(shp, data_cleaned, by = "County_FIPS", suffixes = c("_shp", ""))

# drop Z dimension
data_shp <- st_zm(data_shp, drop = TRUE, what = "ZM")
# Verify all are now XY
dims <- sapply(st_geometry(data_shp), function(g) class(g)[1])
table(dims)  # should show only XY: 3103

pred_cols <- c('total_mean_smoking', 'unemployed_2014', 'SVI_2014',
               'inactivity_2014',
               'uninsured_2012_2016', 'diabetes_2014', 'obesity_2014')
X <- data_cleaned[, pred_cols]
X[] <- lapply(X, scale)
p <- ncol(X)
X <- as.matrix(cbind(1.0, X))
y <- as.vector(scale(data_cleaned$mortality2014))
N <- nrow(X)

qr_res <- qr(X)
Q_x <- qr.Q(qr_res)
R_x <- qr.R(qr_res)


# 1. Create a neighborhood list based on polygon contiguity
# Note: Extra steps required to convert/read data if it is not in sf format
nb <- spdep::poly2nb(data_shp, queen = FALSE)

# 2. Set the location of the adjacency file for INLA
map.adj <- file.path(getwd(), "..", "output", "RDA", "INLA", "map.graph")

# 3. Convert the neighbor list to INLA adjacency format
spdep::nb2INLA(map.adj, nb)

# 4. Load adjacency matrix as a graph for INLA spatial models
g <- inla.read.graph(filename = map.adj)

# index for spatial effects
data_shp$re_u <- 1:nrow(data_shp)

# specify model formula
formula <- mortality2014 ~ total_mean_smoking + unemployed_2014 + SVI_2014 + inactivity_2014 +
  uninsured_2012_2016 + diabetes_2014 + obesity_2014 + f(re_u, model = "bym2", graph = g)

# run model
t0 <- Sys.time()
res <- inla(formula, family = "gaussian", data = data_shp,
            control.predictor = list(compute = TRUE),
            control.compute = list(return.marginals.predictor = TRUE, config = TRUE),
            verbose = TRUE)
t1 <- Sys.time()
print(t1 - t0)          # wall time for just the inla() call itself
res$cpu.used            # INLA's own breakdown, in R-visible form



# draw n samples from the joint posterior (latent field + hyperparameters)
n_samples <- 2000
samples <- inla.posterior.sample(n_samples, res)


# check the exact hyperparameter names -- they must match what's used below
hyperpar_names <- names(samples[[1]]$hyperpar)
print(hyperpar_names)

# pull phi (mixing parameter) and precision for the BYM2 term
phi_name  <- grep("^Phi for re_u", hyperpar_names, value = TRUE)
prec_name <- grep("^Precision for re_u", hyperpar_names, value = TRUE)

phi_samples  <- sapply(samples, function(s) s$hyperpar[[phi_name]])
prec_samples <- sapply(samples, function(s) s$hyperpar[[prec_name]])

# convert precision -> marginal variance/sd of the combined spatial effect, if useful
var_samples <- 1 / prec_samples
sd_samples  <- sqrt(var_samples)

# summarize
hyper_summary <- data.frame(
  parameter = c("phi", "precision", "variance", "sd"),
  mean      = c(mean(phi_samples), mean(prec_samples), mean(var_samples), mean(sd_samples)),
  lower     = c(quantile(phi_samples, 0.025), quantile(prec_samples, 0.025),
                quantile(var_samples, 0.025), quantile(sd_samples, 0.025)),
  upper     = c(quantile(phi_samples, 0.975), quantile(prec_samples, 0.975),
                quantile(var_samples, 0.975), quantile(sd_samples, 0.975))
)
print(hyper_summary)

# combine with re_u draws from before into one data frame for downstream use
posterior_draws <- list(
  re_u      = re_u_samples,    # matrix: n_areas x n_samples
  phi       = phi_samples,     # vector: length n_samples
  precision = prec_samples     # vector: length n_samples
)

# rerun model


# example analysis of new datawith warm start

data_shp$mortality2014_v2 <- data_cleaned$mortality2014 + rnorm(nrow(data_cleaned), 0, 0.01)

formula_y2 <- mortality2014_v2 ~ total_mean_smoking + unemployed_2014 + SVI_2014 +
  inactivity_2014 + uninsured_2012_2016 + diabetes_2014 + obesity_2014 +
  f(re_u, model = "bym2", graph = g)
t0 <- Sys.time()
res_y2 <- inla(formula_y2, family = "gaussian", data = data_shp,
               control.predictor = list(compute = TRUE),
               control.compute = list(config = TRUE),
               control.mode = list(result = res, restart = TRUE),  # warm start
               verbose = TRUE)
t1 <- Sys.time()
print(t1 - t0)          # wall time for just the inla() call itself
res$cpu.used            # INLA's own breakdown, in R-visible form
