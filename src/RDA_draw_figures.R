library(sf)
library(ggplot2)
library(data.table)
library(classInt)
library(xtable)
library(ggpubr)

shp_fp <- file.path(getwd(), "..", "output", "RDA", "us_mainland_data.shp")
shp <- st_read(shp_fp)
shp <- st_zm(shp, drop = TRUE, what = "ZM")
output_fp <- file.path(getwd(), "..", "output", "RDA", "joint_network_v8")
diff_probs <- fread(file.path(output_fp, "diff_probs.csv"))
# number of disparities
npe_disp <- with(diff_probs, (approx_diff_prob >= cutoff_prob))
# number of disparities using MCMC
mcmc_disp <- with(diff_probs, (mcmc_diff_prob >= cutoff_prob))

cor(diff_probs$approx_diff_prob, diff_probs$mcmc_diff_prob)
cor(diff_probs$approx_diff_prob, diff_probs$vague_diff_prob)
cor(diff_probs$mcmc_diff_prob, diff_probs$vague_diff_prob)

RDET_df <- fread(file.path(output_fp, "RDET_df.csv"))
beta_summary <- fread(file.path(output_fp, "beta_summary.csv"))

p1 <- ggplot(data = diff_probs) +
  geom_point(aes(x = approx_diff_prob, y = vague_diff_prob),
             alpha = 0.3, color = "dodgerblue", size = 0.4) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(x = "ABI Network Difference Probability",
       y = "MCMC Difference Probability (Vague Prior)") +
  theme_bw()

p2 <- ggplot(data = diff_probs) +
  geom_point(aes(x = approx_diff_prob, y = mcmc_diff_prob),
             alpha = 0.3, color = "dodgerblue", size = 0.4) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(x = "ABI Network Difference Probability",
       y = "MCMC Difference Probability (Same Prior)") +
  theme_bw()

p <- ggpubr::ggarrange(p1, p2)

ggsave(file.path(output_fp, "vague_mcmc_comparison.png"), p1, dpi = 300)
ggsave(file.path(output_fp, "mcmc_comparison.png"), p2, dpi = 300)
ggsave(file.path(output_fp, "mcmc_comparisons.png"), p, dpi = 300)

### draw disparity map ###

rej_indx <- with(diff_probs, approx_diff_prob >= cutoff_prob)
detected_borders <- diff_probs[rej_indx,]
county1_indx <- match(detected_borders$county1, shp$County_FIP)
county2_indx <- match(detected_borders$county2, shp$County_FIP)
node1_all <- shp[county1_indx,]
node2_all <- shp[county2_indx,]
intersections <- lapply(seq_len(sum(rej_indx)), function(i) {
  node1 <- node1_all[i,]
  node2 <- node2_all[i,]
  suppressMessages(st_intersection(st_buffer(node1, 0.0003),
                                   st_buffer(node2, 0.0003)))
}) %>%
  do.call(rbind, .)


rates <- shp$mortality2
breaks <- classInt::classIntervals(rates, n = 6, style = "jenks")$brks
b <- round(breaks, 1)
labels <- paste(b[-length(b)], b[-1], sep = " – ")
cut_rates <- cut(rates, breaks = breaks, include.lowest = TRUE,
                 labels = labels)


map <- ggplot() +
  geom_sf(data = shp, color = "gray95", linewidth = 0.03, aes(fill = cut_rates)) +
  scale_fill_brewer(palette = "YlOrRd", name = "Lung Cancer Mortality Rate") +
  #scale_fill_viridis_c(name = "Lung Cancer Mortality Rate") +
  geom_sf(data = intersections, col = "blue", fill = NA,
          linewidth = 0.3) +
  coord_sf(crs = st_crs(5070)) +
  theme_bw() +
  theme(legend.position = "bottom", legend.title=element_text(size=10))
#map

ggsave(file.path(output_fp, "disparities_map.png"), map, dpi = 300,
       width = 8, height = 5)

### RDETs ###

max_RDETs <- RDET_df[, .(max_RDET = max(RDET_percent),
                         reduction_factor = min(reduction_factor)),
                     by = .(higher_gamma_mean_county)]

indx <- match(max_RDETs$higher_gamma_mean_county, shp$County_FIP)
shp$RDET <- NA
shp[indx,]$RDET <- max_RDETs$max_RDET

RDET_map <- ggplot() +
  geom_sf(data = shp, color = "gray",
          linewidth = 0.10, aes(fill = RDET)) +
  scale_fill_viridis_c(option = "mako", direction = -1,
                       trans = "sqrt",
                       name = "RDET (% decrease)", na.value = "gray90") +
  coord_sf(crs = st_crs(5070)) +
  theme_bw() +
  theme(legend.position = "bottom",
        legend.key.width = unit(1.5, "cm"))
#RDET_map

ggsave(file.path(output_fp, "RDET_map.png"), RDET_map, dpi = 300,
       width = 8, height = 5)

indx <- match(max_RDETs$higher_gamma_mean_county, shp$County_FIP)
county_names = shp$location[indx]
RDET_rates <- round(max_RDETs$reduction_factor * shp$mortality2[indx], 1)
p_decrease <- round(max_RDETs$max_RDET, 1)
RDET_f <- paste0(RDET_rates, " (", p_decrease, "%)")
output_df <- data.frame(
  high_mortality_county = county_names,
  min_RDET = RDET_f
)
output_df <- output_df[order(p_decrease, decreasing = TRUE),]
colnames(output_df) <- c("High Residual Mortality County", "RDET (% decrease)")
caption <- paste0("Counties that have a detected disparity with a neighboring ",
                  "county and carry a higher posterior mean spatial ",
                  "residual are reported as high residual mortality counties. ",
                  "For each detected disparity, we compute the Residual Disparity ",
                  "Elimination Target (RDET), defined as the minimum mortality rate ",
                  "at which the high residual mortality county would not be classified ",
                  "as a disparity with a neighboring county, using the trained ",
                  "neural posterior estimator. The reported RDET is the minimum ",
                  "value across all disparities for a given high residual mortality county.")
print(xtable::xtable(output_df, caption = caption,
                     label = "tab:min_RDETs", align = rep("c", ncol(output_df) + 1)),
      type = "latex", include.rownames = FALSE, booktabs = TRUE, escape = FALSE,
      file.path(output_fp, "min_RDETs.tex"))


var_names <- c("Intercept", "Smoking", "Unemployment", "SVI", "Physical Inactivity",
               "Uninsured Rate", "Diabetes Prevalence", "Obesity Prevalence")
beta_df <- with(beta_summary, data.frame(
  `Variable` = var_names,
  `Posterior Mean` = sprintf("%.3f", posterior_mean),
  # Combine quantiles into the standard 95% CrI format
  `95\\% Credible Interval` = sprintf("[%.3f, %.3f]", quantile_0_025, quantile_0_975)
))

colnames(beta_df) <- c("Variable", "Posterior Mean", "95% Credible Interval")

beta_caption = paste0("Standardized regression coefficient estimates ",
                      "from a Bayesian spatial regression model predicting 2014 US county-level ",
                      "tracheal, bronchus, and lung cancer mortality rates. ",
                      "Posterior samples were generated via a neural posterior estimator, and ",
                      "95\\% credible intervals were derived from the 2.5\\% and 97.5\\% ",
                      "posterior quantiles for each health risk factor.")
print(xtable::xtable(beta_df, caption = beta_caption,
                     label = "tab:beta_summary", align = c("l", rep("c", ncol(beta_df)))),
      type = "latex", include.rownames = FALSE, booktabs = TRUE, escape = FALSE,
      file.path(output_fp, "beta_summary.tex"))

# %% beta comparison

mcmc_samps <- readRDS(file.path(getwd(), '..', 'output', 'RDA', 'mcmc_samps.rds'))
vague_samps <- readRDS(file.path(getwd(), '..', 'output', 'RDA', 'mcmc_samps_vague_prior.rds'))
beta_mcmc <- mcmc_samps$beta
beta_vague <- vague_samps$beta

summarize <- function(samps, method_name, var_names) {
  data.frame(
    Method = method_name,
    variable = var_names,
    posterior_mean = colMeans(samps),
    quantile_0_025 = apply(samps, MARGIN = 2, function(x) quantile(x, 0.025)),
    quantile_0_975 = apply(samps, MARGIN = 2, function(x) quantile(x, 0.975))
  )
}
var_names <- c("Intercept", "Smoking Prevalence",
               "Unemployment", "SVI", "Physical Inactivity",
               "Uninsured Rate", "Diabetes Prevalence", "Obesity Prevalence")
beta_mcmc_summary <- summarize(beta_mcmc, "MCMC (Equivalent Prior)", var_names)
beta_vague_summary <- summarize(beta_vague, "MCMC (Noninformative Prior)", var_names)
all_beta_summary <- rbind(
  cbind(data.frame("Method" = "NPE", variable = var_names), beta_summary),
  beta_mcmc_summary,
  beta_vague_summary
)
all_beta_summary$variable <- factor(all_beta_summary$variable,
                                    levels = rev(var_names), ordered = TRUE)

beta_CI_graph <- ggplot(data = all_beta_summary) +
  geom_errorbar(aes(x = variable, ymin = quantile_0_025, ymax = quantile_0_975,
                    color = Method), position = "dodge") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  coord_flip() +
  labs(x = "Standardized Regression Coefficient", y = "Value") +
  theme_bw(base_size = 13)

ggsave(file.path(output_fp, "beta_CIs_graph.png"), beta_CI_graph, dpi = 300)

library(latex2exp)
var_names <- c("$\\sigma^2$", "$\\rho$")
var_mcmc_summary <- summarize(cbind(mcmc_samps$sigma2, mcmc_samps$rho),
                              "MCMC (Equivalent Prior)",
                              var_names)
var_vague_summary <- summarize(cbind(vague_samps$sigma2, vague_samps$rho),
                              "MCMC (Noninformative Prior)",
                              var_names)
npe_var_summary <- fread(file.path(output_fp, "var_summary.csv"))
npe_var_summary$variable <- var_names
npe_var_summary <- cbind(data.frame(Method = "NPE"), npe_var_summary)
all_var_summary <- rbind(npe_var_summary, var_mcmc_summary, var_vague_summary)

var_CI_graph <- ggplot(data = all_var_summary) +
  geom_errorbar(aes(x = variable, ymin = quantile_0_025, ymax = quantile_0_975,
                    color = Method), position = "dodge") +
  scale_x_discrete(labels = TeX) +
  scale_y_continuous(limits = c(0, 1)) +
  coord_flip() +
  labs(x = "Variance Parameter", y = "Value") +
  theme_bw(base_size = 13)
#var_CI_graph

CI_graphs <- ggpubr::ggarrange(beta_CI_graph, var_CI_graph,
                  legend = "bottom", common.legend = TRUE)

ggsave(file.path(output_fp, "CI_graphs.png"), CI_graphs, dpi = 300)
