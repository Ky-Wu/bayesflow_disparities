data {
  int<lower=0> N;
  matrix[N, N] Sigma_chol;
  //vector[N] mu_phi;
  vector[N] Y; // response
  int<lower=1> p; // num covariates
  matrix[N, p+1] Q_x; // design matrix X = Q_x R_x (QR decomposition)
}
parameters {
  vector[p+1] theta; // R_x * beta
  real<lower=0> sigma2; // overall standard deviation
  real<lower=0, upper=1> rho; // proportion unstructured vs. spatially structured variance
  vector[N] eta; // for spatial error
}
transformed parameters {
  real sigma = sqrt(sigma2);
  vector[N] gamma = (sigma * sqrt(rho)) * (Sigma_chol * eta);
}
model {
  // Likelihood using GLM primitive for speed
  // residual sd = sigma * sqrt(1 - rho)
  real sigma_resid = sigma * sqrt(1 - rho);
  target += normal_id_glm_lpdf(Y | Q_x, gamma, theta, sigma_resid);
  theta ~ normal(0.0, sigma * sqrt(N / 10.0));
  eta ~ std_normal();
  sigma2 ~ normal(0, 1);
  rho ~ beta(1, 1);
}
//generated quantities {
  //real logit_rho = log(rho / (1.0 - rho));
  //real YFit[N] = normal_rng(mu, sqrt(sigma2 * (1 - rho)));
//}
