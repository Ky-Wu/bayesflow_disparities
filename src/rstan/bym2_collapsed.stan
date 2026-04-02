data {
  int<lower=0> N;
  matrix[N, N] Sigma_chol;
  vector[N] Y;
  int<lower=1> p;
  matrix[N, p+1] Q_x;
}

transformed data {
  matrix[N, N] Sigma = tcrossprod(Sigma_chol);
  vector[N] eigenvalues = eigenvalues_sym(Sigma);
  matrix[N, N] eigenvectors = eigenvectors_sym(Sigma);

  // Rotate Y and the Design Matrix into the Eigen-space
  // This happens once at the start of the chain
  vector[N] Y_rot = eigenvectors' * Y;
  matrix[N, p+1] Q_x_rot = eigenvectors' * Q_x;
}

parameters {
  vector[p+1] theta;
  real<lower=0> sigma2;
  real<lower=0, upper=1> rho;
}

model {
  // Priors
  theta ~ normal(0.0, sqrt(sigma2) * sqrt(N / 10.0));
  sigma2 ~ normal(0, 1);
  rho ~ beta(1, 1);

  // Marginal Likelihood in rotated space
  // Y_rot[i] ~ N( (Q_x_rot * theta)[i], sqrt(sigma2 * rho * eigenvalues[i] + sigma2 * (1 - rho)) )

  vector[N] mu_rot = Q_x_rot * theta;
  vector[N] combined_sd;

  for (n in 1:N) {
    // The variance for each observation in the rotated space
    combined_sd[n] = sqrt(sigma2 * (rho * eigenvalues[n] + (1 - rho)));
  }

  target += normal_lpdf(Y_rot | mu_rot, combined_sd);
}
