fdr_estimate <- function(v, t) {
  v_s <- v[v >= t]
  sum(1 - v_s) / max(length(v_s), 1)
}

BYM2_StdDiff <- function(sim_phi, sim_rho, Q, X, ij_list) {
  # sim indexed by row
  # assume Q positive definite, apply simultaneous diagonalization
  N <- nrow(X)
  k <- nrow(ij_list)

  XtX <- t(X) %*% X
  Q_eigen <- eigen(Q)
  if (any(Q_eigen$values <= 0)) stop("Q not positive definite")
  Q_neghalf <- Q_eigen$vectors %*% diag(Q_eigen$values^(-0.5)) %*% t(Q_eigen$vectors)
  H <- chol(XtX)
  H <- solve(t(H), t(X))
  H <- t(H) %*% H
  I_H <- diag(N) - H
  B <- Q_neghalf %*% I_H %*% Q_neghalf
  B_eigen <- eigen(B)
  D <- B_eigen$values
  O <- B_eigen$vectors
  U <- Q_neghalf %*% O
  n_sim <- nrow(sim_phi)

  U2_contrasts <- vapply(seq_len(k), function(pair_indx) {
    i <- ij_list[pair_indx,]$i
    j <- ij_list[pair_indx,]$j
    (U[i,] - U[j,])^2
  }, numeric(N))
  var_core <- vapply(sim_rho, function(target_rho) {
    1 / (1 + target_rho / (1 - target_rho) * D)
  }, numeric(N))
  sds <- sqrt(t(var_core) %*% U2_contrasts)
  diffs <- vapply(seq_len(k), function(pair_indx) {
    i <- ij_list[pair_indx,]$i
    j <- ij_list[pair_indx,]$j
    sim_phi[,i] - sim_phi[,j]
  }, numeric(n_sim))
  diffs / sds
}

compute_diff_prob <- function(d, epsilon) {
  colMeans(abs(d) > epsilon)
}

compute_fdr_cutoff <- function(diff_prob, delta) {
  t_seq <- sort(unique(diff_prob), decreasing = FALSE)
  t_FDR <- vapply(t_seq, function(t) fdr_estimate(diff_prob, t), numeric(1))
  optim_t <- min(c(t_seq[t_FDR <= delta], 1), na.rm = TRUE)
  list(cutoff = optim_t, FDR_estimate = fdr_estimate(diff_prob, t = optim_t))
}

conditional_entropy_loss <- function(v) {
  neg_entropy <- ifelse(v == 0 | v == 1, 0.0, v * log(v) + (1 - v) * log(1 - v))
  sum(neg_entropy)
}
