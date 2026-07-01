#!/usr/bin/env Rscript
# Forest plots for the DiaTrend subject-cluster bootstrap + influence results.
#
# Reads the text logs written by run_bootstrap_grid.sh (it parses the printed
# tables -- the estimates are reproducible, so no re-run is needed) and renders:
#   fig_boot_primary.{pdf,png}     breakfast vs pooled (ACME/ADE/Total)
#   fig_boot_hba1c.{pdf,png}       HbA1c strata + High-Low moderation contrast
#   fig_boot_sex.{pdf,png}         sex strata + Male-Female moderation contrast
#   fig_boot_influence.{pdf,png}   leave-one-subject-out stability of pooled ACME
#
# Usage:
#   Rscript cma_cluster/diatrend/plot_bootstrap_results.R \
#       [--log-dir mediation_results/diatrend/bootstrap_logs] \
#       [--fig-dir mediation_results/diatrend/figures]
#
# Base graphics only -- no ggplot/optparse dependency.

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default) {
  i <- which(args == flag)
  if (length(i) == 1 && i < length(args)) args[i + 1] else default
}
log_dir <- get_arg("--log-dir", "mediation_results/diatrend/bootstrap_logs")
fig_dir <- get_arg("--fig-dir", "mediation_results/diatrend/figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# ---- Parsers -------------------------------------------------------------
# Bootstrap table rows look like:
#   all                ACME         1.24  [  -8.51,  12.40]  p=0.690
#                      ADE         15.30  [  -2.74,  36.44]  p=0.091
#   High_minus_Low  <-- moderation ACME  -3.10  [ -11.05,  4.95]  p=0.442
parse_boot <- function(path) {
  if (!file.exists(path)) stop(sprintf("missing log: %s", path))
  lines <- readLines(path, warn = FALSE)
  pat <- "(ACME|ADE|TOTAL)\\s+(-?[0-9.]+)\\s+\\[\\s*(-?[0-9.]+),\\s*(-?[0-9.]+)\\]\\s+p=([0-9.]+)"
  out <- list(); last_label <- NA_character_
  for (ln in lines) {
    g <- regmatches(ln, regexec(pat, ln))[[1]]
    if (length(g) == 0) next
    effect <- g[2]
    pre <- sub(paste0("\\s*", effect, ".*$"), "", ln)      # text before the effect keyword
    pre <- trimws(gsub("<--\\s*moderation", "", pre))
    if (nzchar(pre)) last_label <- pre
    out[[length(out) + 1]] <- data.frame(
      label = last_label, effect = effect,
      point = as.numeric(g[3]), lo = as.numeric(g[4]),
      hi = as.numeric(g[5]), p = as.numeric(g[6]),
      stringsAsFactors = FALSE)
  }
  if (!length(out)) stop(sprintf("no estimate rows parsed from %s", path))
  do.call(rbind, out)
}

parse_influence <- function(path) {
  lines <- readLines(path, warn = FALSE)
  num <- function(re, x) as.numeric(sub(re, "\\1", grep(sub("\\\\1.*", "", re), x, value = TRUE)[1]))
  baseline <- as.numeric(sub(".*:\\s*(-?[0-9.]+).*", "\\1", grep("baseline ACME", lines, value = TRUE)[1]))
  rng <- grep("leave-one-out ACME range", lines, value = TRUE)[1]
  lo <- as.numeric(sub(".*\\[\\s*(-?[0-9.]+),.*", "\\1", rng))
  hi <- as.numeric(sub(".*,\\s*(-?[0-9.]+)\\s*\\].*", "\\1", rng))
  pat <- "^\\s*(Subject[0-9]+)\\s+([0-9]+)\\s*\\|\\s*(-?[0-9.]+)\\s+([+-]?[0-9.]+)\\s+(\\S+)\\s*\\|\\s*(-?[0-9.]+)\\s+([+-]?[0-9.]+)"
  rows <- list()
  for (ln in lines) {
    g <- regmatches(ln, regexec(pat, ln))[[1]]
    if (length(g) == 0) next
    rows[[length(rows) + 1]] <- data.frame(
      subject = g[2], n_ep = as.integer(g[3]), acme_s = as.numeric(g[4]),
      dacme = as.numeric(g[5]), flip = g[6], ade_s = as.numeric(g[7]),
      dade = as.numeric(g[8]), stringsAsFactors = FALSE)
  }
  list(baseline = baseline, lo = lo, hi = hi, subj = do.call(rbind, rows))
}

# ---- Forest-plot primitive ----------------------------------------------
# rows: data.frame(row_label, point, lo, hi, p, header=FALSE). header rows draw
# a bold section title and no interval.
RED <- "#B2182B"; GREY <- "#7F7F7F"; DGREY <- "#404040"
forest <- function(rows, title, xlab, file_base, w = 10, h = NULL) {
  n <- nrow(rows)
  if (is.null(h)) h <- 1.6 + 0.42 * n
  ys <- rev(seq_len(n))
  fin <- is.finite(rows$lo) & is.finite(rows$hi)
  xr <- range(c(rows$lo[fin], rows$hi[fin], rows$point[fin], 0), na.rm = TRUE)
  xpad <- 0.08 * diff(xr); xlim <- c(xr[1] - xpad, xr[2] + xpad)
  draw <- function() {
    par(mar = c(4.5, 11.5, 3.2, 9.5), xpd = FALSE)
    plot(NA, xlim = xlim, ylim = c(0.4, n + 0.6), yaxt = "n", xlab = xlab,
         ylab = "", main = title, cex.main = 1.05)
    abline(v = 0, lty = 2, col = "grey55")
    for (i in seq_len(n)) {
      y <- ys[i]
      if (isTRUE(rows$header[i])) {
        text(xlim[1], y, rows$row_label[i], adj = c(0, 0.5), font = 2, cex = 0.95)
        next
      }
      sig <- is.finite(rows$p[i]) && rows$p[i] < 0.05
      col <- if (sig) RED else DGREY
      segments(rows$lo[i], y, rows$hi[i], y, lwd = 2.4, col = if (sig) RED else GREY)
      points(rows$point[i], y, pch = if (sig) 19 else 21, bg = "white",
             col = col, cex = 1.35, lwd = 1.6)
      mtext(rows$row_label[i], side = 2, at = y, las = 1, line = 0.4, cex = 0.85,
            adj = 1)
      if (is.finite(rows$p[i]))
        mtext(sprintf("%.1f [%.1f, %.1f]  p=%.3f", rows$point[i], rows$lo[i],
                      rows$hi[i], rows$p[i]),
              side = 4, at = y, las = 1, line = 0.2, cex = 0.72,
              col = if (sig) RED else "grey30")
    }
  }
  pdf(file.path(fig_dir, paste0(file_base, ".pdf")), width = w, height = h); draw(); dev.off()
  tryCatch({ png(file.path(fig_dir, paste0(file_base, ".png")), width = w, height = h,
                 units = "in", res = 200); draw(); dev.off() },
           error = function(e) message("png skipped: ", conditionMessage(e)))
  cat(sprintf("wrote %s.{pdf,png}\n", file_base))
}

hdr <- function(lab) data.frame(row_label = lab, point = NA, lo = NA, hi = NA,
                                p = NA, header = TRUE, stringsAsFactors = FALSE)
band <- function(df, strat, eff, relabel = NULL) {
  r <- df[df$label == strat & df$effect == eff, ]
  if (!nrow(r)) return(NULL)
  data.frame(row_label = if (is.null(relabel)) eff else relabel,
             point = r$point, lo = r$lo, hi = r$hi, p = r$p, header = FALSE,
             stringsAsFactors = FALSE)
}

XLAB <- expression("Effect on " * Delta * "Glucose at 120 min per +30 g carb (mg/dL)")

# ---- Fig 1: primary (breakfast vs pooled) -------------------------------
bk <- parse_boot(file.path(log_dir, "step2_breakfast.log"))
pl <- parse_boot(file.path(log_dir, "step1_pooled.log"))
f1 <- do.call(rbind, list(
  hdr("Breakfast"),
  band(bk, "all", "TOTAL", "Total"), band(bk, "all", "ADE", "ADE"), band(bk, "all", "ACME", "ACME"),
  hdr("Pooled (all meals)"),
  band(pl, "all", "TOTAL", "Total"), band(pl, "all", "ADE", "ADE"), band(pl, "all", "ACME", "ACME")))
forest(f1, "DiaTrend mediation (subject-cluster bootstrap, B=1000, full cohort)",
       XLAB, "fig_boot_primary")

# ---- Fig 2: HbA1c moderation --------------------------------------------
a1 <- parse_boot(file.path(log_dir, "step4_hba1c.log"))
mod_fig <- function(df, lo_lab, hi_lab, contrast, title, fbase) {
  mk <- function(strat, show) {
    rs <- list(hdr(show))
    for (e in c("TOTAL", "ADE", "ACME"))
      rs[[length(rs) + 1]] <- band(df, strat, e, c(TOTAL = "Total", ADE = "ADE", ACME = "ACME")[e])
    do.call(rbind, rs)
  }
  rows <- do.call(rbind, c(
    list(mk(lo_lab, paste0(lo_lab, " HbA1c"))),
    if (hi_lab %in% df$label) list(mk("Mid", "Mid HbA1c")) else NULL,
    list(mk(hi_lab, paste0(hi_lab, " HbA1c"))),
    list(hdr(paste0(gsub("_minus_", " - ", contrast), "  (moderation)"))),
    list(band(df, contrast, "TOTAL", "Total"),
         band(df, contrast, "ADE", "ADE"),
         band(df, contrast, "ACME", "ACME"))))
  forest(rows, title, XLAB, fbase, h = 1.6 + 0.42 * nrow(rows))
}
mod_fig(a1, "Low", "High", "High_minus_Low",
        "DiaTrend HbA1c moderation (subject-cluster bootstrap, B=1000)", "fig_boot_hba1c")

# ---- Fig 3: sex moderation ----------------------------------------------
sx <- parse_boot(file.path(log_dir, "step5_sex.log"))
sex_rows <- do.call(rbind, list(
  hdr("Female"),
  band(sx, "Female", "TOTAL", "Total"), band(sx, "Female", "ADE", "ADE"), band(sx, "Female", "ACME", "ACME"),
  hdr("Male"),
  band(sx, "Male", "TOTAL", "Total"), band(sx, "Male", "ADE", "ADE"), band(sx, "Male", "ACME", "ACME"),
  hdr("Male - Female  (moderation)"),
  band(sx, "Male_minus_Female", "TOTAL", "Total"),
  band(sx, "Male_minus_Female", "ADE", "ADE"),
  band(sx, "Male_minus_Female", "ACME", "ACME")))
forest(sex_rows, "DiaTrend sex moderation (subject-cluster bootstrap, B=1000)",
       XLAB, "fig_boot_sex")

# ---- Fig 4: LOSO influence on pooled ACME -------------------------------
inf <- parse_influence(file.path(log_dir, "step6_influence.log"))
s <- inf$subj; s <- s[order(abs(s$dacme)), ]    # least to most influential (top of plot = most)
draw_inf <- function() {
  n <- nrow(s); ys <- seq_len(n)
  xr <- range(c(s$acme_s, inf$baseline, inf$lo, inf$hi)); xpad <- 0.15 * diff(xr)
  par(mar = c(4.5, 8, 3.2, 6), xpd = FALSE)
  plot(NA, xlim = c(xr[1] - xpad, xr[2] + xpad), ylim = c(0.5, n + 0.5), yaxt = "n",
       xlab = "Pooled ACME with one subject removed (mg/dL)", ylab = "",
       main = "DiaTrend: leave-one-subject-out stability of pooled ACME")
  rect(inf$lo, 0.4, inf$hi, n + 0.6, col = "#E8EEF6", border = NA)   # LOSO range band
  abline(v = inf$baseline, lwd = 2, col = RED)
  abline(v = 0, lty = 2, col = "grey55")
  points(s$acme_s, ys, pch = 19, col = DGREY, cex = 1.2)
  axis(2, at = ys, labels = sprintf("%s (n=%d)", s$subject, s$n_ep), las = 1, tick = FALSE)
  mtext(sprintf("d=%+.2f", s$dacme), side = 4, at = ys, las = 1, line = 0.4, cex = 0.75, col = "grey30")
  legend("topleft", bty = "n", cex = 0.8,
         legend = c(sprintf("baseline ACME = %.2f", inf$baseline),
                    sprintf("LOSO range [%.2f, %.2f]", inf$lo, inf$hi)),
         lwd = c(2, 8), col = c(RED, "#E8EEF6"))
}
pdf(file.path(fig_dir, "fig_boot_influence.pdf"), width = 9, height = 5.2); draw_inf(); dev.off()
tryCatch({ png(file.path(fig_dir, "fig_boot_influence.png"), width = 9, height = 5.2,
               units = "in", res = 200); draw_inf(); dev.off() },
         error = function(e) message("png skipped: ", conditionMessage(e)))
cat("wrote fig_boot_influence.{pdf,png}\n")
cat(sprintf("\nAll figures written to %s/\n", fig_dir))
