#######################################################################
# This code is computing the koopman operator for the logistic map
# Author : Daniel Fassler
#######################################################################
include("koopman_helper_functions.jl")

# Parameters for the stochastic logistic map
p = [2]
m_list = Int.(ceil.(LinRange(2, 30, 18)))
N = 1

scaling(x) = Int(ceil(x^2*log(x)))
n_list = 3*scaling.(m_list)

N_trials = 20
n = 3

# Jacobi parameters
α = 1
β = 0

# weight functions
unif_weight(x) = 0.5
cheb_weight(x) = 1/(π*sqrt(1-x[1]^2))
jacobi_weight(x; α = α, β = β) = (1-x[1])^α*(1+x[1])^β

L2_residuals_unif_leg = zeros(length(n_list), N_trials)
L2_residuals_unif_cheb = zeros(length(n_list), N_trials)
L2_residuals_cheb_leg = zeros(length(n_list), N_trials)
L2_residuals_cheb_cheb = zeros(length(n_list), N_trials)
L2_residuals_christ_leg = zeros(length(n_list), N_trials)
L2_residuals_christ_cheb = zeros(length(n_list), N_trials)
L2_residuals_unif_mon = zeros(length(n_list), N_trials)
L2_residuals_unif_fou = zeros(length(n_list), N_trials)
L2_residuals_cheb_mon = zeros(length(n_list), N_trials)

Linf_residuals_unif_leg = zeros(length(n_list), N_trials)
Linf_residuals_unif_cheb = zeros(length(n_list), N_trials)
Linf_residuals_cheb_leg = zeros(length(n_list), N_trials)
Linf_residuals_cheb_cheb = zeros(length(n_list), N_trials)
Linf_residuals_christ_leg = zeros(length(n_list), N_trials)
Linf_residuals_christ_cheb = zeros(length(n_list), N_trials)
Linf_residuals_unif_mon = zeros(length(n_list), N_trials)
Linf_residuals_unif_fou = zeros(length(n_list), N_trials)
Linf_residuals_cheb_mon = zeros(length(n_list), N_trials)


for i = 1:N_trials
    for (j, n) in enumerate(n_list)
        @info "n = $n, trial = $i"

        l = n
        dict_Ψ_Leg = Legendre1(n)
        dict_Φ_Leg = Legendre1(l)
        dict_Ψ_Cheb = Chebyshev1(n)
        dict_Φ_Cheb = Chebyshev1(l)
        dict_Ψ_mon = mon1(n)
        dict_Φ_mon = mon1(l)


        ΨX, ΦY, K_unif_leg = koopman(uniform_sampling_sym, shifted_logistic, Int(m), N, dict_Ψ_Leg, dict_Φ_Leg, p; scale = scaling, verbose = true)
        ΨX, ΦY, K_unif_cheb = koopman(uniform_sampling_sym, shifted_logistic, Int(m), N, dict_Ψ_Cheb, dict_Φ_Cheb, p; weight = x -> cheb_weight(x)/unif_weight(x), scale = scaling)
        ΨX, ΦY, K_cheb_leg = koopman(cheb_sampling_sym, shifted_logistic, Int(m), N, dict_Ψ_Leg, dict_Φ_Leg, p; weight = x -> unif_weight(x)/cheb_weight(x), scale = scaling)
        ΨX, ΦY, K_cheb_cheb = koopman(cheb_sampling_sym, shifted_logistic, Int(m), N, dict_Ψ_Cheb, dict_Φ_Cheb, p; scale = scaling)
        ΨX, ΦY, K_christ_leg = koopman(christofell1_sampling_Legendre5, shifted_logistic, Int(m), N, dict_Ψ_Leg, dict_Φ_Leg, p; weight = x -> length(dict_Ψ_Leg)/(christofell(dict_Ψ_Leg, x)[1]), scale = scaling)
        ΨX, ΦY, K_christ_cheb = koopman(christofell1_sampling_Chebyshev5, shifted_logistic, Int(m), N, dict_Ψ_Cheb, dict_Φ_Cheb, p; weight = x -> length(dict_Ψ_Cheb)/(christofell(dict_Ψ_Cheb, x)[1]),scale = scaling)
        ΨX, ΦY, K_unif_mon = koopman(uniform_sampling_sym, shifted_logistic, Int(m), N, dict_Ψ_mon, dict_Φ_mon, p; scale = scaling)
        ΨX, ΦY, K_cheb_mon = koopman(cheb_sampling_sym, shifted_logistic, Int(m), N, dict_Ψ_mon, dict_Φ_mon, p; scale = scaling)

        L2_residuals_unif_leg[j, i] = L₂_error_Legendre1(shifted_logistic, K_unif_leg, 10*Int(3*n^2*ceil(log(n))), uniform_sampling_sym, p, dict_Ψ_Leg)
        L2_residuals_unif_cheb[j, i] = L₂_error_Chebyshev1(shifted_logistic, K_unif_cheb, 10*Int(3*n^2*ceil(log(n))), uniform_sampling_sym, p, dict_Ψ_Cheb)
        L2_residuals_cheb_leg[j, i] = L₂_error_Legendre1(shifted_logistic, K_cheb_leg, 10*Int(3*n^2*ceil(log(n))), cheb_sampling_sym, p, dict_Ψ_Leg)
        L2_residuals_cheb_cheb[j, i] = L₂_error_Chebyshev1(shifted_logistic, K_cheb_cheb, 10*Int(3*n^2*ceil(log(n))), cheb_sampling_sym, p, dict_Ψ_Cheb)
        L2_residuals_christ_leg[j, i] = L₂_error_Legendre1(shifted_logistic, K_christ_leg, 10*Int(3*n^2*ceil(log(n))), uniform_sampling_sym, p, dict_Ψ_Leg)
        L2_residuals_christ_cheb[j, i] = L₂_error_Chebyshev1(shifted_logistic, K_christ_cheb, 10*Int(3*n^2*ceil(log(n))), cheb_sampling_sym, p, dict_Ψ_Cheb)
        L2_residuals_unif_mon[j, i] = L₂_error_Monomial1(shifted_logistic, K_unif_mon, 10*Int(3*n^2*ceil(log(n))), uniform_sampling_sym, p, dict_Ψ_mon)
        L2_residuals_cheb_mon[j, i] = L₂_error_Monomial1(shifted_logistic, K_cheb_mon, 10*Int(3*n^2*ceil(log(n))), cheb_sampling_sym, p, dict_Ψ_mon)

        Linf_residuals_unif_leg[j, i] = L_inf_error_Legendre1(shifted_logistic, K_unif_leg, 10*Int(3*n^2*ceil(log(n))), uniform_sampling_sym, p, dict_Ψ_Leg)
        Linf_residuals_unif_cheb[j, i] = L_inf_error_Chebyshev1(shifted_logistic, K_unif_cheb, 10*Int(3*n^2*ceil(log(n))), uniform_sampling_sym, p, dict_Ψ_Cheb)
        Linf_residuals_cheb_leg[j, i] = L_inf_error_Legendre1(shifted_logistic, K_cheb_leg, 10*Int(3*n^2*ceil(log(n))), cheb_sampling_sym, p, dict_Ψ_Leg)
        Linf_residuals_cheb_cheb[j, i] = L_inf_error_Chebyshev1(shifted_logistic, K_cheb_cheb, 10*Int(3*n^2*ceil(log(n))), cheb_sampling_sym, p, dict_Ψ_Cheb)
        Linf_residuals_christ_leg[j, i] = L_inf_error_Legendre1(shifted_logistic, K_christ_leg, 10*Int(3*n^2*ceil(log(n))), uniform_sampling_sym, p, dict_Ψ_Leg)
        Linf_residuals_christ_cheb[j, i] = L_inf_error_Chebyshev1(shifted_logistic, K_christ_cheb, 10*Int(3*n^2*ceil(log(n))), cheb_sampling_sym, p, dict_Ψ_Cheb)
        Linf_residuals_unif_mon[j, i] = L_inf_error_Monomial1(shifted_logistic, K_unif_mon, 10*Int(3*n^2*ceil(log(n))), uniform_sampling_sym, p, dict_Ψ_mon)
        Linf_residuals_cheb_mon[j, i] = L_inf_error_Monomial1(shifted_logistic, K_cheb_mon, 10*Int(3*n^2*ceil(log(n))), cheb_sampling_sym, p, dict_Ψ_mon)
    end
end

res_avg_L2_unif_leg, res_std_L2_unif_leg = log_avg_std(L2_residuals_unif_leg)
res_avg_L2_unif_cheb, res_std_L2_unif_cheb = log_avg_std(L2_residuals_unif_cheb)
res_avg_L2_cheb_leg, res_std_L2_cheb_leg = log_avg_std(L2_residuals_cheb_leg)
res_avg_L2_cheb_cheb, res_std_L2_cheb_cheb = log_avg_std(L2_residuals_cheb_cheb)
res_avg_L2_christ_leg, res_std_L2_christ_leg = log_avg_std(L2_residuals_christ_leg)
res_avg_L2_christ_cheb, res_std_L2_christ_cheb = log_avg_std(L2_residuals_christ_cheb)
res_avg_L2_unif_mon, res_std_L2_unif_mon = log_avg_std(L2_residuals_unif_mon)
res_avg_L2_cheb_mon, res_std_L2_cheb_mon = log_avg_std(L2_residuals_cheb_mon)

res_avg_Linf_unif_leg, res_std_Linf_unif_leg = log_avg_std(Linf_residuals_unif_leg)
res_avg_Linf_unif_cheb, res_std_Linf_unif_cheb = log_avg_std(Linf_residuals_unif_cheb)
res_avg_Linf_cheb_leg, res_std_Linf_cheb_leg = log_avg_std(Linf_residuals_cheb_leg)
res_avg_Linf_cheb_cheb, res_std_Linf_cheb_cheb = log_avg_std(Linf_residuals_cheb_cheb)
res_avg_Linf_christ_leg, res_std_Linf_christ_leg = log_avg_std(Linf_residuals_christ_leg)
res_avg_Linf_christ_cheb, res_std_Linf_christ_cheb = log_avg_std(Linf_residuals_christ_cheb)
res_avg_Linf_unif_mon, res_std_Linf_unif_mon = log_avg_std(Linf_residuals_unif_mon)
res_avg_Linf_cheb_mon, res_std_Linf_cheb_mon = log_avg_std(Linf_residuals_cheb_mon)


# Saving
save("data_logistic_map_n_equal_l_exp.jld2", 
    "n_list", n_list, 
    "m_list", m_list,
    "L2_residuals_unif_leg", L2_residuals_unif_leg,
    "L2_residuals_unif_cheb", L2_residuals_unif_cheb,
    "L2_residuals_cheb_leg", L2_residuals_cheb_leg,
    "L2_residuals_cheb_cheb", L2_residuals_cheb_cheb,
    "L2_residuals_christ_leg", L2_residuals_christ_leg,
    "L2_residuals_christ_cheb", L2_residuals_christ_cheb,
    "L2_residuals_unif_mon", L2_residuals_unif_mon,
    "L2_residuals_cheb_mon", L2_residuals_cheb_mon,
    "Linf_residuals_unif_leg", Linf_residuals_unif_leg,
    "Linf_residuals_unif_cheb", Linf_residuals_unif_cheb,
    "Linf_residuals_cheb_leg", Linf_residuals_cheb_leg,
    "Linf_residuals_cheb_cheb", Linf_residuals_cheb_cheb,
    "Linf_residuals_christ_leg", Linf_residuals_christ_leg,
    "Linf_residuals_christ_cheb", Linf_residuals_christ_cheb,
    "Linf_residuals_unif_mon", Linf_residuals_unif_mon,
    "Linf_residuals_cheb_mon", Linf_residuals_cheb_mon
    )

# Plotting L2 errors
fig_L2 = Figure(size = (1800, 600))

# Uniform sampling
ax1_L2 = Axis(fig_L2[1, 1], title = "Uniform Sampling (L2 Error)", xlabel = "n", ylabel = "Error", yscale = log10)
lines!(ax1_L2, n_list, 10 .^res_avg_L2_unif_leg[:], color = :blue, label = "Legendre")
band!(ax1_L2, n_list, 10 .^(res_avg_L2_unif_leg[:] .+ res_std_L2_unif_leg[:]), 10 .^(res_avg_L2_unif_leg[:] .- res_std_L2_unif_leg[:]), color = :blue, alpha = 0.3)
lines!(ax1_L2, n_list, 10 .^res_avg_L2_unif_cheb[:], color = :green, label = "Chebyshev")
band!(ax1_L2, n_list, 10 .^(res_avg_L2_unif_cheb[:] .+ res_std_L2_unif_cheb[:]), 10 .^(res_avg_L2_unif_cheb[:] .- res_std_L2_unif_cheb[:]), color = :green, alpha = 0.3)
lines!(ax1_L2, n_list, 10 .^res_avg_L2_unif_mon[:], color = :black, label = "Monomial")
band!(ax1_L2, n_list, 10 .^(res_avg_L2_unif_mon[:] .+ res_std_L2_unif_mon[:]), 10 .^(res_avg_L2_unif_mon[:] .- res_std_L2_unif_mon[:]), color = :black, alpha = 0.3)
axislegend(ax1_L2, position = :rt)

# Chebyshev sampling
ax2_L2 = Axis(fig_L2[1, 2], title = "Chebyshev Sampling (L2 Error)", xlabel = "n", ylabel = "Error", yscale = log10)
lines!(ax2_L2, n_list, 10 .^res_avg_L2_cheb_leg[:], color = :turquoise, label = "Legendre")
band!(ax2_L2, n_list, 10 .^(res_avg_L2_cheb_leg[:] .+ res_std_L2_cheb_leg[:]), 10 .^(res_avg_L2_cheb_leg[:] .- res_std_L2_cheb_leg[:]), color = :turquoise, alpha = 0.3)
lines!(ax2_L2, n_list, 10 .^res_avg_L2_cheb_cheb[:], color = :purple, label = "Chebyshev")
band!(ax2_L2, n_list, 10 .^(res_avg_L2_cheb_cheb[:] .+ res_std_L2_cheb_cheb[:]), 10 .^(res_avg_L2_cheb_cheb[:] .- res_std_L2_cheb_cheb[:]), color = :purple, alpha = 0.3)
lines!(ax2_L2, n_list, 10 .^res_avg_L2_cheb_mon[:], color = :pink, label = "Monomial")
band!(ax2_L2, n_list, 10 .^(res_avg_L2_cheb_mon[:] .+ res_std_L2_cheb_mon[:]), 10 .^(res_avg_L2_cheb_mon[:] .- res_std_L2_cheb_mon[:]), color = :pink, alpha = 0.3)
axislegend(ax2_L2, position = :rt)

# Christofell sampling
ax3_L2 = Axis(fig_L2[1, 3], title = "Christofell Sampling (L2 Error)", xlabel = "n", ylabel = "Error", yscale = log10)
lines!(ax3_L2, n_list, 10 .^res_avg_L2_christ_leg[:], color = :orange, label = "Legendre")
band!(ax3_L2, n_list, 10 .^(res_avg_L2_christ_leg[:] .+ res_std_L2_christ_leg[:]), 10 .^(res_avg_L2_christ_leg[:] .- res_std_L2_christ_leg[:]), color = :orange, alpha = 0.3)
lines!(ax3_L2, n_list, 10 .^res_avg_L2_christ_cheb[:], color = :yellow4, label = "Chebyshev")
band!(ax3_L2, n_list, 10 .^(res_avg_L2_christ_cheb[:] .+ res_std_L2_christ_cheb[:]), 10 .^(res_avg_L2_christ_cheb[:] .- res_std_L2_christ_cheb[:]), color = :yellow4, alpha = 0.3)
axislegend(ax3_L2, position = :rt)

save("weighted_logistic_L2_exp.png", fig_L2)

# Plotting L_infinity errors
fig_Linf = Figure(size = (1800, 600))

# Uniform sampling
ax1_Linf = Axis(fig_Linf[1, 1], title = "Uniform Sampling (L∞ Error)", xlabel = "n", ylabel = "Error", yscale = log10)
lines!(ax1_Linf, n_list, 10 .^res_avg_Linf_unif_leg[:], color = :blue, label = "Legendre")
band!(ax1_Linf, n_list, 10 .^(res_avg_Linf_unif_leg[:] .+ res_std_Linf_unif_leg[:]), 10 .^(res_avg_Linf_unif_leg[:] .- res_std_Linf_unif_leg[:]), color = :blue, alpha = 0.3)
lines!(ax1_Linf, n_list, 10 .^res_avg_Linf_unif_cheb[:], color = :green, label = "Chebyshev")
band!(ax1_Linf, n_list, 10 .^(res_avg_Linf_unif_cheb[:] .+ res_std_Linf_unif_cheb[:]), 10 .^(res_avg_Linf_unif_cheb[:] .- res_std_Linf_unif_cheb[:]), color = :green, alpha = 0.3)
lines!(ax1_Linf, n_list, 10 .^res_avg_Linf_unif_mon[:], color = :black, label = "Monomial")
band!(ax1_Linf, n_list, 10 .^(res_avg_Linf_unif_mon[:] .+ res_std_Linf_unif_mon[:]), 10 .^(res_avg_Linf_unif_mon[:] .- res_std_Linf_unif_mon[:]), color = :black, alpha = 0.3)
axislegend(ax1_Linf, position = :rt)

# Chebyshev sampling
ax2_Linf = Axis(fig_Linf[1, 2], title = "Chebyshev Sampling (L∞ Error)", xlabel = "n", ylabel = "Error", yscale = log10)
lines!(ax2_Linf, n_list, 10 .^res_avg_Linf_cheb_leg[:], color = :turquoise, label = "Legendre")
band!(ax2_Linf, n_list, 10 .^(res_avg_Linf_cheb_leg[:] .+ res_std_Linf_cheb_leg[:]), 10 .^(res_avg_Linf_cheb_leg[:] .- res_std_Linf_cheb_leg[:]), color = :turquoise, alpha = 0.3)
lines!(ax2_Linf, n_list, 10 .^res_avg_Linf_cheb_cheb[:], color = :purple, label = "Chebyshev")
band!(ax2_Linf, n_list, 10 .^(res_avg_Linf_cheb_cheb[:] .+ res_std_Linf_cheb_cheb[:]), 10 .^(res_avg_Linf_cheb_cheb[:] .- res_std_Linf_cheb_cheb[:]), color = :purple, alpha = 0.3)
lines!(ax2_Linf, n_list, 10 .^res_avg_Linf_cheb_mon[:], color = :pink, label = "Monomial")
band!(ax2_Linf, n_list, 10 .^(res_avg_Linf_cheb_mon[:] .+ res_std_Linf_cheb_mon[:]), 10 .^(res_avg_Linf_cheb_mon[:] .- res_std_Linf_cheb_mon[:]), color = :pink, alpha = 0.3)
axislegend(ax2_Linf, position = :rt)

# Christofell sampling
ax3_Linf = Axis(fig_Linf[1, 3], title = "Christofell Sampling (L∞ Error)", xlabel = "n", ylabel = "Error", yscale = log10)
lines!(ax3_Linf, n_list, 10 .^res_avg_Linf_christ_leg[:], color = :orange, label = "Legendre")
band!(ax3_Linf, n_list, 10 .^(res_avg_Linf_christ_leg[:] .+ res_std_Linf_christ_leg[:]), 10 .^(res_avg_Linf_christ_leg[:] .- res_std_Linf_christ_leg[:]), color = :orange, alpha = 0.3)
lines!(ax3_Linf, n_list, 10 .^res_avg_Linf_christ_cheb[:], color = :yellow4, label = "Chebyshev")
band!(ax3_Linf, n_list, 10 .^(res_avg_Linf_christ_cheb[:] .+ res_std_Linf_christ_cheb[:]), 10 .^(res_avg_Linf_christ_cheb[:] .- res_std_Linf_christ_cheb[:]), color = :yellow4, alpha = 0.3)
axislegend(ax3_Linf, position = :rt)

save("weighted_logistic_Linf_exp.png", fig_Linf)

display(fig_L2)
display(fig_Linf)

