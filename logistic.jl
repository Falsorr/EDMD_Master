#######################################################################
# This code is computing the koopman operator for the  logistic map
# Author : Daniel Fassler
#######################################################################
include("koopman_helper_functions.jl")

# Parameters for the  logistic map
p = [2]
n_list = Int.(ceil.(logrange(1e2, 1e5, 20)))
N = 1

# Jacobi parameters
α = 1
β = 0

m = 5
l = 5
dict_Ψ_Leg = Legendre1(m)
dict_Φ_Leg = Legendre1(l)
dict_Ψ_Cheb = Chebyshev1(m)
dict_Φ_Cheb = Chebyshev1(l)
dict_Ψ_mon = mon1(m)
dict_Φ_mon = mon1(l)
dict_Ψ_fou = fourier1(m)
dict_Φ_fou = fourier1(l)
dict_Ψ_jacobi = jacobi1(m, α, β)
dict_Φ_jacobi = jacobi1(l, α, β)


N_trials = 20

# weight functions
unif_weight(x) = 0.5
cheb_weight(x) = 1/(π*sqrt(1-x[1]^2))
christofell1_Legendre5_weight(x) = 6/(christofell1_Legendre5(x[1]))
christofell1_Chebyshev5_weight(x) = 6/(christofell1_Chebyshev5(x[1]))
jacobi_weight(x; α = α, β = β) = (1-x[1])^α*(1+x[1])^β
christofell1_jacobi5_1_0_weight(x) = 6/(christofell1_jacobi5_1_0(x[1]))
christofell1_fourier5_weight(x) = 11/(christofell1_fourier5(x[1]))


residuals_unif_leg = zeros(length(n_list), N_trials)
residuals_unif_cheb = zeros(length(n_list), N_trials)
residuals_cheb_leg = zeros(length(n_list), N_trials)
residuals_cheb_cheb = zeros(length(n_list), N_trials)
residuals_christ_leg = zeros(length(n_list), N_trials)
residuals_christ_cheb = zeros(length(n_list), N_trials)
residuals_unif_mon = zeros(length(n_list), N_trials)
residuals_unif_fou = zeros(length(n_list), N_trials)
residuals_cheb_mon = zeros(length(n_list), N_trials)
residuals_cheb_fou = zeros(length(n_list), N_trials)
residuals_jacobi_jacobi = zeros(length(n_list), N_trials)
residuals_unif_jacobi = zeros(length(n_list), N_trials)
residuals_cheb_jacobi = zeros(length(n_list), N_trials)
residuals_christ_jacobi = zeros(length(n_list), N_trials)
residuals_christ_fourier = zeros(length(n_list), N_trials)


@info "Computing High fidelity approximations"

ΨX, ΦY, 𝒦_unif_leg_hf = koopman(uniform_sampling_sym, shifted_logistic, Int(1e6), N, dict_Ψ_Leg, dict_Φ_Leg, p ; verbose = true)
ΨX, ΦY, 𝒦_unif_cheb_hf = koopman(uniform_sampling_sym, shifted_logistic, Int(1e6), N, dict_Ψ_Cheb, dict_Φ_Cheb, p ; weight  = x -> cheb_weight(x)/unif_weight(x), verbose = true)
ΨX, ΦY, 𝒦_cheb_leg_hf = koopman(cheb_sampling_sym, shifted_logistic, Int(1e6), N, dict_Ψ_Leg, dict_Φ_Leg, p ; weight = x -> unif_weight(x)/cheb_weight(x), verbose = true)
ΨX, ΦY, 𝒦_cheb_cheb_hf = koopman(cheb_sampling_sym, shifted_logistic, Int(1e6), N, dict_Ψ_Cheb, dict_Φ_Cheb, p ; verbose = true)
ΨX, ΦY, 𝒦_christ_leg_hf = koopman(christofell1_sampling_Legendre5, shifted_logistic, Int(1e6), N, dict_Ψ_Leg, dict_Φ_Leg, p ; weight = christofell1_Legendre5_weight, verbose = true)
ΨX, ΦY, 𝒦_christ_cheb_hf = koopman(christofell1_sampling_Chebyshev5, shifted_logistic, Int(1e6), N, dict_Ψ_Cheb, dict_Φ_Cheb, p ; weight = christofell1_Chebyshev5_weight, verbose = true)
ΨX, ΦY, 𝒦_unif_mon_hf = koopman(uniform_sampling_sym, shifted_logistic, Int(1e6), N, dict_Ψ_mon, dict_Φ_mon, p ; verbose = true)
ΨX, ΦY, 𝒦_unif_fou_hf = koopman(uniform_sampling_sym, shifted_logistic, Int(1e6), N, dict_Ψ_fou, dict_Φ_fou, p ; verbose = true)
ΨX, ΦY, 𝒦_cheb_mon_hf = koopman(cheb_sampling_sym, shifted_logistic, Int(1e6), N, dict_Ψ_mon, dict_Φ_mon, p ; verbose = true)
ΨX, ΦY, 𝒦_cheb_fou_hf = koopman(cheb_sampling_sym, shifted_logistic, Int(1e6), N, dict_Ψ_fou, dict_Φ_fou, p ; weight = x -> unif_weight(x)/cheb_weight(x), verbose = true)
ΨX, ΦY, 𝒦_jacobi_jacobi_hf = koopman(jacobi1_1_0_sampling, shifted_logistic, Int(1e6), N, dict_Ψ_jacobi, dict_Φ_jacobi, p ; verbose = true)
ΨX, ΦY, 𝒦_unif_jacobi_hf = koopman(uniform_sampling_sym, shifted_logistic, Int(1e6), N, dict_Ψ_jacobi, dict_Φ_jacobi, p ; weight = x -> jacobi_weight(x) / unif_weight(x), verbose = true)
ΨX, ΦY, 𝒦_cheb_jacobi_hf = koopman(cheb_sampling_sym, shifted_logistic, Int(1e6), N, dict_Ψ_jacobi, dict_Φ_jacobi, p ; weight = x -> jacobi_weight(x) / cheb_weight(x), verbose = true)
ΨX, ΦY, 𝒦_christ_jacobi_hf = koopman(christofell1_sampling_jacobi5_1_0, shifted_logistic, Int(1e6), N, dict_Ψ_jacobi, dict_Φ_jacobi, p ; weight = christofell1_jacobi5_1_0_weight, verbose = true)
ΨX, ΦY, 𝒦_christ_fourier_hf = koopman(christofell1_sampling_fourier5, shifted_logistic, Int(1e6), N, dict_Ψ_fou, dict_Φ_fou, p ; weight = christofell1_fourier5_weight, verbose = true)

for i = 1:N_trials
    for (j, m) in enumerate(n_list)
        @info "m = $m, trial = $i"
        ΨX, ΦY, K_unif_leg = koopman(uniform_sampling_sym, shifted_logistic, Int(m), N, dict_Ψ_Leg, dict_Φ_Leg, p)
        ΨX, ΦY, K_unif_cheb = koopman(uniform_sampling_sym, shifted_logistic, Int(m), N, dict_Ψ_Cheb, dict_Φ_Cheb, p ; weight = x -> cheb_weight(x)/unif_weight(x))
        ΨX, ΦY, K_cheb_leg = koopman(cheb_sampling_sym, shifted_logistic, Int(m), N, dict_Ψ_Leg, dict_Φ_Leg, p ; weight = x -> unif_weight(x)/cheb_weight(x))
        ΨX, ΦY, K_cheb_cheb = koopman(cheb_sampling_sym, shifted_logistic, Int(m), N, dict_Ψ_Cheb, dict_Φ_Cheb, p)
        ΨX, ΦY, K_christ_leg = koopman(christofell1_sampling_Legendre5, shifted_logistic, Int(m), N, dict_Ψ_Leg, dict_Φ_Leg, p ; weight = christofell1_Legendre5_weight)
        ΨX, ΦY, K_christ_cheb = koopman(christofell1_sampling_Chebyshev5, shifted_logistic, Int(m), N, dict_Ψ_Cheb, dict_Φ_Cheb, p ; weight = christofell1_Chebyshev5_weight)
        ΨX, ΦY, K_unif_mon = koopman(uniform_sampling_sym, shifted_logistic, Int(m), N, dict_Ψ_mon, dict_Φ_mon, p)
        ΨX, ΦY, K_unif_fou = koopman(uniform_sampling_sym, shifted_logistic, Int(m), N, dict_Ψ_fou, dict_Φ_fou, p)
        ΨX, ΦY, K_cheb_mon = koopman(cheb_sampling_sym, shifted_logistic, Int(m), N, dict_Ψ_mon, dict_Φ_mon, p)
        ΨX, ΦY, K_cheb_fou = koopman(cheb_sampling_sym, shifted_logistic, Int(m), N, dict_Ψ_fou, dict_Φ_fou, p ; weight = x -> unif_weight(x)/cheb_weight(x))
        ΨX, ΦY, K_jacobi_jacobi = koopman(jacobi1_1_0_sampling, shifted_logistic, Int(m), N, dict_Ψ_jacobi, dict_Φ_jacobi, p)
        ΨX, ΦY, K_unif_jacobi = koopman(uniform_sampling_sym, shifted_logistic, Int(m), N,  dict_Ψ_jacobi, dict_Φ_jacobi, p ; weight = x -> jacobi_weight(x) / unif_weight(x))
        ΨX, ΦY, K_cheb_jacobi = koopman(cheb_sampling_sym, shifted_logistic, Int(m), N,  dict_Ψ_jacobi, dict_Φ_jacobi, p ; weight = x -> jacobi_weight(x) / cheb_weight(x))
        ΨX, ΦY, K_christ_jacobi = koopman(christofell1_sampling_jacobi5_1_0, shifted_logistic, Int(m), N,  dict_Ψ_jacobi, dict_Φ_jacobi, p ; weight = christofell1_jacobi5_1_0_weight)
        ΨX, ΦY, K_christ_fourier = koopman(christofell1_sampling_fourier5, shifted_logistic, Int(m), N, dict_Ψ_fou, dict_Φ_fou, p ; weight = christofell1_fourier5_weight)

        residuals_unif_leg[j, i] = norm(𝒦_unif_leg_hf - K_unif_leg)/norm(𝒦_unif_leg_hf)
        residuals_unif_cheb[j, i] = norm(𝒦_unif_cheb_hf - K_unif_cheb)/norm(𝒦_unif_cheb_hf)
        residuals_cheb_leg[j, i] = norm(𝒦_cheb_leg_hf - K_cheb_leg)/norm(𝒦_cheb_leg_hf)
        residuals_cheb_cheb[j, i] = norm(𝒦_cheb_cheb_hf - K_cheb_cheb)/norm(𝒦_cheb_cheb_hf)
        residuals_christ_leg[j, i] = norm(𝒦_christ_leg_hf - K_christ_leg)/norm(𝒦_christ_leg_hf)
        residuals_christ_cheb[j, i] = norm(𝒦_christ_cheb_hf - K_christ_cheb)/norm(𝒦_christ_cheb_hf)
        residuals_unif_mon[j, i] = norm(𝒦_unif_mon_hf - K_unif_mon)/norm(𝒦_unif_mon_hf)
        residuals_unif_fou[j, i] = norm(𝒦_unif_fou_hf - K_unif_fou)/norm(𝒦_unif_fou_hf)
        residuals_cheb_mon[j, i] = norm(𝒦_cheb_mon_hf - K_cheb_mon)/norm(𝒦_cheb_mon_hf)
        residuals_cheb_fou[j, i] = norm(𝒦_cheb_fou_hf - K_cheb_fou)/norm(𝒦_cheb_fou_hf)
        residuals_jacobi_jacobi[j, i] = norm(𝒦_jacobi_jacobi_hf - K_jacobi_jacobi)/norm(𝒦_jacobi_jacobi_hf)
        residuals_unif_jacobi[j, i] = norm(𝒦_unif_jacobi_hf - K_unif_jacobi)/norm(𝒦_unif_jacobi_hf)
        residuals_cheb_jacobi[j, i] = norm(𝒦_cheb_jacobi_hf - K_cheb_jacobi)/norm(𝒦_cheb_jacobi_hf)
        residuals_christ_jacobi[j, i] = norm(𝒦_christ_jacobi_hf - K_christ_jacobi)/norm(𝒦_christ_jacobi_hf)
        residuals_christ_fourier[j, i] = norm(𝒦_christ_fourier_hf - K_christ_fourier)/norm(𝒦_christ_fourier_hf)
    end
end

res_avg_unif_leg, res_std_unif_leg = log_avg_std(residuals_unif_leg)
res_avg_unif_cheb, res_std_unif_cheb = log_avg_std(residuals_unif_cheb)
res_avg_cheb_leg, res_std_cheb_leg = log_avg_std(residuals_cheb_leg)
res_avg_cheb_cheb, res_std_cheb_cheb = log_avg_std(residuals_cheb_cheb)
res_avg_christ_leg, res_std_christ_leg = log_avg_std(residuals_christ_leg)
res_avg_christ_cheb, res_std_christ_cheb = log_avg_std(residuals_christ_cheb)
res_avg_unif_mon, res_std_unif_mon = log_avg_std(residuals_unif_mon)
res_avg_unif_fou, res_std_unif_fou = log_avg_std(residuals_unif_fou)
res_avg_cheb_mon, res_std_cheb_mon = log_avg_std(residuals_cheb_mon)
res_avg_cheb_fou, res_std_cheb_fou = log_avg_std(residuals_cheb_fou)
res_avg_jacobi_jacobi, res_std_jacobi_jacobi = log_avg_std(residuals_jacobi_jacobi)
res_avg_unif_jacobi, res_std_unif_jacobi = log_avg_std(residuals_unif_jacobi)
res_avg_cheb_jacobi, res_std_cheb_jacobi = log_avg_std(residuals_cheb_jacobi)
res_avg_christ_jacobi, res_std_christ_jacobi = log_avg_std(residuals_christ_jacobi)
res_avg_christ_fourier, res_std_christ_fourier = log_avg_std(residuals_christ_fourier)

# Saving
save("data_logistic_map_n_equal_l.jld2", 
    "n_list", n_list, 
    "residuals_unif_leg", residuals_unif_leg, 
    "residuals_unif_cheb", residuals_unif_cheb, 
    "residuals_cheb_leg", residuals_cheb_leg, 
    "residuals_cheb_cheb", residuals_cheb_cheb, 
    "residuals_christ_leg", residuals_christ_leg, 
    "residuals_christ_cheb", residuals_christ_cheb, 
    "residuals_unif_mon", residuals_unif_mon, 
    "residuals_unif_fou", residuals_unif_fou, 
    "residuals_cheb_mon", residuals_cheb_mon, 
    "residuals_cheb_fou", residuals_cheb_fou, 
    "residuals_jacobi_jacobi", residuals_jacobi_jacobi, 
    "residuals_unif_jacobi", residuals_unif_jacobi, 
    "residuals_cheb_jacobi", residuals_cheb_jacobi, 
    "residuals_christ_jacobi", residuals_christ_jacobi, 
    "residuals_christ_fourier", residuals_christ_fourier, 
    "res_avg_unif_leg", res_avg_unif_leg, 
    "res_std_unif_leg", res_std_unif_leg, 
    "res_avg_unif_cheb", res_avg_unif_cheb, 
    "res_std_unif_cheb", res_std_unif_cheb, 
    "res_avg_cheb_leg", res_avg_cheb_leg, 
    "res_std_cheb_leg", res_std_cheb_leg, 
    "res_avg_cheb_cheb", res_avg_cheb_cheb, 
    "res_std_cheb_cheb", res_std_cheb_cheb, 
    "res_avg_christ_leg", res_avg_christ_leg, 
    "res_std_christ_leg", res_std_christ_leg, 
    "res_avg_christ_cheb", res_avg_christ_cheb, 
    "res_std_christ_cheb", res_std_christ_cheb, 
    "res_avg_unif_mon", res_avg_unif_mon, 
    "res_std_unif_mon", res_std_unif_mon, 
    "res_avg_unif_fou", res_avg_unif_fou, 
    "res_std_unif_fou", res_std_unif_fou, 
    "res_avg_cheb_mon", res_avg_cheb_mon, 
    "res_std_cheb_mon", res_std_cheb_mon, 
    "res_avg_cheb_fou", res_avg_cheb_fou, 
    "res_std_cheb_fou", res_std_cheb_fou, 
    "res_avg_jacobi_jacobi", res_avg_jacobi_jacobi, 
    "res_std_jacobi_jacobi", res_std_jacobi_jacobi, 
    "res_avg_unif_jacobi", res_avg_unif_jacobi, 
    "res_std_unif_jacobi", res_std_unif_jacobi, 
    "res_avg_cheb_jacobi", res_avg_cheb_jacobi, 
    "res_std_cheb_jacobi", res_std_cheb_jacobi, 
    "res_avg_christ_jacobi", res_avg_christ_jacobi, 
    "res_std_christ_jacobi", res_std_christ_jacobi, 
    "res_avg_christ_fourier", res_avg_christ_fourier, 
    "res_std_christ_fourier", res_std_christ_fourier)

# Plotting
fig = Figure(size = (1600, 1200))

# Legendre Basis
ax1 = Axis(fig[1, 1], xlabel = "Number of samples", ylabel = "Error", yscale = log10, xscale = log10, title = "Legendre Basis")
lines!(ax1, n_list, 10 .^res_avg_unif_leg[:], color = :blue, label = "Uniform Sampling")
band!(ax1, n_list, 10 .^(res_avg_unif_leg[:] .+ res_std_unif_leg[:]), 10 .^(res_avg_unif_leg[:] .- res_std_unif_leg[:]), color = :blue, alpha = 0.3)
lines!(ax1, n_list, 10 .^res_avg_cheb_leg[:], color = :green, label = "Chebyshev Sampling")
band!(ax1, n_list, 10 .^(res_avg_cheb_leg[:] .+ res_std_cheb_leg[:]), 10 .^(res_avg_cheb_leg[:] .- res_std_cheb_leg[:]), color = :green, alpha = 0.3)
lines!(ax1, n_list, 10 .^res_avg_christ_leg[:], color = :red, label = "Christofell Sampling")
band!(ax1, n_list, 10 .^(res_avg_christ_leg[:] .+ res_std_christ_leg[:]), 10 .^(res_avg_christ_leg[:] .- res_std_christ_leg[:]), color = :red, alpha = 0.3)
lines!(ax1, n_list, n_list.^(-1/2), color = :black, label = "m⁻¹/²", linestyle = :dash)
Legend(fig[2, 1], ax1, "Legendre Basis - Sampling Methods")

# Chebyshev Basis
ax2 = Axis(fig[1, 2], xlabel = "Number of samples", ylabel = "Error", yscale = log10, xscale = log10, title = "Chebyshev Basis")
lines!(ax2, n_list, 10 .^res_avg_unif_cheb[:], color = :blue, label = "Uniform Sampling")
band!(ax2, n_list, 10 .^(res_avg_unif_cheb[:] .+ res_std_unif_cheb[:]), 10 .^(res_avg_unif_cheb[:] .- res_std_unif_cheb[:]), color = :blue, alpha = 0.3)
lines!(ax2, n_list, 10 .^res_avg_cheb_cheb[:], color = :green, label = "Chebyshev Sampling")
band!(ax2, n_list, 10 .^(res_avg_cheb_cheb[:] .+ res_std_cheb_cheb[:]), 10 .^(res_avg_cheb_cheb[:] .- res_std_cheb_cheb[:]), color = :green, alpha = 0.3)
lines!(ax2, n_list, 10 .^res_avg_christ_cheb[:], color = :red, label = "Christofell Sampling")
band!(ax2, n_list, 10 .^(res_avg_christ_cheb[:] .+ res_std_christ_cheb[:]), 10 .^(res_avg_christ_cheb[:] .- res_std_christ_cheb[:]), color = :red, alpha = 0.3)
lines!(ax2, n_list, n_list.^(-1/2), color = :black, label = "m⁻¹/²", linestyle = :dash)
Legend(fig[2, 2], ax2, "Chebyshev Basis - Sampling Methods")

# Monomial Basis
ax3 = Axis(fig[3, 1], xlabel = "Number of samples", ylabel = "Error", yscale = log10, xscale = log10, title = "Monomial Basis")
lines!(ax3, n_list, 10 .^res_avg_unif_mon[:], color = :blue, label = "Uniform Sampling")
band!(ax3, n_list, 10 .^(res_avg_unif_mon[:] .+ res_std_unif_mon[:]), 10 .^(res_avg_unif_mon[:] .- res_std_unif_mon[:]), color = :blue, alpha = 0.3)
lines!(ax3, n_list, 10 .^res_avg_cheb_mon[:], color = :green, label = "Chebyshev Sampling")
band!(ax3, n_list, 10 .^(res_avg_cheb_mon[:] .+ res_std_cheb_mon[:]), 10 .^(res_avg_cheb_mon[:] .- res_std_cheb_mon[:]), color = :green, alpha = 0.3)
lines!(ax3, n_list, n_list.^(-1/2), color = :black, label = "m⁻¹/²", linestyle = :dash)
Legend(fig[4, 1], ax3, "Monomial Basis - Sampling Methods")

# Fourier Basis
ax4 = Axis(fig[3, 2], xlabel = "Number of samples", ylabel = "Error", yscale = log10, xscale = log10, title = "Fourier Basis")
lines!(ax4, n_list, 10 .^res_avg_unif_fou[:], color = :blue, label = "Uniform Sampling")
band!(ax4, n_list, 10 .^(res_avg_unif_fou[:] .+ res_std_unif_fou[:]), 10 .^(res_avg_unif_fou[:] .- res_std_unif_fou[:]), color = :blue, alpha = 0.3)
lines!(ax4, n_list, 10 .^res_avg_cheb_fou[:], color = :green, label = "Chebyshev Sampling")
band!(ax4, n_list, 10 .^(res_avg_cheb_fou[:] .+ res_std_cheb_fou[:]), 10 .^(res_avg_cheb_fou[:] .- res_std_cheb_fou[:]), color = :green, alpha = 0.3)
lines!(ax4, n_list, 10 .^res_avg_christ_fourier[:], color = :red, label = "Christofell Sampling")
band!(ax4, n_list, 10 .^(res_avg_christ_fourier[:] .+ res_std_christ_fourier[:]), 10 .^(res_avg_christ_fourier[:] .- res_std_christ_fourier[:]), color = :red, alpha = 0.3)
lines!(ax4, n_list, n_list.^(-1/2), color = :black, label = "m⁻¹/²", linestyle = :dash)
Legend(fig[4, 2], ax4, "Fourier Basis - Sampling Methods")

# Jacobi Basis
ax5 = Axis(fig[5, 1], xlabel = "Number of samples", ylabel = "Error", yscale = log10, xscale = log10, title = "Jacobi Basis")
lines!(ax5, n_list, 10 .^res_avg_jacobi_jacobi[:], color = :blue, label = "Uniform Sampling")
band!(ax5, n_list, 10 .^(res_avg_jacobi_jacobi[:] .+ res_std_jacobi_jacobi[:]), 10 .^(res_avg_jacobi_jacobi[:] .- res_std_jacobi_jacobi[:]), color = :blue, alpha = 0.3)
lines!(ax5, n_list, 10 .^res_avg_unif_jacobi[:], color = :green, label = "Chebyshev Sampling")
band!(ax5, n_list, 10 .^(res_avg_unif_jacobi[:] .+ res_std_unif_jacobi[:]), 10 .^(res_avg_unif_jacobi[:] .- res_std_unif_jacobi[:]), color = :green, alpha = 0.3)
lines!(ax5, n_list, 10 .^res_avg_cheb_jacobi[:], color = :red, label = "Christofell Sampling")
band!(ax5, n_list, 10 .^(res_avg_cheb_jacobi[:] .+ res_std_cheb_jacobi[:]), 10 .^(res_avg_cheb_jacobi[:] .- res_std_cheb_jacobi[:]), color = :red, alpha = 0.3)
lines!(ax5, n_list, 10 .^res_avg_christ_jacobi[:], color = :purple, label = "Jacobi Sampling")
band!(ax5, n_list, 10 .^(res_avg_christ_jacobi[:] .+ res_std_christ_jacobi[:]), 10 .^(res_avg_christ_jacobi[:] .- res_std_christ_jacobi[:]), color = :purple, alpha = 0.3)
lines!(ax5, n_list, n_list.^(-1/2), color = :black, label = "m⁻¹/²", linestyle = :dash)
Legend(fig[6, 1], ax5, "Jacobi Basis - Sampling Methods")

save("weighted_logistic_n_equal_l.png", fig)
fig

