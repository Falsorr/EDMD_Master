###############################################################################
# This code performs Basis pursuit to approximate the Koopman Operator of the Lorenz 96 model
# It will compute the data for the convergence plot of Basis pursuit
# Author: Daniel Fassler
###############################################################################

using LinearAlgebra, Random, CairoMakie, JLD2, DynamicalSystems, Convex, COSMO
include("lorenz96_helper_functions.jl")


# Define the Lorenz 96 model
n = 10 # The number of variables, in the main paper, n = 50 is used
F = 8 # The forcing term, usually, F = 8 causes chaos

# Parameters of the simulation
dt = 0.001 # The time step
m = 10 # The number of points in a burst
trials = 30 # The number of repeated experiments

# Parameters for the BP
σ = 10^(-2) # The tolerance

# Generate the basis functions
basis_Φ = generate_legendre_basis(n)
basis_Ψ = generate_legendre2_basis(n)


# We choose our range of bursts to avoid oversampling
k_list = LinRange(2, length(basis_Φ) * length(basis_Ψ) / m - 1, 40) 

𝒦_list = Array{Matrix{Float64}}(undef, length(k_list), trials)
residual_list = zeros(length(k_list), trials)

# Measuring time of execution
elapsed_time = @elapsed begin
    for k_iter = 1:length(k_list)
        k = round(Int, k_list[k_iter])
        println(k / (length(basis_Φ) * length(basis_Ψ) / m))
        for t in 1:trials
        
            # Generate the data matrices Ψ and Φ for training
            X, Y = generate_data(k, m, n, dt)
            Ψ₁, Φ₁= generate_observables_matrices(X, Y, basis_Φ, basis_Ψ)

            # Call the comvex solver
            𝒦 = Variable(length(basis_Φ), length(basis_Ψ))
            problem = minimize(norm(𝒦, 1), [norm(Φ₁ - 𝒦*Ψ₁) <= σ])
            solve!(problem, COSMO.Optimizer; silent = false)
            K₁ = 𝒦.value

            # Save the results
            𝒦_list[k_iter, t] = K₁
            residual_list[k_iter, t] = norm(Φ₁ - K₁*Ψ₁, 2)
        end
    end
end

println("Elapsed time: ", elapsed_time)

# Check recovery of the Koopman operator using Cross Validation
validation_residuals_list = zeros(length(k_list), trials)
for k_iter = 1:length(k_list)
    k = round(Int, k_list[k_iter])
    for t in 1:trials
        X_val, Y_val = generate_data(k, m, n, dt)
        Ψ₂, Φ₂ = generate_observables_matrices(X_val, Y_val, basis_Φ, basis_Ψ)
        validation_residuals_list[k_iter, t] = norm(Φ₂ - 𝒦_list[k_iter, t]*Ψ₂, 2)
    end
end

mean_res_val = mean_row(validation_residuals_list)
std_res_val = std_row(validation_residuals_list)
undersampling_rate = k_list ./ (length(basis_Φ) * length(basis_Ψ) / m)

# Save the data
save("BP_data_run_1.jld2", "K_list", 𝒦_list, "residual_list", residual_list, "undersampling_rate", undersampling_rate, "mean_validation_residuals_list", mean_res_val, "std_validation_residuals_list", std_res_val)
