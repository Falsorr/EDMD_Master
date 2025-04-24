###############################################################################
# This code performs LASSO to approximate the Koopman Operator of the Lorenz 96 model
# It will produce the data to figure out an adequate tuning parameter
# Author: Daniel Fassler
###############################################################################

using LinearAlgebra, Random, CairoMakie, JLD2, DynamicalSystems, Convex, COSMO
include("lorenz96_helper_functions.jl")

# Define the Lorenz 96 model
n = 10 # The number of variables, in the main paper, n = 50 is used
F = 8 # The forcing term, usually, F = 8 causes chaos

# Parameters of the simulation
dt = 0.001 # The time step
m = 5 # The number of points in a burst
k = 20 # The number of bursts
trials = 15 # The number of repeated experiments

# Parameters for the LASSO
λ_list = logrange(1e-16, 10, 34) # The tolerance

# Noise level
ϵ_list = [0, 1e-6, 1e-4, 1e-2]

basis_Φ = generate_legendre_basis(n)
basis_Ψ = generate_legendre2_basis(n)

𝒦_list = Array{Matrix{Float64}}(undef, length(λ_list), trials, 4)

validation_residuals_list1 = zeros(length(λ_list), trials)
validation_residuals_list2 = zeros(length(λ_list), trials)
validation_residuals_list3 = zeros(length(λ_list), trials)
validation_residuals_list4 = zeros(length(λ_list), trials)

# Measuring time of execution
elapsed_time = @elapsed begin
    for ϵ_iter = 1:4
        ϵ = ϵ_list[ϵ_iter]
        for λ_iter = 1:length(λ_list)
            λ = λ_list[λ_iter]
            println("ϵ = ", ϵ, ", λ = ", λ)
            for t in 1:trials
                # Generate the data matrices Φ and Ψ for training
                X, Y = generate_data(k, m, n, dt)
                Ψ₁, Φ₁= generate_observables_matrices(X, Y, basis_Φ, basis_Ψ; ϵ)

                # Generate data for validation
                𝒦 = Variable(length(basis_Φ), length(basis_Ψ))
                problem = minimize(sumsquares(Φ₁ - 𝒦*Ψ₁) + λ*norm(𝒦, 1))
                solve!(problem, COSMO.Optimizer; silent = false)
                K₁ = 𝒦.value                
                𝒦_list[λ_iter, t, ϵ_iter] = K₁
            end
        end
    end
end

println("Elapsed time: ", elapsed_time)

# Check recovery of the Koopman operator using Cross Validation

for λ_iter = 1:length(λ_list)
     for t in 1:trials
        X_val, Y_val = generate_data(k, m, n, dt)
        Ψ₂, Φ₂ = generate_observables_matrices(X_val, Y_val, basis_Φ, basis_Ψ; ϵ = 0)
        validation_residuals_list1[λ_iter, t] = norm(Φ₂ - 𝒦_list[λ_iter, t, 1]*Ψ₂, 2) / norm(𝒦_list[λ_iter, t, 1])
    end
end

for λ_iter = 1:length(λ_list)
    for t in 1:trials
        X_val, Y_val = generate_data(k, m, n, dt)
       Ψ₂, Φ₂ = generate_observables_matrices(X_val, Y_val, basis_Φ, basis_Ψ; ϵ = 0)
       validation_residuals_list2[λ_iter, t] = norm(Φ₂ - 𝒦_list[λ_iter, t, 2]*Ψ₂, 2) / norm(𝒦_list[λ_iter, t, 2])
    end
end

for λ_iter = 1:length(λ_list)
    for t in 1:trials
        X_val, Y_val = generate_data(k, m, n, dt)
       Ψ₂, Φ₂ = generate_observables_matrices(X_val, Y_val, basis_Φ, basis_Ψ; ϵ = 0)
       validation_residuals_list3[λ_iter, t] = norm(Φ₂ - 𝒦_list[λ_iter, t, 3]*Ψ₂, 2) / norm(𝒦_list[λ_iter, t, 3])
    end
end

for λ_iter = 1:length(λ_list)
    for t in 1:trials
        X_val, Y_val = generate_data(k, m, n, dt)
       Ψ₂, Φ₂ = generate_observables_matrices(X_val, Y_val, basis_Φ, basis_Ψ; ϵ = 0)
       validation_residuals_list4[λ_iter, t] = norm(Φ₂ - 𝒦_list[λ_iter, t, 4]*Ψ₂, 2) / norm(𝒦_list[λ_iter, t, 4])
    end
end

mean_res_val1 = mean_row(validation_residuals_list1)
mean_res_val2 = mean_row(validation_residuals_list2)
mean_res_val3 = mean_row(validation_residuals_list3)
mean_res_val4 = mean_row(validation_residuals_list4)

std_res_val1 = std_row(validation_residuals_list1)
std_res_val2 = std_row(validation_residuals_list2)
std_res_val3 = std_row(validation_residuals_list3)
std_res_val4 = std_row(validation_residuals_list4)

# Save the data
save("LASSO_data_tuning.jld2", "epsilon_list", ϵ_list, "lambda_list", λ_list, 
    "mean_validation_residuals_list1", mean_res_val1, 
    "mean_validation_residuals_list2", mean_res_val2, 
    "mean_validation_residuals_list3", mean_res_val3, 
    "mean_validation_residuals_list4", mean_res_val4, 
    "std_validation_residuals_list1", std_res_val1, 
    "std_validation_residuals_list2", std_res_val2, 
    "std_validation_residuals_list3", std_res_val3, 
    "std_validation_residuals_list4", std_res_val4)
