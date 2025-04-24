###############################################################################
# This code performs Basis pursuit to approximate the Koopman Operator of the Lorenz 96 model
# This code generates the data for the plot on the tuning parameter
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
trials = 15 # The number of repeated experiments

# Parameters for the BP
σ_list = logrange(1e-16, 1, 30) # The tolerance

# Noise level
ϵ_list = [0, 1e-6, 1e-4, 1e-2]

# Generate the basis functions
basis_Φ = generate_legendre_basis(n)
basis_Ψ = generate_legendre2_basis(n)

k = 20

𝒦_list = Array{Matrix{Float64}}(undef, length(σ_list), trials, 4)

validation_residuals_list1 = zeros(length(σ_list), trials)
validation_residuals_list2 = zeros(length(σ_list), trials)
validation_residuals_list3 = zeros(length(σ_list), trials)
validation_residuals_list4 = zeros(length(σ_list), trials)

# Measuring time of execution
elapsed_time = @elapsed begin
    for ϵ_iter = 1:4
        ϵ = ϵ_list[ϵ_iter]
        for σ_iter = 1:length(σ_list)
            σ = σ_list[σ_iter]
            println("ϵ = ", ϵ, ", σ = ", σ)
            for t in 1:trials
                # Generate the data matrices Φ and Ψ for training
                X, Y = generate_data(k, m, n, dt)
                Ψ₁, Φ₁= generate_observables_matrices(X, Y, basis_Φ, basis_Ψ; ϵ)

                # Generate data for validation
                𝒦 = Variable(length(basis_Φ), length(basis_Ψ))
                problem = minimize(norm(𝒦, 1), [norm(Φ₁ - 𝒦*Ψ₁) <= σ])
                solve!(problem, COSMO.Optimizer; silent = false)
                K₁ = 𝒦.value                
                𝒦_list[σ_iter, t, ϵ_iter] = K₁
            end
        end
    end
end

println("Elapsed time: ", elapsed_time)

# Check recovery of the Koopman operator using Cross Validation

for σ_iter = 1:length(σ_list)
     for t in 1:trials
        X_val, Y_val = generate_data(k, m, dt)
        Ψ₂, Φ₂ = generate_observables_matrices(X_val, Y_val, basis_Φ, basis_Ψ; ϵ = 0)
        validation_residuals_list1[σ_iter, t] = norm(Φ₂ - 𝒦_list[σ_iter, t, 1]*Ψ₂, 2) / norm(𝒦_list[σ_iter, t, 1])
    end
end

for σ_iter = 1:length(σ_list)
    for t in 1:trials
        X_val, Y_val = generate_data(k, m, dt)
       Ψ₂, Φ₂ = generate_observables_matrices(X_val, Y_val, basis_Φ, basis_Ψ; ϵ = 0)
       validation_residuals_list2[σ_iter, t] = norm(Φ₂ - 𝒦_list[σ_iter, t, 2]*Ψ₂, 2) / norm(𝒦_list[σ_iter, t, 2])
   end
end

for σ_iter = 1:length(σ_list)
    for t in 1:trials
        X_val, Y_val = generate_data(k, m, dt)
       Ψ₂, Φ₂ = generate_observables_matrices(X_val, Y_val, basis_Φ, basis_Ψ; ϵ = 0)
       validation_residuals_list3[σ_iter, t] = norm(Φ₂ - 𝒦_list[σ_iter, t, 3]*Ψ₂, 2) / norm(𝒦_list[σ_iter, t, 3])
   end
end

for σ_iter = 1:length(σ_list)
    for t in 1:trials
        X_val, Y_val = generate_data(k, m, dt)
       Ψ₂, Φ₂ = generate_observables_matrices(X_val, Y_val, basis_Φ, basis_Ψ; ϵ = 0)
       validation_residuals_list4[σ_iter, t] = norm(Φ₂ - 𝒦_list[σ_iter, t, 4]*Ψ₂, 2) / norm(𝒦_list[σ_iter, t, 4])
   end
end

mean_res_val1 = mean_row(validation_residuals_list1)
std_res_val1 = std_row(validation_residuals_list1)

mean_res_val2 = mean_row(validation_residuals_list2)
std_res_val2 = std_row(validation_residuals_list2)

mean_res_val3 = mean_row(validation_residuals_list3)
std_res_val3 = std_row(validation_residuals_list3)

mean_res_val4 = mean_row(validation_residuals_list4)
std_res_val4 = std_row(validation_residuals_list4)

# Save the data
save("BP_data_tuning_1.jld2", "epsilon_list", ϵ_list, "sigma_list", σ_list, 
    "mean_validation_residuals_list1", mean_res_val1, "std_validation_residuals_list1", std_res_val1, 
    "mean_validation_residuals_list2", mean_res_val2, "std_validation_residuals_list2", std_res_val2, "mean_validation_residuals_list3", mean_res_val3, 
    "std_validation_residuals_list3", std_res_val3, "mean_validation_residuals_list4", mean_res_val4, "std_validation_residuals_list4", std_res_val4)
