using Distributions
using Random
using Statistics
using JLD
using NLopt

include("agents.jl")
include("environments.jl")
include("auxiliary.jl")
include("opt.jl")
include("plot.jl")

function main()

	#=
	n_steps = 40
	n_sessions = 10
	n_bandits = 2

	μ_v = [0.0, 0.0]
	σ_v = [1.0, 1.0]
	γ_v = [0.01, 0.01]

	env_v = [initialise_OU_bandit_environment(n_steps, n_bandits, n_sessions, μ_v, σ_v, γ_v),
			initialise_OU_bandit_distribution_outlier_environment(n_steps, n_bandits, n_sessions, μ_v, σ_v, γ_v),
			initialise_OU_bandit_frequency_outlier_environment(n_steps, n_bandits, n_sessions, μ_v, σ_v, γ_v),
			initialise_OU_bandit_delay_outlier_environment(n_steps, n_bandits, n_sessions, μ_v, σ_v, γ_v)]

	run_opt(env_v)	
	=#

	#=
	file_name = "opt_res_9254.jld"

	env_v = load(file_name, "env_v")
	d_agent = load(file_name, "d_agent")
	prob_d_agent = load(file_name, "prob_d_agent")

	println("γ = ", env_v[1].γ_v[1])

	plot_performance([env_v[3]], [d_agent, prob_d_agent], 100)
	=#

	#plot_OU([0.01, 0.1, 0.2, 0.5], [1.0, 1.0, 1.0, 1.0] ; t = 10.0)
end

main()


