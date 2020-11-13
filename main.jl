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
	
	n_warmup_steps = 0
	n_steps = 40
	n_sessions = 10
	n_bandits = 2

	μ_v = [0.0, 0.0]
	σ_v = [1.0, 1.0]
	γ_v = [0.5, 0.5]

	env_v = repeat([initialise_OU_bandit_environment(n_warmup_steps, n_steps, n_bandits, n_sessions, μ_v, σ_v, γ_v),
					initialise_OU_bandit_distribution_outlier_environment(n_warmup_steps, n_steps, n_bandits, n_sessions, μ_v, σ_v, γ_v),
					initialise_OU_bandit_frequency_outlier_environment(n_warmup_steps, n_steps, n_bandits, n_sessions, μ_v, σ_v, γ_v),
					initialise_OU_bandit_delay_outlier_environment(n_warmup_steps, n_steps, n_bandits, n_sessions, μ_v, σ_v, γ_v)], 
					1)

	run_opt(env_v)	
	
	#=
	file_name = "opt_res_3879.jld"

	env_v = load(file_name, "env_v")
	d_agent_bias = load(file_name, "d_agent_bias")
	d_agent_Q = load(file_name, "d_agent_Q")
	d_agent_no_offline = load(file_name, "d_agent_no_offline")

	println("γ = ", env_v[1].γ_v[1])

	plot_performance(env_v, [d_agent_bias, d_agent_Q, d_agent_no_offline]; n_runs = 100)
	=#
	
end

main()


