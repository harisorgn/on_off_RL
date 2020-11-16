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

	r_0_v = [0.0, 0.0]
	μ_v = [0.0, 0.0]
	σ_v = [1.0, 1.0]
	γ_v = [0.5, 0.5]

	env_v = [bandit_environment(OU_process(n_warmup_steps, n_steps, n_bandits, n_sessions, r_0_v, μ_v, σ_v, γ_v),
								no_out_process(n_steps, n_bandits, n_sessions), 
								MersenneTwister()),
			bandit_environment(OU_process(n_warmup_steps, n_steps, n_bandits, n_sessions, r_0_v, μ_v, σ_v, γ_v),
								initialise_distribution_out_process(n_steps, n_bandits, n_sessions), 
								MersenneTwister()),
			bandit_environment(OU_process(n_warmup_steps, n_steps, n_bandits, n_sessions, r_0_v, μ_v, σ_v, γ_v),
								initialise_frequency_out_process(n_steps, n_bandits, n_sessions), 
								MersenneTwister()),
			bandit_environment(OU_process(n_warmup_steps, n_steps, n_bandits, n_sessions, r_0_v, μ_v, σ_v, γ_v),
								initialise_delay_out_process(n_steps, n_bandits, n_sessions), 
								MersenneTwister())]

	run_opt(env_v)	
	
end

main()


