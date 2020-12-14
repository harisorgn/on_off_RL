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

	r_out = 10.0

	μ_v = [3.0, 3.0]
	σ_v = [1.0, 1.0]

	r_0_v = μ_v

	γ_v = [0.01, 0.01]

	env_v = repeat([bandit_environment(OU_process(n_warmup_steps, n_steps, n_bandits, n_sessions, r_0_v, γ_v, μ_v, σ_v),
										initialise_delay_out_process(n_steps, n_bandits, n_sessions, r_out), 
										MersenneTwister())],
					100)

	run_opt(env_v)	
	
	γ_v = [0.2, 0.2]

	env_v = repeat([bandit_environment(OU_process(n_warmup_steps, n_steps, n_bandits, n_sessions, r_0_v, γ_v, μ_v, σ_v),
										initialise_delay_out_process(n_steps, n_bandits, n_sessions, r_out), 
										MersenneTwister())],
					100)

	run_opt(env_v)	

	γ_v = [0.5, 0.5]

	env_v = repeat([bandit_environment(OU_process(n_warmup_steps, n_steps, n_bandits, n_sessions, r_0_v, γ_v, μ_v, σ_v),
										initialise_delay_out_process(n_steps, n_bandits, n_sessions, r_out), 
										MersenneTwister())],
					100)

	run_opt(env_v)	
	

	γ_v = [0.8, 0.8]

	env_v = repeat([bandit_environment(OU_process(n_warmup_steps, n_steps, n_bandits, n_sessions, r_0_v, γ_v, μ_v, σ_v),
										initialise_delay_out_process(n_steps, n_bandits, n_sessions, r_out), 
										MersenneTwister())],
					100)

	run_opt(env_v)	


	γ_v = [1.0, 1.0]

	env_v = repeat([bandit_environment(OU_process(n_warmup_steps, n_steps, n_bandits, n_sessions, r_0_v, γ_v, μ_v, σ_v),
										initialise_delay_out_process(n_steps, n_bandits, n_sessions, r_out), 
										MersenneTwister())],
					100)

	run_opt(env_v)

	#=
	env_v = repeat([bandit_environment(OU_process(n_warmup_steps, n_steps, n_bandits, n_sessions, r_0_v, γ_v, μ_v, σ_v),
										no_out_process(n_steps, n_bandits, n_sessions), 
										MersenneTwister()),
					bandit_environment(OU_process(n_warmup_steps, n_steps, n_bandits, n_sessions, r_0_v, γ_v, μ_v, σ_v),
										initialise_distribution_out_process(n_steps, n_bandits, n_sessions, r_out), 
										MersenneTwister()),
					bandit_environment(OU_process(n_warmup_steps, n_steps, n_bandits, n_sessions, r_0_v, γ_v, μ_v, σ_v),
										initialise_frequency_out_process(n_steps, n_bandits, n_sessions, r_out), 
										MersenneTwister()),
					bandit_environment(OU_process(n_warmup_steps, n_steps, n_bandits, n_sessions, r_0_v, γ_v, μ_v, σ_v),
										initialise_delay_out_process(n_steps, n_bandits, n_sessions, r_out), 
										MersenneTwister())],
					100)
	=#

	#=
	dir = "./"
	file_v = filter(x -> occursin(".jld", x), readdir(dir))

	file_v = [string(dir, file) for file in file_v]

	
	plot_performance(file_v)

	agent_keys_v = ["bias_agent", "Q_agent", "no_offline_agent"]

	for file in file_v

		d = load(file)
		
		println("-------------------")
		println("γ = ", d["env_v"][1].reward_process.γ_v[1])

		for key in agent_keys_v

			agent = d[key]

			println(key)
			println("η_r = ", agent.η, "   ", "η_off = ", agent.offline.η, "   ", "ε = ", agent.policy.ε)
			println("-------------------")
		end
	end
	=#

	#=
	for file in file_v

		env_v = load(file, "env_v")[1:2]

		bias_agent = load(file, "bias_agent")
		Q_agent = load(file, "Q_agent")
		no_offline_agent = load(file, "no_offline_agent")

		for agent in [bias_agent, Q_agent, no_offline_agent]

			plot_reward_split(env_v, agent, 1 : 10 ; save_plot = true)
		end
	end
	=#
end	

main()


