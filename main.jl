using Distributions
using Random
using Statistics

include("agents.jl")
include("environments.jl")
include("auxiliary.jl")
include("plot.jl")
include("opt.jl")

function main()
	
	
	env = initialise_OU_bandit_delay_outlier_environment()

	#___agent parameter vector = [n_bias_steps, bias_buffer_length, decay_bias, decay_reward]___
	agent_param_v = Any[Int(floor(0.5env.n_steps)), Int(floor(0.2env.n_steps)), 0.01, 0.01]

	optimal_agent = get_optimal_agent(env)

	random_agent = delta_agent(env.n_steps, env.n_bandits, env.n_sessions, 0.0, 0.0,
								offline_bias(10, env.n_bandits, env.n_sessions, 10, 0.0, 0.0), 
								ε_greedy_policy(1.0))

	run_environment!(env, random_agent)
	run_environment!(env, optimal_agent)
	

	#___objective function patameter vector = [sum performance of optimal agent, sum performance of random agent]___
	obj_param_v = [sum(optimal_agent.accumulated_r_v), sum(random_agent.accumulated_r_v)]

	d_agent_x_v = run_delta_agent_opt(env, agent_param_v, obj_param_v)
	prob_d_agent_x_v = run_prob_delta_agent_opt(env, agent_param_v, obj_param_v)

	d_agent = delta_agent(env.n_steps, env.n_bandits, env.n_sessions, d_agent_x_v[2], agent_param_v[4], 
						offline_bias(agent_param_v[1], env.n_bandits, env.n_sessions, agent_param_v[2], d_agent_x_v[1], agent_param_v[3]), 
						ε_greedy_policy(d_agent_x_v[3]))

	prob_d_agent = probabilistic_delta_agent(env.n_steps, env.n_bandits, env.n_sessions, prob_d_agent_x_v[2], agent_param_v[4], 
											prob_d_agent_x_v[3], prob_d_agent_x_v[4],
											offline_bias(agent_param_v[1], env.n_bandits, env.n_sessions, agent_param_v[2], 
														prob_d_agent_x_v[1], agent_param_v[3]),
											ε_greedy_policy(prob_d_agent_x_v[5]))

	run_example(env, [d_agent, prob_d_agent] ; plot_flag = true)
	

	#=
	env = initialise_ABT_environment()

	agent = probabilistic_delta_agent(env.n_trials, env.n_bandits, env.n_sessions, 1.0, 0.01, 0.5, 1.0, 
									offline_bias(10, env.n_bandits, env.n_sessions, 5, 0.1, 0.01),
									softmax_policy(5.5))

	run_environment!(env, agent)

	plot_ABT(agent.r_m, agent.bias.b_m, env.available_action_m)
	=#
end

main()


