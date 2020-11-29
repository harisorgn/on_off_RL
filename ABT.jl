
struct ABT_environment <: abstract_environment
	n_trials::Int64
	n_bandits::Int64
	n_sessions::Int64
	r_v::Array{Float64, 1}					# actions x 1
	manipulation_δr_v::Array{Float64, 1}	# sessions x 1
	available_action_m::Array{Int64}		# available actions x sessions
end

(env::ABT_environment)(action) = env.r_v[action]

function initialise_ABT_environment()

	n_trials = 20
	n_bandits = 3							# total number of bandits, union of all available
	n_sessions = 4

	r_v = [1.0, 1.0, 0.0]
	manipulation_δr_v = [0.0, -5.0, 0.0, -5.0]

	available_action_m = repeat([[1,3] [2,3]], outer = [1, Int(n_sessions / 2)])

	return ABT_environment(n_trials, n_bandits, n_sessions, r_v, manipulation_δr_v, available_action_m)

end

function run_environment!(env::ABT_environment, agent::abstract_bandit_agent)

	for session = 1 : env.n_sessions

		agent.action_m[1, session] = env.available_action_m[agent.policy(agent.r_m[1, env.available_action_m[:, session], session] .+ 
																		agent.bias.b_m[1, env.available_action_m[:, session], session]), 
															session]

		r = env(agent.action_m[1, session])	

		for trial = 2 : env.n_trials

			action = agent(r, trial, session, env.available_action_m[:, session])

			r = env(action)
		end

		agent.bias.Δr_v += env.manipulation_δr_v[session]
		
		agent.bias(session)
	end	
end

function run_ABT_example(plot_flag = false)

	env = initialise_ABT_environment()

	agent = probabilistic_delta_agent(env.n_trials, env.n_bandits, env.n_sessions, 1.0, 0.01, 0.5, 1.0, 
									offline_bias(10, env.n_bandits, env.n_sessions, 5, 0.1, 0.01),
									softmax_policy(5.5))

	run_environment!(env, agent)

	if plot_flag
		plot_ABT(agent.r_m, agent.bias.b_m, env.available_action_m)
	end
end

function plot_ABT_results(env::ABT_environment, agent::abstract_bandit_agent, available_action_m::Array{Int64})

	(n_trials, n_actions, n_sessions) = size(agent.r_m)

	n_bias_steps = size(agent.bias.b_m)[1]

	plotted_label_v = Array{Int64, 1}()

	figure()
	ax = gca()

	for session = 1 : n_sessions
		
		x_range_r = collect(((session - 1)*(n_trials + n_bias_steps) + 1) : 
							((session - 1)*(n_trials + n_bias_steps) + n_trials))

		x_range_b = collect(((session - 1)*(n_trials + n_bias_steps) + n_trials + 1) : 
							((session - 1)*(n_trials + n_bias_steps) + n_trials + n_bias_steps))

		for bandit in available_action_m[:, session]

			if bandit in plotted_label_v

				plot(x_range_b, agent.bias.b_m[:, bandit, session], color = b_colour_v[bandit])
				plot(x_range_r, agent.r_m[:, bandit, session], color = r_colour_v[bandit])

			else

				plot(x_range_b, agent.bias.b_m[:, bandit, session], color = b_colour_v[bandit], label = latexstring("bias_{$(label_v[bandit])}"))
				plot(x_range_r, agent.r_m[:, bandit, session], color = r_colour_v[bandit], label = latexstring("reward_{$(label_v[bandit])}"))

				append!(plotted_label_v, bandit)
			end
		end

	end

	ax.set_xlabel("session", fontsize = 20)
	ax.set_ylabel("value", fontsize = 20)

	ax.set_xticks(1:(n_trials + n_bias_steps) * Int64(ceil(0.2*n_sessions)):n_sessions*(n_trials + n_bias_steps))
	ax.set_xticklabels(string.(collect(1:Int64(ceil(0.2*n_sessions)):n_sessions)))

	ax.legend(fontsize = 20, frameon = false)
	show()
end