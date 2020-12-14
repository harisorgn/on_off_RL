using PyPlot
using LaTeXStrings

const r_colour_v = ["r", "b", "g"]
const b_colour_v = ["orange", "c", "k"]

function plot_bandit_results(env::abstract_bandit_environment, agent::abstract_bandit_agent,
							sessions_to_plot::Union{AbstractRange{Int64}, Int64})

	(n_steps, n_bandits, n_sessions) = size(agent.r_m[:,:, sessions_to_plot])

	n_bias_steps = agent.offline.n_steps + 1

	plotted_label_v = Array{Int64, 1}()

	fig, ax = subplots(3, 1, sharex = true)

	for session in sessions_to_plot
		
		x_range_r = collect(((session - 1)*(n_steps + n_bias_steps) + 1) : 
							((session - 1)*(n_steps + n_bias_steps) + n_steps))

		x_range_b = collect(((session - 1)*(n_steps + n_bias_steps) + n_steps + 1) : 
							((session - 1)*(n_steps + n_bias_steps) + n_steps + n_bias_steps))

		for bandit in 1:n_bandits

			if bandit in plotted_label_v

				ax[1].plot(x_range_b, agent.offline.r_m[:, bandit, session], color = b_colour_v[bandit])
				ax[1].plot(x_range_r, agent.r_m[:, bandit, session], color = r_colour_v[bandit])

			else

				ax[1].plot(x_range_b, agent.offline.r_m[:, bandit, session], color = b_colour_v[bandit], label = latexstring("b_{$bandit}"))
				ax[1].plot(x_range_r, agent.r_m[:, bandit, session], color = r_colour_v[bandit], label = latexstring("r_{$bandit}"))

				append!(plotted_label_v, bandit)
			end

			ax[2].plot(x_range_r, env.reward_process.r_m[:, bandit, session] .+ env.out_process.r_m[:, bandit, session], color = r_colour_v[bandit])

			idx_v = findall(x -> x == bandit, agent.action_m[:, session])

			ax[3].hlines(fill(bandit, length(idx_v)), vcat(x_range_r, x_range_b)[idx_v], vcat(x_range_r, x_range_b)[idx_v .+ 1],
						color = r_colour_v[bandit])
		end

	end

	ax[1].set_ylabel("value", fontsize = 20)

	ax[1].legend(fontsize = 20, frameon = false)

	ax[2].set_ylabel("reward", fontsize = 20)

	ax[3].set_yticks(1:n_bandits)
	ax[3].set_yticklabels(string.(1:n_bandits))

	ax[3].set_ylabel("choice", fontsize = 20)
	ax[3].set_xlabel("session", fontsize = 20)

	ax[3].set_xticks(((sessions_to_plot[1] - 1)*(n_steps + n_bias_steps) + 1) : (n_steps + n_bias_steps) * Int(ceil(0.2*n_sessions)) :
					((sessions_to_plot[end] - 1)*(n_steps + n_bias_steps) + 1))

	ax[3].set_xticklabels(string.(sessions_to_plot[1] : Int(ceil(0.2*n_sessions)) : sessions_to_plot[end]))

	show()
end

function plot_performance(file_v::Array{String, 1}; n_runs = 100)

	agent_keys_v = ["bias_agent", "Q_agent", "no_offline_agent"]

	figure()
	ax = gca()

	for file in file_v

		d = load(file)

		env_v = d["env_v"][1:2]

		agent_v = [d[agent_key] for agent_key in agent_keys_v]

		score_m = zeros(n_runs, length(agent_v))

		for i = 1:n_runs
			for env in env_v

				agent_v = initialise_new_instance.(agent_v, env.reward_process.n_steps, env.reward_process.n_bandits, env.reward_process.n_sessions)

				env = initialise_new_instance(env)

				score_m[i, :] += run_bandit_example(env, agent_v)

			end
		end

		ax.errorbar(1:length(agent_v), mean(score_m; dims = 1)[:], yerr = std(score_m; dims = 1)[:], fmt = "o", 
					label = latexstring("\\gamma = $(env_v[1].reward_process.γ_v[1])"))
	end

	ax.set_xticks(1:length(agent_keys_v))

	ax.set_xticklabels(agent_keys_v, fontsize = 20)

	ax.set_ylabel("score", fontsize = 20)

	ax.legend(fontsize = 20, frameon = false)

	show()
end

function plot_performance_split(file_v::Array{String, 1}; n_runs = 100)

	agent_keys_v = ["bias_agent", "Q_agent", "no_offline_agent"]

	fig, ax = subplots(4, 1, sharex = true)

	for file in file_v

		d = load(file)

		env_v = d["env_v"][1:4]

		agent_v = [d[agent_key] for agent_key in agent_keys_v]

		j = 1

		for env in env_v

			score_m = zeros(n_runs, length(agent_v))

			for i = 1 : n_runs

				agent_v = initialise_new_instance.(agent_v, env.reward_process.n_steps, env.reward_process.n_bandits, env.reward_process.n_sessions)

				env = initialise_new_instance(env)

				score_v = run_bandit_example(env, agent_v)

				score_m[i, :] = score_v

			end

			ax[j].errorbar(1:length(agent_v), mean(score_m; dims = 1)[:], yerr = std(score_m; dims = 1)[:], fmt = "o", 
					label = latexstring("\\gamma = $(env_v[1].reward_process.γ_v[1])"))

			ax[j].set_title(string(typeof(env)))

			ax[j].set_ylabel("score", fontsize = 16)

			j += 1
		end
	end

	ax[1].legend(fontsize = 12, frameon = false, bbox_to_anchor=(1, 1))

	ax[end].set_xticks(1:length(agent_keys_v))

	ax[end].set_xticklabels(agent_keys_v, fontsize = 16)

	show()
end

function plot_reward_split(env_v::Array{T, 1}, agent::delta_agent{offline_bias}, 
							sessions_to_plot::Union{AbstractRange{Int64}, Int64};
							save_plot = false) where T <: abstract_bandit_environment

	fig, ax = subplots(length(env_v), 1, sharex = true, figsize = [18,12])

	plotted_label_v = Array{Int64, 1}()

	(n_steps, n_bandits, n_sessions) = size(env_v[1].reward_process.r_m[:, :, sessions_to_plot])

	n_bias_steps = Int(floor(0.5*n_steps))

	j = 1

	for env in env_v

		agent = initialise_new_instance(agent, env.reward_process.n_steps, env.reward_process.n_bandits, env.reward_process.n_sessions)

		run_environment!(env, agent)

		for session in sessions_to_plot
			
			x_range_r = collect(((session - 1)*(n_steps + n_bias_steps) + 1) : 
								((session - 1)*(n_steps + n_bias_steps) + n_steps))

			for bandit in 1:n_bandits

				if bandit in plotted_label_v

					ax[j].plot(x_range_r, agent.r_m[:, bandit, session] .+ agent.offline.r_m[1, bandit, session], 
								color = r_colour_v[bandit])

					ax[j].plot(x_range_r, env.reward_process.r_m[:, bandit, session] + expected_out(env.out_process, n_steps, bandit), 
								color = b_colour_v[bandit])
				else

					ax[j].plot(x_range_r, agent.r_m[:, bandit, session] .+ agent.offline.r_m[1, bandit, session], 
								color = r_colour_v[bandit], label = latexstring("\\hat{r}_{$bandit}"))

					ax[j].plot(x_range_r, env.reward_process.r_m[:, bandit, session] + expected_out(env.out_process, n_steps, bandit), 
								color = b_colour_v[bandit], label = latexstring("r_{$bandit}"))

					append!(plotted_label_v, bandit)
				end


			end
		end

		ax[j].set_title(string(typeof(env.out_process)), fontsize = 18)

		j += 1
	end

	η_r = round(agent.η, digits = 2)
	η_off = round(agent.offline.η, digits = 2)
	ε = round(agent.policy.ε, digits = 2)

	fig.suptitle(string(typeof(agent), "\n", 
						latexstring("\\gamma = $(env_v[1].reward_process.γ_v[1]),", "\\quad", "\\eta_{r} = $η_r,", "\\quad",
									"\\eta_{off} = $η_off,", "\\quad", "\\epsilon = $ε")), 
				fontsize = 20)

	ax[1].legend(fontsize = 20, frameon = false, bbox_to_anchor=(1, 1))

	ax[end].set_xlabel("session", fontsize = 18)

	ax[end].set_xticks(((sessions_to_plot[1] - 1)*(n_steps + n_bias_steps) + 1) : (n_steps + n_bias_steps) * Int(ceil(0.2*n_sessions)) :
					((sessions_to_plot[end] - 1)*(n_steps + n_bias_steps) + 1))

	ax[end].set_xticklabels(string.(sessions_to_plot[1] : Int(ceil(0.2*n_sessions)) : sessions_to_plot[end]))

	if save_plot
		savefig(string("reward_split_", rand(MersenneTwister(), 1000:9999), ".png"))
	else
		show()
	end
end

function plot_reward_split(env_v::Array{T, 1}, agent::delta_agent{offline_Q}, 
							sessions_to_plot::Union{AbstractRange{Int64}, Int64};
							save_plot = false) where T <: abstract_bandit_environment

	fig, ax = subplots(length(env_v), 1, sharex = true, figsize = [18,12])

	plotted_label_v = Array{Int64, 1}()

	(n_steps, n_bandits, n_sessions) = size(env_v[1].reward_process.r_m[:, :, sessions_to_plot])

	n_bias_steps = Int(floor(0.5*n_steps))

	j = 1

	for env in env_v

		agent = initialise_new_instance(agent, env.reward_process.n_steps, env.reward_process.n_bandits, env.reward_process.n_sessions)

		run_environment!(env, agent)

		for session in sessions_to_plot
			
			x_range_r = collect(((session - 1)*(n_steps + n_bias_steps) + 1) : 
								((session - 1)*(n_steps + n_bias_steps) + n_steps))

			for bandit in 1:n_bandits

				if bandit in plotted_label_v

					ax[j].plot(x_range_r, agent.r_m[:, bandit, session], 
								color = r_colour_v[bandit])

					ax[j].plot(x_range_r, env.reward_process.r_m[:, bandit, session] + expected_out(env.out_process, n_steps, bandit), 
								color = b_colour_v[bandit])
				else

					ax[j].plot(x_range_r, agent.r_m[:, bandit, session], 
								color = r_colour_v[bandit], label = latexstring("\\hat{r}_{$bandit}"))

					ax[j].plot(x_range_r, env.reward_process.r_m[:, bandit, session] + expected_out(env.out_process, n_steps, bandit), 
								color = b_colour_v[bandit], label = latexstring("r_{$bandit}"))

					append!(plotted_label_v, bandit)
				end


			end
		end

		ax[j].set_title(string(typeof(env.out_process)), fontsize = 18)

		j += 1
	end

	η_r = round(agent.η, digits = 2)
	η_off = round(agent.offline.η, digits = 2)
	ε = round(agent.policy.ε, digits = 2)

	fig.suptitle(string(typeof(agent), "\n", 
						latexstring("\\gamma = $(env_v[1].reward_process.γ_v[1]),", "\\quad", "\\eta_{r} = $η_r,", "\\quad",
									"\\eta_{off} = $η_off,", "\\quad", "\\epsilon = $ε")), 
				fontsize = 20)

	ax[1].legend(fontsize = 20, frameon = false, bbox_to_anchor=(1, 1))

	ax[end].set_xlabel("session", fontsize = 18)

	ax[end].set_xticks(((sessions_to_plot[1] - 1)*(n_steps + n_bias_steps) + 1) : (n_steps + n_bias_steps) * Int(ceil(0.2*n_sessions)) :
					((sessions_to_plot[end] - 1)*(n_steps + n_bias_steps) + 1))

	ax[end].set_xticklabels(string.(sessions_to_plot[1] : Int(ceil(0.2*n_sessions)) : sessions_to_plot[end]))

	if save_plot
		savefig(string("reward_split_", rand(MersenneTwister(), 1000:9999), ".png"))
	else
		show()
	end
end

function plot_OU(γ_v, σ_v; t = 40.0, t_0 = 0.0, x_0 = 0.0)

	x_r = -20.0:0.01:20.0

	figure()
	ax = gca()

	for i = 1 : length(γ_v)

		plot(x_r, OU_distr.(x_r, t, γ_v[i], σ_v[i]), label = "γ = $(γ_v[i])")

	end

	ax.legend(fontsize = 20, frameon = false)
	show()
end





