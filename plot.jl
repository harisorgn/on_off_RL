using PyPlot
using LaTeXStrings

const r_colour_v = ["r", "b", "g"]
const b_colour_v = ["orange", "c", "k"]
const label_v = ["A", "B", "blank"]

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

			ax[2].plot(x_range_r, env.r_m[:, bandit, session] .+ env.r_outlier_m[:, bandit, session], color = r_colour_v[bandit])

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

function plot_performance(env_v::Array{Y,1}, agent_v::Array{T, 1}; n_runs = 100) where {T <: abstract_bandit_agent, 
																						Y <: abstract_bandit_environment}

	score_m = zeros(n_runs, length(agent_v))

	for i = 1:n_runs
		for env in env_v

			agent_v = initialise_new_instance.(agent_v, env.n_steps, env.n_bandits, env.n_sessions)

			env = initialise_new_instance(env)

			score_m[i, :] .+= run_bandit_example(env, agent_v)

		end
	end

	figure()
	ax = gca()

	for i = 1:length(agent_v)
		scatter(fill(i, n_runs), score_m[:, i])
	end

	ax.set_xticks(1:length(agent_v))

	ax.set_xticklabels([L"\delta \ bias", L"\delta \ Q", L"\delta"], fontsize = 20)

	ax.set_ylabel("score", fontsize = 20)

	show()
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





