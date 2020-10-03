using PyPlot
using LaTeXStrings

const r_colour_v = ["r", "b", "g"]
const b_colour_v = ["orange", "c", "k"]
const label_v = ["A", "B", "blank"]

function plot_timeseries(r_m::Array{Float64}, b_m::Array{Float64}, available_action_m::Array{Int64},
						sessions_to_plot::Union{AbstractRange{Int64}, Int64})

	(n_steps, n_bandits, n_sessions) = size(r_m[:,:, sessions_to_plot])

	n_bias_steps = size(b_m)[1]

	plotted_label_v = Array{Int64, 1}()

	figure()
	ax = gca()

	for i = 1 : n_sessions
		
		x_range_r = collect(((i - 1)*(n_steps + n_bias_steps) + 1) : 
							((i - 1)*(n_steps + n_bias_steps) + n_steps))

		x_range_b = collect(((i - 1)*(n_steps + n_bias_steps) + n_steps + 1) : 
							((i - 1)*(n_steps + n_bias_steps) + n_steps + n_bias_steps))

		for j in available_action_m[:, i]

			if j in plotted_label_v

				plot(x_range_b, b_m[:, j, i], color = b_colour_v[j])
				plot(x_range_r, r_m[:, j, i], color = r_colour_v[j])

			else

				plot(x_range_b, b_m[:, j, i], color = b_colour_v[j], label = latexstring("b_{$j}"))
				plot(x_range_r, r_m[:, j, i], color = r_colour_v[j], label = latexstring("r_{$j}"))

				append!(plotted_label_v, j)
			end
		end

	end

	ax.set_xlabel("session", fontsize = 20)
	ax.set_ylabel("value", fontsize = 20)

	ax.set_xticks(1:(n_steps + n_bias_steps) * Int64(ceil(0.2*n_sessions)):n_sessions*(n_steps + n_bias_steps))
	ax.set_xticklabels(string.(collect(1:Int64(ceil(0.2*n_sessions)):n_sessions)))

	ax.legend(fontsize = 20, frameon = false)
	show()
end

function plot_ABT(r_m::Array{Float64}, b_m::Array{Float64}, available_action_m::Array{Int64})

	(n_trials, n_actions, n_sessions) = size(r_m)

	n_bias_steps = size(b_m)[1]

	plotted_label_v = Array{Int64, 1}()

	figure()
	ax = gca()

	for i = 1 : n_sessions
		
		x_range_r = collect(((i - 1)*(n_trials + n_bias_steps) + 1) : 
							((i - 1)*(n_trials + n_bias_steps) + n_trials))

		x_range_b = collect(((i - 1)*(n_trials + n_bias_steps) + n_trials + 1) : 
							((i - 1)*(n_trials + n_bias_steps) + n_trials + n_bias_steps))

		for j in available_action_m[:, i]

			if j in plotted_label_v

				plot(x_range_b, b_m[:, j, i], color = b_colour_v[j])
				plot(x_range_r, r_m[:, j, i], color = r_colour_v[j])

			else

				plot(x_range_b, b_m[:, j, i], color = b_colour_v[j], label = latexstring("bias_{$(label_v[j])}"))
				plot(x_range_r, r_m[:, j, i], color = r_colour_v[j], label = latexstring("reward_{$(label_v[j])}"))

				append!(plotted_label_v, j)
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