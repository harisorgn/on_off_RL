using PyPlot
using LaTeXStrings

const r_colour_v = ["r", "b"]
const b_colour_v = ["g", "m"]

function plot_timeseries(r_m::Array{Float64}, b_m::Array{Float64}, sessions_to_plot::Union{AbstractRange{Int64}, Int64})

	(n_steps, n_actions, n_sessions) = size(r_m[:,:, sessions_to_plot])

	n_bias_steps = size(b_m)[1]

	figure()
	ax = gca()

	for i = 1 : n_sessions - 1
		
		x_range_r = collect(((i - 1)*(n_steps + n_bias_steps) + 1) : 
							((i - 1)*(n_steps + n_bias_steps) + n_steps))

		x_range_b = collect(((i - 1)*(n_steps + n_bias_steps) + n_steps + 1) : 
							((i - 1)*(n_steps + n_bias_steps) + n_steps + n_bias_steps))

		for j = 1 : n_actions

			plot(x_range_b, b_m[:, j, i], color = b_colour_v[j])

			plot(x_range_r, r_m[:, j, i], color = r_colour_v[j])

		end

	end

	i = n_sessions
 
	x_range_r = collect(((i - 1)*(n_steps + n_bias_steps) + 1) : 
						((i - 1)*(n_steps + n_bias_steps) + n_steps))

	x_range_b = collect(((i - 1)*(n_steps + n_bias_steps) + n_steps + 1) : 
						((i - 1)*(n_steps + n_bias_steps) + n_steps + n_bias_steps))

	for j = 1 : n_actions

		plot(x_range_b, b_m[:, j, i], color = b_colour_v[j], label = latexstring("b_$j"))

		plot(x_range_r, r_m[:, j, i], color = r_colour_v[j], label = latexstring("r_$j"))

	end

	ax.set_xlabel("session", fontsize = 20)
	ax.set_ylabel("value", fontsize = 20)

	ax.set_xticks(1:(n_steps + n_bias_steps) * Int64(ceil(0.2*n_sessions)):n_sessions*(n_steps + n_bias_steps))
	ax.set_xticklabels(string.(collect(1:Int64(ceil(0.2*n_sessions)):n_sessions)))

	ax.legend(fontsize = 20, frameon = false)
	show()
end

function plot_timeseries(r_m::Array{Float64}, n_bias_steps::Int64, sessions_to_plot::Union{AbstractRange{Int64}, Int64})

	(n_steps, n_actions, n_sessions) = size(r_m[:,:, sessions_to_plot])

	figure()
	ax = gca()

	for i = 1 : n_sessions - 1
		
		x_range_r = collect(((i - 1)*(n_steps + n_bias_steps) + 1) : 
							((i - 1)*(n_steps + n_bias_steps) + n_steps))

		for j = 1 : n_actions

			plot(x_range_r, r_m[:, j, i], color = r_colour_v[j])

		end

	end

	i = n_sessions

	x_range_r = collect(((i - 1)*(n_steps + n_bias_steps) + 1) : 
						((i - 1)*(n_steps + n_bias_steps) + n_steps))

	for j = 1 : n_actions

		plot(x_range_r, r_m[:, j, i], color = r_colour_v[j], label = latexstring("r_$j"))

	end

	ax.set_xlabel("session", fontsize = 20)
	ax.set_ylabel("value", fontsize = 20)

	ax.set_xticks(1:(n_steps + n_bias_steps) * Int64(ceil(0.2*n_sessions)):n_sessions*(n_steps + n_bias_steps))
	ax.set_xticklabels(string.(collect(1:Int64(ceil(0.2*n_sessions)):n_sessions)))

	ax.legend(fontsize = 20, frameon = false)
	show()
end