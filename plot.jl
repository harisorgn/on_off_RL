using PyPlot
using LaTeXStrings
using Printf

function plot_abt_bias(b1_m, b2_m, b01_m, b02_m, bias)

	(n_trials, n_sessions) = size(b1_m)

	figure()
	ax = gca()

	x_start = 1

	for i = 1 : n_sessions - 1

		plot(x_start : x_start + n_trials - 1, b1_m[:, i], "-b")
		plot(x_start : x_start + n_trials - 1, b01_m[:, i], "-k")

		plot(x_start + n_trials : x_start + 2 * n_trials - 1, b2_m[:, i], "-r")
		plot(x_start + n_trials : x_start + 2 * n_trials - 1, b02_m[:, i], "-k")

		x_start += 2 * n_trials
	end

	plot(x_start : x_start + n_trials - 1, b1_m[:, n_sessions], "-b", label = L"b_1")
	plot(x_start : x_start + n_trials - 1, b01_m[:, n_sessions], "-k", label = L"b_0")

	plot(x_start + n_trials : x_start + 2 * n_trials - 1, b2_m[:, n_sessions], "-r", label = L"b_2")
	plot(x_start + n_trials : x_start + 2 * n_trials - 1, b02_m[:, n_sessions], "-k")

	ax.set_ylabel("bias", fontsize = 20)
	ax.set_xlabel("trials", fontsize = 20)
	ax.set_title(@sprintf("choice bias = %.2f %%", 100.0*bias), fontsize = 20)
	ax.legend(fontsize = 20, frameon = false)

	show()
end

function plot_abt_reward(b1_m, b2_m, b01_m, b02_m, bias)

	(n_trials, n_sessions) = size(b1_m)

	figure()
	ax = gca()

	x_start = 1

	for i = 1 : n_sessions - 1

		plot(x_start : x_start + n_trials - 1, b1_m[:, i], "-b")
		plot(x_start : x_start + n_trials - 1, b01_m[:, i], "-k")

		plot(x_start + n_trials : x_start + 2 * n_trials - 1, b2_m[:, i], "-r")
		plot(x_start + n_trials : x_start + 2 * n_trials - 1, b02_m[:, i], "-k")

		x_start += 2 * n_trials
	end

	plot(x_start : x_start + n_trials - 1, b1_m[:, n_sessions], "-b", label = L"r_1")
	plot(x_start : x_start + n_trials - 1, b01_m[:, n_sessions], "-k", label = L"r_0")

	plot(x_start + n_trials : x_start + 2 * n_trials - 1, b2_m[:, n_sessions], "-r", label = L"r_2")
	plot(x_start + n_trials : x_start + 2 * n_trials - 1, b02_m[:, n_sessions], "-k")

	ax.set_ylabel("reward", fontsize = 20)
	ax.set_xlabel("trials", fontsize = 20)
	ax.set_title(@sprintf("choice bias = %.2f %%", 100.0*bias), fontsize = 20)
	ax.legend(fontsize = 20, frameon = false)

	show()
end

function plot_reward_bias(r1_m, r2_m, b1_m, b2_m, sessions_to_plot = 0)

	if sessions_to_plot == 0

		(n_trials, n_sessions) = size(r1_m)
		n_off = size(b1_m)[1]

	else

		n_trials = size(r1_m[:, sessions_to_plot])[1]
		n_off = size(b1_m[:, sessions_to_plot])[1]
		n_sessions = length(sessions_to_plot)

	end

	r1_v = Array{Float64, 1}(undef, n_trials * n_sessions)
	r2_v = Array{Float64, 1}(undef, n_trials * n_sessions)
	b1_v = Array{Float64, 1}(undef, n_off * n_sessions)
	b2_v = Array{Float64, 1}(undef, n_off * n_sessions)

	x_range_r = Array{Int64, 1}(undef, n_trials * n_sessions)
	x_range_b = Array{Int64, 1}(undef, n_off * n_sessions)

	for i = 1 : n_sessions
		
		x_range_r[(1:n_trials) .+ (i - 1)*n_trials] = collect(((i - 1)*(n_trials + n_off) + 1) : 
															((i - 1)*(n_trials + n_off) + n_trials))

		x_range_b[(1:n_off) .+ (i - 1)*n_off] = collect(((i - 1)*(n_trials + n_off) + n_trials + 1) : 
														((i - 1)*(n_trials + n_off) + n_trials + n_off))

		r1_v[(1:n_trials) .+ (i - 1)*n_trials] = r1_m[:, i]
		r2_v[(1:n_trials) .+ (i - 1)*n_trials] = r2_m[:, i]
		b1_v[(1:n_off) .+ (i - 1)*n_off] = b1_m[:, i]
		b2_v[(1:n_off) .+ (i - 1)*n_off] = b2_m[:, i]

	end

	figure()
	ax = gca()

	#plot(x_range_r, r1_v, alpha = 0.5, "-b", label = L"r_1")

	#plot(x_range_r, r2_v, alpha = 0.5, "-r", label = L"r_2")

	plot(x_range_b, b1_v, "-c", label = L"b_1")

	plot(x_range_b, b2_v, "-", color = "orange", label = L"b_2")

	ax.set_xlabel("trials", fontsize = 20)
	ax.set_ylabel("r", fontsize = 20)
	ax.legend(fontsize = 20, frameon = false)
	show()
end


function plot_reward_learning(n_RW, n_prob_RW, m_prob_RW, s_prob_RW, r1_v, r2_v, 
				r1_RW_v, r2_RW_v, r1_prob_RW_v, r2_prob_RW_v, 
				choice_RW_v, choice_prob_RW_v)

	choice_RW_r1_v = findall(x -> x == 1, choice_RW_v)
	choice_RW_r2_v = findall(x -> x == 2, choice_RW_v)

	choice_prob_RW_r1_v = findall(x -> x == 1, choice_prob_RW_v)
	choice_prob_RW_r2_v = findall(x -> x == 2, choice_prob_RW_v)

	#----------------------
	# probabilistic RW plot 
	#----------------------

	fig, ax_r = subplots()

	ax = ax_r.twinx()

	ax_r.set_ylabel("trials", fontsize = 20)

	# Plot reward time series
	ax_r.plot(r1_v, 1:length(r1_v), "-b", label = L"r_1")
	ax_r.plot(r2_v, 1:length(r2_v), "-r", label = L"r_2")

	# Plot estimated reward time series
	ax_r.plot(r1_prob_RW_v, 1:length(r1_prob_RW_v), "-c", label = L"\hat{r}_1")
	ax_r.plot(r2_prob_RW_v, 1:length(r2_prob_RW_v), "-", color = "orange", label = L"\hat{r}_2")

	# Plot choice columns
	ax_r.vlines(fill(ax_r.get_xlim()[1]*0.9, length(choice_prob_RW_r1_v)), 
				choice_prob_RW_r1_v, choice_prob_RW_r1_v .+ 1, 
				"b", lw = 10)

	ax_r.vlines(fill(ax_r.get_xlim()[2]*0.9, length(choice_prob_RW_r2_v)), 
				choice_prob_RW_r2_v, choice_prob_RW_r2_v .+ 1, 
				"r", lw = 10)

	# Plot learning rate
	ax.plot((ax.get_xlim()[1] : 0.01: ax.get_xlim()[2]), 
			n_prob_RW * gauss_pdf.((ax.get_xlim()[1] : 0.01: ax.get_xlim()[2]), m_prob_RW, s_prob_RW), "-k")

	# Plot options
	ax_r.legend(fontsize = 14, frameon = false)

	ax_r.set_xlabel("r", fontsize = 20)
	ax.set_ylabel("learning rate", fontsize = 20)
	ax_r.set_title("probabilistic RW", fontsize = 20)
	ax.set_ylim((0.0, n_prob_RW * gauss_pdf(m_prob_RW, m_prob_RW, s_prob_RW)))

	#-----------------
	# classic RW plot 
	#-----------------

	fig, ax_r = subplots()

	ax = ax_r.twinx()

	ax_r.set_ylabel("steps", fontsize = 20)

	# Plot reward time series
	ax_r.plot(r1_v, 1:length(r1_v), "-b", label = L"r_1")
	ax_r.plot(r2_v, 1:length(r2_v), "-r", label = L"r_2")

	# Plot estimated reward time series
	ax_r.plot(r1_RW_v, 1:length(r1_RW_v), "-c", label = L"\hat{r}_1")
	ax_r.plot(r2_RW_v, 1:length(r2_RW_v), "-", color = "orange", label = L"\hat{r}_2")

	# Plot choice columns
	ax_r.vlines(fill(ax_r.get_xlim()[1]*0.9, length(choice_RW_r1_v)), 
				choice_RW_r1_v, choice_RW_r1_v .+ 1, 
				"b", lw = 10)

	ax_r.vlines(fill(ax_r.get_xlim()[2]*0.9, length(choice_RW_r2_v)), 
				choice_RW_r2_v, choice_RW_r2_v .+ 1, 
				"r", lw = 10)

	# Plot learning rate
	ax.plot((ax.get_xlim()[1] : ax.get_xlim()[2]), 
			fill(n_RW, length((ax.get_xlim()[1] : ax.get_xlim()[2]))), "-k", lw = 4)

	# Plot options
	ax_r.legend(fontsize = 14, frameon = false)

	ax_r.set_xlabel("r", fontsize = 20)
	ax.set_ylabel("learning rate", fontsize = 20)
	ax_r.set_title("RW", fontsize = 20)
	ax.set_ylim((0.0, n_RW))

	show()

end