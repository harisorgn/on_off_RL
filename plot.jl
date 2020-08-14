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
	ax.set_title(@sprintf("action bias = %.2f %%", 100.0*bias), fontsize = 20)
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

	figure()
	ax = gca()

	for i = 1 : n_sessions - 1
		
		x_range_r = collect(((i - 1)*(n_trials + n_off) + 1) : 
							((i - 1)*(n_trials + n_off) + n_trials))

		x_range_b = collect(((i - 1)*(n_trials + n_off) + n_trials + 1) : 
							((i - 1)*(n_trials + n_off) + n_trials + n_off))

		plot(x_range_b, b1_m[:, i], "-c")
		plot(x_range_b, b2_m[:, i], "-", color = "orange")

		plot(x_range_r, r1_m[:, i], "-b")
		plot(x_range_r, r2_m[:, i], "-r")

	end

	i = n_sessions

	x_range_r = collect(((i - 1)*(n_trials + n_off) + 1) : 
						((i - 1)*(n_trials + n_off) + n_trials))

	x_range_b = collect(((i - 1)*(n_trials + n_off) + n_trials + 1) : 
						((i - 1)*(n_trials + n_off) + n_trials + n_off))

	plot(x_range_b, b1_m[:, i], "-c", label = L"b_1")
	plot(x_range_b, b2_m[:, i], "-", color = "orange", label = L"b_2")

	plot(x_range_r, r1_m[:, i], "-b", label = L"r_1")
	plot(x_range_r, r2_m[:, i], "-r", label = L"r_2")

	ax.set_xlabel("trials", fontsize = 20)
	ax.set_ylabel("reward", fontsize = 20)
	ax.legend(fontsize = 20, frameon = false)
	show()
end

function plot_reward(r1_m, r2_m, r1_exp_m, r2_exp_m, n_off, sessions_to_plot = 0)

	if sessions_to_plot == 0

		(n_trials, n_sessions) = size(r1_m)

	else

		n_trials = size(r1_m[:, sessions_to_plot])[1]
		n_sessions = length(sessions_to_plot)

	end

	figure()
	ax = gca()

	for i = 1 : n_sessions - 1
		
		x_range = collect(((i - 1)*(n_trials + n_off) + 1) : 
						((i - 1)*(n_trials + n_off) + n_trials))

		plot(x_range, r1_m[:, i], "-b")
		plot(x_range, r2_m[:, i], "-r")

		plot(x_range, r1_exp_m[:, i], "-c")
		plot(x_range, r2_exp_m[:, i], "-", color = "orange")
	end

	i = n_sessions

	x_range = collect(((i - 1)*(n_trials + n_off) + 1) : 
					((i - 1)*(n_trials + n_off) + n_trials))

	plot(x_range, r1_m[:, i], "-b", label = L"r_1")
	plot(x_range, r2_m[:, i], "-r", label = L"r_2")

	plot(x_range, r1_exp_m[:, i], "-c", label = L"\hat{r}_1")
	plot(x_range, r2_exp_m[:, i], "-", color = "orange", label = L"\hat{r}_2")

	ax.set_xlabel("trials", fontsize = 20)
	ax.set_ylabel("reward", fontsize = 20)
	ax.legend(fontsize = 20, frameon = false)
	show()
end