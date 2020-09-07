
using Distributions
using Random
using Statistics

include("agents.jl")
include("environments.jl")
include("plot.jl")

function main()

	out1_d = MixtureModel(Normal, [(0.0, 0.01), (10.0, 0.1), (-10.0, 0.1)], 
								  [0.8, 		0.15, 		0.05])

	out2_d = MixtureModel(Normal, [(0.0, 0.01), (10.0, 0.1), (-10.0, 0.1)], 
								  [0.9, 		0.05, 		0.05])
	
	n_trials = 20
	n_sessions = 100
	n_actions = 2

	μ_OU_bandit_v = [0.0, 0.0]
	σ_OU_bandit_v = [0.5, 0.5]
	γ_OU_bandit_v = [0.01, 0.01]

	env = OU_bandit_environment(n_trials, n_sessions, γ_OU_bandit_v, μ_OU_bandit_v, σ_OU_bandit_v, [out1_d, out2_d])
	
	optimal_r_m = reshape(maximum(env.r_outlier_m, dims = 2), (n_trials, n_sessions))	# trials x sessions

	optimal_accumulated_r_v = sum(optimal_r_m, dims = 1)[:]

	random_r_m = reshape([env.r_outlier_m[j, rand(1:n_actions), i] for i = 1:n_sessions for j = 1:n_trials], (n_trials, n_sessions))

	random_accumulated_r_v = sum(random_r_m, dims = 1)[:]

	n_bias_timesteps = 20
	bias_buffer_length = 10

	n_b = 0.1
	decay_b = 0.01

	bias = offline_bias(n_bias_timesteps, n_actions, n_sessions, bias_buffer_length, n_b, decay_b)

	n_r_delta = 0.1
	decay_r = 0.01
	β = 2.5

	d_agent = delta_agent(n_trials, n_actions, n_sessions, n_r_delta, decay_r, bias, β)

	μ_prob_delta = 0.0
	σ_prob_delta = 0.8
	n_r_prob_delta = 10.0 * n_r_delta / pdf(Normal(μ_prob_delta, σ_prob_delta), μ_prob_delta)

	@assert(n_r_prob_delta <= 1.0/pdf(Normal(μ_prob_delta, σ_prob_delta), μ_prob_delta), "n_r_prob_delta out of bounds")

	prob_d_agent = probabilistic_delta_agent(n_trials, n_actions, n_sessions, n_r_prob_delta, decay_r, μ_prob_delta, σ_prob_delta, bias, β)

	run_environment!(env, d_agent)

	run_environment!(env, prob_d_agent)

	println("delta agent score : ", (median(d_agent.accumulated_r_v) - median(random_accumulated_r_v)) / 
								 	(median(optimal_accumulated_r_v) - median(random_accumulated_r_v)))

	println("probabilistic delta agent score : ", (median(prob_d_agent.accumulated_r_v) - median(random_accumulated_r_v)) / 
								 			   	  (median(optimal_accumulated_r_v) - median(random_accumulated_r_v)))

	plot_timeseries(d_agent.r_m, d_agent.bias.b_m, 1:n_sessions)

	plot_timeseries(prob_d_agent.r_m, prob_d_agent.bias.b_m, 1:n_sessions)

end

main()


