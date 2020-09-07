
abstract type abstract_environment end

struct OU_bandit_environment <:abstract_environment
	γ_v::Array{Float64, 1}
	μ_v::Array{Float64, 1}
	σ_v::Array{Float64, 1}
	outlier_distr_v::Array{Distribution, 1}
	r_m::Array{Float64}				# trials x actions x sessions
	r_outlier_m::Array{Float64}		# trials x actions x sessions

	function OU_bandit_environment(n_trials, n_sessions, γ_v, μ_v, σ_v, outlier_distr_v)

		rng = RandomDevice()

		n_actions = length(μ_v)

		r_m = zeros(n_trials, n_actions, n_sessions)
		r_outlier_m = zeros(n_trials, n_actions, n_sessions)

		for i = 1 : n_sessions
			for j = 2 : n_trials

				r_m[j, :, i] = r_m[j - 1, :, i] .+ γ_v .* (μ_v .- r_m[j - 1, :, i]) .+ σ_v .* rand(rng, Normal(0, 1), n_actions)

				r_outlier_m[j, :, i] = r_m[j, :, i] .+ [rand(rng, outlier_distr) for outlier_distr in outlier_distr_v]

			end
		end

		new(γ_v, μ_v, σ_v, outlier_distr_v, r_m, r_outlier_m)
	end
end

(OU_bandit_env::OU_bandit_environment)(action, trial, session) = OU_bandit_env.r_outlier_m[trial, action, session]

function run_environment!(env::OU_bandit_environment, agent::abstract_agent)

	(n_trials, n_actions, n_sessions) = size(env.r_m)

	for i = 1 : n_sessions

		p_first_action = agent.policy(agent.r_m[1, :, i] .+ agent.bias.b_m[1, :, i])
		
		agent.action_m[1, i] = rand(Categorical(p_first_action))

		r = env(agent.action_m[1, i], 1, i)

		for j = 2 : n_trials

			action = agent(r, j, i)

			r = env(action, j, i)
		end

		agent.bias(i)
	end	
end