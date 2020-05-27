using Distributions
using Random
using Statistics

include("models.jl")
include("plot.jl")

function run()

	rng = MersenneTwister();

	m_r1 = 1.0
	s_r1 = 0.05

	d_r1 = Normal(m_r1, s_r1)

	m_r2 = 1.0
	s_r2 = s_r1

	d_r2 = Normal(m_r2, s_r2)

	m_r0 = 0.0
	s_r0 = s_r1

	d_r0 = Normal(m_r0, s_r0)

	dr_offset = -5.0

	m_prob_RW = 0.5
	s_prob_RW = 0.5

	n_RW = 0.1			
	n_prob_RW = 1.0/gauss_pdf(m_prob_RW, m_prob_RW, s_prob_RW)

	println("Max probabilistc RW learning rate = ", 1.0/gauss_pdf(m_prob_RW, m_prob_RW, s_prob_RW))
	@assert(n_prob_RW <= 1.0/gauss_pdf(m_prob_RW, m_prob_RW, s_prob_RW), "n_prob_RW out of bounds")

	beta = 2.5		
	decay = 0.01

	off_capacity = 5
	n_off_samples = 5	
	n_b = 0.1			

	n_sessions = 2
	n_trials = 20

	n_test_sessions = 1
	n_test_trials = 10

	r1_RW_m = zeros(n_trials, n_sessions)
	r2_RW_m = zeros(n_trials, n_sessions)
	r01_RW_m = zeros(n_trials, n_sessions)
	r02_RW_m = zeros(n_trials, n_sessions)

	r1_prob_RW_m = zeros(n_trials, n_sessions)
	r2_prob_RW_m = zeros(n_trials, n_sessions)
	r01_prob_RW_m = zeros(n_trials, n_sessions)
	r02_prob_RW_m = zeros(n_trials, n_sessions)

	choice_RW_m = Matrix{Int64}(undef, n_trials, n_sessions)
	choice_prob_RW_m = Matrix{Int64}(undef, n_trials, n_sessions)

	off_surprise_RW_m = zeros(off_capacity, n_sessions)
	off_choice_RW_m = Matrix{Int64}(undef, off_capacity, n_sessions)
	off_dr_RW_m = Matrix{Float64}(undef, off_capacity, n_sessions)

	off_surprise_prob_RW_m = zeros(off_capacity, n_sessions)
	off_choice_prob_RW_m = Matrix{Int64}(undef, off_capacity, n_sessions)
	off_dr_prob_RW_m = Matrix{Float64}(undef, off_capacity, n_sessions)

	b1_RW_m = zeros(n_off_samples + 1, n_sessions)
	b2_RW_m = zeros(n_off_samples + 1, n_sessions)
	b01_RW_m = zeros(n_off_samples + 1, n_sessions)
	b02_RW_m = zeros(n_off_samples + 1, n_sessions)

	b1_prob_RW_m = zeros(n_off_samples + 1, n_sessions)
	b2_prob_RW_m = zeros(n_off_samples + 1, n_sessions)
	b01_prob_RW_m = zeros(n_off_samples + 1, n_sessions)
	b02_prob_RW_m = zeros(n_off_samples + 1, n_sessions)

	accum_r_RW_v = zeros(n_sessions)
	accum_r_prob_RW_v = zeros(n_sessions)

	for i = 1 : n_sessions 

		#-----------
		# r1 VS r0
		#-----------

		off_surprise_RW_v = zeros(off_capacity)
		off_choice_RW_v = zeros(Int, off_capacity)
		off_dr_RW_v = zeros(off_capacity)

		off_surprise_prob_RW_v = zeros(off_capacity)
		off_choice_prob_RW_v = zeros(Int, off_capacity)
		off_dr_prob_RW_v = zeros(off_capacity)

		if i > 1
			r1_RW_m[1, i] = r1_RW_m[end, i - 1]
			r2_RW_m[1, i] = r2_RW_m[end, i - 1]
			r01_RW_m[1, i] = r02_RW_m[end, i - 1]

			r1_prob_RW_m[1, i] = r1_prob_RW_m[end, i - 1]
			r2_prob_RW_m[1, i] = r2_prob_RW_m[end, i - 1]
			r01_prob_RW_m[1, i] = r02_prob_RW_m[end, i - 1]

			b1_RW_m[1, i] = b1_RW_m[end, i - 1]
			b2_RW_m[1, i] = b2_RW_m[end, i - 1]
			b01_RW_m[1, i] = b02_RW_m[end, i - 1]

			b1_prob_RW_m[1, i] = b1_prob_RW_m[end, i - 1]
			b2_prob_RW_m[1, i] = b2_prob_RW_m[end, i - 1]
			b01_prob_RW_m[1, i] = b02_prob_RW_m[end, i - 1]
		end

		b1_RW = b1_RW_m[1, i]
		b2_RW = b2_RW_m[1, i]
		b0_RW = b01_RW_m[1, i]

		b1_prob_RW = b1_prob_RW_m[1, i]
		b2_prob_RW = b2_prob_RW_m[1, i]
		b0_prob_RW = b01_prob_RW_m[1, i]

		for j = 2 : n_trials

			r1 = rand(rng, d_r1)
			r0 = rand(rng, d_r0)

			(r1_RW_m[j, i], r01_RW_m[j,i], 
			choice_RW_m[j - 1, i], r_trial_RW, dr_RW) = RW_update!(r1, r0, 
																r1_RW_m[j - 1, i], r01_RW_m[j - 1, i], 
																n_RW, beta, decay, rng, 
																b1_RW, b0_RW)

			accum_r_RW_v[i] += r_trial_RW

			(r1_prob_RW_m[j, i], r01_prob_RW_m[j,i], choice_prob_RW_m[j - 1, i], 
			r_trial_prob_RW, dr_prob_RW, surprise_prob_RW) = prob_RW_update!(r1, r0, 
																			r1_prob_RW_m[j - 1, i], 
																			r01_prob_RW_m[j - 1, i], 
																			n_prob_RW, m_prob_RW, s_prob_RW, 
																			beta, decay, rng, 
																			b1_prob_RW, b0_prob_RW)

			accum_r_prob_RW_v[i] += r_trial_prob_RW

			off_buffer_update!(off_surprise_RW_v, off_choice_RW_v, off_dr_RW_v,
							abs(dr_RW), choice_RW_m[j - 1, i], dr_RW)

			off_buffer_update!(off_surprise_prob_RW_v, off_choice_prob_RW_v, off_dr_prob_RW_v,
						surprise_prob_RW, choice_prob_RW_m[j - 1, i], dr_prob_RW)
   
		end

		off_value_update!(off_surprise_RW_v, off_choice_RW_v, sum(off_dr_RW_v), 
						b1_RW_m, b01_RW_m, n_b, n_off_samples, rng, i)

		off_value_update!(off_surprise_prob_RW_v, off_choice_prob_RW_v, sum(off_dr_prob_RW_v), 
						b1_prob_RW_m, b01_prob_RW_m, n_b, n_off_samples, rng, i)

		println("Preference of substrate 1 over 0 : ", 
				100.0 * count(x -> x == 1, choice_prob_RW_m[:, i]) / n_trials)

		#-----------
		# r2 VS r0
		#-----------

		off_surprise_RW_v = zeros(off_capacity)
		off_choice_RW_v = zeros(Int, off_capacity)
		off_dr_RW_v = zeros(off_capacity)

		off_surprise_prob_RW_v = zeros(off_capacity)
		off_choice_prob_RW_v = zeros(Int, off_capacity)
		off_dr_prob_RW_v = zeros(off_capacity)

		r02_RW_m[1, i] = r01_RW_m[end, i]
		r02_prob_RW_m[1, i] = r01_prob_RW_m[end, i]

		b02_RW_m[1, i] = b01_RW_m[end, i]
		b02_prob_RW_m[1, i] = b01_prob_RW_m[end, i]

		b0_RW = b02_RW_m[1, i]
		b0_prob_RW = b02_prob_RW_m[1, i]

		for j = 2 : n_trials

			r2 = rand(rng, d_r2)
			r0 = rand(rng, d_r0)

			(r2_RW_m[j, i], r02_RW_m[j,i], 
			choice_RW_m[j - 1, i], r_trial_RW, dr_RW) = RW_update!(r2, r0, 
																r2_RW_m[j - 1, i], r02_RW_m[j - 1, i], 
																n_RW, beta, decay, rng, 
																b2_RW, b0_RW)

			accum_r_RW_v[i] += r_trial_RW

			(r2_prob_RW_m[j, i], r02_prob_RW_m[j,i], choice_prob_RW_m[j - 1, i], 
			r_trial_prob_RW, dr_prob_RW, surprise_prob_RW) = prob_RW_update!(r2, r0, 
																			r2_prob_RW_m[j - 1, i], 
																			r02_prob_RW_m[j - 1, i], 
																			n_prob_RW, m_prob_RW, s_prob_RW, 
																			beta, decay, rng, 
																			b2_prob_RW, b0_prob_RW)

			accum_r_prob_RW_v[i] += r_trial_prob_RW

			off_buffer_update!(off_surprise_RW_v, off_choice_RW_v, off_dr_RW_v,
							abs(dr_RW), choice_RW_m[j - 1, i], dr_RW)

			off_buffer_update!(off_surprise_prob_RW_v, off_choice_prob_RW_v, off_dr_prob_RW_v,
						surprise_prob_RW, choice_prob_RW_m[j - 1, i], dr_prob_RW)
   
		end

		off_value_update!(off_surprise_RW_v, off_choice_RW_v, sum(off_dr_RW_v) + dr_offset, 
						b2_RW_m, b02_RW_m, n_b, decay, n_off_samples, rng, i)

		off_value_update!(off_surprise_prob_RW_v, off_choice_prob_RW_v, sum(off_dr_prob_RW_v) + dr_offset, 
						b2_prob_RW_m, b02_prob_RW_m, n_b, decay, n_off_samples, rng, i)

		println("Preference of substrate 2 over 0 : ", 
				100.0 * count(x -> x == 1, choice_prob_RW_m[:, i]) / n_trials)
	end

	println("P(r2 over r1) = ", 1.0 - softmax_p(beta, r1_prob_RW_m[end, end], r2_prob_RW_m[end, end], 
												b1_prob_RW_m[end,end], b2_prob_RW_m[end,end]))

	bias = 1.0 - 0.5 - softmax_p(beta, r1_prob_RW_m[end, end], r2_prob_RW_m[end, end], 
								b1_prob_RW_m[end,end], b2_prob_RW_m[end,end])

	plot_abt_reward(r1_prob_RW_m, r2_prob_RW_m, r01_prob_RW_m, r02_prob_RW_m, bias)

	plot_abt_bias(b1_prob_RW_m, b2_prob_RW_m, b01_prob_RW_m, b02_prob_RW_m, bias)

end

run()


