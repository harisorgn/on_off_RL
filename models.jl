
# Gaussian pdf
gauss_pdf(x, m, s) = exp(-((x - m)^2.0)/(2.0*s^2.0)) / (s*sqrt(2.0*pi))

# Kalman Filter prediction 
KF_predict(x, x_ss, revert_coeff, s, s_x) = (x + revert_coeff * (x_ss - x), 
											sqrt((s^2.0)*(1.0 - revert_coeff)^2.0 + s_x^2.0))

# softmax P(choose r1)
softmax_p(beta, r1, r2, b1 = 0.0, b2 = 0.0) = exp(beta * (r1 + b1)) / (exp(beta * (r1 + b1)) + exp(beta * (r2 + b2)))

function off_buffer_update!(off_surprise_v, off_action_v, off_dr_v, surprise, action, dr)

	min_surprise = minimum(off_surprise_v)

	if surprise > min_surprise

		min_idx = findfirst(x -> x == min_surprise, off_surprise_v)

		off_surprise_v[min_idx] = surprise
		off_action_v[min_idx] = action
		off_dr_v[min_idx] = dr
	end
end


function off_value_update!(off_surprise_v, off_action_v, off_dr, b1_m, b2_m, n, decay, n_off_samples, rng, session_idx)

	weight_v = off_surprise_v ./ sum(off_surprise_v)
	
	d = Categorical(weight_v)

	i = 2
	
	for s in rand(rng, d, n_off_samples)

		if off_action_v[s] == 1

			b1_m[i, session_idx] = b1_m[i - 1, session_idx] +  n * ((off_dr / n_off_samples) - b1_m[i - 1, session_idx])

			b2_m[i, session_idx] = b2_m[i - 1, session_idx] * (1.0 - decay)

		elseif off_action_v[s] == 2

			b2_m[i, session_idx] = b2_m[i - 1, session_idx] + n * ((off_dr / n_off_samples) - b2_m[i - 1, session_idx])

			b1_m[i, session_idx] = b1_m[i - 1, session_idx] * (1.0 - decay)

		end
		i += 1
	end
end

function RW_update!(r1, r2, r1_exp, r2_exp, n, beta, decay, rng, b1 = 0.0, b2 = 0.0)

	r = 0.0
	dr = 0.0
	action = 0

	r1_updated = 0.0
	r2_updated = 0.0

	p_r1 = softmax_p(beta, r1_exp, r2_exp, b1, b2)

	if p_r1 > rand(rng)

		r = r1

		r1_updated = r1_exp + n * (r1 - r1_exp)
		
		r2_updated = (1.0 - decay) * r2_exp

		action = 1

		dr = r1 - r1_exp - b1

	else
		
		r = r2

		r2_updated = r2_exp + n * (r2 - r2_exp)
		
		r1_updated = (1.0 - decay) * r1_exp

		action = 2

		dr = r2 - r2_exp - b2

	end

	return (r1_updated, r2_updated, action, r, dr)
end

function prob_RW_update!(r1, r2, r1_exp, r2_exp, n, m, s, beta, decay, rng, b1 = 0.0, b2 = 0.0)

	r = 0.0
	dr = 0.0
	surprise = 0.0
	action = 0

	r1_updated = 0.0
	r2_updated = 0.0

	p_r1 = softmax_p(beta, r1_exp, r2_exp, b1, b2)

	if p_r1 > rand(rng)

		r = r1

		r1_updated = r1_exp + n * gauss_pdf(r1, m, s) * (r1 - r1_exp)
		
		r2_updated = (1.0 - decay) * r2_exp

		action = 1

		dr = r1 - r1_exp - b1

		surprise = -log(gauss_pdf(r1, m, s))

	else
		
		r = r2

		r2_updated = r2_exp + n * gauss_pdf(r2, m, s) * (r2 - r2_exp)
		
		r1_updated = (1.0 - decay) * r1_exp

		action = 2

		dr = r2 - r2_exp - b2

		surprise = -log(gauss_pdf(r2, m, s))
	end

	return (r1_updated, r2_updated, action, r, dr, surprise)
end


function stoch_RW_update!(r1, r2, r1_exp, r2_exp, n, beta, decay, rng, b1 = 0.0, b2 = 0.0)

	r = 0.0
	dr = 0.0
	action = 0

	r1_updated = 0.0
	r2_updated = 0.0

	p_r1 = softmax_p(beta, r1_exp, r2_exp, b1, b2)

	if p_r1 > rand(rng)

		r = r1

		r1_updated = r1_exp + n * rand(rng) * (r1 - r1_exp)
		
		r2_updated = (1.0 - decay) * r2_exp

		action = 1

		dr = r1 - r1_exp - b1

	else
		
		r = r2

		r2_updated = r2_exp + n * rand(rng) * (r2 - r2_exp)
		
		r1_updated = (1.0 - decay) * r1_exp

		action = 2

		dr = r2 - r2_exp - b2

	end

	return (r1_updated, r2_updated, action, r, dr)
end

function KF_update!(r1, r2, r1_exp, r2_exp, s_r1, s_r2, 
					m, s, revert_coeff, s_observe, beta, decay, rng, b1 = 0.0, b2 = 0.0)

	r = 0.0
	dr = 0.0
	action = 0

	r1_updated = 0.0
	r2_updated = 0.0

	(r1_exp, s_r1) = KF_predict(r1_exp, m, revert_coeff, s_r1, s)

	(r2_exp, s_r2) = KF_predict(r2_exp, m, revert_coeff, s_r2, s)

	p_r1 = softmax_p(beta, r1_exp, r2_exp, b1, b2)

	if p_r1 > rand(rng)

		r = r1

		r1_updated = r1_exp + (s^2.0 / (s^2.0 + s_observe^2.0)) * (r1 - r1_exp)
		
		s_r1_updated = sqrt((1.0 - (s^2.0 / (s^2.0 + s_observe^2.0)))*(s^2.0))

		r2_updated = (1.0 - decay) * r2_exp

		s_r2_updated = s_r2

		action = 1

		dr = r1 - r1_exp - b1

	else
		
		r = r2

		r2_updated = r2_exp + (s^2.0 / (s^2.0 + s_observe^2.0)) * (r2 - r2_exp)
		
		s_r2_updated = sqrt((1.0 - (s^2.0 / (s^2.0 + s_observe^2.0)))*(s^2.0))

		r1_updated = (1.0 - decay) * r1_exp

		s_r1_updated = s_r1

		action = 2

		dr = r2 - r2_exp - b2

	end

	return (r1_updated, r2_updated, s_r1_updated, s_r2_updated, action, r, dr)
end
