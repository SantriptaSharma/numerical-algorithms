def trapezoid_integral(f, a: float, b: float, h = 0.5) -> tuple[float, float]:
	""" integrate f: R -> R in the interval a to b using the trapezoid method, x_i = a + ih, returning the answer and the number of iterations """

	x = a
	left = f(a)
	area = 0
	iterations = 0

	while x < b:
		next_x = x + h
		right = f(next_x)

		height = (left + right) / 2
		area += height * h

		x = next_x
		left = f(x)
		iterations += 1

	return area, iterations

def solve_euler(f_prime, xmin: float, init_val: float, xmax: float, h = 0.5) -> dict[int, float]:
	""" solve a (ordinary, first order) diffeq using euler's method with a given initial value, init_val = f_prime(xmin), step size = h """
	assert xmax >= xmin
	steps = int((xmax - xmin) / h)

	xs = [xmin + i * h for i in range(steps + 1)]
	res = {xmin: init_val}

	# i is shifted by 1 due to slicing
	for i, x in enumerate(xs[1:]):
		# which leads to this weird index here
		prev_x = xs[i]
		res[x] = res[prev_x] + h * f_prime(prev_x)
	
	return res

def newton_rhapson(f, f_prime, init_val: float, steps: int = 30):
	""" estimate the root of an at least once differentiable function f, f_prime = f', using the newton-rhapson fixed point iteration """
	xs = [init_val]

	for i in range(1, steps):
		x_p = xs[i - 1]

		x_next = x_p - f(x_p) / f_prime(x_p)
		xs.append(x_next)


	return xs

def solve_quadratic_normal(a: float, b: float, c: float) -> tuple[float, float] | None:
	""" solve a quadratic in standard form ax^2 + bx + c = 0 using the normal quadratic formula """

	disc = b**2 - 4*a*c

	if disc < 0:
		return None
	
	denom = 2*a
	A = -b/denom
	B = disc/denom

	return (A + B, A - B)

def solve_quadratic_stable(a: float, b: float, c: float) -> tuple[float, float] | None:
	""" solve a quadratic in standard form ax^2 + bx + c = 0 using the more stable quadratic formula """

	disc = b**2 - 4*a*c

	if disc < 0:
		return None
	
	numerator = 2 * c
	d1 = -b + disc
	d2 = -b - disc

	return (numerator / d1, numerator / d2)

