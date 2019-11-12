using DataFrames
using Sobol

# looks good and very good analyses at the end! you get a check plus!

# Q1
function calculate_π(n::Int64)
  p = [rand(n) rand(n)]
  calculate_π(p)
end

function calculate_π(p)
  n = size(p)[1]
  count = 0 
  for i in 1:n
    if p[i,1]^2 + p[i,2]^2 <= 1
      count += 1
    end
  end

  return 4*count/n
end

# Pseudo Random
calculate_π(n)

# Equidistributed Numbers
s = SobolSeq(2)
p = hcat([next!(s) for _ in 1:10000]...)'
calculate_π(p)

# Q2
function integrate(f, a, b, N::Int64)
  if(a > b)
    stop("a must be lower than b")
  end
  if(N < 0)
    stop("N must be positive")
  end
    
  if N % 2 == 1
    N += 1
  end

  h = (b - a)/N
  X = a:h:b

  w = 2*ones(N+1)
  evens = 2:2:N
  w[evens] .= 4
  w[1] = 1
  w[N+1] = 1
  w=w*(h/3)

  fv = f.(X);

  return fv'*w;
end

# Q2 
Ind(x::Float64, y::Float64) = x^2 + y^2 <= 1

4*integrate(x -> integrate(y -> Ind(x, y), 0, 1, 10000), 0, 1, 10000)

# Q3
# The same as one?

# Q4

circle(x) = sqrt(1 - x^2)
4*integrate(circle, 0, 1, 10000)

# Q5
MSE = function(β, b)
  β_bar = mean(β)
  β_bias  = (β_bar - b)^2
  β_var   = var(β) 
  β_mse   = β_bias + β_var

  return hcat(β_bias, β_var, β_mse)
end

res = DataFrame(n = Int64[], pMC_MSE = Float64[], qMC_error = Float64[],
                NC_error = Float64[])

for n in [100 1000 10000]
  β  = [calculate_π(n) for i in 1:200]
  x  = MSE(β, π)

  s  = SobolSeq(2)
  p  = hcat([next!(s) for _ in 1:n]...)'
  β2 = calculate_π(p)
  e1 = (β2 - π)^2

  e2 = (π - 4*integrate(circle, 0, 1, n))^2

  push!(res, (n, x[3], e1, e2))
end
res

# NC seem to have the lowest error of them all
# not truly random number is better than random
# numbers
