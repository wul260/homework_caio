using CSV
using DataFrames
using Optim
using LsqFit
using LinearAlgebra

X = CSV.read("X.csv", header=false)
y = CSV.read("y.csv", header=false)

X = convert(Matrix, X)
y = y[:,1]

function logLik(β, X, y)
  # Sanity check
  if(size(X)[1] != size(y)[1])
    error("X and y must have compatible dimensions")
  end

  return -sum(-exp.(X * β) + y.*(X*β) - log.(factorial.(y)))
end

function ∇(β, X, y)
  # Sanity check
  if(size(X)[1] != size(y)[1])
    error("X and y must have compatible dimensions")
  end

  G = zeros(6)
  for i in 1:size(y)[1]
    G += exp(X[i, :]' * β)*X[i,:] - y[i]*X[i,:]
  end
  return G
end

# Question 1
res_1 = optimize(β -> logLik(β, X, y), fill(0.4, 6), NelderMead())
min_1 = Optim.minimizer(res_1)

# Question 2
res_2 = optimize(β -> logLik(β, X, y), β -> ∇(β, X, y), [1.0;0;0;0;0;0], LBFGS(), Optim.Options(show_trace = true); inplace = false)
min_2 = Optim.minimizer(res_2)

# Test fully Newton

function h(β, X)
  H = zeros(6, 6)
  for i in 1:size(X)[1]
    H += exp(X[i, :]' * β)*X[i,:]*X[i,:]'
  end
  return H
end

res_new = optimize(β -> logLik(β, X, y), β -> ∇(β, X, y), β -> h(β, X), min_2, Newton(), Optim.Options(show_trace = true); inplace = false)
min_new = Optim.minimizer(res_new)

#same as BFGS

### RSS
function rss(β, X, y)
  # Sanity check
  if(size(X)[1] != size(y)[1])
    error("X and y must have compatible dimensions")
  end

  rss = 0
  for i in 1:size(y)[1]
    rss += (y[i] - exp(β'*X[i, :]))^2
  end
  return rss
end

# Question 3
# This question is MATLAB-dependant seems it use a particular MATLAB function,
# searching in MATLAB help page I found out that MATLAB claims to use the 
# "trust-region-reflective", looking further I discover that this is a 
# adaptation of the "Levenberg-Marquardt" algorithm. A equally named function
# (in the better documented R) also uses the "Levenberg-Marquardt".
# This method is translated to julia by the following package/method:
@. model(x, β) = exp(x[:,1]*β[1] + x[:, 2]*β[2] + x[:, 3]*β[3] + x[:, 4]*β[4] + x[:, 5]*β[5] + x[:, 6]*β[6])
res_3 = curve_fit(model, X, y, zeros(6))
coef(res_3)

# Question 4
res_4 = optimize(β -> rss(β, X, y), fill(0.4, 6), NelderMead())
min_4 = Optim.minimizer(res_4)

# Question 5

init = []
for i in -1.5:0.2:1.5
  push!(init, i * min_new)
end

init2 = []
for i in 1:100
  push!(init2, 6*(rand(6) .- 0.5))
end

function parse_results(res, new, sec = "")
  ini  = Optim.initial_state(res)[1]/new[1]
  alg  = string(summary(res), sec)
  bool = Optim.converged(res)
  dist = norm(new - Optim.minimizer(res))
  return (ini, alg, bool, dist)
end

final_res = DataFrame(init = Float64[], alg = String[], converged = Bool[], distance = Float64[])
for i in init2
  try 
    res_1 = optimize(β -> logLik(β, X, y), i, NelderMead())
    push!(final_res, parse_results(res_1, min_new))
    res_2 = optimize(β -> logLik(β, X, y), β -> ∇(β, X, y), i, LBFGS(); inplace = false)
    push!(final_res, parse_results(res_2, min_new))
    res_3 = curve_fit(model, X, y, i)
    push!(final_res, (i[1]/min_new[1], "LSQ", true, norm(min_new - coef(res_3))))
    res_4 = optimize(β -> rss(β, X, y), i, NelderMead())
    push!(final_res, parse_results(res_4, min_new, "2"))
  catch
    println(string("Initials", i, " are singular"))
  end
end
final_res.useful_res = final_res.converged .& (final_res.distance .< 0.2)
by(final_res, :alg, :useful_res => mean)
useful_res = filter(r -> r[:useful_res], final_res)
by(useful_res, :alg, :distance   => mean)

# L-BFGS is more sensible to initial conditions but when it converges it 
# get closer to the Newton Method answer.
# Rather surpringly to me Loglik Nelder-Mead is the only one to always converge.
# When I use initial conditions that are proportional to the "true" minimun
# all algoritms perform better.
