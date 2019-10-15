using CSV
using DataFrames
using Optim
import Base.!

!(x::Int64) = factorial(x)

X = CSV.read("X.csv", header=false)
y = CSV.read("y.csv", header=false)

X = convert(Matrix, X)
y = convert(Matrix, y)
plot(y = y, x = X[:, 4], Geom.point)

function logLik(β, X, y)
  # Sanity check
  if(size(X)[1] != size(y)[1])
    error("X and y must have compatible dimensions")
  end

  logLikvalue = 0
  for i in 1:size(y)[1]
    logLikvalue += -exp(X[i, :]' * β) + y[i]*X[i,:]'*β - log(!y[i])
  end
  return -logLikvalue
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

res_new = optimize(β -> logLik(β, X, y), β -> ∇(β, X, y), β -> h(β, X), [1.0;0;0;0;0;0], Newton(), Optim.Options(show_trace = true); inplace = false)
min_new = Optim.minimizer(res_new)

#same as BFGS
