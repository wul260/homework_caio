using CSV
using DataFrames
using Optim
import Base.!

!(x::Int64) = factorial(x)

X = CSV.read("X.csv", header=false)
y = CSV.read("y.csv", header=false)

X = convert(Matrix, X)
y = convert(Matrix, y)

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

# Question 1
optimize(β -> logLik(β, X, y), fill(0.4, 6), NelderMead())

