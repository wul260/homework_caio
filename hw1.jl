# LOOKS GOOD

# Ex 1
#
X = [1 1.5 3 4 5 7 8 9]
Y1 = -2 .+ 0.5*X
Y2 = -2 .+ 0.5*X.^2

# Ex 2
X = range(-10, stop=20, length=200)
sum(X)

# Ex 3
A = [2 4 6; 1 7 5; 3 12 4]
b = [-2 3 10]'

C = A'*b
D = inv((A'*A))*b
E = [repeat(vec(A),3)[i] * repeat(b, inner=(9,1))[i] for i=1:27]
E = sum(E)
F = A[1:2, 1:2]
x = A\b

# Ex 4
[A zeros(3,12);
 zeros(3,3)  A zeros(3,9);
 zeros(3,6)  A zeros(3,6);
 zeros(3,9)  A zeros(3,3);
 zeros(3,12) A]

# Ex 5
using Plots
A = 10 .+ 5* randn((5,3))
for i in 1:15
  if(A[i] < 10)
    A[i] = 0
  else
    A[i] = 1
  end
end

# Ex 6
# using Pkg
# Pkg.add("DataFrames")
# Pkg.add("CSV")
# Pkg.add("LinearAlgebra")
using CSV
using DataFrames
using LinearAlgebra
data = CSV.read("/home/caio/Homework/Homeworks/hw1/datahw1.csv", header=false)
data = dropmissing(data)
n = nrow(data)
X = [ones(n) data[:,3] data[:,4] data[:,6]]
Y = data[:, 5]
β = inv(X'X)*(X'Y)
ε = Y - X*β
σ_ε = ε'*ε/(n - 4)
σ_β = σ_ε*inv(X'X)
sd = diag(σ_β).^(1/2)
