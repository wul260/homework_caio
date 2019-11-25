using MAT
using Distributions
using QuadGK
using Optim
using NLsolve
using LinearAlgebra

### Data Load {{{
file = matopen("hw5.mat")
data = read(file)["data"]
close(file)

F(x) = cdf(Logistic(), x)
slog(x) = x < 1e-100 ? log(1e-100) : log(x) 
### }}}

### Calculating moments {{{
# It's hard to apreciate the beauty of this function when I lost
# so much time on something so superfluous
function moment_generator(n)
  if n == 0
    return [Dict(:μ_power => 0, :σ_power => 0, :mult => 1)]
  else
    expr = moment_generator(n - 1) # lagged expression
    tmp = []

    for term in expr
      if(term[:μ_power] > 0)
        tmp_term = copy(term)
        tmp_term[:mult] *= tmp_term[:μ_power]
        tmp_term[:μ_power] -= 1
        tmp_term[:σ_power] += 1
        push!(tmp, tmp_term)
      end
      term[:μ_power] += 1
    end

    # Cleaning redundant terms
    for tmp_term in tmp
      include = true
      for term in expr
        if(tmp_term[:μ_power] == term[:μ_power] &&
           tmp_term[:σ_power] == term[:σ_power])
          term[:mult] += tmp_term[:mult]
          include = false
          break
        end
      end
      if include
        push!(expr, tmp_term)
      end
    end

    return expr
  end
end

function normal_moments(μ, σ, n)
  expr = moment_generator(n)
  val = 0
  for term in expr
    val += term[:mult] * μ ^ term[:μ_power] * σ ^ term[:σ_power]
  end
  return val
end
### }}}

function gauss(μ, σ, x, w)
  n = length(x)
  m = 2*n-1
  mom = normal_moments.(μ, σ, 0:m);
  rhs = [w'*x.^i for i in 0:m]
  
  return mom - rhs;
end
gauss(0, 1, [-1.732050; 0; 1.732050], [1/6; 2/3; 1/6])

function fn(x,n)
  return sum(gauss(0, 1, x[1:n], x[(n+1):(2*n)]) .^2)
end
f(x) = fn(x, 20)

function gn(x,n)
  m = 2*n
  J2 = zeros(m,m)
  for i in 1:m
    if i == 1
      J2[i, 1:n] = zeros(n)
    else
      J2[i, 1:n] = (i - 1) .* x[(n+1):m] .* x[1:n] .^ (i - 2)
    end
    J2[i, (n+1):m] = x[1:n] .^ (i - 1) 
  end

  y = gauss(0, 1, x[1:n], x[(n+1):m])
  return -2 .* J2' * y
end

function hn(x, n)
  m = 2*n
  J2 = zeros(m,m)
  for i in 1:m
    if i == 1
      J2[i, 1:n] = zeros(n)
    else
      J2[i, 1:n] = (i - 1) .* x[(n+1):m] .* x[1:n] .^ (i - 2)
    end
    J2[i, (n+1):m] = x[1:n] .^ (i - 1) 
  end

  H = zeros(m,m,m)
  for i in 1:m
    if i <= n
      H[i,3:m,i] = [factorial(j-1) * x[i] ^ (j-3) * x[n+i] for j in 3:m]
      H[(n+i),2:m,i] = [(j-1) * x[i] ^ (j-2) for j in 2:m]
    else
      H[i-n,2:m,i] = [(j-1) * x[i-n] ^ (j-2) for j in 2:m]
    end
  end

  Final = zeros(m,m)
  for i in 1:m, j in 1:m
    # Final[i, j] = sum(J2[i
  end
end



g(x) = gn(x, 20)
wx = [-1.732050; 0; 1.732050; 1/6; 2/3; 1/6] 

w2(n) = 1/(n+1):1/(n+1):n/(n+1)
w = w2(9)
d = Normal(0, 1)
x = quantile.(d, w)
res = nlsolve(f, g, vcat(x, w))
res.zero

best = optimize(f, g, vcat(randn(20),rand(20)); inplace = false)
for i in 1:100
  println(i)
  res = optimize(f, g, vcat(randn(20),rand(20)); inplace = false)
  if(res.minimum < 1)
    break
  elseif res.minimum < best.minimum
    best = res
  end
end
res = nlsolve(x -> gauss(0.1, 1, x[1:5], x[6:10]), vcat(x, w))
gauss(0.1, 1, res.zero[1:5], res.zero[6:10])

function lik_ind(β, γ, u, Y, X, Z)
  value = 1
  for t in eachindex(Y)
    ε = β*X[t] + γ*Z[t] + u
    value *= F(ε)^Y[t] * (1 - F(ε))^(1 - Y[t])
  end

  return value
end

# Question 1
function loglike_Gauss(γ, μ, u, Y, X, Z)
  β, w = QuadGK.gauss(20)
  β .+= μ

  val = 0
  for i in axes(Y)[2]
    ll(β) = lik_ind(β, γ, u, Y[:,i], X[:,i], Z[:,i])
    val += log(w' * ll.(β))
  end

  return val
end

loglike_Gauss(0, 0.1, 0, data["Y"], data["X"], data["Z"])

#Question 2
function loglike_MC(γ, μ, u, Y, X, Z)
  β = rand(Normal(0.1, 1), 100)

  val = 0
  for i in axes(Y)[2]
    ll(β) = lik_ind(β, γ, u, Y[:,i], X[:,i], Z[:,i])
    val   += log(sum(ll.(β))/100)
  end

  return val
end
loglike_MC(0, 0.1, 0, data["Y"], data["X"], data["Z"])

#Question 3
ll(x) = loglike_Gauss(x[1], x[2], 0, data["Y"], data["X"], data["Z"])

d = Normal()
quantile(d, 1/20:1/20:19/20)
