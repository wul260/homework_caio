using MAT
using Distributions
using QuadGK
using Optim
using LinearAlgebra
using PDMats
using Random
using DataFrames

### Data Load {{{
file = matopen("hw5.mat")
data = read(file)["data"]
close(file)

F(x) = cdf(Logistic(), x)
slog(x) = x < 1e-100 ? log(1e-100) : log(x) 
### }}}

# {{{
# I tried very hard to make a gauss quadrature myself, julia standard gaussian
# quadrature assumes we are integrating betwean 0 and 1 and that we are using
# a standard normal distribution for it. And I don't really know how to
# modify it. MATLAB function that I found let you pick the limits of
# the integration but..... I still don't know how to integrate to infinity.
# I didn't seem that using a very high number helps, so it look like I 
# just missunderstood the algorithm.
# I already spent way to much time on this, so I will go with the standard
# function.
### Calculating moments {{{2
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
### 2}}}

function gauss(μ, σ, x, w)
  n = length(x)
  m = 2*n-1
  mom = normal_moments.(μ, σ, 0:m);
  rhs = [w'*x.^i for i in 0:m]
  
  return mom - rhs;
end
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
g(x) = gn(x, 20)
gkres = QuadGK.gauss(20)
init = vcat(gkres[1] * 2, gkres[2])
global best = optimize(f, g, init; inplace = false)
for i in 1:100
  println(i)
  res = optimize(f, g, vcat(randn(20),rand(20)); inplace = false)
  if(res.minimum < 1)
    break
  elseif res.minimum < best.minimum
    best = res
  end
end
# }}}

function lik_ind(β, γ, u, Y, X, Z)
  value = 1
  for t in eachindex(Y)
    ε = β*X[t] + γ*Z[t] + u
    value *= F(ε)^Y[t] * (1 - F(ε))^(1 - Y[t])
  end

  return value
end

### Question 1 {{{
function loglike_gauss(γ, μ, σ_β, Y, X, Z)
  β, w = QuadGK.gauss(20)
  β = β .* 3 * σ_β .+ μ

  val = 0
  for i in axes(Y)[2]
    ll(β) = lik_ind(β, γ, 0, Y[:,i], X[:,i], Z[:,i])
    val += log(w' * ll.(β))
  end

  return val
end
loglike_gauss(0, 0.1, 1, 0, data["Y"], data["X"], data["Z"])
### }}}

### Question 2 {{{
function loglike_MC(γ, μ, σ_β, σ_u, σ_βu, Y, X, Z)
  if σ_u == 0
    n = 100
    Random.seed!(1234)
    β = rand(Normal(μ, σ_β), n)
    u = zeros(n)
  else
    n = 300
    Σ = [σ_β σ_βu; σ_βu σ_u]
    Σ = PDMat(Cholesky(Σ, :U, 0))
    Random.seed!(1234)
    R = rand(MvNormal(μ, Σ) , n)'
    β = R[:,1]
    u = R[:,2]
  end

  val = 0
  for i in axes(Y)[2]
    ll(β, u) = lik_ind(β, γ, u, Y[:,i], X[:,i], Z[:,i])
    tmp = 0
    for j in 1:n
      tmp   += ll(β[j], u[j])/n
    end
    val += log(tmp)
  end
  return val
end
loglike_MC(0, 0.1, 1, 0, 0, data["Y"], data["X"], data["Z"])

### }}}

### Using Julia Package for comparison {{{
function loglike_GK(γ, μ, σ_β, σ_u, σ_βu, Y, X, Z)
  val = 0
  for i in axes(Y)[2]
    ll(β, u) = lik_ind(β, γ, u, Y[:,i], X[:,i], Z[:,i]) * pdf(Normal(μ, σ_β), β)
    int, err = quadgk(β -> ll(β, 0), -3, 3)
    val += log(int)
  end
  return val
end
loglike_GK(0, 0.1, 1, 0, 0, data["Y"], data["X"], data["Z"])
### }}}

final_res = DataFrame(method = String[], γ_init = Float64[],
                      β_init = Float64[], u_init = Float64[], σ_β_init = Float64[],
                      σ_u_init = Float64[], σ_βu_init = Float64[], γ_argmax = Float64[],
                      β_argmax = Float64[], u_argmax = Float64[], σ_β_argmax = Float64[], 
                      σ_u_argmax = Float64[], σ_βu_argmax = Float64[],
                      loglike = Float64[])
### Question 3 {{{
ll(x) = -loglike_gauss(x[1], x[2], x[3], data["Y"], data["X"], data["Z"])
res_gauss = optimize(ll, [0.0, 0.1, 1.0])
rg = res_gauss.minimizer
push!(final_res, ("GaussQuad", 0, 0.1, 0, 1, 0, 0, rg[1], rg[2], 0, rg[3], 0, 0, -res_gauss.minimum))

ll2(x) = -loglike_MC(x[1], x[2], x[3], 0, 0, data["Y"], data["X"], data["Z"])
res_MC = optimize(ll2, [0.0, 0.1, 1.0], Optim.Options(show_trace = true))
rm = res_MC.minimizer 
push!(final_res, ("MonteCarlo", 0, 0.1, 0, 1, 0, 0, rm[1], rm[2], 0, rm[3], 0, 0, -res_MC.minimum))

llgk(x) = -loglike_GK(x[1], x[2], x[3], 0, 0, data["Y"], data["X"], data["Z"])
res_GK = optimize(llgk, [0.0, 0.1, 1.0], Optim.Options(show_trace = true))
rj = res_GK.minimizer
push!(final_res, ("Julia", 0, 0.1, 0, 1, 0, 0, rj[1], rj[2], 0, rj[3], 0, 0, -res_GK.minimum))
### }}}

### Question 4 {{{
ll3(x) = -loglike_MC(x[1], x[2:3], x[4], x[5], x[6], data["Y"], data["X"], data["Z"])
res_Q4 = optimize(ll3, [0.0, 0.1, 0.0, 1.0, 1.0, 0.1], Optim.Options(show_trace = true))
Q4 = res_Q4.minimizer
Σ = [Q4[4] Q4[5];Q4[5] Q4[6]]
Σ = PDMat(Cholesky(Σ, :U, 0)).mat
push!(final_res, ("Question4", 0, 0.1, 0, 1, 1, 0.1, Q4[1], Q4[2], Q4[3], Σ[1,1], Σ[2,2], Σ[1,2], -res_Q4.minimum))
### }}}

### Question 5 and Final Remarks
final_res
# Gaussian Quadrature seems to have a remarkably good results (higher log-likelihood) and
# a much faster convergence rate, even though I was not entirely sure of what I was
# doing. My answer are similar to Julia method for integration in Question 3.
# Question 4 don't seems to improve the log-likelihood by a lot even though we
# greatly increased the degrees of freedom.
