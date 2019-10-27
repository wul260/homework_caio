using Distributions
using DataFrames
using Optim
using Weave
using JLD

include("aux.jl")

### Q2

# Defining Kernels
k1(x) = φ(x)
k2(x) = φ(x)*(3 - x^2)/2
k3(x) = 2*φ(x) - φ(x/sqrt(2))/sqrt(2)
k4(x) = sin(π*x)/π*x
k = Dict(:1 => k1, :2 => k2, :3 => k3, :4 => k4)

if isfile("Q2-cvs.jld")
  cv_value = load("Q2-cvs.jld")["cv_value"]
else
  cv_value = Dict(i => zeros(4) for i in [10 100 1000])
  for n in [10 100 1000], j in 1:4
    cv_temp = zeros(200)
    for i in 1:200
      x = rand(Γ, n)
      cv_temp[i] = cv(x, k[j])
    end
    cv_value[n][j] = mean(cv_temp)
  end
  save("Q2-cvs.jld", "cv_value", cv_value)
end

## Q2 - a Bootstrap Simulation and Estimation
k_est= DataFrame(n = Int16[], i = Int16[], h = Float64[], method=Symbol[], est = Float64[])
for n in [10 100 1000], i in 1:1000 
  x = rand(Γ, n)
  for j in 1:4, h in 0.1:0.05:1.5
    push!(k_est, (n, i, h, Symbol(j), kernel(1, x, h, k[j])))
  end
end

k_mse = by(k_est, [:n, :h, :method], :est => β -> MSE(β, pdf(Γ, 1)))

draw(PNG("Q2-a.png", 1200px, 600px), q2_plot(k_mse, cv_value))

# Q2-b
if isfile("Q2-b.jld")
  q2b = load("Q2-b.jld")["q2b"]
else
  r_est = DataFrame(n = Int16[], i = Int16[], h = Float64[], method=Symbol[], kest = Float64[])
  for n in [10 100 1000], i in 1:1000 
    x = rand(Γ, n)
    y = log.(x.^2 .+ 1)
    for j in 1:4, h in 0.1:0.05:1.5
      push!(r_est, (n, i, h, Symbol(j), kreg(1, x, y, h, k[j])))
    end
  end

  r_mse = by(r_est, [:n, :h, :method], :kest => β -> MSE(β, log(2)))

  ll_est = DataFrame(n = Int16[], i = Int16[], h = Float64[], loclin_0 = Float64[], loclin_1 = Float64[])
  for n in [10 100 1000], i in 1:200
    x = rand(Γ, n)
    y = log.(x.^2 .+ 1)
    for h in 0.1:0.05:1.5
      row =  (n, i, h)
      row = tuple(row..., loclin(1, x, y, h, k[1])...)
      push!(ll_est, row)
    end
  end

  ll_mse = by(ll_est, [:n, :h], :loclin_1 => β -> MSE(β, 1.0))
  ll_mse.method = 5
  q2b    = vcat(r_mse, ll_mse)
  save("Q2-b.jld", "q2b", res4_2b)
end
draw(PNG("Q2-b.png", 1200px, 600px), q2_plot(q2b, cv_value))
