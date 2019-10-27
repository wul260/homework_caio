using Distributions
using DataFrames
using Optim
using JLD

include("aux.jl")

### Q1 {{{

x = rand(Γ, 100000)
y = exp.(x)
p1 = plot(h -> (kernel(1, x, h, φ) - pdf(Γ, 1))^2, 0.001, 1,
          Guide.xlabel("h"), Guide.ylabel("Square Difference"), Guide.title("Kernel"))
p2 = plot(h -> (kreg(1, x, y, h, φ) - exp(1))^2, 0.001, 1,
          Guide.xlabel("h"), Guide.ylabel("Square Difference"), Guide.title("Regression"))
p = hstack(p1, p2)
draw(PNG("Q1.png",16cm, 8cm), p)

### }}}

### Q2 {{{
# Defining Kernels {{{2
k1(x) = φ(x)
k2(x) = φ(x)*(3 - x^2)/2
k3(x) = 2*φ(x) - φ(x/sqrt(2))/sqrt(2)
k4(x) = sin(π*x)/π*x
k = Dict(:1 => k1, :2 => k2, :3 => k3, :4 => k4)
# }}}
## Cross validation {{{2
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
## 2}}}}
## Q2 - a {{{2
k_est = DataFrame(n = Int16[], i = Int16[], h = Float64[], method=Symbol[], est = Float64[])
for n in [10 100 1000], i in 1:1000 
  x = rand(Γ, n)
  for j in 1:4, h in 0.1:0.05:1.5
    push!(k_est, (n, i, h, Symbol(j), kernel(1, x, h, k[j])))
  end
end

k_mse = by(k_est, [:n, :h, :method], :est => β -> MSE(β, pdf(Γ, 1)))

draw(PNG("Q2-a.png", 16cm, 16cm), q2_plot(k_mse, cv_value))

# 2}}}
## Q2 - b {{{2
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

  ll_mse  = by(ll_est, [:n, :h], :loclin_1 => β -> MSE(β, 1.0))
  ll_mse.method = 5
  q2b_mse = vcat(r_mse, ll_mse)
  save("Q2-b.jld", "q2b_mse", q2b_mse)
end
draw(PNG("Q2-b.png", 16cm, 16cm), q2_plot(q2b, cv_value))

### }}}
### }}}
