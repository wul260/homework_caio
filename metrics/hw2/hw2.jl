using Distributions
using DataFrames
using Optim
using JLD

include("aux.jl")

### Q1 {{{

x = rand(Γ, 1000)
y = exp.(x)
p = plot(h -> (kernel(1, x, h, φ) - pdf(Γ, 1))^2, 0.1, 1)
p = plot(h -> (kreg(1, x, y, h, φ) - exp(1))^2, 0.1, 1)


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
res = DataFrame(n = Int16[], i = Int16[], h = Float64[], method=Symbol[], est = Float64[])
for n in [10 100 1000], i in 1:1000 
  x = rand(Γ, n)
  for j in 1:4, h in 0.1:0.05:1.5
    push!(res, (n, i, h, Symbol(j), kernel(1, x, h, k[j])))
  end
end

res2 = by(res, [:n, :h, :method], :est => β -> MSE(β, pdf(Γ, 1)))

draw(PNG("Q2-a.png", 16cm, 16cm), q2_plot(res2, cv_value))

# 2}}}
## Q2 - b {{{2
if isfile("Q2-b.jld")
  q2b = load("Q2-b.jld")["q2b"]
else
  res2b = DataFrame(n = Int16[], i = Int16[], h = Float64[], method=Symbol[], kest = Float64[])
  for n in [10 100 1000], i in 1:1000 
    x = rand(Γ, n)
    y = log.(x.^2 .+ 1)
    for j in 1:4, h in 0.1:0.05:1.5
      push!(res2b, (n, i, h, Symbol(j), kreg(1, x, y, h, k[j])))
    end
  end

  res2_2b = by(res2b, [:n, :h, :method], :kest => β -> MSE(β, log(2)))

  res2b_loclin = DataFrame(n = Int16[], i = Int16[], h = Float64[], loclin_0 = Float64[], loclin_1 = Float64[])
  for n in [10 100 1000], i in 1:200
    x = rand(Γ, n)
    y = log.(x.^2 .+ 1)
    for h in 0.1:0.05:1.5
      row =  (n, i, h)
      row = tuple(row..., loclin(1, x, y, h, k[1])...)
      push!(res2b_loclin, row)
    end
  end

  res3_2b = by(res2b_loclin, [:n, :h], :loclin_1 => β -> MSE(β, 1.0))
  res3_2b.method = 5
  q2b     = vcat(res2_2b, res3_2b)
  save("Q2-b.jld", "q2b", res4_2b)
end
draw(PNG("Q2-b.png", 16cm, 16cm), q2_plot(q2b, cv_value))

### }}}
### }}}
