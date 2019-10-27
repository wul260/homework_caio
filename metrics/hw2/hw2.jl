using Distributions
using DataFrames
using Optim
using Weave
using JLD

d = Gumbel()
φ(x) = pdf.(Normal(), x)
theme = Theme(background_color = "white");


MSE = function(β, b::Float64)
  β = convert(Array{Float64}, β) 
  β_bar = mean(β)
  β_bias  = (β_bar - b)^2
  β_var   = var(β) 
  β_mse   = β_bias + β_var

  return (bias = β_bias, var = β_var, mse = β_mse)
end

### Q2

# Defining Kernels
k1(x) = φ(x)
k2(x) = φ(x)*(3 - x^2)/2
k3(x) = 2*φ(x) - φ(x/sqrt(2))/sqrt(2)
k4(x) = sin(π*x)/π*x
k = Dict(:1 => k1, :2 => k2, :3 => k3, :4 => k4)

# Defining Plot
function q2_plot(data, cv_value)
  p = []
  for n in [10 100 1000]
    x    = filter(r -> r.n == n && r.method != Symbol(4), data)
    cv   = cv_value[n][1:3]
    xmin =  minimum(x.mse)
    p1 = plot(x, x=:h, y=:mse, color=:method, group=:n, Geom.line, theme,
              Guide.annotation(compose(context(), text(cv, fill(xmin, 3), ["1"; "2"; "3"]))))
    push!(p, p1)
  end
  return hstack(p[1], p[2], p[3])
end

# Kernel Estimators
function kernel(x, X, h, k)
  n = length(X)
  1/(n*h) * sum(k.((X .- x) ./ h))
end

function a(x, X, Y, h, k)
  n = length(X)
  1/(n*h) * sum(k.((X .- x) ./ h).*Y)
end

function kreg(x, X, Y, h, k)
  n = length(X)
  a(x, X, Y, h, k)/kernel(x, X, h, k)
end

function loclin(x, X, Y, h, k)
  step(γ) = sum((Y .- γ[1] - γ[2].*(X .- x)).^2 .* k.((X .- x)./h))

  res = optimize(step, [1.0 1.0])
  res = Optim.minimizer(res)
  return (res[1], res[2])
end
### Calculating CVs [long estimation]
function cv_step(x, k, h)
  n = length(x)
  val = 0
  for i in 1:n
    y = x[1:end .!= i]
    val += log(max(0, 1/(h*(n - 1))*sum(k.((y .- x[i])./h))))
  end
  return val
end

function cv(x, k)
  res = optimize(h -> -cv_step(x, k, h), 0, 2)
  return Optim.minimizer(res)
end

if isfile("Q2-cvs.jld")
  cv_value = load("Q2-cvs.jld")["cv_value"]
else
  cv_value = Dict(i => zeros(4) for i in [10 100 1000])
  for n in [10 100 1000], j in 1:4
    cv_temp = zeros(200)
    for i in 1:200
      x = rand(d, n)
      cv_temp[i] = cv(x, k[j])
    end
    cv_value[n][j] = mean(cv_temp)
  end
  save("Q2-cvs.jld", "cv_value", cv_value)
end

## Q2 - a Bootstrap Simulation and Estimation
res = DataFrame(n = Int16[], i = Int16[], h = Float64[], method=Symbol[], est = Float64[])
for n in [10 100 1000], i in 1:1000 
  x = rand(d, n)
  for j in 1:4, h in 0.1:0.05:1.5
    push!(res, (n, i, h, Symbol(j), kernel(1, x, h, k[j])))
  end
end

res2 = by(res, [:n, :h, :method], :est => β -> MSE(β, pdf(d, 1)))

draw(PNG("Q2-a.png", 1200px, 600px), q2_plot(res2, cv_value))

# Q2-b
if isfile("Q2-b.jld")
  q2b = load("Q2-b.jld")["q2b"]
else
  res2b = DataFrame(n = Int16[], i = Int16[], h = Float64[], method=Symbol[], kest = Float64[])
  for n in [10 100 1000], i in 1:1000 
    x = rand(d, n)
    y = log.(x.^2 .+ 1)
    for j in 1:4, h in 0.1:0.05:1.5
      push!(res2b, (n, i, h, Symbol(j), kreg(1, x, y, h, k[j])))
    end
  end

  res2_2b = by(res2b, [:n, :h, :method], :kest => β -> MSE(β, log(2)))

  res2b_loclin = DataFrame(n = Int16[], i = Int16[], h = Float64[], loclin_0 = Float64[], loclin_1 = Float64[])
  for n in [10 100 1000], i in 1:200
    x = rand(d, n)
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
draw(PNG("Q2-b.png", 1200px, 600px), q2_plot(q2b, cv_value))

### EXPORT
# weave("hw2.jmd", out_path=:pwd)
