using Distributions
using DataFrames
using Optim
using JLD

Γ = Gumbel()
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

# Defining Plot
function q2_plot(data, cv_value)
  p = []
  for n in [10 100 1000]
    x    = filter(r -> r.n == n && r.method != Symbol(4), data)
    cv   = cv_value[n][1:3]
    xmin =  minimum(x.mse)
    if(n != 1000)
      theme = Theme(background_color = "white", key_position=:none)
    else
      theme = Theme(background_color = "white")
    end
    p1 = plot(x, x=:h, y=:mse, color=:method, group=:n, Geom.line, theme,
              Guide.title(string("n = ", n)),
              Guide.annotation(compose(context(), text(cv, fill(xmin, 3), ["1"; "2"; "3"]))))
    push!(p, p1)
  end
  return vstack(hstack(p[1], p[2]), p[3])
end

### Kernel Estimators {{{
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
###}}}
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
