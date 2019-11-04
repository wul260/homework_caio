### packages {{{
using Distributions
using DataFrames
using Optim
using JLD
using CSV
### }}}

### Variables {{{
Γ = Gumbel()
theme = Theme(background_color = "white");
### }}}

### General utility function {{{
φ(x)  = pdf(Normal(), x)
Φ(x)  = cdf(Normal(), x)
dφ(x) = -x/sqrt(2*π) * exp(-x^2/2)

MSE = function(β, b::Float64)
  β = convert(Array{Float64}, β) 
  β_bar = mean(β)
  β_bias  = (β_bar - b)^2
  β_var   = var(β) 
  β_mse   = β_bias + β_var

  return (bias = β_bias, var = β_var, mse = β_mse)
end
###}}}

### Plot functions {{{
## Q2 Plot {{{2
function q2_plot(data, cv_value)
  p = []
  for n in [10 100 1000]
    x    = filter(r -> r.n == n, data)
    cv   = cv_value[n][1:4]
    xmin =  minimum(x.mse)
    if(n != 1000)
      theme = Theme(background_color = "white", key_position=:none)
    else
      theme = Theme(background_color = "white")
    end
    p1 = plot(x, x=:h, y=:mse, color=:method, Geom.line, theme,
              Guide.title(string("n = ", n)),
              Coord.cartesian(ymax=0.06),
              Guide.annotation(compose(context(), text(cv, fill(xmin, 4), ["1"; "2"; "3"; "4"]))))
    push!(p, p1)
  end
  return vstack(hstack(p[1], p[2]), p[3])
end
## 2}}}
## Q4 Plot {{{
function plot_kernel(h, p = true)
  x    = 1000:1000:500000
  
  density = []
  for xi in x
    push!(density, kernel(xi, inc, h, φ))
  end

  if p
    return plot(x = x, y = density, Geom.line, 
                Guide.ylabel("density"), Guide.xlabel("Household income"))
  else
    return plot(x = x, y = density, Geom.line,
                Guide.ylabel(""), Guide.xlabel(""),
                Guide.title(string("h = ", h)),
                Guide.yticks(label = false), Guide.xticks(label = false),
                Theme(plot_padding = [1mm], background_color = "white"))
  end
end
## }}}
### }}}

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

function aderivative(X, Y, h)
  n = size(X)[1]
  d = size(X)[2]
  val = zeros(d)
  for i in 1:n
    for j in 1:n
      val += dφ.((X[i,:] - X[j,:]) ./ h).*Y[i]
    end
  end
  return -2/(n^2*h^d).*val
end

function loclin(x, X, Y, h, k)
  step(γ) = sum((Y .- γ[1] - γ[2].*(X .- x)).^2 .* k.((X .- x)./h))

  res = optimize(step, [1.0 1.0])
  res = Optim.minimizer(res)
  return (res[1], res[2])
end
###}}}

### Calculating CVs {{{
function cv_step(x, k, h)
  n = length(x)
  val = 0
  for i in 1:n
     y = x[1:end .!= i]
     z = 1/(h[1]*(n - 1))*sum(k.((y .- x[i])./h[1]))
    if z > 0
      val += log(z)
    else
      val = -Inf
      break
    end
  end
  return val
end
-
# infruitforeus attempt to speed cv calculation {{{3
function dcv(x, h)
  n = length(x)
  α = 1/(h[1]*(n-1))
  β = 0
  γ = 0
  for i in 1:n
    y = x[1:end .!= i]
    β += sum(φ.((y .- x[i])./h[1]))
    γ += sum(dφ.((y .- x[i])./h[1]) .* (y .- x[i]))
  end

  1/(h[1]^2*α*β) * (β/(n-1) + α*γ)
end
function dφ2(x)
  (-1 + x^2)/sqrt(2*π) * exp(-x^2/2)
end
#3}}}

function cv(x, k, lower = 0, upper = 2)
  res = optimize(h -> -cv_step(x, k, h), lower, upper)
  return Optim.minimizer(res)
k4(x) = sin(π*x)/π*x
end
### }}}
