using LinearAlgebra
using Statistics
using Distributions
using DataFrames
using Dates
using Optim
using CSV
using Gadfly, Cairo, Fontconfig

theme = Theme(line_width = 2pt, minor_label_font_size = 16pt)
# Exercise 1:
MSE = function(β, b)
  β_bar = mean(β, dims=1)
  β_bias  = (β_bar - b').^2
  β_var   = var(β, dims=1) 
  β_mse   = β_bias + β_var

  return hcat(β_bias, β_var, β_mse)
end

function procedure(n, d_z)
  β = zeros(1000, 2)
  σ = zeros(2,2,1000)
  θ = ones(2)
  Σ = [1   1/2 1/2;
       1/2 1   0  ;
       1/2 0   1  ]
  Σ = sqrt(Σ)
  Π = zeros(d_z, 2)
  Π[1,1] = 1;  Π[2,2] = 1;  
  for k in 1:1000
    z = randn(n, d_z); u = randn(n); v = randn(n, 2);

    E = [u v] * Σ
    u = E[:,1]
    v = E[:,2:3]

    x = z * Π + v
    y = x*θ + u

    Pz  = z*inv(z'z)*z';
    β[k,:] = inv(x'*Pz*x)*x'*Pz*y;
  end

  return n .* MSE(β, θ)
end

res = DataFrame(n = Int16[], dz = Int8[], SBias = Float64[], SBias2 = Float64[],
                Var = Float64[], Var2 = Float64[], MSE = Float64[], MSE2 = Float64[])
res = load("res.jld", "res")


@time for n in [100 1000]
  for dz in 3:3:36
    println(n)
    println(dz)
    push!(res, [n dz procedure(n,dz)])
  end
end

resStack = stack(res, [:SBias, :SBias2])
p = plot(resStack[resStack.n .== 100, :], x = :dz, y = :value, color = :variable, 
         Geom.bar(position = :dodge), Guide.title("MSE"),
         Guide.xlabel(nothing), Guide.ylabel(nothing),
         Theme(major_label_font_size = 24pt, minor_label_font_size=16pt, background_color = "white",
              key_position = :none));
p
p = plot(resStack[resStack.n .== 1000, :], x = :dz, y = :value, color = :variable, Geom.bar(position = :dodge), Theme(minor_label_font_size=16pt));
draw(SVG("Ex1 - NEW - Bias100.svg", 30cm, 15cm), p)

resStack = stack(res, [:Var, :Var2])
p = plot(resStack[resStack.n .== 100, :], x = :dz, y = :value, color = :variable, Geom.bar(position = :dodge), Theme(minor_label_font_size=16pt));
p = plot(resStack[resStack.n .== 1000, :], x = :dz, y = :value, color = :variable, Geom.bar(position = :dodge), Theme(minor_label_font_size=16pt));
draw(PNG("Ex1 - Var100.png", 30cm, 15cm), p)

resStack = stack(res, [:MSE, :MSE2])
p = plot(resStack[resStack.n .== 100, :], x = :dz, y = :value, color = :variable, Geom.bar(position = :dodge), Theme(minor_label_font_size=16pt));
p = plot(resStack[resStack.n .== 1000, :], x = :dz, y = :value, color = :variable, Geom.bar(position = :dodge), Theme(minor_label_font_size=16pt));
draw(PNG("Ex1 - MSE1000.png", 30cm, 15cm), p)
# Bias rise with the number of intruments, Variance descreses, Mean Square Error increses.

#Exercise 2
# a

res2 = DataFrame(x = Float64[], y = Float64[], lnInvΦ = Float64[], 
                 mlnΦ = Float64[], lnPoly = Float64[]) 

Φ = x -> cdf(Normal(),x)
for x in 1:0.5:5
  y = exp(x)
  push!(res2, [x y log(1 - Φ(y)) log(Φ(-y)) -log(y)-y^2/2])
end
res2

# Exercise 2
# b

res2b = DataFrame(τ = Int8[], mseβ = Float64[], mseβ2 = Float64[],
                  mseS = Float64[], mseS2 = Float64[])

θ = ones(2)
β   = zeros(1000, 2)
sol = zeros(1000, 2)
for τ in 10:20
  for i in 1:1000
    σ_x = exp(τ/2)
    x   = 1 .+ randn(100).*σ_x
    x   = hcat(ones(100), x)
    y   = x*θ
    β[i,:]   = inv(x'x)*x'y
    sol[i,:] = x\y
  end

  mseβ = MSE(β, θ)[5:6]
  mseS = MSE(sol, θ)[5:6]
  push!(res2b, vcat(τ, mseβ, mseS))
end

# Exercize 2
# c

res2c = DataFrame(i = Time(0), ii = Time(0), iii = Time(0), iv = Time(0))

θ = ones(2)
for r in 1:1000
  # I used n = 1.000 instead of 10.000 since my laptop was struggling with the bigger n
  x = randn(1000,2)
  z = randn(1000,3)
  y = x*θ

  time = now()
  i    = inv(x'*z*inv(z'z)*z'*x)*x'*z*inv(z'z)*z'*y
  res2c.i .+= now() - time
  
  time = now()
  xhat = z*inv(z'z)*z'*x 
  ii   = inv(xhat'xhat)*xhat'y 
  res2c.ii .+= now() - time

  time = now()
  xhat = z * (z\x)
  iii  = xhat\y 
  res2c.iii .+= now() - time

  time = now()
  Pz   = z*inv(z'z)*z'
  iv   = inv(x'*Pz*x)*x'*Pz*y
  res2c.iv .+= now() - time
end

res2c

# Exercise 3

φ = x -> pdf(Normal(), x)
f = x -> 2*φ(x)*(x >= 0)

θ = 1 
n = 1000
u = abs.(randn(n))
y = θ .+ log.(u)

mlefun(θ) = -sum(log.(f.(exp.(y .- θ))))
optimize(mlefun, 0, 10)
x = 0:0.5:15; z = -mlefun.(x)
p = plot(x = x, y = z, Geom.line, Guide.xticks(ticks=collect(x)),
         Guide.xlabel(nothing), Guide.ylabel(nothing),
         Theme(line_width = 2pt, minor_label_font_size = 16pt));
draw(PNG("mle.png", 30cm, 15cm), p)

# Criteria function has no maximum or minimun, maximization goes to upper bound
# We can modify it to solve the problem:

mlefun(θ) = θ - 1/n*sum(log.(f.(exp.(y .- θ))))
optimize(mlefun, 0, 10)

# Exercise 4

data = CSV.read("ass1q42.csv", header=false)
names!(data, [:x, :y])

p = plot(data, x=:x, y=:y, Guide.xticks(ticks=collect(0:25:400)), Guide.yticks(ticks=collect(13:24)),
         Guide.xlabel(nothing), Guide.ylabel(nothing),
         Theme(line_width = 2pt, minor_label_font_size = 16pt));
draw(PNG("Cloud.png", 30cm, 15cm), p)
p = plot(data, x=:y, Geom.histogram(bincount = 12), Guide.xticks(ticks=collect(13:25)),
         Guide.xlabel(nothing), Guide.ylabel(nothing),
         Theme(line_width = 2pt, minor_label_font_size = 16pt));
draw(PNG("Ex4-y.png", 30cm, 15cm), p)
p = Plot[]
for i in 13:24 
  push!(p, plot(data[data.y .== i, :], x=:x, Geom.histogram(bincount=406), 
                Guide.xlabel(string("y = ", i)),
                Guide.xticks(ticks=collect(0:25:406)),
                Guide.yticks(ticks=collect(1:3))));
end
p2 = plot(data, x = :x, Geom.histogram(bincount=400),
          Guide.xticks(ticks=collect(0:25:400)));
t = vstack(p2, gridstack(reshape(p, 4,3)), Theme(background_color = "white"));
img = PNG("Ex4-x.PNG", 34cm,40cm)
import Cairo, Fontconfig
draw(img, t)

function f(x,p)
  if x in 14:23
    p
  elseif x in [13,24]
    (1 - p*10)/2
  else
    0
  end
end

mle(p) = -sum(log.(f.(data.y, p)))
o = optimize(mle, 0, 0.1)
p = Optim.minimizer(o)
(1 - 10*p)/2


res4 = DataFrame(y = Int8[], uncon_prob = Float64[], given1 = Float64[], given0 = Float64[])

sum(data.y .== 16)
unique(rand(0:406, 70))
for y in 13:24
  crite = data[data.y .== y, :x]
  ndata = DataFrame(t = 1:406, x = 0)
  g0data = DataFrame(t = Int64[], x = Int64[])
  g1data = DataFrame(t = Int64[], x = Int64[])

  for i in 1:406
    if i in crite
      ndata[i,:x] = 1
    end
  end

  for i in 1:405
    if ndata[i,:x] == 1
      push!(g1data, ndata[i+1, :])
    else
      push!(g0data, ndata[i+1, :])
    end
  end

  up  = sum(ndata.x)/406
  pg1 = sum(g1data.x)/size(g1data)[1]
  pg0 = sum(g0data.x)/size(g0data)[1]

  push!(res4, [y up pg1 pg0])
end
res4

let
i = 0
for k in 1:1000000
  x = rand(1:406, 250)
  if(length(unique(x)) == 250)
    i += 1
  end
end
println(i)
end


# Exercise 6

θhat = zeros(10000)
μhat = zeros(10000)
n    = 10000
for r in 1:10000
  x       = rand(Normal(1, 100), n)
  μhat[r] = mean(x)
  σhat    = var(x)
  θhat[r] = n*μhat[r]^3 / (n*μhat[r]^2 + σhat) 
end

mseμ = 100*MSE(μhat, [1])
# Note tha all MSE is coming from the estimator variance and not from the bias.

mseθ = 100*MSE(θhat, [1])

# Exercise 7

data = CSV.read("ass1q7.csv", header = false)
data = data[:, :Column1]
data = nothing

function mle(param, data)
  λ,β,σ = param
  A = (1+σ/λ^2)*(1+σ/β^2);
  if log(A) < 0  
    return Inf
  else
    B = log(sqrt(log(A)));
    E = (λ/sqrt(1+σ/λ^2))*(β/sqrt(1+σ/β^2))
    if E < 0
      return Inf
    else
      C = (log.(data) .- log(E)).^2
      D = 2*log(A);
      return sum(B .+ C/D);
    end
  end
end


o = optimize(x->mle(x, data), [200 300 200.0]) 
res = Optim.minimizer(o)
mle([170.9453, 94.7704, 1.1637])

# c
sim = function(σ)
  λ₀ = 150
  β₀ = 100
  θ = zeros(1000, 3)
  for r in 1:1000
    μ_l = log(λ₀) - 1/2*σ 
    μ_b = log(β₀) - 1/2*σ
    s_l = log(σ/λ₀^2 + 1)
    s_b = log(σ/β₀^2 + 1)

    n = 1000
    a = rand(LogNormal(μ_l + μ_b, s_l + s_b), n)
    data = a

    o = optimize(x->mle(x, data), [125 125 1.0]) 
    θ[r,:] = Optim.minimizer(o) 
  end
  smse = sqrt(sum(MSE(θ, [λ₀ β₀ σ]')[7:9]))
  return smse
end

σ = 0.1:0.1:2
smse = sim.(σ)
smse = load("smse.jld", "smse")

using Plots;
plot(collect(σ), smse);
savefig("Ex7.png");

# Exercise 8

φ = x -> pdf(Normal(), x)
Φ = x -> cdf(Normal(),x)
f(x, y) = φ(x)*φ(y)*(1 - 0.5*(1 - 2*Φ(x))*(1 - 2*Φ(y)))
x = -2:0.02:2
y = -2:0.02:2
z = f.(x,y)
Plots.scalefontsizes(2)
plot(x,y,f, st=:surface, c = :blues, size = (1400, 1080));
savefig("3dplot.png")

zy0 = f.(x,0)
plot(x, zy0, size = (1400, 1080));
savefig("zy0.png")
zy1 = f.(x,1)
plot(x, zy1, size = (1400, 1080));
savefig("zy1.png")
zym1 = f.(x,-1)
plot(x, zym1, size = (1400, 1080));
savefig("zym1.png")
zy5 = f.(x,0.5)
plot(x, zy5, size = (1400, 1080));
savefig("zy5.png")
zym5 = f.(x,-1.5)
plot(x, zym5, size = (1400, 1080));
savefig("zym5.png")
