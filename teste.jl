function P(θ, γ, ε)
  return 1/2 - ε + θ*γ
end

function μ(η, θ, γ, y)
  if y == 1
    return P(θ, γ, 0)*η / (P(θ,γ,0)*η + P(-θ, γ, 0)*(1 - η))
  elseif y == 0
    return (1 - P(θ, γ, 0))*η / (1 - (P(θ,γ,0)*η + P(-θ, γ, 0)*(1 - η)))
  end
end


let
  T = 10000
  μ0 = 1/2
  μl = zeros(25,1)' .+ μ0
  μf = μl
  ε = 1/32
  for i in 1:T
    for j in 1:25
      # γ = 1/4 - (μl[j] - 1/2)^2
      γ = μl[j] < 3/4 && μl[j] > 1/4 ? 1/4 : 1/64
      y = rand() < P(1, γ, ε) ? 1 : 0
      μl[j] = μ(μl[j], 1, γ, y)
    end
    μf = vcat(μf, μl)
  end
  x = convert(DataFrame, μf)
  x. t = 0:T
  df = stack(x,1:25)
  plot(df, x=:t, y = :value, color=:variable, Geom.line, 
       Coord.cartesian(ymax=1.5, ymin = -0.5))
end

function sim(a)
  T = 1000
  μl = 1/2
  δ = 0.99
  β = 0.1
  u(a, θ) = a*θ
  γf(μ) = max(0, min(1/4, a[1]*μ^2 + a[2]*μ + a[3]))
  val = 0
  for i in 0:T
      γ = γf(μl)
      y = rand() < P(1, γ, 0) ? 1 : 0
      μl = μ(μl, 1, γ, y)
      if μl >= 1/2
        val += δ^i*(u(1,1) - β*γ)
      else
        val += δ^i*(u(0,1) - β*γ)
      end
  end
  return -val
end
using Optim
optimize(sim, [1 1 1.0])
  

μ(1/2, 1, 1/8, 1)

