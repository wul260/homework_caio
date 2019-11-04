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
k4(x) = sin(π*x)/(π*x)
k = Dict(:1 => k1, :2 => k2, :3 => k3, :4 => k4)
# }}}
## Cross validation [long estimation] {{{2
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
  for j in 1:4, h in 0.1:0.1:2
    push!(k_est, (n, i, h, Symbol(j), kernel(1, x, h, k[j])))
  end
end

k_mse = by(k_est, [:n, :h, :method], :est => β -> MSE(β, pdf(Γ, 1)))
cv_value

draw(PNG("Q2-a.png", 16cm, 16cm), 
     q2_plot(k_mse, cv_value)
    )

# 2}}}
## Q2 - b {{{2
if isfile("Q2-b.jld")
  q2b = load("Q2-b.jld")["q2b"]
else
  r_est = DataFrame(n = Int16[], i = Int16[], h = Float64[], method=Symbol[], kest = Float64[])
  for n in [10 100 1000], i in 1:1000 
    x = rand(Γ, n)
    y = log.(x.^2 .+ 1) + randn(n)./4
    for j in 1:4, h in 0.01:0.1:2.01
      push!(r_est, (n, i, h, Symbol(j), kreg(1, x, y, h, k[j])))
    end
  end

  r_mse = by(r_est, [:n, :h, :method], :kest => β -> MSE(β, log(2)))

  ll_est = DataFrame(n = Int16[], i = Int16[], h = Float64[], loclin_0 = Float64[], loclin_1 = Float64[])
  for n in [10 100 1000], i in 1:200
    x = rand(Γ, n)
    y = log.(x.^2 .+ 1) + randn(n)./4
    for h in 0.01:0.01:2.01
      row =  (n, i, h)
      row = tuple(row..., loclin(1, x, y, h, k[1])...)
      push!(ll_est, row)
    end
  end

  ll_mse  = by(ll_est, [:n, :h], :loclin_0 => β -> MSE(β, log(2)))
  ll_mse.method = 5
  q2b = vcat(r_mse, ll_mse)
  save("Q2-b.jld", "q2b", q2b)
end
draw(PNG("Q2-b.png", 16cm, 16cm), 
     q2_plot(q2b, cv_value)
    )

### }}}
### Q3 {{{
## Q3 - a {{{2
x = rand(Γ, 1000)
p1 = plot(z -> kernel(z, x, 1e-6, φ), -5, 20,
          Guide.ylabel("density"), Guide.yticks(ticks=[-0.0001 0 0.0001]))
p2 = plot(z -> kernel(z, x, 1e+6, φ), -5, 20,
          Guide.ylabel("density"), Guide.yticks(ticks=[-0.0001 0 0.0001]))
draw(PNG("Q3-a.png", 16cm, 8cm), hstack(p1, p2))
### 2}}}
## Q3 - b {{{2
y = log.(x.^2 .+ 1)
p1 = plot(layer(z -> kreg(z, x, y, 1e-6, φ), -5, 20),
          layer(z -> log(z^2 + 1), -5 , 20, Theme(default_color="red")),
          Guide.ylabel("f(x)"), Guide.xlabel("x"),  Guide.yticks(ticks=-1:6))
p2 = plot(layer(z -> kreg(z, x, y, 1e+6, φ), -5, 20),
          layer(z -> log(z^2 + 1), -5 , 20, Theme(default_color="red")),
          Guide.ylabel("f(x)"), Guide.xlabel("x"),  Guide.yticks(ticks=-1:6))
draw(PNG("Q3-b.png", 16cm, 8cm), hstack(p1, p2))
### 2}}}
## Q3 - c {{{2
x = rand(1000)
σ = sqrt(var(x))
p1 = plot(z -> kernel(z, x, σ*1000^(-1/5), φ), 0, 1, Guide.ylabel("density"))
draw(PNG("Q3-c.png", 8cm, 8cm), p1)
## 2}}}
### }}}

### Q4 {{{
data = CSV.read("psam_h08.csv")
data = CSV.read("/home/caio/share/kansas_hincp.csv")
inc  = data.HINCP[.!ismissing.(data.HINCP)]

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

## Q4 - a {{{2
h  = sqrt(var(inc))*length(inc)^(-1/5)
pa = plot_kernel(h)
push!(pa, Guide.title("Silverman"))
## 2}}}
## Q4 - b {{{2
# h    = cv(inc, φ, 0, 20000)
h  = 9918.5355
pb = plot_kernel(h)
push!(pb, Guide.title("Cross Validation"))
draw(PNG("Q4-ab.png", 16cm, 8cm), hstack(pa, pb))
## 2}}}
## Q4 - c {{{2
plots = Array{Plot}(undef, 0)
for h in 1000:3000:25000
  push!(plots, plot_kernel(h, false))
end
plots = reshape(plots, 3, 3)
p = gridstack(plots)
draw(PNG("Q4-c.png", 24cm, 24cm), p)
## 2}}}


### }}}

### Q6 {{{
## Simulation {{{2
σ = 1.0*convert(Matrix, Diagonal(5:5:25))
X = rand(MvNormal(σ), 1000)
β = [1; 2; 1; 2; 1] ./ [5; 10; 15; 20; 25]
ε = randn(1000)
ε = ε .* (X'*ones(5)).^2 ./ 50
f = X'*β + randn(1000)
Y = f .> 0.4
## }}}
## Probit {{{2
probit(β) = -sum(Y.*log.(Φ.(X'*β)) + (1 .- Y).*log.((1 .- Φ.(X'*β))))
probit_res = optimize(probit, zeros(5))
probit_est = Optim.minimizer(probit_res)
## }}}
## 
## Average Derivative {{{2
h = var(X, dims = 2) .* 1000^(-1/5)
aderivative(X', Y, h)
## }}}

##  {{{2

## 2}}}

### }}} 
