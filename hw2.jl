# If you wish to run, install Gadfly and DataFrames run everything and check
# the results at the and of file.

using LinearAlgebra
using Dates
# Pkg.add("DataFrames")
# Pkg.add("Gadfly")
using DataFrames
using Gadfly

# Exercise 1
function D(w, v, p)
  if w < 3
    A = exp(v[w] - p[w])
  else
    A = 1
  end

  return A / (1 + exp(v[1] - p[1]) + exp(v[2] - p[2]))
end

v = [2.0; 2.0]
p = [1.0; 1.0]

input  = [:a;:b;:o]
output = D.(1:3, Ref(v), Ref(p))
res    = DataFrame(input = input, output = output)

# Exercise 2
# Answer
dπ(w, v, p) = D(w, v, p)*(1 - p[w]*(1 - D(w, v, p)))
dπ(p) = [dπ(1, v, p), dπ(2, v, p)]

function Jacobian(dπ, v, p)
  h  = 1e-5
  # Calculating Jacobian wrt a
  Ja = (dπ(p + [h; 0]) - dπ(p))./h
  # Calculating Jacobian wrt a
  Jb = (dπ(p + [0; h]) - dπ(p))./h
  # Full Jacobian
  return hcat(Ja, Jb)
end

function broyden(f, init, J)
  xo , x  = init
  fvo, fv = f.(init)
  invJ = inv(J)

  for i in 1:1000
    println(string("Interation: ", i, "; x = ", x, "; f(x) = ", fv))
    if norm(fv) < 1e-8
      return x
    else
      xo , x  = (x, x - invJ*fv)
      fvo, fv = (fv, f(x))

      Δx = x - xo
      Δf = fv - fvo

      invJ = invJ + (Δx - invJ*Δf) * Δf' / (Δx'invJ*Δf)
    end
  end
  return x
end

J = Jacobian(dπ, v, p)
t = now()
res2 = broyden(dπ, [[1;1], [2;2]] , J)
res2 = DataFrame(pa = res2[1], pb = res2[2], time = now() - t)

# Exercise 3
function secant(f, init)
  xo, x   = init
  fvo, fv = f.(init)

  for i in 1:1000
    println(string("Interation: ", i, "; x = ", x, "; f(x) = ", fv))
    if norm(fv) < 1e-8
      return x
    else
      xo , x  = (x, x - fv*(x - xo)/(fv - fvo))
      fvo, fv = (fv, f(x))
    end
  end
  return x
end

function gauss_seidel(f1, f2, xo, yo, init1, init2) 
  x, y = xo, yo
  for i in 1:1000
    println(string("Seidel: (x, y) = ", [x, y]))
    xo, x = x, secant(p -> f1(p,y), init1)
    yo, y = y, secant(p -> f2(p,x), init2)

    Δx = x - xo
    Δy = y - yo
    if norm([Δx Δy]) < 1e-8
      break
    end
  end
  return [x, y]
end

f1(p1,p2) = dπ(1, v, [p1 p2])
f2(p1,p2) = dπ(2, v, [p2 p1])
t = now()
@time res3 = gauss_seidel(f1, f2, 1, 1, [1;2], [1;2])
res3 = DataFrame(pa = res3[1], pb = res3[2], time = now() - t)

# Seidel is faster, but by almost nothing, still intriguing because there is way more interations

# Exercise 3! - 4

function ex4(v, init)
  println(v)
  po = p = init 
  for i in 1:1000
    po, p = p, [1 / (1 - D(1, v, p)) ; 1 / (1 - D(2, v, p))]
    if norm(p - po) < 1e-8
      break
    end
  end
  return p
end

t = now()
res4 = ex4(v, [0; 5])
res4 = DataFrame(pa = res4[1], pb = res4[2], time = now() - t)

# Exercise 5

vb = 0:0.2:3
v5  = vcat.(Ref(2), vb)

res5_GS = DataFrame(vb = Float64[], A = Float64[], B = Float64[]) 
@time for v in v5
  f1(p1,p2) = dπ(1, v, [p1 p2])
  f2(p1,p2) = dπ(2, v, [p2 p1])
  p1, p2  = gauss_seidel(f1, f2, 1, 1, [1;2], [1;2])
  push!(res5_GS, [v[2] p1 p2])
end

res5_broyden = DataFrame(vb = Float64[], A = Float64[], B = Float64[]) 
@time for v in v5
  f(p) = [dπ(1, v, p), dπ(2, v, p)]
  J = Jacobian(f, v, [2; 2])
  push!(res5_broyden, vcat(v[2], broyden(f, [[1;1], [2;2]] , J)))
end

res5_GS
res5_broyden

#Now it easier to compare the two methods since te computation is more demanding
# and broyden seems to be faster. Probably due to less interations.
# But Broyden finds strange solutions for pb close to zero 

ex4.(v5, Ref([2; 2]))

# Ex4 method goes crazy for p[:b] > 2, why?

res5 = stack(res5_GS, [:A, :B])
res5_plot = plot(res5, x = :vb, y = :value, color = :variable, Geom.line, Guide.ylabel("Price"),
     Guide.colorkey("Firm"))

### RECAP
# EX1:
res
# EX2:
res2 # Second fastest - Unstable vb < 0.4
# EX3:
res3 # Slowest - The most realiable?
# EX4:
res4 # Actually the fastest - Unstable vb > 2
# EX5:
res5_plot
