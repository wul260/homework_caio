using DataFrames
using LinearAlgebra

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

function Jacobian(v, p)
  h  = 1e-5
  # Calculating Jacobian wrt a
  Ja = (dπ(p + [h; 0]) - dπ(p))./h
  # Calculating Jacobian wrt a
  Jb = (dπ(p + [0; h]) - dπ(p))./h
  # Full Jacobian
  return hcat(Ja, Jb)
end

Jacobian(v, p)

function broyden(f, init, J)
  xo , x  = init
  fvo, fv = f.(init)
  invJ = inv(J)

  for i in 1:100
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

@time res = broyden(dπ, [[1;1], [2;2]] , Jacobian(v, p))

# Exercise 3
function secant(f, init)
  xo, x   = init
  fvo, fv = f.(init)

  for i in 1:100
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
  for i in 1:100
    println(string("Seidel: (x, y) = ", [x, y]))
    xo, x = x, secant(p -> f1(p,y), init1)
    yo, y = y, secant(p -> f2(p,x), init2)

    Δx = x - xo
    Δy = y - yo
    if norm([Δx Δy]) < 1e-8
      break
    end
  end
  return x, y
end

f1(p1,p2) = dπ(1, v, [p1 p2])
f2(p1,p2) = dπ(2, v, [p2 p1])
@time gauss_seidel(f1, f2, 1, 1, [1;2], [1;2])

# Seidel is faster, but by almost nothing, still intriguing because there is way more interations
