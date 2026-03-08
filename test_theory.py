import torch as th
x = th.randn(10, requires_grad=True)
y = th.randn(10)
# Strict Bregman
diff = x - y
D_strict = th.cosh(y) - th.cosh(x) + th.sinh(x) * diff
D_strict.mean().backward()
print("Strict grad wrt x:", x.grad)

x.grad.zero_()
D_pseudo = th.cosh(diff) - 1
D_pseudo.mean().backward()
print("Pseudo grad wrt x:", x.grad)
