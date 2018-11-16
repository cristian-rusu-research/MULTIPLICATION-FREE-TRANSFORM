function Y = applyReflectorOnRight(X, u)
Y = X-2*(X*u)*u';
