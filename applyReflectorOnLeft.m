function Y = applyReflectorOnLeft(u, X)
Y = X-2*u*(u'*X);
