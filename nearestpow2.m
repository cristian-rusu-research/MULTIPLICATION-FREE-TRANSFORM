function y = nearestpow2(x)

a = nextpow2(x); % Find the nearest power of 2 higher than or equal to x.

if x-2^(a-1) == 2^a-x % Check if x lies in the middle of 2^(a-1) and 2^a
   y = [a-1 a];
else % If not, find the closer of the two powers
   [j, k] = intersect([x-2^(a-1) 2^a-x], min(x-2^(a-1), 2^a-x));
   if k == 1 % And define y accordingly
      y = a-1;
   else
      y = a;
   end 
end

y = y(1);
