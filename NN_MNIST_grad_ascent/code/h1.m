function [ h, grad_h ] = h1( a )
%ACTIVATION_FUNCTION Summary of this function goes here
% log(1 + exp(a)) = log(exp(0) + exp(a)) = 0 + log(exp(-0) + exp(a-0))

% numerical stable implementation
m = max(0, a);  % rectifier activation function
h = m + log(exp(-m) + exp(a-m));

% compute the gradient of: h(a) = log(1 + exp(a));
% h'(a) = (1 / (1 + exp(a))) * exp(a) =>
% h'(a) = exp(a) / (1 + exp(a)) =>
% if we divide the numerator and the denominator with epx(a):
% h'(a) = exp(a)/exp(a) / (1 + exp(a)) / exp(a) =>
% h'(a) = 1 / (1/exp(a) + 1) =>
% h'(a) = 1 / (exp(-a) +1) -> sigmoid function
if nargout > 1
    grad_h = 1 ./ (1 + exp(-a));
end

end
